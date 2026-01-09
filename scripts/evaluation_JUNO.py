import os
import json
import sys
from argparse import ArgumentParser
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch_scatter import scatter

sys.path.append("./")
from hpst.utils.options import Options
from hpst.trainers.point_set_trainer import PointSetTrainer


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def _get_scatter_reduce(model: PointSetTrainer) -> str:
    return str(getattr(model, "scatter_reduce", "mean"))


# -------------------------
# Inference: full-set (for metrics) and single-event (for visualization)
# -------------------------

def pmt_overall_accuracy_2ch(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute PMT-level overall accuracy for 2-channel binary labels.
    Expects y_true and y_pred shapes like (M,2) or (N,2) containing 0/1.
    Returns fraction of matching entries (float in [0,1]).
    """
    if y_true is None or y_pred is None:
        return 0.0
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.size == 0:
        return 0.0
    # broadcast/reshape safety: require last dim == 2 or treat flat as single-sample
    if yt.ndim == 1:
        yt = yt.reshape(1, -1)
    if yp.ndim == 1:
        yp = yp.reshape(1, -1)
    # ensure same shape
    if yt.shape != yp.shape:
        # try to coerce if second dim mismatched
        min0 = min(yt.shape[0], yp.shape[0])
        yt = yt[:min0]
        yp = yp[:min0]
    return float((yt == yp).mean())

@torch.no_grad()
def run_inference_collect_full(
    model: PointSetTrainer,
    device: torch.device,
    max_batches: int = -1,
) -> Dict[str, np.ndarray]:
    model.eval()
    model.to(device)

    if getattr(model, "testing_dataset", None) is not None:
        loader = model.test_dataloader()
        split_name = "test"
    else:
        loader = model.val_dataloader()
        split_name = "val"

    all_true: List[torch.Tensor] = []
    all_prob: List[torch.Tensor] = []
    all_pred: List[torch.Tensor] = []

    for bi, batch in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break

        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        hit_features = batch["hit_features"]   # (N,5)
        hit_pmt_ids = batch["hit_pmt_ids"]     # (N,)
        pmt_labels = batch["pmt_labels"]       # (M,2)

        coords = hit_features[:, :4]
        feats = hit_features[:, 4:]
        ev_batch = batch["batch"]

        logits_hit = model.forward(coords, feats, ev_batch)  # (N,2)
        logits_pmt = scatter(
            logits_hit, hit_pmt_ids, dim=0, reduce=_get_scatter_reduce(model), dim_size=pmt_labels.shape[0]
        )  # (M,2)

        prob = torch.sigmoid(logits_pmt)
        pred = (prob > 0.5).to(dtype=torch.int64)
        true = (pmt_labels > 0.5).to(dtype=torch.int64)

        all_true.append(true.detach().cpu())
        all_prob.append(prob.detach().cpu())
        all_pred.append(pred.detach().cpu())

    y_true_bin = torch.cat(all_true, dim=0).numpy()
    y_prob = torch.cat(all_prob, dim=0).numpy().astype(np.float32)
    y_pred_bin = torch.cat(all_pred, dim=0).numpy()

    return {
        "split": np.array([split_name]),
        "y_true_bin": y_true_bin,
        "y_prob": y_prob,
        "y_pred_bin": y_pred_bin,
    }


@torch.no_grad()
def run_inference_single_event_from_dataset(
    model: PointSetTrainer,
    device: torch.device,
    event_idx: int = 0,
    split: str = "val",  # "val" | "test"
) -> Dict[str, np.ndarray]:
    model.eval()
    model.to(device)

    if split == "test" and getattr(model, "testing_dataset", None) is not None:
        ds = model.testing_dataset
        split_name = "test"
    else:
        ds = model.validation_dataset
        split_name = "val"

    # unwrap Subset
    if hasattr(ds, "dataset") and hasattr(ds, "indices"):
        base_ds = ds.dataset
        indices = ds.indices
        actual_idx = int(indices[event_idx]) if event_idx < len(indices) else int(indices[0])
    else:
        base_ds = ds
        actual_idx = int(event_idx)

    sample = base_ds[actual_idx]

    def as_tensor(x, dtype, dev):
        if torch.is_tensor(x):
            return x.clone().detach().to(dev).to(dtype)
        return torch.tensor(x, dtype=dtype, device=dev)

    hit_features = as_tensor(sample["hit_features"], torch.float32, device)  # (N,5)
    hit_pmt_ids = as_tensor(sample["hit_pmt_ids"], torch.int64, device)      # (N,)
    pmt_labels = as_tensor(sample["pmt_labels"], torch.float32, device)      # (M,2)

    # PMT-level npe sum from hit-level feature column 4
    hit_npe = hit_features[:, 4].clamp_min(0.0)  # (N,)
    pmt_npe = scatter(hit_npe, hit_pmt_ids, dim=0, reduce="sum", dim_size=pmt_labels.shape[0])  # (M,)
    pmt_npe_np = pmt_npe.detach().cpu().numpy().astype(np.float32)

    unique_pmt_ids = sample.get("unique_pmt_ids", None)
    if unique_pmt_ids is not None:
        if torch.is_tensor(unique_pmt_ids):
            unique_pmt_ids_t = unique_pmt_ids.clone().detach().cpu().to(torch.int64)
        else:
            unique_pmt_ids_t = torch.tensor(unique_pmt_ids, dtype=torch.int64, device="cpu")
    else:
        unique_pmt_ids_t = None

    coords = hit_features[:, :4]
    feats = hit_features[:, 4:]
    ev_batch = torch.zeros(coords.shape[0], dtype=torch.int64, device=device)

    logits_hit = model.forward(coords, feats, ev_batch)
    logits_pmt = scatter(
        logits_hit, hit_pmt_ids, dim=0, reduce=_get_scatter_reduce(model), dim_size=pmt_labels.shape[0]
    )
    prob = torch.sigmoid(logits_pmt).detach().cpu().numpy().astype(np.float32)
    pred = (prob > 0.5).astype(np.int64)
    true = (pmt_labels.detach().cpu().numpy() > 0.5).astype(np.int64)

    return {
        "split": np.array([split_name]),
        "event_idx": np.array([actual_idx], dtype=np.int64),
        "y_true_bin": true,
        "y_prob": prob,
        "y_pred_bin": pred,
        "unique_pmt_ids": unique_pmt_ids_t.numpy() if unique_pmt_ids_t is not None else None,
        "pmt_npe": pmt_npe_np,
    }

@torch.no_grad()
def run_inference_collect_test_event_accuracy(
    model: PointSetTrainer,
    device: torch.device,
    max_batches: int = -1,
) -> Dict[str, np.ndarray]:
    """
    Compute per-event PMT overall accuracy by calling the single-event inference
    for each event in the test/val dataset. Returns arrays:
      - event_idx: global dataset index for each event
      - acc: per-event PMT accuracy (mean over PMT*2)
    This is slightly slower (one forward per event) but robust and guarantees a full mapping.
    """
    model.eval()
    model.to(device)

    # choose dataset (test preferred)
    if getattr(model, "testing_dataset", None) is not None:
        ds = model.testing_dataset
        split = "test"
    else:
        ds = model.validation_dataset
        split = "val"

    n_events = len(ds)
    if max_batches and max_batches > 0:
        # interpret max_batches as max events when using this function
        n_events = min(n_events, int(max_batches))

    ev_ids: List[int] = []
    accs: List[float] = []

    for local_i in range(n_events):
        ev = run_inference_single_event_from_dataset(model, device=device, event_idx=local_i, split=split)
        actual_idx = int(ev["event_idx"][0])
        y_true = ev["y_true_bin"].astype(np.int64)
        y_pred = ev["y_pred_bin"].astype(np.int64)
        acc = pmt_overall_accuracy_2ch(y_true, y_pred)
        ev_ids.append(actual_idx)
        accs.append(float(acc))

    ev_ids_np = np.array(ev_ids, dtype=np.int64)
    accs_np = np.array(accs, dtype=np.float32)
    return {"event_idx": ev_ids_np, "acc": accs_np}

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-t", "--training_file",
        type=str,
        default="/disk_pool1/houyh/data/scattered",
        help="JUNO dataset root dir"
    )
    parser.add_argument(
        "-o", "--options_file",
        type=str,
        default="config/pst/pst_small_tune.json",
        help="JSON config used in training"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/disk_pool1/houyh/results/Noise_run/version_2/checkpoints/last.ckpt",
        help="Path to Lightning checkpoint (*.ckpt)."
    )
    # NOTE: evaluation 固定输出到 results/JUNO_run/version0
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/disk_pool1/houyh/results/Noise_run/version_2",
        help="Output directory for evaluation artifacts (.npz/.txt)."
    )
    # NEW: only export a single-event npz, and use seed as the event id suffix.
    parser.add_argument(
        "--single_only",
        action="store_true",
        help="If set, only export eval_single_event_{seed}.npz and skip eval_full.npz.",
    )

    parser.add_argument("--max_batches", type=int, default=-1, help="Limit number of batches for debug.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--event_idx", type=int, default=-1, help="Single event index for visualization export; -1 means random.")
    parser.add_argument("--seed", type=int, default=279, help="RNG seed for random event selection.")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)

    # Options
    options = Options(args.training_file)
    if args.options_file:
        with open(args.options_file, "r", encoding="utf-8") as f:
            options.update_options(json.load(f))

    # Device
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Load model
    model = PointSetTrainer.load_from_checkpoint(args.ckpt, options=options)

    # Full inference (for metrics/plots input)
    if not args.single_only:
        data = run_inference_collect_full(model, device=device, max_batches=args.max_batches)
        split = str(data["split"][0])
        y_true2 = data["y_true_bin"]
        y_prob2 = data["y_prob"]
        y_pred2 = data["y_pred_bin"]

        # np.savez_compressed(
        #     os.path.join(out_dir, "eval_full.npz"),
        #     split=np.array([split]),
        #     y_true_bin=y_true2.astype(np.int8),
        #     y_pred_bin=y_pred2.astype(np.int8),
        #     y_prob=y_prob2.astype(np.float32),
        # )

    # Single-event inference (for Mollweide plots input)
    plot_split = "test" if getattr(model, "testing_dataset", None) is not None else "val"
    ds = model.testing_dataset if plot_split == "test" else model.validation_dataset
    n_events = len(ds)

    if args.event_idx >= 0:
        chosen = int(args.event_idx)
    else:
        rng = np.random.default_rng(int(args.seed))
        chosen = int(rng.integers(0, n_events))

    ev = run_inference_single_event_from_dataset(model, device=device, event_idx=chosen, split=plot_split)

    # NEW: output name controlled by single_only; suffix uses seed
    single_name = (
        f"eval_single_event_{int(args.seed)}.npz"
        if args.single_only
        else f"eval_single_event_{plot_split}.npz"
    )

    np.savez_compressed(
        os.path.join(out_dir, single_name),
        split=np.array([plot_split]),
        event_idx=ev["event_idx"].astype(np.int64),
        y_true_bin=ev["y_true_bin"].astype(np.int8),
        y_pred_bin=ev["y_pred_bin"].astype(np.int8),
        y_prob=ev["y_prob"].astype(np.float32),
        unique_pmt_ids=ev["unique_pmt_ids"],   # may be None
        pmt_npe=ev["pmt_npe"].astype(np.float32),
    )

    # if getattr(model, "testing_dataset", None) is not None:
    #     acc_pack = run_inference_collect_test_event_accuracy(model, device=device, max_batches=args.max_batches)
    #     np.savez_compressed(os.path.join(out_dir, "eval_test_event_acc.npz"), event_idx=acc_pack["event_idx"], acc=acc_pack["acc"])
    #     print(f"[OK] saved per-event acc cache: {os.path.join(out_dir, 'eval_test_event_acc.npz')}")
        
    if args.single_only:
        print(f"[OK] single-event artifact saved to: {os.path.join(out_dir, single_name)}")
        print(f"[OK] split(event)={plot_split}, event_idx={int(ev['event_idx'][0])}, seed={int(args.seed)}")
    else:
        print(f"[OK] evaluation artifacts saved to: {out_dir}")
        print(f"[OK] split(event)={plot_split}, event_idx={int(ev['event_idx'][0])}")


if __name__ == "__main__":
    main()
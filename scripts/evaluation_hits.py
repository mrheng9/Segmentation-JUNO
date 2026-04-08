# import os
# import json
# import sys
# from argparse import ArgumentParser
# from typing import Dict, List

# import numpy as np
# import torch

# sys.path.append("./")
# from hpst.utils.options import Options
# from hpst.trainers.point_set_trainer import PointSetTrainer


# def ensure_dir(p: str) -> str:
#     os.makedirs(p, exist_ok=True)
#     return p


# @torch.no_grad()
# def run_inference_collect_full_apme(
#     model: PointSetTrainer,
#     device: torch.device,
#     max_batches: int = -1,
# ) -> Dict[str, np.ndarray]:
#     model.eval()
#     model.to(device)

#     if getattr(model, "testing_dataset", None) is not None:
#         loader = model.test_dataloader()
#         split_name = "test"
#     else:
#         loader = model.val_dataloader()
#         split_name = "val"

#     ys_true: List[torch.Tensor] = []
#     ys_prob: List[torch.Tensor] = []
#     ys_pred: List[torch.Tensor] = []

#     for bi, batch in enumerate(loader):
#         if max_batches > 0 and bi >= max_batches:
#             break

#         batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

#         hit_features = batch["hit_features"]  # (N,5)
#         hit_labels = batch["hit_labels"]      # (N,)
#         ev_batch = batch["batch"]             # (N,)

#         coords = hit_features[:, :4]
#         feats = hit_features[:, 4:]

#         logits_hit = model.forward(coords, feats, ev_batch)  # (N,2)
#         prob_me = torch.softmax(logits_hit, dim=-1)[:, 1]    # (N,)
#         pred = (prob_me > 0.5).to(torch.int64)

#         ys_true.append(hit_labels.detach().cpu().to(torch.int64))
#         ys_prob.append(prob_me.detach().cpu().to(torch.float32))
#         ys_pred.append(pred.detach().cpu().to(torch.int64))

#     y_true = torch.cat(ys_true, dim=0).numpy().astype(np.int64)
#     y_prob = torch.cat(ys_prob, dim=0).numpy().astype(np.float32)
#     y_pred = torch.cat(ys_pred, dim=0).numpy().astype(np.int64)

#     return {
#         "split": np.array([split_name]),
#         "y_true": y_true,
#         "y_prob": y_prob,
#         "y_pred": y_pred,
#     }


# def main():
#     parser = ArgumentParser()
#     parser.add_argument(
#         "-t", "--training_file",
#         type=str,
#         default="/disk_pool1/houyh/data/AP_ME",
#         help="APME dataset root dir (contains hits/)",
#     )
#     parser.add_argument(
#         "-o", "--options_file",
#         type=str,
#         default="config/pst/pst_small_tune.json",
#         help="JSON config used in training",
#     )
#     parser.add_argument(
#         "--ckpt",
#         type=str,
#         default="/home/houyh/Segmentation-JUNO-C/results/ME_AP_run_new/version_1/checkpoints/epoch=15-step=3000.ckpt",
#         help="Path to Lightning checkpoint (*.ckpt).",
#     )
#     parser.add_argument(
#         "--out_dir",
#         type=str,
#         default="/home/houyh/Segmentation-JUNO-C/results/ME_AP_run_new",
#         help="Output directory for evaluation artifacts (.npz).",
#     )
#     parser.add_argument("--max_batches", type=int, default=-1, help="Limit number of batches for debug.")
#     parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
#     args = parser.parse_args()

#     out_dir = ensure_dir(args.out_dir)

#     # Options (IMPORTANT: must set dataset type to apme)
#     options = Options(args.training_file)
#     if args.options_file:
#         with open(args.options_file, "r", encoding="utf-8") as f:
#             options.update_options(json.load(f))
#     options.juno_dataset_type = "apme"

#     # Device
#     if args.device == "cuda" and not torch.cuda.is_available():
#         device = torch.device("cpu")
#     else:
#         device = torch.device(args.device)

#     # Load model
#     model = PointSetTrainer.load_from_checkpoint(args.ckpt, options=options)

#     # Full inference (hit-level)
#     data = run_inference_collect_full_apme(model, device=device, max_batches=args.max_batches)
#     split = str(data["split"][0])

#     np.savez_compressed(
#         os.path.join(out_dir, "eval_full_apme_3000.npz"),
#         split=np.array([split]),
#         y_true=data["y_true"].astype(np.int64),
#         y_pred=data["y_pred"].astype(np.int64),
#         y_prob=data["y_prob"].astype(np.float32),
#     )

#     print(f"[OK] saved: {os.path.join(out_dir, 'eval_full_apme_3000.npz')}  split={split}  N={data['y_true'].size}")


# if __name__ == "__main__":
#     main()

import os
import json
import sys
from argparse import ArgumentParser
from typing import Dict, List

import numpy as np
import torch

sys.path.append("./")
from hpst.utils.options import Options
from hpst.trainers.point_set_trainer import PointSetTrainer


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


@torch.no_grad()
def run_inference_collect_full_apme(
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

    ys_true: List[torch.Tensor] = []
    ys_prob: List[torch.Tensor] = []
    ys_pred: List[torch.Tensor] = []

    xs_chunk: List[torch.Tensor] = []
    xs_local: List[torch.Tensor] = []

    for bi, batch in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break

        # move tensors to device (non-tensors kept as-is)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        if "hit_features" not in batch or "hit_labels" not in batch or "batch" not in batch:
            raise KeyError("batch missing required keys: hit_features/hit_labels/batch")

        hit_features = batch["hit_features"]  # (N,5)
        hit_labels = batch["hit_labels"]      # (N,)
        ev_batch = batch["batch"]             # (N,)

        coords = hit_features[:, :4]
        feats = hit_features[:, 4:]

        logits_hit = model.forward(coords, feats, ev_batch)  # (N,2)
        prob_me = torch.softmax(logits_hit, dim=-1)[:, 1]    # (N,)
        pred = (prob_me > 0.5).to(torch.int64)

        ys_true.append(hit_labels.detach().cpu().to(torch.int64))
        ys_prob.append(prob_me.detach().cpu().to(torch.float32))
        ys_pred.append(pred.detach().cpu().to(torch.int64))

        # IMPORTANT: require per-hit event ids from collate_fn_hitcls
        if "chunk_index" not in batch or "local_event" not in batch:
            raise KeyError(
                "batch missing chunk_index/local_event. "
                "Fix Dataset+collate_fn_hitcls to provide per-hit event ids."
            )

        xs_chunk.append(batch["chunk_index"].detach().cpu().to(torch.int64).reshape(-1))
        xs_local.append(batch["local_event"].detach().cpu().to(torch.int64).reshape(-1))

    y_true = torch.cat(ys_true, dim=0).numpy().astype(np.int64)
    y_prob = torch.cat(ys_prob, dim=0).numpy().astype(np.float32)
    y_pred = torch.cat(ys_pred, dim=0).numpy().astype(np.int64)

    chunk_index = torch.cat(xs_chunk, dim=0).numpy().astype(np.int64)
    local_event = torch.cat(xs_local, dim=0).numpy().astype(np.int64)

    if not (y_true.shape[0] == y_prob.shape[0] == y_pred.shape[0] == chunk_index.shape[0] == local_event.shape[0]):
        raise ValueError(
            f"Length mismatch: y_true={y_true.shape} y_prob={y_prob.shape} y_pred={y_pred.shape} "
            f"chunk_index={chunk_index.shape} local_event={local_event.shape}"
        )

    return {
        "split": np.array([split_name]),
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "chunk_index": chunk_index,
        "local_event": local_event,
    }


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-t", "--training_file",
        type=str,
        default="/disk_pool1/houyh/data/AP_ME",
        help="APME dataset root dir (contains hits/)",
    )
    parser.add_argument(
        "-o", "--options_file",
        type=str,
        default="config/pst/pst_small_tune.json",
        help="JSON config used in training",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/home/houyh/Segmentation-JUNO-C/results/ME_AP_run_new/version_1/checkpoints/epoch=15-step=3000.ckpt",
        help="Path to Lightning checkpoint (*.ckpt).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/houyh/Segmentation-JUNO-C/results/ME_AP_run_new",
        help="Output directory for evaluation artifacts (.npz).",
    )
    parser.add_argument("--max_batches", type=int, default=-1, help="Limit number of batches for debug.")
    parser.add_argument("--device", type=str, default="cpu", help="cuda or cpu")
    parser.add_argument("--out_name", type=str, default="eval_full_apme_3000_new2.npz")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)

    options = Options(args.training_file)
    if args.options_file:
        with open(args.options_file, "r", encoding="utf-8") as f:
            options.update_options(json.load(f))
    options.juno_dataset_type = "apme"

    # device (safe)
    if args.device.lower() == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("[warn] CUDA requested but not available; using CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    # load model
    model = PointSetTrainer.load_from_checkpoint(args.ckpt, options=options)

    data = run_inference_collect_full_apme(model, device=device, max_batches=args.max_batches)
    split = str(data["split"][0])

    out_path = os.path.join(out_dir, args.out_name)
    np.savez_compressed(out_path, **data)

    print(f"[OK] saved: {out_path} split={split} N={int(data['y_true'].size)} +event_ids")


if __name__ == "__main__":
    main()
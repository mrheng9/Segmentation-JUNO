import os
import json
from argparse import ArgumentParser
from typing import Dict, Tuple, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def compute_binary_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, fp, tn, fn


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp, fp, tn, fn = compute_binary_counts(y_true, y_pred)
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = (2 * prec * rec) / max(1e-12, (prec + rec))
    return {
        "tp": float(tp), "fp": float(fp), "tn": float(tn), "fn": float(fn),
        "accuracy": float(acc),
        "precision(ME)": float(prec),
        "recall(ME)": float(rec),
        "f1(ME)": float(f1),
    }


def plot_confusion(cm: np.ndarray, labels, out_path: str, title: str, normalize: bool = False):
    fig, ax = plt.subplots(figsize=(6.3, 5.4))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    if normalize:
        disp.plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=True)
    else:
        disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=True)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_roc_pr(y_true: np.ndarray, y_prob: np.ndarray, out_dir: str):
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5.2, 4.7))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.5f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC (positive=ME)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "roc_me.png"), dpi=170)
    plt.close(fig)

    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5.2, 4.7))
    ax.plot(rec, prec, label=f"AP={ap:.5f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR curve (positive=ME)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pr_me.png"), dpi=170)
    plt.close(fig)

    return float(roc_auc), float(ap)


def plot_prob_hist(y_true: np.ndarray, y_prob: np.ndarray, out_dir: str):
    y_true = y_true.astype(np.int64)
    y_prob = y_prob.astype(np.float32)

    p0 = y_prob[y_true == 0]
    p1 = y_prob[y_true == 1]

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    bins = np.linspace(0.0, 1.0, 51)
    ax.hist(p0, bins=bins, alpha=0.65, label=f"AP (0), n={p0.size}", color="tab:blue", density=True)
    ax.hist(p1, bins=bins, alpha=0.65, label=f"ME (1), n={p1.size}", color="tab:orange", density=True)
    ax.set_xlabel("Predicted P(ME)")
    ax.set_ylabel("Density")
    ax.set_title("Hit-level probability histogram by class")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "prob_hist_by_class.png"), dpi=180)
    plt.close(fig)


def plot_threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray, out_dir: str):
    y_true = y_true.astype(np.int64)
    y_prob = y_prob.astype(np.float32)

    thrs = np.linspace(0.0, 1.0, 201)
    accs = []
    f1s = []
    precs = []
    recs = []

    for thr in thrs:
        pred = (y_prob >= thr).astype(np.int64)
        accs.append(accuracy_score(y_true, pred))
        f1s.append(f1_score(y_true, pred, zero_division=0))
        precs.append(precision_score(y_true, pred, zero_division=0))
        recs.append(recall_score(y_true, pred, zero_division=0))

    accs = np.asarray(accs)
    f1s = np.asarray(f1s)
    precs = np.asarray(precs)
    recs = np.asarray(recs)

    best_i = int(np.argmax(f1s))
    best_thr = float(thrs[best_i])
    best = {
        "best_thr_by_f1": best_thr,
        "best_f1": float(f1s[best_i]),
        "acc_at_best_f1": float(accs[best_i]),
        "prec_at_best_f1": float(precs[best_i]),
        "rec_at_best_f1": float(recs[best_i]),
    }

    fig, ax = plt.subplots(figsize=(7.0, 4.9))
    ax.plot(thrs, accs, label="Accuracy")
    ax.plot(thrs, f1s, label="F1(ME)")
    ax.plot(thrs, precs, label="Precision(ME)", alpha=0.9)
    ax.plot(thrs, recs, label="Recall(ME)", alpha=0.9)
    ax.axvline(best_thr, color="k", linestyle="--", linewidth=1.0, label=f"best thr={best_thr:.3f}")
    ax.set_xlabel("Threshold on P(ME)")
    ax.set_ylabel("Score")
    ax.set_title("Threshold sweep (hit-level)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "threshold_sweep.png"), dpi=180)
    plt.close(fig)

    return best


def load_ap_event_nhits(ap_raw_dir: str) -> Dict[Tuple[int, int], int]:
    """
    Load nhits for each AP raw event from AP_chunk_*.npy
    Return: (chunk_index, local_event) -> nhits
    """
    out: Dict[Tuple[int, int], int] = {}
    for name in sorted(os.listdir(ap_raw_dir)):
        if not (name.startswith("AP_chunk_") and name.endswith(".npy")):
            continue
        try:
            ci = int(name[len("AP_chunk_"):-len(".npy")])
        except ValueError:
            continue

        path = os.path.join(ap_raw_dir, name)
        arr = np.load(path, allow_pickle=True)
        if arr.ndim != 1 or arr.dtype != object:
            raise ValueError(f"Bad AP chunk format: {path} shape={arr.shape} dtype={arr.dtype}")

        for ei in range(arr.shape[0]):
            ev = np.array(arr[ei], dtype=object)
            if ev.ndim != 2 or ev.shape[1] != 3:
                raise ValueError(f"Bad raw event table: {path} local_event={ei}, shape={ev.shape}")
            out[(ci, ei)] = int(ev.shape[0])
    return out

def compute_event_class_accuracy_from_hits(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    chunk_index: np.ndarray,
    local_event: np.ndarray,
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """
    Per event, compute per-class hit accuracy:
      acc_ap: accuracy on hits with true label 0 (AP)
      acc_me: accuracy on hits with true label 1 (ME)
    Returns: (chunk_index, local_event) -> {"acc_ap":..., "acc_me":..., "n_ap":..., "n_me":...}
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    chunk_index = chunk_index.astype(np.int64)
    local_event = local_event.astype(np.int64)

    if not (y_true.shape == y_pred.shape == chunk_index.shape == local_event.shape):
        raise ValueError("Shapes mismatch among y_true/y_pred/chunk_index/local_event")

    key = chunk_index * 1_000_000 + local_event
    uniq = np.unique(key)

    out: Dict[Tuple[int, int], Dict[str, float]] = {}
    for k in uniq:
        m = (key == k)
        yt = y_true[m]
        yp = y_pred[m]

        m_ap = (yt == 0)
        m_me = (yt == 1)

        n_ap = int(m_ap.sum())
        n_me = int(m_me.sum())

        acc_ap = float((yp[m_ap] == 0).mean()) if n_ap > 0 else float("nan")
        acc_me = float((yp[m_me] == 1).mean()) if n_me > 0 else float("nan")

        ci = int(k // 1_000_000)
        ei = int(k % 1_000_000)
        out[(ci, ei)] = {"acc_ap": acc_ap, "acc_me": acc_me, "n_ap": float(n_ap), "n_me": float(n_me)}

    return out

def compute_event_balanced_accuracy_from_hits(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    chunk_index: np.ndarray,
    local_event: np.ndarray,
) -> Dict[Tuple[int, int], float]:
    """
    For each event (chunk_index, local_event):
      acc_AP = accuracy on hits with true label 0 within this event
      acc_ME = accuracy on hits with true label 1 within this event
      balanced_acc = (acc_AP + acc_ME)/2  (average over existing classes)
    Returns: (chunk_index, local_event) -> balanced_acc
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    chunk_index = chunk_index.astype(np.int64)
    local_event = local_event.astype(np.int64)

    if not (y_true.shape == y_pred.shape == chunk_index.shape == local_event.shape):
        raise ValueError("Shapes mismatch among y_true/y_pred/chunk_index/local_event")

    key = chunk_index * 1_000_000 + local_event
    uniq = np.unique(key)

    out: Dict[Tuple[int, int], float] = {}
    for k in uniq:
        m = (key == k)
        yt = y_true[m]
        yp = y_pred[m]

        m_ap = (yt == 0)
        m_me = (yt == 1)

        acc_ap = float((yp[m_ap] == 0).mean()) if m_ap.any() else float("nan")
        acc_me = float((yp[m_me] == 1).mean()) if m_me.any() else float("nan")

        vals = [v for v in (acc_ap, acc_me) if np.isfinite(v)]
        bal = float(np.mean(vals)) if vals else float("nan")

        ci = int(k // 1_000_000)
        ei = int(k % 1_000_000)
        out[(ci, ei)] = bal

    return out

def plot_ap_nhits_vs_me_acc(
    ap_event_nhits: Dict[Tuple[int, int], int],
    event_acc: Dict[Tuple[int, int], Dict[str, float]],
    out_dir: str,
    split: str,
    min_me_hits: int = 1,
):
    """
    Scatter: x = AP raw event nhits, y = ME accuracy within that event.
    Only includes events with at least min_me_hits ME hits (otherwise acc_me is NaN).
    """
    xs: List[int] = []
    ys: List[float] = []
    dropped_missing = 0
    dropped_no_me = 0

    for key, nh in ap_event_nhits.items():
        info = event_acc.get(key, None)
        if info is None:
            dropped_missing += 1
            continue

        acc_me = float(info["acc_me"])
        n_me = int(info["n_me"])
        if not np.isfinite(acc_me) or n_me < int(min_me_hits):
            dropped_no_me += 1
            continue

        xs.append(int(nh))
        ys.append(acc_me)

    xs = np.asarray(xs, dtype=np.int64)
    ys = np.asarray(ys, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.scatter(xs, ys, s=10, alpha=0.35, linewidths=0)
    ax.set_xlabel("nhits (AP raw event)")
    ax.set_ylabel("ME accuracy within event")
    ax.set_title(f"ME hit accuracy vs AP event nhits ({split})")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ap_event_nhits_vs_me_acc.png"), dpi=180)
    plt.close(fig)

    with open(os.path.join(out_dir, "ap_event_nhits_vs_me_acc.meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "split": split,
                "num_points": int(xs.size),
                "num_ap_events_total": int(len(ap_event_nhits)),
                "dropped_missing_in_eval": int(dropped_missing),
                "dropped_no_or_few_me_hits": int(dropped_no_me),
                "min_me_hits": int(min_me_hits),
                "x_min": int(xs.min()) if xs.size else None,
                "x_max": int(xs.max()) if xs.size else None,
                "y_min": float(ys.min()) if ys.size else None,
                "y_max": float(ys.max()) if ys.size else None,
            },
            f,
            indent=2,
        )

def plot_ap_nhits_vs_event_balanced_acc(
    ap_event_nhits: Dict[Tuple[int, int], int],
    event_bal_acc: Dict[Tuple[int, int], float],
    out_dir: str,
    split: str,
):
    xs: List[int] = []
    ys: List[float] = []
    missing_in_eval = 0

    for key, nh in ap_event_nhits.items():
        bal = event_bal_acc.get(key, None)
        if bal is None:
            missing_in_eval += 1
            continue
        if not np.isfinite(bal):
            continue
        xs.append(int(nh))
        ys.append(float(bal))

    xs = np.asarray(xs, dtype=np.int64)
    ys = np.asarray(ys, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.scatter(xs, ys, s=10, alpha=0.35, linewidths=0)
    ax.set_xlabel("nhits (AP raw event)")
    ax.set_ylabel("event balanced accuracy = (acc_AP + acc_ME)/2")
    ax.set_title(f"AP event nhits vs balanced hit accuracy ({split})")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ap_event_nhits_vs_balanced_acc.png"), dpi=180)
    plt.close(fig)

    with open(os.path.join(out_dir, "ap_event_nhits_vs_balanced_acc.meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "split": split,
                "num_points": int(xs.size),
                "num_ap_events_total": int(len(ap_event_nhits)),
                "num_missing_in_eval": int(missing_in_eval),
                "x_min": int(xs.min()) if xs.size else None,
                "x_max": int(xs.max()) if xs.size else None,
                "y_min": float(ys.min()) if ys.size else None,
                "y_max": float(ys.max()) if ys.size else None,
            },
            f,
            indent=2,
        )


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--eval_dir",
        type=str,
        default="/home/houyh/Segmentation-JUNO-C/results/ME_AP_run_new",
        help="Directory containing eval_full_apme.npz produced by evaluation_hits.py",
    )
    parser.add_argument(
        "--npz",
        type=str,
        default="eval_full_apme_3000_new2.npz",
        help="Name of the npz file inside eval_dir",
    )
    parser.add_argument("--out_subdir", type=str, default="plots_apme_3000_new2", help="Output subdir")

    # NEW: for your required plot
    parser.add_argument(
        "--ap_raw_dir",
        type=str,
        default="/disk_pool1/houyh/data/AP_ME_raw/AP_chunks",
        help="Directory containing AP_chunk_*.npy",
    )
    parser.add_argument(
        "--plot_ap_nhits_vs_acc",
        action="store_true",
        help="Plot: x=AP raw event nhits, y=event balanced accuracy=(acc_AP+acc_ME)/2",
    )

    parser.add_argument(
        "--plot_me_acc_vs_ap_nhits",
        action="store_true",
        help="Plot: x=AP raw event nhits, y=ME accuracy (hit-level) within same event",
    )
    parser.add_argument(
        "--min_me_hits",
        type=int,
        default=1,
        help="Minimum ME hits required in an event to include it in ME-accuracy plot",
    )

    args = parser.parse_args()

    eval_dir = str(args.eval_dir)
    out_dir = ensure_dir(os.path.join(eval_dir, args.out_subdir))

    npz_path = os.path.join(eval_dir, args.npz)
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing: {npz_path}. Run evaluation_hits.py first.")

    d = np.load(npz_path, allow_pickle=True)
    split = str(d["split"][0]) if "split" in d.files else "unknown"
    y_true = d["y_true"].astype(np.int64)
    y_prob = d["y_prob"].astype(np.float32)
    y_pred = d["y_pred"].astype(np.int64) if "y_pred" in d.files else (y_prob >= 0.5).astype(np.int64)

    if y_true.ndim != 1 or y_prob.ndim != 1 or y_true.shape[0] != y_prob.shape[0]:
        raise ValueError(f"Bad shapes: y_true={y_true.shape}, y_prob={y_prob.shape}")

    # core metrics
    m = binary_metrics(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plot_confusion(
        cm,
        labels=["AP(0)", "ME(1)"],
        out_path=os.path.join(out_dir, f"confusion_counts_{split}.png"),
        title=f"Confusion Matrix (Counts) - {split}",
        normalize=False,
    )

    cm_norm = cm.astype(np.float32)
    cm_norm = cm_norm / np.maximum(cm_norm.sum(axis=1, keepdims=True), 1.0)
    plot_confusion(
        cm_norm,
        labels=["AP(0)", "ME(1)"],
        out_path=os.path.join(out_dir, f"confusion_norm_{split}.png"),
        title=f"Confusion Matrix (Row-normalized) - {split}",
        normalize=True,
    )

    roc_auc, ap = plot_roc_pr(y_true, y_prob, out_dir=out_dir)
    m["roc_auc(ME)"] = float(roc_auc)
    m["average_precision(ME)"] = float(ap)

    plot_prob_hist(y_true, y_prob, out_dir=out_dir)
    # best = plot_threshold_sweep(y_true, y_prob, out_dir=out_dir)
    # m.update(best)

    print("[debug] overall AP recall =", float(((y_pred == 0) & (y_true == 0)).sum() / max(1, (y_true == 0).sum())))
    print("[debug] overall ME recall =", float(((y_pred == 1) & (y_true == 1)).sum() / max(1, (y_true == 1).sum())))
        # NEW plot requested
    if args.plot_ap_nhits_vs_acc or args.plot_me_acc_vs_ap_nhits:
        if ("chunk_index" not in d.files) or ("local_event" not in d.files):
            raise KeyError(
                "eval npz missing chunk_index/local_event. "
                "Please re-run evaluation_hits.py (scheme B) to generate them."
            )
        chunk_index = d["chunk_index"].astype(np.int64)
        local_event = d["local_event"].astype(np.int64)

        ap_event_nhits = load_ap_event_nhits(args.ap_raw_dir)

        if args.plot_ap_nhits_vs_acc:
            event_bal_acc = compute_event_balanced_accuracy_from_hits(y_true, y_pred, chunk_index, local_event)
            plot_ap_nhits_vs_event_balanced_acc(
                ap_event_nhits=ap_event_nhits,
                event_bal_acc=event_bal_acc,
                out_dir=out_dir,
                split=split,
            )

        if args.plot_me_acc_vs_ap_nhits:
            event_acc = compute_event_class_accuracy_from_hits(y_true, y_pred, chunk_index, local_event)
            plot_ap_nhits_vs_me_acc(
                ap_event_nhits=ap_event_nhits,
                event_acc=event_acc,
                out_dir=out_dir,
                split=split,
                min_me_hits=args.min_me_hits,
            )

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"split": split, "num_hits": int(y_true.size), "metrics": m}, f, indent=2)

    print(f"[OK] saved plots to: {out_dir}")
    print(json.dumps({"split": split, "num_hits": int(y_true.size), "metrics": m}, indent=2))


if __name__ == "__main__":
    main()
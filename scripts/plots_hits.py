import os
import json
from argparse import ArgumentParser
from typing import Dict, Tuple

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


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--eval_dir",
        type=str,
        required=True,
        help="Directory containing eval_full_apme.npz produced by evaluation_JUNO.py",
    )
    parser.add_argument(
        "--npz",
        type=str,
        default="eval_full_apme.npz",
        help="Name of the npz file inside eval_dir",
    )
    parser.add_argument("--out_subdir", type=str, default="plots_apme", help="Output subdir")
    args = parser.parse_args()

    eval_dir = str(args.eval_dir)
    out_dir = ensure_dir(os.path.join(eval_dir, args.out_subdir))

    npz_path = os.path.join(eval_dir, args.npz)
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing: {npz_path}. Run evaluation_JUNO.py first.")

    d = np.load(npz_path, allow_pickle=True)
    split = str(d["split"][0]) if "split" in d.files else "unknown"
    y_true = d["y_true"].astype(np.int64)
    y_prob = d["y_prob"].astype(np.float32)
    y_pred = d["y_pred"].astype(np.int64) if "y_pred" in d.files else (y_prob >= 0.5).astype(np.int64)

    # sanity
    if y_true.ndim != 1 or y_prob.ndim != 1 or y_true.shape[0] != y_prob.shape[0]:
        raise ValueError(f"Bad shapes: y_true={y_true.shape}, y_prob={y_prob.shape}")

    # core metrics at default threshold
    m = binary_metrics(y_true, y_pred)

    # confusion matrices
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plot_confusion(cm, labels=["AP(0)", "ME(1)"], out_path=os.path.join(out_dir, f"confusion_counts_{split}.png"),
                   title=f"Confusion Matrix (Counts) - {split}", normalize=False)

    cm_norm = cm.astype(np.float32)
    cm_norm = cm_norm / np.maximum(cm_norm.sum(axis=1, keepdims=True), 1.0)
    plot_confusion(cm_norm, labels=["AP(0)", "ME(1)"], out_path=os.path.join(out_dir, f"confusion_norm_{split}.png"),
                   title=f"Confusion Matrix (Row-normalized) - {split}", normalize=True)

    # ROC/PR
    roc_auc, ap = plot_roc_pr(y_true, y_prob, out_dir=out_dir)
    m["roc_auc(ME)"] = float(roc_auc)
    m["average_precision(ME)"] = float(ap)

    # probability hist
    plot_prob_hist(y_true, y_prob, out_dir=out_dir)

    # threshold sweep
    best = plot_threshold_sweep(y_true, y_prob, out_dir=out_dir)
    m.update(best)

    # save metrics
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"split": split, "num_hits": int(y_true.size), "metrics": m}, f, indent=2)

    print(f"[OK] saved plots to: {out_dir}")
    print(json.dumps({"split": split, "num_hits": int(y_true.size), "metrics": m}, indent=2))


if __name__ == "__main__":
    main()
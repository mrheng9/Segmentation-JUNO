import os
import sys
import json
from argparse import ArgumentParser
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

import re
from pathlib import Path
ID_RE = re.compile(r".*_(\d+)\.npy$")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

sys.path.append("./")
from hpst.utils.options import Options
from hpst.trainers.point_set_trainer import PointSetTrainer


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b != 0 else 0.0


# -------------------------
# Efficiency / purity (binary)
# -------------------------
def compute_binary_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, fp, tn, fn


def binary_eff_pur_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp, fp, tn, fn = compute_binary_counts(y_true, y_pred)
    eff = _safe_div(tp, tp + fn)
    pur = _safe_div(tp, tp + fp)
    fpr = _safe_div(fp, fp + tn)
    tnr = _safe_div(tn, tn + fp)
    acc = _safe_div(tp + tn, tp + tn + fp + fn)
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "efficiency(recall)": eff,
        "purity(precision)": pur,
        "fpr": fpr,
        "tnr(specificity)": tnr,
        "accuracy": acc,
    }


# -------------------------
# Label mapping: (2,) -> 4-class / 3-class (split 11)
# -------------------------
def bin2to4(y2: np.ndarray) -> np.ndarray:
    y2 = y2.astype(np.int64)
    return y2[:, 0] * 1 + y2[:, 1] * 2


def to3class_split11_pairwise(y_true2: np.ndarray, y_pred2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    def cls_single(v: np.ndarray) -> int:
        a, b = int(v[0]), int(v[1])
        if a == 0 and b == 0:
            return 0
        if a == 1 and b == 0:
            return 1
        if a == 0 and b == 1:
            return 2
        return -1

    yt_list: List[int] = []
    yp_list: List[int] = []

    y_true2 = y_true2.astype(np.int64)
    y_pred2 = y_pred2.astype(np.int64)

    for t, p in zip(y_true2, y_pred2):
        t11 = (t[0] == 1 and t[1] == 1)
        p11 = (p[0] == 1 and p[1] == 1)

        if (not t11) and (not p11):
            yt_list.append(cls_single(t))
            yp_list.append(cls_single(p))
        else:
            yt_list.append(1 if t11 else cls_single(t))
            yp_list.append(1 if p11 else cls_single(p))
            yt_list.append(2 if t11 else cls_single(t))
            yp_list.append(2 if p11 else cls_single(p))

    return np.asarray(yt_list, dtype=np.int64), np.asarray(yp_list, dtype=np.int64)


# -------------------------
# Confusion matrix plot helpers
# -------------------------
def confusion_efficiency(cm_counts: np.ndarray) -> np.ndarray:
    row_sum = cm_counts.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    return cm_counts.astype(np.float32) / row_sum


def confusion_purity(cm_counts: np.ndarray) -> np.ndarray:
    col_sum = cm_counts.sum(axis=0, keepdims=True)
    col_sum[col_sum == 0] = 1
    return cm_counts.astype(np.float32) / col_sum


def plot_confusion_counts(cm: np.ndarray, class_names: List[str], save_path: str, title: str):
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    disp.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=25, colorbar=True)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_confusion_float(cm: np.ndarray, class_names: List[str], save_path: str, title: str):
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    disp.plot(ax=ax, cmap="Blues", values_format=".2f", xticks_rotation=25, colorbar=True)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


# -------------------------
# Binary ROC/PR
# -------------------------
def plot_roc_pr(y_true: np.ndarray, y_prob: np.ndarray, class_name: str, out_dir: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"ROC - {class_name}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"roc_{class_name}.png"), dpi=160)
    plt.close(fig)

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    ax.plot(rec, prec, label=f"AP={ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR - {class_name}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"pr_{class_name}.png"), dpi=160)
    plt.close(fig)


# -------------------------
# PMT projection (Mollweide ellipse)
# -------------------------
def xyz_to_lonlat(coords_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = coords_xyz[:, 0]
    y = coords_xyz[:, 1]
    z = coords_xyz[:, 2]
    r = np.sqrt(x * x + y * y + z * z)
    r[r == 0] = 1.0
    lon = np.arctan2(y, x)
    lat = np.arcsin(np.clip(z / r, -1, 1))
    return lon, lat


def _wrap_lon(lon: np.ndarray) -> np.ndarray:
    return (lon + np.pi) % (2 * np.pi) - np.pi


def _hist2d_on_lonlat(
    lon: np.ndarray,
    lat: np.ndarray,
    nbins_lon: int = 180,
    nbins_lat: int = 90,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon = _wrap_lon(lon)
    lon_edges = np.linspace(-np.pi, np.pi, nbins_lon + 1, dtype=np.float32)
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, nbins_lat + 1, dtype=np.float32)
    H, _, _ = np.histogram2d(lat, lon, bins=[lat_edges, lon_edges])
    return H.astype(np.float32), lon_edges, lat_edges


def _hist2d_weighted_on_lonlat(
    lon: np.ndarray,
    lat: np.ndarray,
    w: np.ndarray,
    nbins_lon: int = 180,
    nbins_lat: int = 90,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon = _wrap_lon(lon)
    lon_edges = np.linspace(-np.pi, np.pi, nbins_lon + 1, dtype=np.float32)
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, nbins_lat + 1, dtype=np.float32)
    H, _, _ = np.histogram2d(lat, lon, bins=[lat_edges, lon_edges], weights=w)
    return H.astype(np.float32), lon_edges, lat_edges


def plot_mollweide_npe_overlay_rgb(
    lon: np.ndarray,
    lat: np.ndarray,
    pmt_npe: np.ndarray,
    mask_e: np.ndarray,
    mask_c: np.ndarray,
    out_path: str,
    title: str,
    nbins_lon: int = 140,
    nbins_lat: int = 70,
    smooth_sigma: float = 2.0,
    use_log1p: bool = True,
    clip_percentile: float = 99.5,
    gamma_intensity: float = 0.85,
    alpha_max: float = 0.95,
    show_colorbar: bool = True,
    color_e: str = "tab:orange",
    color_c: str = "tab:blue",
    background: str = "white",
    alpha_floor: float = 0.0,
    mode: str = "npe",  # "npe" or "count"
):
    lon = _wrap_lon(lon)

    if mode not in ("npe", "count"):
        raise ValueError(f"mode must be 'npe' or 'count', got: {mode}")

    if mode == "npe":
        w = np.clip(pmt_npe.astype(np.float32), 0.0, None)
        if use_log1p:
            w = np.log1p(w)
        He, lon_edges, lat_edges = _hist2d_weighted_on_lonlat(
            lon[mask_e], lat[mask_e], w[mask_e], nbins_lon=nbins_lon, nbins_lat=nbins_lat
        )
        Hc, _, _ = _hist2d_weighted_on_lonlat(
            lon[mask_c], lat[mask_c], w[mask_c], nbins_lon=nbins_lon, nbins_lat=nbins_lat
        )
        cbar_label = "total intensity per bin (sum of log1p(npe))" if use_log1p else "total intensity per bin (sum of npe)"
    else:
        He, lon_edges, lat_edges = _hist2d_on_lonlat(
            lon[mask_e], lat[mask_e], nbins_lon=nbins_lon, nbins_lat=nbins_lat
        )
        Hc, _, _ = _hist2d_on_lonlat(
            lon[mask_c], lat[mask_c], nbins_lon=nbins_lon, nbins_lat=nbins_lat
        )
        He = np.log1p(He)
        Hc = np.log1p(Hc)
        cbar_label = "total intensity per bin (log1p(count))"

    if smooth_sigma and smooth_sigma > 0:
        if gaussian_filter is None:
            print("[WARN] scipy not available; skip gaussian smoothing.")
        else:
            He = gaussian_filter(He, sigma=float(smooth_sigma))
            Hc = gaussian_filter(Hc, sigma=float(smooth_sigma))

    Ht = He + Hc
    vmax = float(np.percentile(Ht.ravel(), clip_percentile)) if Ht.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(Ht.max() if Ht.size else 1.0)
    if vmax <= 0:
        vmax = 1.0

    xe = np.clip(He / vmax, 0.0, 1.0)
    xc = np.clip(Hc / vmax, 0.0, 1.0)
    xt = np.clip(Ht / vmax, 0.0, 1.0)

    if mode == "count":
        t_low = 0.10
        t_high = 0.25
        gate = (xt - t_low) / max(1e-6, (t_high - t_low))
        gate = np.clip(gate, 0.0, 1.0)
    else:
        gate = np.ones_like(xt, dtype=np.float32)

    if gamma_intensity and gamma_intensity != 1.0:
        xe = np.power(xe, float(gamma_intensity))
        xc = np.power(xc, float(gamma_intensity))

    ae = np.clip(xe * gate * float(alpha_max), 0.0, float(alpha_max))
    ac = np.clip(xc * gate * float(alpha_max), 0.0, float(alpha_max))

    if alpha_floor and alpha_floor > 0:
        ae = np.where(He > 0, np.maximum(ae, float(alpha_floor)), 0.0)
        ac = np.where(Hc > 0, np.maximum(ac, float(alpha_floor)), 0.0)

    Lon, Lat = np.meshgrid(lon_edges, lat_edges)

    fig = plt.figure(figsize=(12.5, 6.6))
    ax = fig.add_subplot(1, 1, 1, projection="mollweide")
    ax.set_facecolor(background)

    rgba_e = np.zeros((nbins_lat, nbins_lon, 4), dtype=np.float32)
    rgba_e[..., :3] = matplotlib.colors.to_rgb(color_e)
    rgba_e[..., 3] = ae
    ax.pcolormesh(Lon, Lat, rgba_e, shading="auto")

    rgba_c = np.zeros((nbins_lat, nbins_lon, 4), dtype=np.float32)
    rgba_c[..., :3] = matplotlib.colors.to_rgb(color_c)
    rgba_c[..., 3] = ac
    ax.pcolormesh(Lon, Lat, rgba_c, shading="auto")

    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.20)

    if show_colorbar:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=vmax), cmap="Greys")
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.05)
        cb.set_label(cbar_label)

    import matplotlib.patches as mpatches
    ax.legend(
        handles=[mpatches.Patch(color=color_e, label="e+"), mpatches.Patch(color=color_c, label="C14")],
        loc="lower left",
        frameon=True,
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=240, facecolor=background)
    plt.close(fig)


def plot_mollweide_npe_single_channel(
    lon: np.ndarray,
    lat: np.ndarray,
    pmt_npe: np.ndarray,
    mask: np.ndarray,
    out_path: str,
    title: str,
    nbins_lon: int = 140,
    nbins_lat: int = 70,
    smooth_sigma: float = 2.0,
    use_log1p: bool = True,
    clip_percentile: float = 99.5,
    cmap: str = "Blues",
    background: str = "white",
    show_colorbar: bool = True,
    mode: str = "npe",  # "npe" or "count"
):
    lon = _wrap_lon(lon)

    if mode not in ("npe", "count"):
        raise ValueError(f"mode must be 'npe' or 'count', got: {mode}")

    lon_m = lon[mask]
    lat_m = lat[mask]
    if lon_m.size == 0:
        print(f"[WARN] plot_mollweide_npe_single_channel: empty mask -> {out_path}")
        return

    if mode == "npe":
        w = np.clip(pmt_npe.astype(np.float32), 0.0, None)
        if use_log1p:
            w = np.log1p(w)
        w_m = w[mask]
        H, lon_edges, lat_edges = _hist2d_weighted_on_lonlat(
            lon_m, lat_m, w_m, nbins_lon=nbins_lon, nbins_lat=nbins_lat
        )
        cbar_label = "sum of log1p(npe) per bin" if use_log1p else "sum of npe per bin"
    else:
        H, lon_edges, lat_edges = _hist2d_on_lonlat(
            lon_m, lat_m, nbins_lon=nbins_lon, nbins_lat=nbins_lat
        )
        H = np.log1p(H)
        cbar_label = "log1p(count) per bin"

    if smooth_sigma and smooth_sigma > 0:
        if gaussian_filter is None:
            print("[WARN] scipy not available; skip gaussian smoothing.")
        else:
            H = gaussian_filter(H, sigma=float(smooth_sigma))

    vmax = float(np.percentile(H.ravel(), clip_percentile)) if H.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(H.max() if H.size else 1.0)
    if vmax <= 0:
        vmax = 1.0

    Lon, Lat = np.meshgrid(lon_edges, lat_edges)

    fig = plt.figure(figsize=(12.5, 6.6))
    ax = fig.add_subplot(1, 1, 1, projection="mollweide")
    ax.set_facecolor(background)

    Hm = np.ma.masked_where(H <= 0, H)
    im = ax.pcolormesh(Lon, Lat, Hm, shading="auto", cmap=cmap, vmin=0.0, vmax=vmax)

    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.20)

    if show_colorbar:
        cb = fig.colorbar(im, ax=ax, shrink=0.55, pad=0.05)
        cb.set_label(cbar_label)

    fig.tight_layout()
    fig.savefig(out_path, dpi=240, facecolor=background)
    plt.close(fig)

def plot_mollweide_scatter_points(
    lon: np.ndarray,
    lat: np.ndarray,
    y2: np.ndarray,
    out_path: str,
    title: str,
    s: float = 6.0,
    alpha: float = 0.85,
    show_legend: bool = True,
    background: str = "white",
):
    """
    Scatter-point Mollweide plot.
    lon, lat: (M,)
    y2: (M,2) with channels [eplus, c14] in {0,1}
    """
    lon = _wrap_lon(lon.astype(np.float32))
    lat = lat.astype(np.float32)
    y2 = y2.astype(np.int64)

    if lon.ndim != 1 or lat.ndim != 1 or lon.shape[0] != lat.shape[0]:
        raise ValueError(f"lon/lat shape mismatch: {lon.shape} vs {lat.shape}")
    if y2.ndim != 2 or y2.shape[0] != lon.shape[0] or y2.shape[1] != 2:
        raise ValueError(f"y2 must be (M,2), got {y2.shape}")

    # classes
    mask_none = (y2[:, 0] == 0) & (y2[:, 1] == 0)
    mask_e = (y2[:, 0] == 1) & (y2[:, 1] == 0)
    mask_c = (y2[:, 0] == 0) & (y2[:, 1] == 1)
    mask_both = (y2[:, 0] == 1) & (y2[:, 1] == 1)

    fig = plt.figure(figsize=(12.5, 6.6))
    ax = fig.add_subplot(1, 1, 1, projection="mollweide")
    ax.set_facecolor(background)
    ax.grid(True, alpha=0.20)
    ax.set_title(title, fontsize=13)

    # draw in layers (none first, then signals)
    if np.any(mask_none):
        ax.scatter(lon[mask_none], lat[mask_none], s=s, c="lightgray", alpha=0.25, linewidths=0, label="00 none")
    if np.any(mask_e):
        ax.scatter(lon[mask_e], lat[mask_e], s=s, c="tab:orange", alpha=alpha, linewidths=0, label="10 e+")
    if np.any(mask_c):
        ax.scatter(lon[mask_c], lat[mask_c], s=s, c="tab:blue", alpha=alpha, linewidths=0, label="01 C14")
    if np.any(mask_both):
        ax.scatter(lon[mask_both], lat[mask_both], s=s, c="purple", alpha=alpha, linewidths=0, label="11 both")

    if show_legend:
        ax.legend(loc="lower left", frameon=True, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=240, facecolor=background)
    plt.close(fig)

def pack_y2_from_masks(mask_e: np.ndarray, mask_c: np.ndarray) -> np.ndarray:
    """
    Build (M,2) label array from boolean masks.
    Useful for 'single-class scatter' plots:
      - e+ only:  mask_e=True,  mask_c=False
      - C14 only: mask_e=False, mask_c=True
    """
    mask_e = np.asarray(mask_e, dtype=bool)
    mask_c = np.asarray(mask_c, dtype=bool)
    if mask_e.shape != mask_c.shape:
        raise ValueError(f"mask_e/mask_c shape mismatch: {mask_e.shape} vs {mask_c.shape}")

    y2 = np.zeros((mask_e.shape[0], 2), dtype=np.int64)
    y2[mask_e, 0] = 1
    y2[mask_c, 1] = 1
    return y2
# -------------------------
# time-space analysis for pair events
# -------------------------
def earliest_positive_time_1d(t: np.ndarray) -> float:
    t = np.asarray(t, dtype=np.float32).reshape(-1)
    m = t > 0
    if not np.any(m):
        return float("nan")
    return float(t[m].min())


def extract_dt_eplus_c14_from_tq_event(tq_ev: np.ndarray) -> float:
    """
    tq_ev: (Npmt, 2, 2)
      - dim1 index 0: e+
      - dim1 index 1: C14
      - dim2 feature 0: t
      - dim2 feature 1: q
    Returns abs(min_t_eplus - min_t_c14) using earliest positive t.
    """
    tq_ev = np.asarray(tq_ev)
    if tq_ev.ndim != 3 or tq_ev.shape[1:] != (2, 2):
        raise ValueError(f"Expect tq_ev shape (Npmt,2,2), got {tq_ev.shape}")

    t_e = tq_ev[:, 0, 0]
    t_c = tq_ev[:, 1, 0]
    me = earliest_positive_time_1d(t_e)
    mc = earliest_positive_time_1d(t_c)
    if not np.isfinite(me) or not np.isfinite(mc):
        return float("nan")
    return float(abs(me - mc))


def extract_vertex_distance_from_y_pair_ev(y_pair_ev: np.ndarray) -> float:
    """
    y_pair_ev: (30,)
      first 15: particle1 (e+)
      last  15: particle2 (C14)
    indices inside 15-dim:
      [theta,phi,pid,energy,depE,visE,inX,inY,inZ,inPx,inPy,inPz,exX,exY,exZ]
       0     1   2   3      4    5    6   7   8   ...
    """
    y_pair_ev = np.asarray(y_pair_ev)
    if y_pair_ev.shape[-1] != 30:
        raise ValueError(f"Expect y_pair_ev dim=30, got {y_pair_ev.shape}")

    p1 = y_pair_ev[:15].astype(np.float32)
    p2 = y_pair_ev[15:].astype(np.float32)
    v1 = p1[6:9]
    v2 = p2[6:9]
    return float(np.linalg.norm(v1 - v2))


def pmt_overall_accuracy_2ch(y_true2: np.ndarray, y_pred2: np.ndarray) -> float:
    """
    y_true2/y_pred2: (M,2) in {0,1}
    Returns mean match over all PMT and both channels.
    """
    yt = np.asarray(y_true2, dtype=np.int64)
    yp = np.asarray(y_pred2, dtype=np.int64)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: y_true2 {yt.shape} vs y_pred2 {yp.shape}")
    if yt.size == 0:
        return float("nan")
    return float((yt == yp).mean())


@torch.no_grad()
def plot_pair_vertex_dt_scatter_from_test_loader(
    model: PointSetTrainer,
    device: torch.device,
    out_dir: str,
    max_batches: int = -1,
    alpha: float = 0.75,
    s: float = 10.0,
):
    """
    Compute pair scatter points from model.test_dataloader():
      x = vertex distance between e+ and C14 (from y_pair)
      y = abs( earliest_t(e+) - earliest_t(C14) ) from tq_pair
      color = PMT-level overall accuracy of prediction vs target (mean over M*2)

    Requirements: batch must contain:
      - "tq_pair":   (B, Npmt, 2, 2)  (t,q per particle)
      - "y_pair":    (B, 5, 30) or (B,30) depending on dataset; we handle both
      - "target":    (B, Npmt, 2) or "pmt_labels"
      - "unique_pmt_ids" or direct PMT indexing compatible with target
    """
    ensure_dir(out_dir)

    model.eval()
    model.to(device)

    if getattr(model, "testing_dataset", None) is None:
        raise RuntimeError("model.testing_dataset is None; cannot run pair scatter on test split.")

    loader = model.test_dataloader()

    xs: List[float] = []
    ys: List[float] = []
    cs: List[float] = []

    for bi, batch in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break

        # move tensors to device
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        # try common keys
        if "tq_pair" not in batch:
            raise KeyError("batch missing key 'tq_pair' (expected (B,Npmt,2,2)).")
        tq_pair = batch["tq_pair"]  # torch, (B,Npmt,2,2)

        # labels: prefer 'target' then 'pmt_labels'
        if "target" in batch:
            y_true = batch["target"]
        elif "pmt_labels" in batch:
            y_true = batch["pmt_labels"]
        else:
            raise KeyError("batch missing 'target' or 'pmt_labels' (expected (B,Npmt,2)).")

        # y_pair for vertex distance
        if "y_pair" not in batch:
            raise KeyError("batch missing key 'y_pair'.")
        y_pair = batch["y_pair"]

        # forward pass to PMT-level prediction
        # Here we assume pair model takes something else? If your model is the same PointSetTrainer,
        # you likely have hit-level inputs ("hit_features"/"hit_pmt_ids"). If not, tell me your batch keys.
        if "hit_features" in batch and "hit_pmt_ids" in batch and "batch" in batch:
            hit_features = batch["hit_features"]
            hit_pmt_ids = batch["hit_pmt_ids"]
            ev_batch = batch["batch"]
            coords = hit_features[:, :4]
            feats = hit_features[:, 4:]
            logits_hit = model.forward(coords, feats, ev_batch)  # (N,2)
            # aggregate to PMT-level as in evaluation_JUNO.py
            from torch_scatter import scatter
            logits_pmt = scatter(logits_hit, hit_pmt_ids, dim=0, reduce="mean", dim_size=y_true.shape[1])
            prob = torch.sigmoid(logits_pmt)
            y_pred = (prob > 0.5).to(torch.int64)  # (Npmt,2) for a single event only if batch==1
            # If batch has multiple events in one loader output, this path is not correct.
            # We'll guard below.
        else:
            raise RuntimeError(
                "Don't know how to run model forward for pair scatter. "
                "Batch missing hit_features/hit_pmt_ids/batch. Please tell me your test batch keys."
            )

        # At this point, we need per-event predictions. The above only works for single-event-per-batch.
        # If your loader is B>1 events, we need a different aggregation path.
        # We'll enforce B==1 for now to avoid wrong results.
        B = int(tq_pair.shape[0])
        if B != 1:
            raise RuntimeError(f"Expected test loader batch size B==1 for pair scatter, got B={B}. Need per-event aggregation logic.")

        tq_ev = tq_pair[0].detach().cpu().numpy()  # (Npmt,2,2)

        # y_true: (B,Npmt,2)
        y_true_ev = y_true[0].detach().cpu().numpy().astype(np.int64)

        # y_pred currently: (Npmt,2)
        y_pred_ev = y_pred.detach().cpu().numpy().astype(np.int64)

        # y_pair maybe (B,5,30) or (B,30)
        yp = y_pair[0].detach().cpu().numpy()
        if yp.ndim == 2 and yp.shape[0] == 5 and yp.shape[1] == 30:
            # choose k=0 by default (or you can loop k=0..4)
            # If you want 5 points per event, loop over k and use tq_ev? (tq is not per-k here)
            y_pair_ev = yp[0]
        elif yp.ndim == 1 and yp.shape[0] == 30:
            y_pair_ev = yp
        else:
            raise RuntimeError(f"Unexpected y_pair[0] shape: {yp.shape} (expected (5,30) or (30,))")

        x = extract_vertex_distance_from_y_pair_ev(y_pair_ev)
        y = extract_dt_eplus_c14_from_tq_event(tq_ev)
        acc = pmt_overall_accuracy_2ch(y_true_ev, y_pred_ev)

        if np.isfinite(x) and np.isfinite(y) and np.isfinite(acc):
            xs.append(x)
            ys.append(y)
            cs.append(acc)

    if len(xs) == 0:
        raise RuntimeError("No points collected for pair scatter from test loader. Check batch keys and data validity.")

    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    cs = np.asarray(cs, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    sc = ax.scatter(xs, ys, c=cs, cmap="viridis", vmin=0.0, vmax=1.0, s=float(s), alpha=float(alpha), linewidths=0)
    cb = fig.colorbar(sc, ax=ax, shrink=0.90, pad=0.02)
    cb.set_label("PMT overall accuracy (mean over M*2)")

    ax.set_xlabel("vertex distance |v_e+ - v_C14| (inX/inY/inZ units)")
    ax.set_ylabel("| earliest t(e+) - earliest t(C14) | (ns)")
    ax.set_title("Pair scatter (test): vertex distance vs earliest-t gap (color=accuracy)")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    save_path = os.path.join(out_dir, "pair_scatter_vertexDist_dt_colorAcc_test.png")
    fig.savefig(save_path, dpi=220)
    plt.close(fig)
    print(f"[OK] saved: {save_path} (points={xs.size})")


# ---------- load event-accuracy cache produced by evaluation_JUNO.py ----------
def load_event_acc_cache_npz(acc_npz_path: str) -> Dict[int, float]:
    if not os.path.exists(acc_npz_path):
        raise FileNotFoundError(f"Missing acc cache: {acc_npz_path}")
    d = np.load(acc_npz_path, allow_pickle=True)
    ev = d["event_idx"].astype(np.int64)
    acc = d["acc"].astype(np.float32)
    return {int(e): float(a) for e, a in zip(ev, acc)}


def _id_from_path_str(p: str) -> Optional[int]:
    m = ID_RE.match(os.path.basename(p))
    return int(m.group(1)) if m else None


# def _load_one_pair_for_event(mixed_root: str, ev_idx: int):
#     """
#     Load the pair corresponding to a global event index ev_idx.
#     Mapping: base_id = ev_idx // 5, local_k = ev_idx % 5.
#     Opens y_pair_{base_id}.npy and tq_pair_{base_id}.npy, picks candidate local_k.
#     Returns (dist_mm, dt_ns) or (None, None) if missing/invalid.
#     """
#     try:
#         ev = int(ev_idx)
#     except Exception:
#         return None, None

#     base_id = ev // 5
#     local_k = ev % 5

#     yfp = os.path.join(mixed_root, "y_pair", f"y_pair_{int(base_id)}.npy")
#     tqfp = os.path.join(mixed_root, "tq_pair", f"tq_pair_{int(base_id)}.npy")
#     if not os.path.exists(yfp) or not os.path.exists(tqfp):
#         return None, None

#     try:
#         y = np.load(yfp).astype(np.float32)
#         tq = np.load(tqfp).astype(np.float32)
#     except Exception:
#         return None, None

#     # select the local candidate from y
#     if y.ndim == 2 and y.shape[1] == 30:
#         # y shape (K,30) where K is num candidates (often 5)
#         if local_k < y.shape[0]:
#             y_ev = y[local_k]
#         else:
#             return None, None
#     elif y.ndim == 1 and y.shape[0] == 30:
#         # single-candidate file; only local_k==0 valid
#         if local_k == 0:
#             y_ev = y
#         else:
#             return None, None
#     else:
#         return None, None

#     # select the local candidate from tq
#     # tq may be (K, Npmt, 2, 2) or (Npmt, 2, 2)
#     if tq.ndim == 4:
#         if local_k < tq.shape[0]:
#             tq_ev = tq[local_k]
#         else:
#             return None, None
#     elif tq.ndim == 3 and tq.shape[1:] == (2, 2):
#         # single per-base tq
#         tq_ev = tq
#     else:
#         return None, None

#     try:
#         dist = extract_vertex_distance_from_y_pair_ev(y_ev)
#         dt = extract_dt_eplus_c14_from_tq_event(tq_ev)
#     except Exception:
#         return None, None

#     if not (np.isfinite(dist) and np.isfinite(dt)):
#         return None, None
#     return float(dist), float(dt)


# def plot_pair_scatter_from_mixed_root_using_cache(
#     mixed_root: str,
#     out_dir: str,
#     acc_npz_path: str,
#     max_files: int = -1,
#     alpha: float = 0.75,
#     s: float = 10.0,
# ):
#     """
#     Revised: iterate over acc_cache keys (event_idx) and produce one point per event.
#     """
#     ensure_dir(out_dir)
#     acc_cache = load_event_acc_cache_npz(acc_npz_path)
#     ev_keys = sorted(acc_cache.keys())
#     if max_files and max_files > 0:
#         ev_keys = ev_keys[:max_files]

#     xs, ys, cs = [], [], []
#     missing = 0
#     for ev in ev_keys:
#         dist, dt = _load_one_pair_for_event(mixed_root, ev)
#         if dist is None:
#             missing += 1
#             continue
#         xs.append(dist); ys.append(dt); cs.append(float(acc_cache[int(ev)]))

#     xs = np.asarray(xs, dtype=np.float32)
#     ys = np.asarray(ys, dtype=np.float32)
#     cs = np.asarray(cs, dtype=np.float32)

#     if xs.size == 0:
#         raise RuntimeError(f"No points collected (missing_events={missing}).")

#     from matplotlib.colors import Normalize
#     vmin = float(np.nanmin(cs)); vmax = float(np.nanmax(cs))
#     if abs(vmax - vmin) < 1e-6:
#         vmax = vmin + 1e-6
#     norm = Normalize(vmin=vmin, vmax=vmax)

#     fig, ax = plt.subplots(figsize=(8.0, 6.0))
#     sc = ax.scatter(xs, ys, c=cs, cmap="viridis", norm=norm, s=float(s), alpha=float(alpha), linewidths=0)
#     cb = fig.colorbar(sc, ax=ax, shrink=0.9, pad=0.02)
#     cb.set_label("PMT overall accuracy (mean over PMT*2)")
#     ax.set_xlabel("vertex distance |v_e+ - v_C14| (mm)")
#     ax.set_ylabel("| min_t(e+) - min_t(C14) | (ns)")
#     ax.set_title("Pair scatter: vertex distance vs earliest-t gap (color=event acc) — 1 point per event")
#     ax.grid(True, alpha=0.25)
#     fig.tight_layout()
#     save_path = os.path.join(out_dir, "pair_scatter_vertexDist_dt_colorAcc.png")
#     fig.savefig(save_path, dpi=220)
#     plt.close(fig)
#     print(f"[OK] saved: {save_path} (points={xs.size}, missing_events={missing})")

# def collect_pair_points_from_mixed_root_using_cache(
#     mixed_root: str,
#     acc_npz_path: str,
#     max_files: int = -1,
# ):
#     """
#     Revised collector: returns one (dist,dt,acc) per acc_cache event key.
#     """
#     acc_cache = load_event_acc_cache_npz(acc_npz_path)
#     ev_keys = sorted(acc_cache.keys())
#     if max_files and max_files > 0:
#         ev_keys = ev_keys[:max_files]

#     xs, ys, cs = [], [], []
#     missing = 0
#     for ev in ev_keys:
#         dist, dt = _load_one_pair_for_event(mixed_root, ev)
#         if dist is None:
#             missing += 1
#             continue
#         xs.append(dist); ys.append(dt); cs.append(float(acc_cache[int(ev)]))

#     if len(xs) == 0:
#         return np.empty(0), np.empty(0), np.empty(0)
#     return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), np.asarray(cs, dtype=np.float32)

def _find_files_by_base(dir_path: str) -> Dict[int, str]:
    """
    Scan dir_path for any '*.npy' and return mapping base_id -> first matched filepath.
    Matches filenames that end with '_<int>.npy' (e.g. y_0.npy, y_pair_0.npy, tq_pair_0.npy).
    """
    if not os.path.isdir(dir_path):
        return {}
    mapping: Dict[int, str] = {}
    for p in sorted(Path(dir_path).glob("*.npy")):
        bid = _id_from_path_str(str(p))
        if bid is None:
            continue
        b = int(bid)
        # keep first seen file for this base id
        if b not in mapping:
            mapping[b] = str(p)
    return mapping


def _build_available_event_set_from_y_pair(mixed_root: str, per_base_candidates: int = 5) -> set:
    """
    Build set of expanded global event ids available under mixed_root/y_pair.
    Uses actual filenames discovered (not hardcoded 'y_pair_<id>.npy').
    """
    y_dir = os.path.join(mixed_root, "y_pair")
    if not os.path.isdir(y_dir):
        return set()
    base_map = _find_files_by_base(y_dir)  # base_id -> actual filepath
    if len(base_map) == 0:
        return set()
    available = set()
    for base, yfp in base_map.items():
        try:
            y = np.load(yfp)
        except Exception:
            continue
        if y.ndim == 2:
            K = int(y.shape[0])
        elif y.ndim == 1 and y.shape[0] == 30:
            K = 1
        else:
            # unknown format: skip
            continue
        for k in range(K):
            available.add(int(base) * per_base_candidates + k)
    return available


def _load_one_pair_for_event(mixed_root: str, ev_idx: int):
    """
    Load the pair corresponding to a global event index ev_idx.
    Mapping: base_id = ev_idx // per_base_candidates, local_k = ev_idx % per_base_candidates.
    Uses discovered filenames in y_pair/ and tq_pair/ (supports y_*.npy, y_pair_*.npy, tq_*.npy, tq_pair_*.npy).
    Returns (dist_mm, dt_ns) or (None, None) if missing/invalid.
    """
    try:
        ev = int(ev_idx)
    except Exception:
        return None, None

    per_base = 5
    base_id = ev // per_base
    local_k = ev % per_base

    # find y file for this base id
    y_dir = os.path.join(mixed_root, "y_pair")
    tq_dir = os.path.join(mixed_root, "tq_pair")
    y_candidates = list(Path(y_dir).glob(f"*_{base_id}.npy")) if os.path.isdir(y_dir) else []
    tq_candidates = list(Path(tq_dir).glob(f"*_{base_id}.npy")) if os.path.isdir(tq_dir) else []

    if len(y_candidates) == 0 or len(tq_candidates) == 0:
        return None, None

    # prefer filenames that contain 'y_pair' / 'tq_pair' if present
    def pick_best(cands, prefer_substr: str = ""):
        if not cands:
            return None
        if prefer_substr:
            for p in cands:
                if prefer_substr in p.name:
                    return str(p)
        return str(cands[0])

    yfp = pick_best(y_candidates, prefer_substr="y_pair")
    tqfp = pick_best(tq_candidates, prefer_substr="tq_pair")

    try:
        y = np.load(yfp).astype(np.float32)
        tq = np.load(tqfp).astype(np.float32)
    except Exception:
        return None, None

    # select the local candidate from y
    if y.ndim == 2 and y.shape[1] == 30:
        if local_k < y.shape[0]:
            y_ev = y[local_k]
        else:
            return None, None
    elif y.ndim == 1 and y.shape[0] == 30:
        if local_k == 0:
            y_ev = y
        else:
            return None, None
    else:
        return None, None

    # select the local candidate from tq
    # tq may be (K, Npmt, 2, 2) or (Npmt, 2, 2)
    if tq.ndim == 4:
        if local_k < tq.shape[0]:
            tq_ev = tq[local_k]
        else:
            return None, None
    elif tq.ndim == 3 and tq.shape[1:] == (2, 2):
        tq_ev = tq
    else:
        return None, None

    try:
        dist = extract_vertex_distance_from_y_pair_ev(y_ev)
        dt = extract_dt_eplus_c14_from_tq_event(tq_ev)
    except Exception:
        return None, None

    if not (np.isfinite(dist) and np.isfinite(dt)):
        return None, None
    return float(dist), float(dt)


def plot_pair_scatter_from_mixed_root_using_cache(
    mixed_root: str,
    out_dir: str,
    acc_npz_path: str,
    max_files: int = -1,
    alpha: float = 0.75,
    s: float = 10.0,
):
    """
    Revised: iterate over acc_cache keys (event_idx) but only those that exist
    in mixed_root/y_pair (expanded by base_id*5 + local_k). Diagnostic prints included.
    """
    ensure_dir(out_dir)
    acc_cache = load_event_acc_cache_npz(acc_npz_path)
    ev_keys = sorted(acc_cache.keys())
    if max_files and max_files > 0:
        ev_keys = ev_keys[:max_files]

    # build available global-event set from y_pair files
    avail = _build_available_event_set_from_y_pair(mixed_root, per_base_candidates=5)
    print(f"[DEBUG] acc_cache keys={len(ev_keys)}, y_pair available global ids={len(avail)}")
    if len(avail) == 0:
        raise RuntimeError(f"No y_pair files found or no valid candidates under {os.path.join(mixed_root,'y_pair')}")

    ev_set = set(ev_keys)
    common = sorted(list(ev_set & avail))
    print(f"[DEBUG] acc_cache ∩ y_pair = {len(common)} events")

    if len(common) == 0:
        sample_acc = sorted(ev_keys)[:10]
        sample_avail = sorted(list(avail))[:10]
        raise RuntimeError(
            "No overlap between eval_test_event_acc.npz event_idx and expanded y_pair entries.\n"
            f"sample acc keys: {sample_acc}\n"
            f"sample available (from y_pair): {sample_avail}\n"
            "Check mapping rule (global = base_id*5 + local_k) and that mixed_root matches eval outputs."
        )

    xs, ys, cs = [], [], []
    missing = 0
    for ev in common:
        dist, dt = _load_one_pair_for_event(mixed_root, ev)
        if dist is None:
            missing += 1
            continue
        xs.append(dist); ys.append(dt); cs.append(float(acc_cache[int(ev)]))

    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    cs = np.asarray(cs, dtype=np.float32)

    if xs.size == 0:
        raise RuntimeError(f"No points collected (missing_events_in_common={missing}).")

    from matplotlib.colors import Normalize
    vmin = float(np.nanmin(cs)); vmax = float(np.nanmax(cs))
    if abs(vmax - vmin) < 1e-6:
        vmax = vmin + 1e-6
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    sc = ax.scatter(xs, ys, c=cs, cmap="viridis", norm=norm, s=float(s), alpha=float(alpha), linewidths=0)
    cb = fig.colorbar(sc, ax=ax, shrink=0.9, pad=0.02)
    cb.set_label("PMT overall accuracy (mean over PMT*2)")
    ax.set_xlabel("vertex distance |v_e+ - v_C14| (mm)")
    ax.set_ylabel("| min_t(e+) - min_t(C14) | (ns)")
    ax.set_title("Pair scatter: vertex distance vs earliest-t gap (color=event acc) — 1 point per event")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_path = os.path.join(out_dir, "pair_scatter_vertexDist_dt_colorAcc.png")
    fig.savefig(save_path, dpi=220)
    plt.close(fig)
    print(f"[OK] saved: {save_path} (points={xs.size}, missing_in_common={missing})")


def collect_pair_points_from_mixed_root_using_cache(
    mixed_root: str,
    acc_npz_path: str,
    max_files: int = -1,
):
    """
    Revised collector: returns one (dist,dt,acc) per acc_cache event key that exists in y_pair expanded set.
    """
    acc_cache = load_event_acc_cache_npz(acc_npz_path)
    ev_keys = sorted(acc_cache.keys())
    if max_files and max_files > 0:
        ev_keys = ev_keys[:max_files]

    avail = _build_available_event_set_from_y_pair(mixed_root, per_base_candidates=5)
    ev_set = set(ev_keys)
    common = sorted(list(ev_set & avail))
    if len(common) == 0:
        return np.empty(0), np.empty(0), np.empty(0)

    xs, ys, cs = [], [], []
    for ev in common:
        dist, dt = _load_one_pair_for_event(mixed_root, ev)
        if dist is None:
            continue
        xs.append(dist); ys.append(dt); cs.append(float(acc_cache[int(ev)]))

    if len(xs) == 0:
        return np.empty(0), np.empty(0), np.empty(0)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), np.asarray(cs, dtype=np.float32)


def plot_pair_accuracy_grid_from_mixed_root_using_cache(
    mixed_root: str,
    out_dir: str,
    acc_npz_path: str,
    nx: int = 6,
    ny: int = 7,
    max_files: int = -1,
    fmt_precision: int = 2,
    cmap: str = "Blues",
):
    """
    Create nx x ny grid, compute mean accuracy per cell and plot a heatmap with text labels.
    Uses a single-blue color map ('Blues') by default.
    """
    ensure_dir(out_dir)
    xs, ys, cs = collect_pair_points_from_mixed_root_using_cache(mixed_root, acc_npz_path, max_files=max_files)
    if xs.size == 0:
        raise RuntimeError("No points to build grid heatmap.")

    # define bin edges
    x_edges = np.linspace(float(xs.min()), float(xs.max()), nx + 1)
    y_edges = np.linspace(float(ys.min()), float(ys.max()), ny + 1)

    # digitize and accumulate
    xi = np.digitize(xs, x_edges) - 1
    yi = np.digitize(ys, y_edges) - 1
    # clamp indices
    xi = np.clip(xi, 0, nx - 1)
    yi = np.clip(yi, 0, ny - 1)

    sums = np.zeros((ny, nx), dtype=np.float64)
    counts = np.zeros((ny, nx), dtype=np.int32)
    for xind, yind, val in zip(xi, yi, cs):
        sums[yind, xind] += float(val)
        counts[yind, xind] += 1

    means = np.full_like(sums, np.nan, dtype=np.float32)
    mask = counts > 0
    means[mask] = (sums[mask] / counts[mask]).astype(np.float32)

    # plot heatmap (single-blue colormap)
    fig, ax = plt.subplots(figsize=(10.0, 6.5))
    vmin = float(np.nanmin(means)) if np.any(mask) else 0.0
    vmax = float(np.nanmax(means)) if np.any(mask) else 1.0
    im = ax.imshow(
        means,
        origin="lower",
        cmap=cmap,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
    )
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("mean PMT overall accuracy")

    # draw grid lines
    for xe in x_edges:
        ax.axvline(x=xe, color="white", linewidth=0.8, alpha=0.7)
    for ye in y_edges:
        ax.axhline(y=ye, color="white", linewidth=0.8, alpha=0.7)

    # annotate cells (centered) with contrast-aware text color
    mid = 0.5 * (vmin + vmax)
    for iy in range(ny):
        for ix in range(nx):
            cx = 0.5 * (x_edges[ix] + x_edges[ix + 1])
            cy = 0.5 * (y_edges[iy] + y_edges[iy + 1])
            if counts[iy, ix] > 0 and np.isfinite(means[iy, ix]):
                txt = f"{means[iy, ix]:.{fmt_precision}f}\n(n={counts[iy, ix]})"
                # choose white text for dark blue (mean > mid), else black
                text_color = "white" if means[iy, ix] > mid else "black"
            else:
                txt = f"-\n(n=0)"
                text_color = "black"
            ax.text(cx, cy, txt, ha="center", va="center", fontsize=9, color=text_color)

    ax.set_xlabel("vertex distance |v_e+ - v_C14| (mm)")
    ax.set_ylabel("| min_t(e+) - min_t(C14) | (ns)")
    ax.set_title(f"Grid {nx}x{ny} mean PMT accuracy per cell")
    fig.tight_layout()
    outp = os.path.join(out_dir, f"pair_acc_grid_{nx}x{ny}.png")
    fig.savefig(outp, dpi=220)
    plt.close(fig)
    print(f"[OK] saved: {outp} (cells_with_data={int(mask.sum())})")

def plot_pair_acc_vs_4d_distance_from_mixed_root_using_cache(
    mixed_root: str,
    out_dir: str,
    acc_npz_path: str,
    max_files: int = -1,
    vg_mm_per_ns: float = 190.0,
    alpha: float = 0.8,
    s: float = 18.0,
):
    """
    Revised: compute 4D distance per event (one point per acc_cache entry).
    """
    ensure_dir(out_dir)
    xs, ys, cs = collect_pair_points_from_mixed_root_using_cache(mixed_root, acc_npz_path, max_files=max_files)
    if xs.size == 0:
        raise RuntimeError("No points collected for 4D plot.")

    d4 = np.sqrt(xs * xs + (vg_mm_per_ns * ys) ** 2)
    accs = cs

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    sc = ax.scatter(d4, accs, c=accs, cmap="viridis", s=float(s), alpha=float(alpha), edgecolors="none")
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("PMT overall accuracy")
    ax.set_xlabel("4D distance sqrt(spatial^2 + (vg*dt)^2) (mm)")
    ax.set_ylabel("PMT overall accuracy (mean over PMT*2)")
    ax.set_title("Accuracy vs 4D spacetime distance (1 point per event)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    outp = os.path.join(out_dir, f"pair_acc_vs_4d_distance_vg{int(vg_mm_per_ns)}.png")
    fig.savefig(outp, dpi=220)
    plt.close(fig)
    print(f"[OK] saved: {outp} (points={d4.size})")

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--eval_dir",
        type=str,
        default="/disk_pool1/houyh/results/Noise_run/version_2",
        help="Directory that contains eval_single_event_*.npz and (optionally) eval_full.npz. Also used as output directory.",
    )
    parser.add_argument(
        "--mixed_root",
        type=str,
        default="/disk_pool1/houyh/data/scattered",
        help="Root directory for mixed pair data (expects tq_pair/, target/, y_pair/).",
    )
    parser.add_argument(
        "--pair_max_files",
        type=int,
        default=-1,
        help="Limit number of tq_pair_*.npy files to scan (debug).",
    )
    parser.add_argument(
        "--use_truth_color",
        action="store_true",
        help="If set, do not load eval_full.npz; color all points as 1.0 (sanity check).",
    )
    parser.add_argument(
        "--options_file",
        type=str,
        default="config/pst/pst_small_tune.json",
        help="JSON config (used to load coords_mm for Mollweide).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="auto",
        choices=["auto", "val", "test"],
        help="Which single-event file to use for Mollweide.",
    )
    args = parser.parse_args()

    eval_dir = str(args.eval_dir)
    out_dir = ensure_dir(args.eval_dir + "/plots/2") 

    # -------------------------
    # 1) Load eval_full and reproduce evaluation_JUNO.py plots (CMs, ROC/PR)
    # -------------------------
    # full_path = os.path.join(eval_dir, "eval_full.npz")
    # if not os.path.exists(full_path):
    #     raise FileNotFoundError(f"Missing {full_path}. Run evaluation_JUNO.py first.")
    # full = np.load(full_path, allow_pickle=True)

    # split_full = str(full["split"][0])
    # y_true2 = full["y_true_bin"].astype(np.int64)
    # y_pred2 = full["y_pred_bin"].astype(np.int64)
    # y_prob2 = full["y_prob"].astype(np.float32)

    # # (1) 3-class CM (split 11)
    # cls3_names = ["00:none", "10:e+", "01:C14"]
    # y_true3, y_pred3 = to3class_split11_pairwise(y_true2, y_pred2)
    # cm3 = confusion_matrix(y_true3, y_pred3, labels=[0, 1, 2])
    # plot_confusion_counts(cm3, cls3_names, os.path.join(out_dir, f"confusion3_counts_{split_full}.png"),
    #                       "3-class Confusion Matrix (Counts)")
    # plot_confusion_float(confusion_efficiency(cm3), cls3_names, os.path.join(out_dir, f"confusion3_eff_{split_full}.png"),
    #                      "3-class Efficiency (Row Normalized)")
    # plot_confusion_float(confusion_purity(cm3), cls3_names, os.path.join(out_dir, f"confusion3_pur_{split_full}.png"),
    #                      "3-class Purity (Column Normalized)")

    # # (2) 4-class CM
    # cls4_names = ["00:none", "10:e+", "01:C14", "11:both"]
    # y_true4 = bin2to4(y_true2)
    # y_pred4 = bin2to4(y_pred2)
    # cm4 = confusion_matrix(y_true4, y_pred4, labels=[0, 1, 2, 3])
    # plot_confusion_counts(cm4, cls4_names, os.path.join(out_dir, f"confusion4_counts_{split_full}.png"),
    #                       "4-class Confusion Matrix (Counts)")
    # plot_confusion_float(confusion_efficiency(cm4), cls4_names, os.path.join(out_dir, f"confusion4_eff_{split_full}.png"),
    #                      "4-class Efficiency (Row Normalized)")
    # plot_confusion_float(confusion_purity(cm4), cls4_names, os.path.join(out_dir, f"confusion4_pur_{split_full}.png"),
    #                      "4-class Purity (Column Normalized)")

    # # (3) ROC/PR
    # for j, name in enumerate(["eplus", "c14"]):
    #     plot_roc_pr(y_true2[:, j], y_prob2[:, j], name, out_dir)

    # -------------------------
    # 2) Load single event and reproduce evaluation_JUNO.py Mollweide plots
    #    NOTE: need coords_mm from dataset -> load PointSetTrainer to access ds.coords_mm
    # -------------------------
    if args.split == "auto":
        path = os.path.join(eval_dir, "eval_single_event_279.npz")
        if not os.path.exists(path):
            path = os.path.join(eval_dir, "eval_single_event_test.npz")
    else:
        path = os.path.join(eval_dir, f"eval_single_event_{args.split}.npz")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run evaluation_JUNO.py first.")

    ev = np.load(path, allow_pickle=True)
    plot_split = str(ev["split"][0]) if "split" in ev.files else "unknown"
    ev_idx = int(ev["event_idx"][0]) if "event_idx" in ev.files else -1

    y_true_ev = ev["y_true_bin"].astype(np.int64)
    y_pred_ev = ev["y_pred_bin"].astype(np.int64)
    uid = ev["unique_pmt_ids"] if "unique_pmt_ids" in ev.files else None
    pmt_npe_ev = ev["pmt_npe"].astype(np.float32) if "pmt_npe" in ev.files else None

    if uid is None or uid is np.array(None):
        raise RuntimeError("unique_pmt_ids missing in eval_single_event npz; cannot draw Mollweide plots.")

    uid = uid.astype(np.int64)

    # Load dataset geometry (coords_mm) via trainer
    options = Options(args.mixed_root)
    with open(args.options_file, "r", encoding="utf-8") as f:
        options.update_options(json.load(f))
    model = PointSetTrainer(options=options)
    ds = model.testing_dataset if plot_split == "test" and getattr(model, "testing_dataset", None) is not None else model.validation_dataset
    base_ds = ds.dataset if hasattr(ds, "dataset") else ds

    coords_xyz = base_ds.coords_mm[uid].cpu().numpy().astype(np.float32)
    lon, lat = xyz_to_lonlat(coords_xyz)

    # Choose PMTs for visualization:
    mask_truth_hit = (y_true_ev[:, 0] == 1) | (y_true_ev[:, 1] == 1)
    mask_pred_hit = (y_pred_ev[:, 0] == 1) | (y_pred_ev[:, 1] == 1)
    mask_vis = mask_truth_hit | mask_pred_hit

    lon_h = lon[mask_vis]
    lat_h = lat[mask_vis]
    y_true_h = y_true_ev[mask_vis]
    y_pred_h = y_pred_ev[mask_vis]

    mask_e_true = (y_true_h[:, 0] == 1)
    mask_c_true = (y_true_h[:, 1] == 1)
    mask_e_pred = (y_pred_h[:, 0] == 1)
    mask_c_pred = (y_pred_h[:, 1] == 1)

    if pmt_npe_ev is None:
        raise RuntimeError("pmt_npe missing in eval_single_event npz; cannot draw npe/count Mollweide plots.")

    pmt_npe_h = pmt_npe_ev[mask_vis]

    # plot_mollweide_npe_overlay_rgb(
    #     lon=lon_h,
    #     lat=lat_h,
    #     pmt_npe=pmt_npe_h,
    #     mask_e=mask_e_pred,
    #     mask_c=mask_c_pred,
    #     out_path=os.path.join(out_dir, f"pmt_event{ev_idx}_pred_overlay_COUNT.png"),
    #     title=f"PMT Mollweide Overlay (COUNT) - Pred (event={ev_idx}, {plot_split})",
    #     nbins_lon=140,
    #     nbins_lat=70,
    #     smooth_sigma=2.0,
    #     clip_percentile=99.5,
    #     gamma_intensity=0.85,
    #     alpha_max=0.95,
    #     show_colorbar=True,
    #     mode="count",
    # )

    # plot_mollweide_npe_single_channel(
    #     lon=lon_h,
    #     lat=lat_h,
    #     pmt_npe=pmt_npe_h,
    #     mask=mask_e_pred,
    #     out_path=os.path.join(out_dir, f"pmt_event{ev_idx}_pred_eplus_COUNT.png"),
    #     title=f"PMT Mollweide COUNT - Pred e+ only (event={ev_idx}, {plot_split})",
    #     nbins_lon=140,
    #     nbins_lat=70,
    #     smooth_sigma=1.2,
    #     clip_percentile=99.5,
    #     cmap="Oranges",
    #     show_colorbar=True,
    #     mode="count",
    # )

    # plot_mollweide_npe_single_channel(
    #     lon=lon_h,
    #     lat=lat_h,
    #     pmt_npe=pmt_npe_h,
    #     mask=mask_c_pred,
    #     out_path=os.path.join(out_dir, f"pmt_event{ev_idx}_pred_c14_COUNT.png"),
    #     title=f"PMT Mollweide COUNT - Pred C14 only (event={ev_idx}, {plot_split})",
    #     nbins_lon=140,
    #     nbins_lat=70,
    #     smooth_sigma=1.2,
    #     clip_percentile=99.5,
    #     cmap="Blues",
    #     show_colorbar=True,
    #     mode="count",
    # )

    # plot_mollweide_npe_single_channel(
    #     lon=lon_h,
    #     lat=lat_h,
    #     pmt_npe=pmt_npe_h,
    #     mask=mask_e_true,
    #     out_path=os.path.join(out_dir, f"pmt_event{ev_idx}_truth_eplus_COUNT.png"),
    #     title=f"PMT Mollweide COUNT - Truth e+ only (event={ev_idx}, {plot_split})",
    #     nbins_lon=140,
    #     nbins_lat=70,
    #     smooth_sigma=1.2,
    #     clip_percentile=99.5,
    #     cmap="Oranges",
    #     show_colorbar=True,
    #     mode="count",
    # )

    # plot_mollweide_npe_single_channel(
    #     lon=lon_h,
    #     lat=lat_h,
    #     pmt_npe=pmt_npe_h,
    #     mask=mask_c_true,
    #     out_path=os.path.join(out_dir, f"pmt_event{ev_idx}_truth_c14_COUNT.png"),
    #     title=f"PMT Mollweide COUNT - Truth C14 only (event={ev_idx}, {plot_split})",
    #     nbins_lon=140,
    #     nbins_lat=70,
    #     smooth_sigma=1.2,
    #     clip_percentile=99.5,
    #     cmap="Blues",
    #     show_colorbar=True,
    #     mode="count",
    # )

    # plot_mollweide_npe_overlay_rgb(
    #     lon=lon_h,
    #     lat=lat_h,
    #     pmt_npe=pmt_npe_h,
    #     mask_e=mask_e_true,
    #     mask_c=mask_c_true,
    #     out_path=os.path.join(out_dir, f"pmt_event{ev_idx}_truth_overlay_COUNT.png"),
    #     title=f"PMT Mollweide Overlay (COUNT) - Truth (event={ev_idx}, {plot_split})",
    #     nbins_lon=140,
    #     nbins_lat=70,
    #     smooth_sigma=2.0,
    #     clip_percentile=99.5,
    #     gamma_intensity=0.85,
    #     alpha_max=0.95,
    #     show_colorbar=True,
    #     mode="count",
    # )


    plot_mollweide_scatter_points(
        lon=lon_h,
        lat=lat_h,
        y2=y_true_h,
        out_path=os.path.join(out_dir, f"pmt_event{ev_idx}_truth_scatter.png"),
        title=f"PMT Mollweide - Truth (event_idx={ev_idx}, {plot_split})",
        s=16.0,
        alpha=0.90,
        show_legend=True,
        background="white",
    )

    plot_mollweide_scatter_points(
        lon=lon_h,
        lat=lat_h,
        y2=y_pred_h,
        out_path=os.path.join(out_dir, f"pmt_event{ev_idx}_pred_scatter.png"),
        title=f"PMT Mollweide - Pred (event_idx={ev_idx}, {plot_split})",
        s=16.0,
        alpha=0.90,
        show_legend=True,
        background="white",
    )

    # (Scatter) Truth: e+ only / C14 only
    y_true_e_only = pack_y2_from_masks(mask_e_true, np.zeros_like(mask_e_true, dtype=bool))
    y_true_c_only = pack_y2_from_masks(np.zeros_like(mask_c_true, dtype=bool), mask_c_true)


    plot_mollweide_scatter_points(
        lon=lon_h,
        lat=lat_h,
        y2=y_true_e_only,
        out_path=os.path.join(out_dir, f"pmt_event{ev_idx}_truth_eplus_scatter.png"),
        title=f"PMT Mollweide - Truth e+ only (event_idx={ev_idx}, {plot_split})",
        s=16.0,
        alpha=0.90,
        show_legend=True,
        background="white",
    )

    plot_mollweide_scatter_points(
        lon=lon_h,
        lat=lat_h,
        y2=y_true_c_only,
        out_path=os.path.join(out_dir, f"pmt_event{ev_idx}_truth_c14_scatter.png"),
        title=f"PMT Mollweide - Truth C14 only (event_idx={ev_idx}, {plot_split})",
        s=16.0,
        alpha=0.90,
        show_legend=True,
        background="white",
    )

    # (Scatter) Pred: e+ only / C14 only
    y_pred_e_only = pack_y2_from_masks(mask_e_pred, np.zeros_like(mask_e_pred, dtype=bool))
    y_pred_c_only = pack_y2_from_masks(np.zeros_like(mask_c_pred, dtype=bool), mask_c_pred)

    plot_mollweide_scatter_points(
        lon=lon_h,
        lat=lat_h,
        y2=y_pred_e_only,
        out_path=os.path.join(out_dir, f"pmt_event{ev_idx}_pred_eplus_scatter.png"),
        title=f"PMT Mollweide - Pred e+ only (event_idx={ev_idx}, {plot_split})",
        s=16.0,
        alpha=0.90,
        show_legend=True,
        background="white",
    )

    plot_mollweide_scatter_points(
        lon=lon_h,
        lat=lat_h,
        y2=y_pred_c_only,
        out_path=os.path.join(out_dir, f"pmt_event{ev_idx}_pred_c14_scatter.png"),
        title=f"PMT Mollweide - Pred C14 only (event_idx={ev_idx}, {plot_split})",
        s=16.0,
        alpha=0.90,
        show_legend=True,
        background="white",
    )

    print(f"[OK] plots saved to: {out_dir}")

    # -------------------------
    # DEBUG: inspect test batch structure (keys + tensor shapes)
    # -------------------------
    # if getattr(model, "testing_dataset", None) is not None:
    #     loader = model.test_dataloader()
    #     batch0 = next(iter(loader))
    #     print("[DEBUG] test batch keys:", list(batch0.keys()))
    #     for k, v in batch0.items():
    #         if torch.is_tensor(v):
    #             print(f"[DEBUG] {k}: shape={tuple(v.shape)} dtype={v.dtype}")
    #         else:
    #             try:
    #                 print(f"[DEBUG] {k}: type={type(v)}")
    #             except Exception:
    #                 print(f"[DEBUG] {k}: <unprintable>")

    # -------------------------
    # 3) Pair scatter 
    # -------------------------

    # acc_npz = os.path.join(out_dir, "eval_test_event_acc.npz")
    # if os.path.exists(acc_npz):
    #     # use mixed_root + precomputed per-event acc cache (fast, no re-inference)
    #     plot_pair_scatter_from_mixed_root_using_cache(
    #         mixed_root=args.mixed_root,
    #         out_dir=out_dir,
    #         acc_npz_path=acc_npz,
    #         max_files=args.pair_max_files,
    #         alpha=0.75,
    #         s=10.0,
    #     )
    # else:
    #     print(f"[WARN] acc cache not found: {acc_npz}. Skipping pair scatter. Run evaluation_JUNO.py to produce eval_test_event_acc.npz if you want colored pair scatter.")

    # if os.path.exists(acc_npz):
    #     plot_pair_accuracy_grid_from_mixed_root_using_cache(
    #         mixed_root=args.mixed_root,
    #         out_dir=out_dir,
    #         acc_npz_path=acc_npz,
    #         nx=2,
    #         ny=2,
    #         max_files=args.pair_max_files,
    #     )

    # if os.path.exists(acc_npz):
    #     plot_pair_acc_vs_4d_distance_from_mixed_root_using_cache(
    #         mixed_root=args.mixed_root,
    #         out_dir=out_dir,
    #         acc_npz_path=acc_npz,
    #         max_files=args.pair_max_files,
    #         vg_mm_per_ns=getattr(options, "juno_vg_mm_per_ns", 190.0),
    #     )

if __name__ == "__main__":
    main()
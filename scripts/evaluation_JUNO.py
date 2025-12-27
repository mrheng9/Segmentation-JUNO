import os
import json
from argparse import ArgumentParser
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

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
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    accuracy_score,
)

from torch_scatter import scatter

import sys
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
    """
    efficiency = recall  = TP/(TP+FN)
    purity     = precision = TP/(TP+FP)
    """
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
    """
    Map (N,2) -> 4 class:
      (0,0)->0
      (1,0)->1
      (0,1)->2
      (1,1)->3
    """
    y2 = y2.astype(np.int64)
    return y2[:, 0] * 1 + y2[:, 1] * 2


def to3class_split11_pairwise(y_true2: np.ndarray, y_pred2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    3-class CM over {none, e+, C14}, splitting (1,1) into two samples (e+ view, C14 view).
    class ids: 0=none(00), 1=e+(10), 2=C14(01)
    """
    def cls_single(v: np.ndarray) -> int:
        a, b = int(v[0]), int(v[1])
        if a == 0 and b == 0:
            return 0
        if a == 1 and b == 0:
            return 1
        if a == 0 and b == 1:
            return 2
        return -1  # (1,1) handled outside

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
            # e+ view
            yt_list.append(1 if t11 else cls_single(t))
            yp_list.append(1 if p11 else cls_single(p))
            # C14 view
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


def plot_pmt_mollweide_overlay_3color(
    lon: np.ndarray,
    lat: np.ndarray,
    y2: np.ndarray,
    out_path: str,
    title: str,
    point_size: float = 3.0,
    show_background: bool = False,
    background_alpha: float = 0.05,
):
    """
    Single Mollweide plot:
      (1,0) -> orange
      (0,1) -> purple
      (1,1) -> green
      (0,0) -> not drawn (keep blank)
    """
    lon = _wrap_lon(lon)

    a = (y2[:, 0] == 1)
    b = (y2[:, 1] == 1)
    mask_11 = a & b
    mask_10 = a & (~b)
    mask_01 = (~a) & b

    fig = plt.figure(figsize=(12.5, 6.6))
    ax = fig.add_subplot(1, 1, 1, projection="mollweide")

    if show_background:
        ax.scatter(lon, lat, s=point_size, c="lightgray", alpha=background_alpha, linewidths=0)

    ax.scatter(lon[mask_10], lat[mask_10], s=point_size, c="tab:orange", alpha=0.90, linewidths=0, label="(1,0) e+")
    ax.scatter(lon[mask_01], lat[mask_01], s=point_size, c="tab:purple", alpha=0.90, linewidths=0, label="(0,1) C14")
    ax.scatter(lon[mask_11], lat[mask_11], s=point_size, c="tab:green",  alpha=0.95, linewidths=0, label="(1,1) both")

    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.35)

    xticks = np.radians(np.array([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150]))
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{d}°" for d in [-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150]])
    yticks = np.radians(np.array([-60, -30, 0, 30, 60]))
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{d}°" for d in [-60, -30, 0, 30, 60]])

    ax.legend(loc="lower left", frameon=True, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)

def _hist2d_on_lonlat(
    lon: np.ndarray,
    lat: np.ndarray,
    nbins_lon: int = 180,
    nbins_lat: int = 90,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns: H (nbins_lat, nbins_lon), lon_edges, lat_edges
    """
    lon = _wrap_lon(lon)
    lon_edges = np.linspace(-np.pi, np.pi, nbins_lon + 1, dtype=np.float32)
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, nbins_lat + 1, dtype=np.float32)
    H, _, _ = np.histogram2d(lat, lon, bins=[lat_edges, lon_edges])  # NOTE: lat first -> rows
    return H.astype(np.float32), lon_edges, lat_edges

def _hist2d_weighted_on_lonlat(
    lon: np.ndarray,
    lat: np.ndarray,
    w: np.ndarray,
    nbins_lon: int = 180,
    nbins_lat: int = 90,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Weighted 2D histogram on lon/lat.
    Returns: H (nbins_lat, nbins_lon), lon_edges, lat_edges
    """
    lon = _wrap_lon(lon)
    lon_edges = np.linspace(-np.pi, np.pi, nbins_lon + 1, dtype=np.float32)
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, nbins_lat + 1, dtype=np.float32)
    H, _, _ = np.histogram2d(lat, lon, bins=[lat_edges, lon_edges], weights=w)  # lat first
    return H.astype(np.float32), lon_edges, lat_edges

def summarize_heatmap(H: np.ndarray, name: str):
    H = H.astype(np.float32)
    mean = float(H.mean())
    mx = float(H.max())
    p99 = float(np.percentile(H, 99))
    p95 = float(np.percentile(H, 95))
    # 一个很直观的“是否有明显亮斑”的指标
    peak_ratio = mx / mean if mean > 0 else 0.0
    print(f"[DBG] {name}: mean={mean:.4f}, p95={p95:.2f}, p99={p99:.2f}, max={mx:.2f}, max/mean={peak_ratio:.2f}")

def plot_mollweide_density(
    lon: np.ndarray,
    lat: np.ndarray,
    mask: np.ndarray,
    out_path: str,
    title: str,
    nbins_lon: int = 180,
    nbins_lat: int = 90,
    smooth_sigma: float = 1.2,
    cmap: str = "magma",
    log_scale: bool = True,
):
    """
    Density heatmap on Mollweide:
      - mask selects points (e+, C14, both...)
      - Uses histogram in lon/lat then pcolormesh on mollweide axes.
    """
    lon_m = lon[mask]
    lat_m = lat[mask]

    if lon_m.size == 0:
        print(f"[WARN] plot_mollweide_density: empty mask -> {out_path}")
        return

    H, lon_edges, lat_edges = _hist2d_on_lonlat(lon_m, lat_m, nbins_lon=nbins_lon, nbins_lat=nbins_lat)

    if smooth_sigma and smooth_sigma > 0:
        if gaussian_filter is None:
            print("[WARN] scipy not available; skip gaussian smoothing.")
        else:
            H = gaussian_filter(H, sigma=float(smooth_sigma))

    if log_scale:
        # avoid log(0)
        H = np.log10(H + 1.0)

    Lon, Lat = np.meshgrid(lon_edges, lat_edges)

    fig = plt.figure(figsize=(12.5, 6.6))
    ax = fig.add_subplot(1, 1, 1, projection="mollweide")
    im = ax.pcolormesh(Lon, Lat, H, shading="auto", cmap=cmap)

    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.35)
    cbar = fig.colorbar(im, ax=ax, shrink=0.86, pad=0.06)
    cbar.set_label("log10(count+1)" if log_scale else "count")

    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)

def plot_mollweide_density_overlay_two(
    lon: np.ndarray,
    lat: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    out_path: str,
    title: str,
    nbins_lon: int = 180,
    nbins_lat: int = 90,
    smooth_sigma: float = 1.8,
    log_scale: bool = True,
    cmap_a: str = "Pinks",
    cmap_b: str = "Blues",
    alpha_a: float = 0.65,
    alpha_b: float = 0.65,
    show_colorbar: bool = False,
    colorbar_shrink: float = 0.4,
):
    """
    Overlay two density heatmaps on Mollweide with shared normalization,
    so both channels are visible and comparable. Overlap blends via alpha.
    """
    lon = _wrap_lon(lon)

    lon_a = lon[mask_a]
    lat_a = lat[mask_a]
    lon_b = lon[mask_b]
    lat_b = lat[mask_b]

    if lon_a.size == 0 and lon_b.size == 0:
        print(f"[WARN] plot_mollweide_density_overlay_two: both masks empty -> {out_path}")
        return

    Ha, lon_edges, lat_edges = _hist2d_on_lonlat(lon_a, lat_a, nbins_lon=nbins_lon, nbins_lat=nbins_lat)
    Hb, _, _ = _hist2d_on_lonlat(lon_b, lat_b, nbins_lon=nbins_lon, nbins_lat=nbins_lat)

    # DEBUG: show peakedness BEFORE smoothing/log
    summarize_heatmap(Ha, "e+ count heatmap (raw)")
    summarize_heatmap(Hb, "C14 count heatmap (raw)")

    if smooth_sigma and smooth_sigma > 0:
        if gaussian_filter is None:
            print("[WARN] scipy not available; skip gaussian smoothing.")
        else:
            Ha = gaussian_filter(Ha, sigma=float(smooth_sigma))
            Hb = gaussian_filter(Hb, sigma=float(smooth_sigma))

    # DEBUG: show peakedness AFTER smoothing BEFORE log
    summarize_heatmap(Ha, "e+ count heatmap (smoothed)")
    summarize_heatmap(Hb, "C14 count heatmap (smoothed)")

    if log_scale:
        Ha = np.log10(Ha + 1.0)
        Hb = np.log10(Hb + 1.0)

    # DEBUG: after log
    summarize_heatmap(Ha, "e+ heatmap (log)")
    summarize_heatmap(Hb, "C14 heatmap (log)")

    # Shared normalization so e+ doesn't disappear
    vmax = float(np.max([Ha.max() if Ha.size else 0.0, Hb.max() if Hb.size else 0.0]))
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=vmax if vmax > 0 else 1.0)

    Lon, Lat = np.meshgrid(lon_edges, lat_edges)

    fig = plt.figure(figsize=(12.5, 6.6))
    ax = fig.add_subplot(1, 1, 1, projection="mollweide")

    im_a = ax.pcolormesh(Lon, Lat, Ha, shading="auto", cmap=cmap_a, alpha=float(alpha_a), norm=norm)
    im_b = ax.pcolormesh(Lon, Lat, Hb, shading="auto", cmap=cmap_b, alpha=float(alpha_b), norm=norm)

    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.35)

    if show_colorbar:
        # smaller colorbars; still two, but much less intrusive
        cbar_a = fig.colorbar(im_a, ax=ax, shrink=float(colorbar_shrink), pad=0.04)
        cbar_a.set_label("e+ (shared scale)")
        cbar_b = fig.colorbar(im_b, ax=ax, shrink=float(colorbar_shrink), pad=0.10)
        cbar_b.set_label("C14 (shared scale)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)

def plot_mollweide_npe_overlay_rgb(
    lon: np.ndarray,
    lat: np.ndarray,
    pmt_npe: np.ndarray,      # (M,) aligned with lon/lat/y2
    mask_e: np.ndarray,       # (M,) boolean
    mask_c: np.ndarray,       # (M,) boolean
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
    cmap_e: str = "Oranges",
    cmap_c: str = "Blues",
    mix_power: float = 0.75,
    show_legend: bool = True,
    legend_loc: str = "lower left",
    save_dominant_map: bool = True,
    dominant_margin: float = 0.12,
):
    """
    Readable 2-channel NPE overlay on Mollweide.

    - Build He/Hc: weighted intensity maps (log1p(npe) optionally), smoothed.
    - Brightness encodes total intensity Ht.
    - Color encodes composition: e+ uses cmap_e, C14 uses cmap_c.
    - Adds legend for e+/C14/mixed.
    - Optionally saves a dominant-class categorical map.
    """
    lon = _wrap_lon(lon)

    w = np.clip(pmt_npe.astype(np.float32), 0.0, None)
    if use_log1p:
        w = np.log1p(w)

    He, lon_edges, lat_edges = _hist2d_weighted_on_lonlat(
        lon[mask_e], lat[mask_e], w[mask_e],
        nbins_lon=nbins_lon, nbins_lat=nbins_lat
    )
    Hc, _, _ = _hist2d_weighted_on_lonlat(
        lon[mask_c], lat[mask_c], w[mask_c],
        nbins_lon=nbins_lon, nbins_lat=nbins_lat
    )

    if smooth_sigma and smooth_sigma > 0 and gaussian_filter is not None:
        He = gaussian_filter(He, sigma=float(smooth_sigma))
        Hc = gaussian_filter(Hc, sigma=float(smooth_sigma))
    elif smooth_sigma and smooth_sigma > 0 and gaussian_filter is None:
        print("[WARN] scipy not available; skip gaussian smoothing.")

    Ht = He + Hc

    # robust scaling for intensity -> [0,1]
    vmax = float(np.percentile(Ht.ravel(), clip_percentile))
    if vmax <= 0:
        vmax = float(Ht.max() if Ht.size else 1.0)
    s = np.clip(Ht / vmax, 0.0, 1.0)
    s_disp = np.power(s, float(gamma_intensity)) if (gamma_intensity and gamma_intensity != 1.0) else s

    # composition in [0,1]
    frac_e = He / (Ht + 1e-6)
    frac_c = 1.0 - frac_e

    # make composition more/less decisive
    if mix_power and mix_power != 1.0:
        # apply symmetric power around 0.5
        frac_e = np.clip(frac_e, 0.0, 1.0)
        frac_c = 1.0 - frac_e
        # power in probability space
        pe = np.power(frac_e + 1e-6, float(mix_power))
        pc = np.power(frac_c + 1e-6, float(mix_power))
        frac_e = pe / (pe + pc)
        frac_c = 1.0 - frac_e

    # Convert channel strengths to RGBA using colormaps (more readable than pure RGB)
    cm_e = matplotlib.colormaps.get_cmap(cmap_e)
    cm_c = matplotlib.colormaps.get_cmap(cmap_c)

    # color from colormap at channel fraction (avoid very pale ends by mapping into [0.25, 1])
    ce = cm_e(0.25 + 0.75 * frac_e)  # RGBA
    cc = cm_c(0.25 + 0.75 * frac_c)

    rgb = np.zeros((nbins_lat, nbins_lon, 3), dtype=np.float32)
    # blend two colors by their fractions, then modulate by intensity
    rgb[..., 0] = (ce[..., 0] * frac_e + cc[..., 0] * frac_c) * s_disp
    rgb[..., 1] = (ce[..., 1] * frac_e + cc[..., 1] * frac_c) * s_disp
    rgb[..., 2] = (ce[..., 2] * frac_e + cc[..., 2] * frac_c) * s_disp

    rgba = np.zeros((nbins_lat, nbins_lon, 4), dtype=np.float32)
    rgba[..., :3] = rgb
    rgba[..., 3] = np.clip(s_disp, 0.0, 1.0) * float(alpha_max)

    Lon, Lat = np.meshgrid(lon_edges, lat_edges)
    fig = plt.figure(figsize=(12.5, 6.6))
    ax = fig.add_subplot(1, 1, 1, projection="mollweide")

    ax.pcolormesh(Lon, Lat, rgba, shading="auto")
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.20)

    if show_colorbar:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=vmax), cmap="gray")
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.05)
        cb.set_label("total intensity per bin (sum of log1p(npe))" if use_log1p else "total intensity per bin (sum of npe)")

    if show_legend:
        import matplotlib.patches as mpatches
        # representative colors
        col_e = matplotlib.cm.get_cmap(cmap_e)(0.85)
        col_c = matplotlib.cm.get_cmap(cmap_c)(0.85)
        col_m = (0.55, 0.25, 0.65, 1.0)  # fixed "mixed" purple for legend only
        handles = [
            mpatches.Patch(color=col_e, label="e+ (orange)"),
            mpatches.Patch(color=col_c, label="C14 (blue)"),
            mpatches.Patch(color=col_m, label="mixed/overlap"),
        ]
        ax.legend(handles=handles, loc=str(legend_loc), frameon=True, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)

    # Optional categorical dominant map (much easier to see who dominates per bin)
    if save_dominant_map:
        dom = np.full((nbins_lat, nbins_lon), 0, dtype=np.int32)  # 0=empty
        # Only where there is intensity
        has = Ht > 1e-6
        # dominance decision
        dom_e = (frac_e >= (0.5 + float(dominant_margin))) & has
        dom_c = (frac_e <= (0.5 - float(dominant_margin))) & has
        dom_m = has & (~dom_e) & (~dom_c)
        dom[dom_e] = 1
        dom[dom_c] = 2
        dom[dom_m] = 3

        from matplotlib.colors import ListedColormap, BoundaryNorm
        cmap_dom = ListedColormap([
            (1, 1, 1, 0.0),                              # 0 empty transparent
            matplotlib.cm.get_cmap(cmap_e)(0.85),        # 1 e+
            matplotlib.cm.get_cmap(cmap_c)(0.85),        # 2 C14
            (0.55, 0.25, 0.65, 0.95),                    # 3 mixed
        ])
        norm_dom = BoundaryNorm([0, 1, 2, 3, 4], cmap_dom.N)

        out2 = os.path.splitext(out_path)[0] + "_dominant.png"
        fig = plt.figure(figsize=(12.5, 6.6))
        ax = fig.add_subplot(1, 1, 1, projection="mollweide")
        ax.pcolormesh(Lon, Lat, dom, shading="auto", cmap=cmap_dom, norm=norm_dom)
        ax.set_title(title + " [dominant map]", fontsize=13)
        ax.grid(True, alpha=0.20)

        import matplotlib.patches as mpatches
        handles = [
            mpatches.Patch(color=matplotlib.cm.get_cmap(cmap_e)(0.85), label="e+ dominant"),
            mpatches.Patch(color=matplotlib.cm.get_cmap(cmap_c)(0.85), label="C14 dominant"),
            mpatches.Patch(color=(0.55, 0.25, 0.65, 0.95), label="mixed"),
        ]
        ax.legend(handles=handles, loc=str(legend_loc), frameon=True, fontsize=10)

        fig.tight_layout()
        fig.savefig(out2, dpi=240)
        plt.close(fig)

# -------------------------
# Inference: full-set (for metrics) and single-event (for visualization)
# -------------------------
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
            logits_hit, hit_pmt_ids, dim=0, reduce="mean", dim_size=pmt_labels.shape[0]
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

    # dataset may return tensors already
    def as_tensor(x, dtype, dev):
        if torch.is_tensor(x):
            return x.clone().detach().to(dev).to(dtype)
        return torch.tensor(x, dtype=dtype, device=dev)

    hit_features = as_tensor(sample["hit_features"], torch.float32, device)  # (N,5)
    hit_pmt_ids = as_tensor(sample["hit_pmt_ids"], torch.int64, device)      # (N,)
    pmt_labels = as_tensor(sample["pmt_labels"], torch.float32, device)      # (M,2)

    # NEW: PMT-level npe sum from hit-level feature column 4
    # hit_features[:, 4] confirmed as npe
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
    logits_pmt = scatter(logits_hit, hit_pmt_ids, dim=0, reduce="mean", dim_size=pmt_labels.shape[0])

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
        "pmt_npe": pmt_npe_np,  # NEW
    }


def save_text_summary(y_true2: np.ndarray, y_pred2: np.ndarray, out_dir: str, split: str):
    lines: List[str] = []
    lines.append(f"Split: {split}")
    lines.append("")

    for j, name in enumerate(["eplus", "c14"]):
        yt = y_true2[:, j]
        yp = y_pred2[:, j]
        lines.append(f"== Binary report: {name} ==")
        lines.append(classification_report(yt, yp, digits=4, zero_division=0))
        m = binary_eff_pur_metrics(yt, yp)
        lines.append(f"counts: TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")
        lines.append(
            "metrics: "
            f"efficiency(recall)={m['efficiency(recall)']:.6f}  "
            f"purity(precision)={m['purity(precision)']:.6f}  "
            f"fpr={m['fpr']:.6f}  "
            f"tnr(specificity)={m['tnr(specificity)']:.6f}  "
            f"accuracy={m['accuracy']:.6f}  "
            f"f1={f1_score(yt, yp, zero_division=0):.6f}"
        )
        lines.append("")

    y_true4 = bin2to4(y_true2)
    y_pred4 = bin2to4(y_pred2)
    lines.append("== 4-class (00/10/01/11) report ==")
    lines.append(
        classification_report(
            y_true4,
            y_pred4,
            labels=[0, 1, 2, 3],
            target_names=["00:none", "10:e+", "01:c14", "11:both"],
            digits=4,
            zero_division=0,
        )
    )
    lines.append(f"4-class accuracy: {accuracy_score(y_true4, y_pred4):.6f}")
    lines.append("")

    y_true3, y_pred3 = to3class_split11_pairwise(y_true2, y_pred2)
    lines.append("== 3-class (none / e+ / C14) with split 11 -> (10)+(01) ==")
    lines.append(
        classification_report(
            y_true3,
            y_pred3,
            labels=[0, 1, 2],
            target_names=["00:none", "10:e+", "01:C14"],
            digits=4,
            zero_division=0,
        )
    )
    lines.append(f"3-class accuracy (split11): {accuracy_score(y_true3, y_pred3):.6f}")
    lines.append(f"3-class sample count: {y_true3.shape[0]} (original PMT count: {y_true2.shape[0]})")

    with open(os.path.join(out_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-t", "--training_file",
        type=str,
        default="/disk_pool1/houyh/data/mixed",
        help="JUNO dataset root dir"
    )
    parser.add_argument(
        "-o", "--options_file",
        type=str,
        default="config/pst/pst_small_tune.json",
        help="JSON config used in training"
    )
    parser.add_argument("--ckpt", type=str, default="results/JUNO_run/version_0/checkpoints/last.ckpt", help="Path to Lightning checkpoint (*.ckpt).")
    parser.add_argument("--out_dir", type=str, default="/home/houyh/Segmentation-JUNO/results/JUNO_run/version_0/plots4", help="Output directory.")
    parser.add_argument("--max_batches", type=int, default=-1, help="Limit number of batches for debug.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    # Ensure the output directory exists
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    out_dir = ensure_dir(args.out_dir)

    # Options
    options = Options(args.training_file)
    if args.options_file:
        import json
        with open(args.options_file, "r", encoding="utf-8") as f:
            options.update_options(json.load(f))

    # Device
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Load model
    model = PointSetTrainer.load_from_checkpoint(args.ckpt, options=options)

    # Full inference for metrics
    data = run_inference_collect_full(model, device=device, max_batches=args.max_batches)
    split = str(data["split"][0])
    y_true2 = data["y_true_bin"]
    y_prob2 = data["y_prob"]
    y_pred2 = data["y_pred_bin"]

    # Save arrays
    # np.save(os.path.join(out_dir, "y_true_bin.npy"), y_true2)
    # np.save(os.path.join(out_dir, "y_pred_bin.npy"), y_pred2)
    # np.save(os.path.join(out_dir, "y_prob.npy"), y_prob2)

    # Text summary
    save_text_summary(y_true2, y_pred2, out_dir, split)

####################################################################
#1. 3-class CM (split 11)
####################################################################
#     cls3_names = ["00:none", "10:e+", "01:C14"]
#     y_true3, y_pred3 = to3class_split11_pairwise(y_true2, y_pred2)
#     cm3 = confusion_matrix(y_true3, y_pred3, labels=[0, 1, 2])
#     plot_confusion_counts(cm3, cls3_names, os.path.join(out_dir, f"confusion3_counts_{split}.png"),
#                           "3-class Confusion Matrix (Counts)")
#     plot_confusion_float(confusion_efficiency(cm3), cls3_names, os.path.join(out_dir, f"confusion3_eff_{split}.png"),
#                          "3-class Efficiency (Row Normalized)")
#     plot_confusion_float(confusion_purity(cm3), cls3_names, os.path.join(out_dir, f"confusion3_pur_{split}.png"),
#                          "3-class Purity (Column Normalized)")
    
# ####################################################################
# #2.  4-class CM
# ####################################################################
#     cls4_names = ["00:none", "10:e+", "01:C14", "11:both"]
#     y_true4 = bin2to4(y_true2)
#     y_pred4 = bin2to4(y_pred2)
#     cm4 = confusion_matrix(y_true4, y_pred4, labels=[0, 1, 2, 3])
#     plot_confusion_counts(cm4, cls4_names, os.path.join(out_dir, f"confusion4_counts_{split}.png"),
#                           "4-class Confusion Matrix (Counts)")
#     plot_confusion_float(confusion_efficiency(cm4), cls4_names, os.path.join(out_dir, f"confusion4_eff_{split}.png"),
#                          "4-class Efficiency (Row Normalized)")
#     plot_confusion_float(confusion_purity(cm4), cls4_names, os.path.join(out_dir, f"confusion4_pur_{split}.png"),
#                          "4-class Purity (Column Normalized)")
    
# ####################################################################
# #3.  ROC/PR
# #################################################################### 
#     for j, name in enumerate(["eplus", "c14"]):
#         plot_roc_pr(y_true2[:, j], y_prob2[:, j], name, out_dir)

####################################################################
#4.  Random single-event Mollweide projection (Truth & Pred)
#################################################################### 
    plot_split = "test" if getattr(model, "testing_dataset", None) is not None else "val"
    ds = model.testing_dataset if plot_split == "test" else model.validation_dataset

    n_events = len(ds)
    rng = np.random.default_rng(999)
    chosen = int(rng.integers(0, n_events))

    ev = run_inference_single_event_from_dataset(model, device=device, event_idx=chosen, split=plot_split)
    ev_idx = int(ev["event_idx"][0])
    y_true_ev = ev["y_true_bin"]
    y_pred_ev = ev["y_pred_bin"]
    uid = ev.get("unique_pmt_ids", None)

    if uid is None:
        print("[WARN] single-event projection skipped: unique_pmt_ids missing from dataset sample.")
    else:
        base_ds = ds.dataset if hasattr(ds, "dataset") else ds
        coords_xyz = base_ds.coords_mm[uid].cpu().numpy().astype(np.float32)
        lon, lat = xyz_to_lonlat(coords_xyz)

        # hits only (remove (0,0))
        mask_hit = (y_true_ev[:, 0] == 1) | (y_true_ev[:, 1] == 1)
        lon_h = lon[mask_hit]
        lat_h = lat[mask_hit]
        y_true_h = y_true_ev[mask_hit]
        y_pred_h = y_pred_ev[mask_hit]

        mask_e_true = (y_true_h[:, 0] == 1)
        mask_c_true = (y_true_h[:, 1] == 1)
        mask_e_pred = (y_pred_h[:, 0] == 1)
        mask_c_pred = (y_pred_h[:, 1] == 1)

        # plot_pmt_mollweide_overlay_3color(
        #     lon_h, lat_h, y_true_h,
        #     out_path=os.path.join(out_dir, f"pmt_event{ev_idx}_truth_mollweide.png"),
        #     title=f"PMT Mollweide - Truth (event={ev_idx}, {plot_split}, hits={int(mask_hit.sum())})",
        #     point_size=4.5,
        #     show_background=False,
        # )
        # plot_pmt_mollweide_overlay_3color(
        #     lon_h, lat_h, y_pred_h,
        #     out_path=os.path.join(out_dir, f"pmt_event{ev_idx}_pred_mollweide.png"),
        #     title=f"PMT Mollweide - Pred (event={ev_idx}, {plot_split}, hits={int(mask_hit.sum())})",
        #     point_size=4.5,
        #     show_background=False,
        # )

        # plot_mollweide_density(
        #     lon=lon_h,
        #     lat=lat_h,
        #     mask=np.ones(lon_h.shape[0], dtype=bool),  # all truth-hit PMTs
        #     out_path=os.path.join(out_dir, f"pmt_event{ev_idx}_truth_density_allhits.png"),
        #     title=f"PMT Mollweide Density - Truth All Hits (event={ev_idx}, {plot_split})",
        #     nbins_lon=180,
        #     nbins_lat=90,
        #     smooth_sigma=1.2,
        #     cmap="magma",
        #     log_scale=True,
        # )

        # plot_mollweide_density_overlay_two(
        #     lon=lon_h,
        #     lat=lat_h,
        #     mask_a=mask_e_true,
        #     mask_b=mask_c_true,
        #     out_path=os.path.join(out_dir, f"pmt_event{ev_idx}_truth_density_overlay_eplus_c14.png"),
        #     title=f"PMT Mollweide Density Overlay - Truth (e+ vs C14) (event={ev_idx}, {plot_split})",
        #     nbins_lon=140,
        #     nbins_lat=70,
        #     smooth_sigma=2.0,
        #     log_scale=True,
        #     cmap_a='OrRd',
        #     cmap_b="PuBu",
        #     alpha_a=0.70,
        #     alpha_b=0.70,
        #     show_colorbar=True,
        # )

        # plot_mollweide_density_overlay_two(
        #     lon=lon_h,
        #     lat=lat_h,
        #     mask_a=mask_e_pred,
        #     mask_b=mask_c_pred,
        #     out_path=os.path.join(out_dir, f"pmt_event{ev_idx}_pred_density_overlay_eplus_c14.png"),
        #     title=f"PMT Mollweide Density Overlay - Pred (e+ vs C14) (event={ev_idx}, {plot_split})",
        #     nbins_lon=140,
        #     nbins_lat=70,
        #     smooth_sigma=2.0,
        #     log_scale=True,
        #     cmap_a='OrRd',
        #     cmap_b="PuBu",
        #     alpha_a=0.70,
        #     alpha_b=0.70,
        #     show_colorbar=True,
        # )

        pmt_npe_ev = ev.get("pmt_npe", None)
        if pmt_npe_ev is None:
            print("[WARN] pmt_npe missing; cannot draw npe-weighted heatmap.")
        else:
            pmt_npe_h = pmt_npe_ev[mask_hit]

            # Truth npe-weighted overlay
            plot_mollweide_npe_overlay_rgb(
                lon=lon_h,
                lat=lat_h,
                pmt_npe=pmt_npe_h,
                mask_e=mask_e_true,
                mask_c=mask_c_true,
                out_path=os.path.join(out_dir, f"pmt_event{ev_idx}_truth_npe_overlay.png"),
                title=f"PMT Mollweide NPE Overlay - Truth (e+:R, C14:B) (event={ev_idx}, {plot_split})",
                nbins_lon=140,
                nbins_lat=70,
                smooth_sigma=2.0,
                use_log1p=True,
                clip_percentile=99.5,
                gamma_intensity=0.85,
                alpha_max=0.95,
                show_colorbar=True,
            )

            # Pred npe-weighted overlay
            plot_mollweide_npe_overlay_rgb(
                lon=lon_h,
                lat=lat_h,
                pmt_npe=pmt_npe_h,
                mask_e=mask_e_pred,
                mask_c=mask_c_pred,
                out_path=os.path.join(out_dir, f"pmt_event{ev_idx}_pred_npe_overlay.png"),
                title=f"PMT Mollweide NPE Overlay - Pred (e+:R, C14:B) (event={ev_idx}, {plot_split})",
                nbins_lon=140,
                nbins_lat=70,
                smooth_sigma=2.0,
                use_log1p=True,
                clip_percentile=99.5,
                gamma_intensity=0.85,
                alpha_max=0.95,
                show_colorbar=True,
            )

        n_e = int((y_true_h[:, 0] == 1).sum())
        n_c = int((y_true_h[:, 1] == 1).sum())
        n_both = int(((y_true_h[:, 0] == 1) & (y_true_h[:, 1] == 1)).sum())
        print("[OK] single-event projection saved:")
        print(f"     event_idx={ev_idx}, hits={int(mask_hit.sum())}")
        print(f"     Truth: e+={n_e}, C14={n_c}, both={n_both}")

    print(f"[OK] split={split}")
    print(f"[OK] saved to: {out_dir}")


if __name__ == "__main__":
    main()
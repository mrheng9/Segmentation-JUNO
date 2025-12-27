import argparse
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt

ID_RE = re.compile(r".*_(\d+)\.npy$")


def parse_id(p: Path) -> int:
    m = ID_RE.match(p.name)
    if not m:
        raise ValueError(f"Cannot parse id from filename: {p}")
    return int(m.group(1))


def load_all_fht(det_feat_dir: Path):
    # 仅统计 PMT 的 FHT
    fht_pmt_files = sorted(det_feat_dir.glob("fht_pmt_*.npy"), key=parse_id)
    if not fht_pmt_files:
        raise RuntimeError(f"No fht_pmt files found in {det_feat_dir}")

    all_values = []
    for fp in fht_pmt_files:
        arr = np.load(fp)  # shape: (E, Npmt)
        vals = arr.reshape(-1)
        # 只保留有效命中，且去除超过 1000 的异常/离群值
        vals = vals[(vals > 0) & (vals <= 1000)]
        if vals.size:
            all_values.append(vals)

    if not all_values:
        raise RuntimeError("No valid FHT values found in PMT (0<FHT<=1000).")
    return np.concatenate(all_values, axis=0)

def load_eventwise_min_fht(det_feat_dir: Path):
    """
    返回每个 event 的最小 FHT（在所有 PMT 上取最小）。
    仅在 0<FHT<=1000 的范围内计算；若某个 event 没有任何有效命中，则跳过该 event。
    """
    fht_pmt_files = sorted(det_feat_dir.glob("fht_pmt_*.npy"), key=parse_id)
    if not fht_pmt_files:
        raise RuntimeError(f"No fht_pmt files found in {det_feat_dir}")

    per_event_min = []
    for fp in fht_pmt_files:
        arr = np.load(fp)  # (E, Npmt)
        # 过滤无效与>1000的值；无效位置设为 +inf，便于取最小
        valid = np.where((arr > 0) & (arr <= 1000), arr, np.inf)  # (E, Npmt)
        mins = np.min(valid, axis=1)  # (E,)
        # 仅保留有至少一个有效 PMT 的 event（min < inf）
        keep = mins[np.isfinite(mins)]
        if keep.size:
            per_event_min.append(keep)
    if not per_event_min:
        raise RuntimeError("No per-event minimum FHT available (after filtering 0<FHT<=1000).")
    return np.concatenate(per_event_min, axis=0)


def summarize(values: np.ndarray):
    stats = {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p1": float(np.percentile(values, 1)),
        "p5": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }
    return stats


def plot_distribution(values: np.ndarray, out_path: Path, bins=200):
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=bins, density=True, alpha=0.6, color="tab:blue", label="Histogram")
    hist, bin_edges = np.histogram(values, bins=bins, density=True)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    window = 5
    if hist.size >= window:
        kernel = np.ones(window) / window
        smooth = np.convolve(hist, kernel, mode="same")
        plt.plot(centers, smooth, color="tab:red", linewidth=2, label="Smoothed density")

    plt.xlabel("FHT (time units)")
    plt.ylabel("Density")
    plt.title("FHT distribution (PMT only, 0<FHT<=1000)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()



def main():
    ap = argparse.ArgumentParser(description="Analyze FHT distribution across det_feat (PMT only).")
    ap.add_argument("--det-feat-dir", default="/disk_pool1/weijsh/e+/det_feat", help="Path to det_feat directory")
    ap.add_argument("--out-dir", default="/home/houyh/Segmentation-JUNO", help="Directory to save outputs")
    ap.add_argument("--bins", type=int, default=200, help="Histogram bins")
    args = ap.parse_args()

    det_feat_dir = Path(args.det_feat_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 全局 FHT 分布（所有 PMT 的所有命中）
    values = load_all_fht(det_feat_dir)
    stats = summarize(values)

    print("FHT stats (PMT, 0<FHT<=1000):")
    for k, v in stats.items():
        print(f"- {k}: {v}")

    np.save(out_dir / "fht_values_sample.npy", values[:min(values.size, 1_000_000)])
    with open(out_dir / "fht_stats.txt", "w", encoding="utf-8") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")

    plot_distribution(values, out_dir / "fht_pmt_distribution.png", bins=args.bins)
    print(f"Saved plot to {out_dir / 'fht_pmt_distribution.png'} and stats to {out_dir / 'fht_stats.txt'}")

    # 每个 event 的最小 FHT 分布
    event_min_values = load_eventwise_min_fht(det_feat_dir)
    event_min_stats = summarize(event_min_values)
    print("Per-event minimum FHT stats (PMT, 0<FHT<=1000):")
    for k, v in event_min_stats.items():
        print(f"- {k}: {v}")
    with open(out_dir / "fht_eventmin_stats.txt", "w", encoding="utf-8") as f:
        for k, v in event_min_stats.items():
            f.write(f"{k}: {v}\n")

    # 画每个 event 的最小 FHT 的分布图
    plt.figure(figsize=(10, 6))
    plt.hist(event_min_values, bins=args.bins, density=True, alpha=0.6, color="tab:green", label="Histogram")
    hist, bin_edges = np.histogram(event_min_values, bins=args.bins, density=True)
    centers = (bin_edges[:-1] + bin_edges[1] ) / 2.0
    window = 5
    if hist.size >= window:
        kernel = np.ones(window) / window
        smooth = np.convolve(hist, kernel, mode="same")
        plt.plot(centers, smooth, color="tab:orange", linewidth=2, label="Smoothed density")
    plt.xlabel("Per-event minimum FHT (time units)")
    plt.ylabel("Density")
    plt.title("Per-event minimum FHT distribution (PMT only, 0<FHT<=1000)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fht_pmt_eventmin_distribution.png")
    plt.close()
    print(f"Saved plot to {out_dir / 'fht_pmt_eventmin_distribution.png'} and stats to {out_dir / 'fht_eventmin_stats.txt'}")

if __name__ == "__main__":
    main()
# import argparse
# from pathlib import Path
# import re
# import numpy as np
# import matplotlib.pyplot as plt
# from typing import List, Dict

# ID_RE = re.compile(r".*_(\d+)\.npy$")

# def parse_id(p: Path) -> int:
#     m = ID_RE.match(p.name)
#     if not m:
#         raise ValueError(f"Cannot parse id from filename: {p}")
#     return int(m.group(1))

# def load_all_pmt_fht(det_feat_dir: Path):
#     files = sorted(det_feat_dir.glob("fht_pmt_*.npy"), key=parse_id)
#     if not files:
#         raise RuntimeError("No fht_pmt files.")
#     events = [np.load(fp) for fp in files]  # list of (E, Npmt)
#     return np.concatenate(events, axis=0)   # (B_total, Npmt)

# def sample_delays(mode: str, mean: float, k: float, size: int, dmin: float, dmax: float, rng: np.random.Generator):
#     if mode == "gamma":
#         theta = mean / max(k, 1e-6)
#         d = rng.gamma(shape=k, scale=theta, size=size)
#     elif mode == "exp":
#         lam = 1.0 / max(mean, 1e-6)
#         d = rng.exponential(scale=1.0/lam, size=size)
#     else:  # fixed provided in sweep
#         raise ValueError("Use fixed delays via --fixed-delays for 'fixed' mode.")
#     return np.clip(d, dmin, dmax)

# def sweep_fixed_delays(fht: np.ndarray, delays: List[float], max_time: float = 1000.0):
#     """
#     fht: (B, Npmt)
#     return:
#       - overflow_ratios_mean: list of mean overflow ratio across events for each delay
#       - earliest_after: dict delay-> per-event earliest FHT array after shift (only considering hits>0)
#       - overflow_per_event: dict delay-> per-event overflow ratios (list/array)
#     """
#     mask_all = fht > 0
#     B = fht.shape[0]
#     overflow_ratios_mean: List[float] = []
#     earliest_after: Dict[float, np.ndarray] = {}
#     overflow_per_event: Dict[float, np.ndarray] = {}
#     for d in delays:
#         shifted = fht.copy()
#         shifted[mask_all] = shifted[mask_all] + d
#         overflow = (shifted > max_time) & mask_all
#         # per-event ratio: (#overflow_hits)/(#valid_hits)
#         denom = np.count_nonzero(mask_all, axis=1)  # (B,)
#         denom = np.where(denom > 0, denom, 1)
#         per_event_ratio = np.count_nonzero(overflow, axis=1) / denom  # (B,)
#         overflow_ratios_mean.append(float(np.mean(per_event_ratio)))
#         overflow_per_event[d] = per_event_ratio
#         # per-event earliest FHT (inf for no hit)
#         valid = np.where((shifted > 0) & (shifted <= max_time), shifted, np.inf)
#         earliest = np.min(valid, axis=1)  # (B,)
#         earliest_after[d] = earliest[np.isfinite(earliest)]
#     return overflow_ratios_mean, earliest_after, overflow_per_event

# def plot_overflow(delays, ratios, out_png: Path):
#     plt.figure(figsize=(8,5))
#     plt.plot(delays, ratios, marker="o")
#     plt.xlabel("Fixed delay (ns)")
#     plt.ylabel("Mean overflow ratio (>1000) across events")
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(out_png)
#     plt.close()

# def plot_earliest_hist(earliest_after: Dict[float, np.ndarray], out_png: Path, bins=100):
#     plt.figure(figsize=(10,6))
#     for d, arr in earliest_after.items():
#         hist, edges = np.histogram(arr, bins=bins, density=True)
#         centers = (edges[:-1] + edges[1:]) / 2
#         plt.plot(centers, hist, label=f"delay={d:g} ns", linewidth=1.5)
#     plt.xlabel("Earliest FHT after shift (ns)")
#     plt.ylabel("Density")
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(out_png)
#     plt.close()

# def plot_overflow_box(overflow_per_event: Dict[float, np.ndarray], out_png: Path):
#     """箱线图：每个 delay 下的 per-event overflow ratio 分布"""
#     delays = sorted(overflow_per_event.keys())
#     data = [overflow_per_event[d] for d in delays]
#     plt.figure(figsize=(10,6))
#     plt.boxplot(data, labels=[f"{int(d)}" for d in delays], showfliers=False)
#     plt.xlabel("Fixed delay (ns)")
#     plt.ylabel("Per-event overflow ratio (>1000)")
#     plt.grid(True, axis="y", alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(out_png)
#     plt.close()

# def plot_overflow_percentiles(overflow_per_event: Dict[float, np.ndarray], out_png: Path):
#     """分位数曲线：p50/p90/p99 随 delay 变化"""
#     delays = sorted(overflow_per_event.keys())
#     p50, p90, p99 = [], [], []
#     for d in delays:
#         arr = overflow_per_event[d]
#         p50.append(float(np.percentile(arr, 50)))
#         p90.append(float(np.percentile(arr, 90)))
#         p99.append(float(np.percentile(arr, 99)))
#     plt.figure(figsize=(10,6))
#     plt.plot(delays, p50, marker="o", label="p50")
#     plt.plot(delays, p90, marker="o", label="p90")
#     plt.plot(delays, p99, marker="o", label="p99")
#     plt.xlabel("Fixed delay (ns)")
#     plt.ylabel("Per-event overflow ratio (>1000)")
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(out_png)
#     plt.close()

# def main():
#     ap = argparse.ArgumentParser("Sweep PMT delays and measure overflow/earliest distributions.")
#     ap.add_argument("--det-feat-dir", default="/disk_pool1/weijsh/e+/det_feat")
#     ap.add_argument("--out-dir", default="./delay_sweep_out")
#     ap.add_argument("--fixed-delays", type=str, default="0,50,100,150,200,300,400,500,600",
#                     help="Comma-separated fixed delays to test (ns)")
#     ap.add_argument("--max-time", type=float, default=1000.0)
#     ap.add_argument("--bins", type=int, default=150)
#     args = ap.parse_args()

#     det_feat_dir = Path(args.det_feat_dir)
#     out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

#     fht = load_all_pmt_fht(det_feat_dir)  # (B, Npmt)
#     delays = [float(x) for x in args.fixed_delays.split(",")]
#     ratios, earliest_after, overflow_per_event = sweep_fixed_delays(fht, delays, max_time=args.max_time)

#     plot_overflow(delays, ratios, out_dir / "overflow_ratio_vs_delay.png")
#     plot_earliest_hist(earliest_after, out_dir / "earliest_after_shift.png", bins=args.bins)
#     plot_overflow_box(overflow_per_event, out_dir / "overflow_ratio_boxplot.png")
#     plot_overflow_percentiles(overflow_per_event, out_dir / "overflow_ratio_percentiles.png")

#     print("Delays:", delays)
#     print("Mean overflow ratios:", ratios)

# if __name__ == "__main__":
#     main()
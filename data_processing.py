import argparse
from pathlib import Path
import re
import numpy as np
from typing import Tuple 

ID_RE = re.compile(r".*_(\d+)\.npy$")

# --- 延迟分布及约束参数（可用命令行覆盖） ---
DELAY_MODE = "gamma"     # "gamma" 或 "exp"
DELAY_MEAN = 300.0       # ns，基于统计均值 ~31
DELAY_MIN = 200.0        # ns
DELAY_MAX = 600.0        # ns
GAMMA_K = 2.0            # gamma 的 shape
MAX_TIME = 1000.0        # FHT 上限
MAX_OVERFLOW_RATIO = 0.03
MAX_DELAY_RETRIES = 4
DELAY_REDUCE_FACTOR = 0.7


def pair_adjacent_events_concat(y10: np.ndarray) -> np.ndarray:
    """
    将 10 个 event 两两相邻拼接：
      输入:  (10, D)
      输出:  (5,  2D)  => [event0||event1, event2||event3, ...]
    """
    if y10.ndim != 2:
        raise ValueError(f"Expect 2D array (10, D), got shape={y10.shape}")
    if y10.shape[0] != 10:
        raise ValueError(f"Expect 10 events, got shape={y10.shape}")
    a = y10[0::2]  # (5, D)
    b = y10[1::2]  # (5, D)
    return np.concatenate([a, b], axis=1)  # (5, 2D)

def parse_id(p: Path) -> int:
    m = ID_RE.match(p.name)
    if not m:
        raise ValueError(f"Cannot parse id from filename: {p}")
    return int(m.group(1))

def find_common_ids(det_feat_dir: Path, y_dir: Path):
    def ids_for(pattern: str):
        return {parse_id(p) for p in det_feat_dir.glob(pattern)}
    ids = (
        ids_for("fht_pix_*.npy")
        & ids_for("npe_pix_*.npy")
        & ids_for("fht_pmt_*.npy")
        & ids_for("npe_pmt_*.npy")
        & {parse_id(p) for p in y_dir.glob("y_*.npy")}
    )
    return sorted(ids)

def combine_fht_min_nonzero(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_ = np.where(a > 0, a, np.inf)
    b_ = np.where(b > 0, b, np.inf)
    m = np.minimum(a_, b_)
    return np.where(np.isinf(m), 0.0, m)

# --- 采样延迟 + 约束 ---
def sample_delay(rng: np.random.Generator) -> float:
    if DELAY_MODE == "gamma":
        theta = DELAY_MEAN / GAMMA_K
        d = rng.gamma(shape=GAMMA_K, scale=theta)
    else:
        lam = 1.0 / max(DELAY_MEAN, 1e-6)
        d = rng.exponential(scale=1.0 / lam)
    return float(np.clip(d, DELAY_MIN, DELAY_MAX))

def apply_delay_to_event_fht(
    fht_event: np.ndarray,
    rng: np.random.Generator,
    max_time: float = MAX_TIME,
    max_overflow_ratio: float = MAX_OVERFLOW_RATIO,
    max_retries: int = MAX_DELAY_RETRIES,
) -> Tuple[np.ndarray, float, float]:
    """
    将该 event 的所有 PMT FHT 右移 delay。
    控制：超 1000 的比例 <= max_overflow_ratio；否则按 DELAY_REDUCE_FACTOR 缩减，且不低于 DELAY_MIN。
    返回 (shifted_fht, used_delay, overflow_ratio)。
    """
    fht = fht_event.copy()
    mask = fht > 0
    if not np.any(mask):
        return fht, 0.0, 0.0

    delay = sample_delay(rng)

    for _ in range(max_retries + 1):
        shifted = fht.copy()
        shifted[mask] = shifted[mask] + delay
        overflow = (shifted > max_time) & mask
        denom = float(np.count_nonzero(mask)) or 1.0
        overflow_ratio = float(np.count_nonzero(overflow)) / denom
        if overflow_ratio <= max_overflow_ratio:
            shifted[overflow] = 0.0
            return shifted, delay, overflow_ratio
        # 缩减 delay，但不低于 DELAY_MIN
        delay = max(DELAY_MIN, delay * DELAY_REDUCE_FACTOR)

    # 兜底：应用最后的 delay，并把超限置 0
    shifted = fht.copy()
    shifted[mask] = shifted[mask] + delay
    overflow = (shifted > max_time) & mask
    shifted[overflow] = 0.0
    denom = float(np.count_nonzero(mask)) or 1.0
    overflow_ratio = float(np.count_nonzero(overflow)) / denom
    return shifted, delay, overflow_ratio

# --- 仅 PMT 合并（对第二个 event 施加延迟），并记录日志 ---
def combine_group_adjacent_pmt_with_delay_return_sources(
    fht_pmt: np.ndarray,
    npe_pmt: np.ndarray,
    rng: np.random.Generator,
    max_time: float = MAX_TIME,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    输入：(10, Npmt)，配对 (0,1),(2,3),...,(8,9)。
    返回：
      - tq_pair: (5, Npmt, 2, 2)  [event(0/1), feature(t=0/q=1)]
      - fht_comb: (5, Npmt)      合并后 FHT（min-非零）
      - npe_comb: (5, Npmt)      合并后 NPE（求和）
      - target:   (5, Npmt, 2)   [event1_hit, event2_hit]，命中=(FHT>0 且 NPE>0)
    """
    assert fht_pmt.shape[0] == 10 and npe_pmt.shape[0] == 10, "Expect 10 events per group"
    pairs = [(i, i + 1) for i in range(0, 10, 2)]
    tq_list, fht_list, npe_list, tgt_list = [], [], [], []

    for i, j in pairs:
        # 第二路施加延迟
        fht_j_shifted, used_delay, overflow_ratio = apply_delay_to_event_fht(
            fht_pmt[j], rng, max_time=max_time
        )

        # 两路源：event1 与 delayed event2
        t2 = np.stack([fht_pmt[i], fht_j_shifted], axis=-1)  # (Npmt, 2)
        q2 = np.stack([npe_pmt[i], npe_pmt[j]], axis=-1)     # (Npmt, 2)
        # 组装为 (Npmt, 2, 2) -> [event, feature(t/q)]
        tq = np.stack([t2, q2], axis=-1)                    # (Npmt, 2, 2)
        tq_list.append(tq)

        # 合并后的 FHT/NPE
        fht_comb = combine_fht_min_nonzero(fht_pmt[i], fht_j_shifted)
        npe_comb = npe_pmt[i] + npe_pmt[j]
        fht_list.append(fht_comb)
        npe_list.append(npe_comb)

        # 命中标签：两路各自 (FHT>0 且 NPE>0)
        hit1 = (t2[:, 0] > 0) & (q2[:, 0] > 0)
        hit2 = (t2[:, 1] > 0) & (q2[:, 1] > 0)
        target = np.stack([hit1.astype(np.int32), hit2.astype(np.int32)], axis=-1)  # (Npmt, 2)
        tgt_list.append(target)

    tq_pair  = np.stack(tq_list,  axis=0)  # (5, Npmt, 2, 2)
    fht_comb = np.stack(fht_list, axis=0)  # (5, Npmt)
    npe_comb = np.stack(npe_list, axis=0)  # (5, Npmt)
    target   = np.stack(tgt_list, axis=0)  # (5, Npmt, 2)
    return tq_pair, fht_comb, npe_comb, target


def main():
    global DELAY_MODE, DELAY_MEAN, DELAY_MIN, DELAY_MAX, MAX_OVERFLOW_RATIO, DELAY_REDUCE_FACTOR
    ap = argparse.ArgumentParser(description="PMT-only adjacent pairing with delayed second event.")
    ap.add_argument("--det-feat-dir", default="/disk_pool1/weijsh/e+/det_feat", help="Path to det_feat directory")
    ap.add_argument("--y-dir", default="/disk_pool1/weijsh/e+/y", help="Path to y directory (for id intersection)")
    ap.add_argument("--out-dir", default="/disk_pool1/houyh/data/mixed/", help="Output directory for combined PMT")
    ap.add_argument("--start-id", type=int, default=None, help="Optional start id")
    ap.add_argument("--end-id", type=int, default=None, help="Optional end id (inclusive)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    # 延迟与约束（取消 max-earliest-fht）
    ap.add_argument("--delay-mode", choices=["gamma", "exp"], default=DELAY_MODE)
    ap.add_argument("--delay-mean", type=float, default=DELAY_MEAN)
    ap.add_argument("--delay-min", type=float, default=DELAY_MIN)
    ap.add_argument("--delay-max", type=float, default=DELAY_MAX)
    ap.add_argument("--overflow-ratio", type=float, default=MAX_OVERFLOW_RATIO)
    ap.add_argument("--delay-reduce-factor", type=float, default=DELAY_REDUCE_FACTOR)  # 新增
    args = ap.parse_args()

    DELAY_MODE = args.delay_mode
    DELAY_MEAN = args.delay_mean
    DELAY_MIN = args.delay_min
    DELAY_MAX = args.delay_max
    MAX_OVERFLOW_RATIO = args.overflow_ratio
    DELAY_REDUCE_FACTOR = args.delay_reduce_factor

    det_feat_dir = Path(args.det_feat_dir)
    y_dir = Path(args.y_dir)
    out_root = Path(args.out_dir)
    out_tq   = out_root / "tq_pair"
    out_tgt  = out_root / "target"
    out_y    = out_root / "y_pair"
    out_meta = out_root / "meta"
    out_tq.mkdir(parents=True, exist_ok=True)
    out_tgt.mkdir(parents=True, exist_ok=True)
    out_y.mkdir(parents=True, exist_ok=True)
    out_meta.mkdir(parents=True, exist_ok=True)


    ids = find_common_ids(det_feat_dir, y_dir)
    if args.start_id is not None:
        ids = [i for i in ids if i >= args.start_id]
    if args.end_id is not None:
        ids = [i for i in ids if i <= args.end_id]

    if len(ids) < 2:
        raise RuntimeError("Need at least two ids to form a 10-event group.")
    groups = [ids[i:i+2] for i in range(0, len(ids) - 1, 2)]
    rng = np.random.default_rng(args.seed)

    for out_idx, g in enumerate(groups):
        if len(g) != 2:
            continue
        a, b = g
        # 只读 PMT 级别
        fht_pmt_a = np.load(det_feat_dir / f"fht_pmt_{a}.npy")  # (5, Npmt)
        npe_pmt_a = np.load(det_feat_dir / f"npe_pmt_{a}.npy")
        fht_pmt_b = np.load(det_feat_dir / f"fht_pmt_{b}.npy")
        npe_pmt_b = np.load(det_feat_dir / f"npe_pmt_{b}.npy")

        # 读取 y：每个 (5, 15)
        y_a = np.load(y_dir / f"y_{a}.npy")
        y_b = np.load(y_dir / f"y_{b}.npy")
        if y_a.shape[0] != 5 or y_b.shape[0] != 5:
            raise ValueError(f"Expect y_* to be (5, 15); got y_{a}={y_a.shape}, y_{b}={y_b.shape}")


        # 拼成 10 个 event
        fht_pmt = np.concatenate([fht_pmt_a, fht_pmt_b], axis=0)
        npe_pmt = np.concatenate([npe_pmt_a, npe_pmt_b], axis=0)
        y10 = np.concatenate([y_a, y_b], axis=0)  # (10, 15)

        # 仅 PMT 合并 + 延迟（含源与标签）
        tq_pair, Fm, Nm, Tm = combine_group_adjacent_pmt_with_delay_return_sources(
            fht_pmt, npe_pmt, rng, max_time=MAX_TIME
        )

        # y 相邻两两拼接 -> (5, 30)
        y_pair = pair_adjacent_events_concat(y10)

        base = f"{out_idx}"
        # np.save(out_tq  / f"tq_pair_{base}.npy", tq_pair)  # (5, Npmt, 2, 2)
        # np.save(out_tgt / f"target_{base}.npy", Tm)        # (5, Npmt, 2)
        # np.save(out_y   / f"y_{base}.npy", y_pair)    # (5, 30)

        # global indices 0..9 correspond to [a:0..4, b:0..4] via:
        #   if gidx<5 => (a, gidx), else => (b, gidx-5)
        pairs = [(0,1),(2,3),(4,5),(6,7),(8,9)]
        src_file = np.zeros((5, 2), dtype=np.int64)
        src_loc  = np.zeros((5, 2), dtype=np.int64)
        for k, (i, j) in enumerate(pairs):
            for t, gidx in enumerate((i, j)):
                if gidx < 5:
                    src_file[k, t] = a
                    src_loc[k, t] = gidx
                else:
                    src_file[k, t] = b
                    src_loc[k, t] = gidx - 5

        np.save(out_meta / f"src_file_ids_{base}.npy", src_file)  # (5,2)
        np.save(out_meta / f"src_local_idx_{base}.npy", src_loc)  # (5,2)

        print(f"Group ({a},{b}) -> saved {base} -> dirs: tq_pair/, target/, y_pair/, meta/")

    print("Done.")

if __name__ == "__main__":
    main()
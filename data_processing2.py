import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np

ID_RE = re.compile(r".*_(\d+)\.npy$")

NPmt = 17612
FHT_WINDOW_MAX = 1000.0


def parse_id(p: Path) -> int:
    m = ID_RE.match(p.name)
    if not m:
        raise ValueError(f"Bad filename: {p}")
    return int(m.group(1))


def find_common_ids(det_feat_dir: Path, y_dir: Path):
    fht_ids = {parse_id(p) for p in det_feat_dir.glob("fht_pmt_*.npy")}
    npe_ids = {parse_id(p) for p in det_feat_dir.glob("npe_pmt_*.npy")}
    y_ids = {parse_id(p) for p in y_dir.glob("y_*.npy")}
    return sorted(fht_ids & npe_ids & y_ids)


def load_per_event_stats_jsonl(jsonl_path: Path) -> Dict[Tuple[int, int], Dict[str, Any]]:
    out: Dict[Tuple[int, int], Dict[str, Any]] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[(int(rec["file_id"]), int(rec["local_event"]))] = rec
    return out


def _draw_npe_from_dist_1_6(rng: np.random.Generator, dist_1_6: Dict[str, float], size: int) -> np.ndarray:
    probs = np.array([float(dist_1_6.get(str(k), 0.0)) for k in range(1, 7)], dtype=np.float64)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0:
        probs = np.ones(6, dtype=np.float64) / 6.0
    else:
        probs = probs / s
    vals = np.arange(1, 7, dtype=np.int64)
    return rng.choice(vals, size=size, replace=True, p=probs)


def sample_noise_hits(
    rng: np.random.Generator,
    hit_e_mask: np.ndarray,
    nhits_noise: int,
    fht_min_hit: float,
    npe_dist_1_6: Dict[str, float],
    n_pmt: int = NPmt,
    fht_hi: float = FHT_WINDOW_MAX,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回 noise 的 (pmt_ids, t, q) 的 hit-list。允许同一 PMT 多次命中（pmt_ids 可重复）。
    """
    if nhits_noise <= 0:
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    lo = float(fht_min_hit) if (fht_min_hit is not None and np.isfinite(fht_min_hit)) else 0.0
    lo = max(0.0, min(lo, fht_hi))
    hi = float(fht_hi)
    if hi <= lo:
        lo = 0.0

    free_idx = np.flatnonzero(~hit_e_mask)
    if free_idx.size > 0:
        # 优先选不被 e+ 命中的 PMT；不足就回退到全体
        if free_idx.size >= nhits_noise:
            chosen_pmt = rng.choice(free_idx, size=nhits_noise, replace=False).astype(np.int32)
        else:
            chosen_pmt = rng.choice(np.arange(n_pmt), size=nhits_noise, replace=True).astype(np.int32)
    else:
        chosen_pmt = rng.choice(np.arange(n_pmt), size=nhits_noise, replace=True).astype(np.int32)

    t = rng.uniform(lo, hi, size=nhits_noise).astype(np.float32)
    q = _draw_npe_from_dist_1_6(rng, npe_dist_1_6, nhits_noise).astype(np.float32)
    return chosen_pmt, t, q


def sort_hits_within_pmt(pmt: np.ndarray, t: np.ndarray, q: np.ndarray, src: np.ndarray):
    """
    全局排序：按 (pmt_id, t) 升序，使得每个 PMT 内 hits 时间有序。
    """
    order = np.lexsort((t, pmt))
    return pmt[order], t[order], q[order], src[order]


def main():
    ap = argparse.ArgumentParser("Generate hit-list dataset (variable-length per PMT) for JUNO.")
    ap.add_argument("--det-feat-dir", default="/disk_pool1/weijsh/e+/det_feat")
    ap.add_argument("--y-dir", default="/disk_pool1/weijsh/e+/y")
    ap.add_argument("--out-dir", default="/disk_pool1/houyh/data/scattered")
    ap.add_argument("--stats-dir", default="/disk_pool1/houyh/data/scattered")
    ap.add_argument("--start-id", type=int, default=None)
    ap.add_argument("--end-id", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--clip-eplus-to-1000", action="store_true", help="Optionally drop e+ hits with t>1000 (set to 0).")
    args = ap.parse_args()

    det_feat_dir = Path(args.det_feat_dir)
    y_dir = Path(args.y_dir)
    out_root = Path(args.out_dir)
    stats_dir = Path(args.stats_dir)

    out_hits = out_root / "hits"
    out_hits.mkdir(parents=True, exist_ok=True)

    stats_path = stats_dir / "per_event_stats.jsonl"
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing stats: {stats_path} (run test.py first)")
    stats_map = load_per_event_stats_jsonl(stats_path)

    ids = find_common_ids(det_feat_dir, y_dir)
    if args.start_id is not None:
        ids = [i for i in ids if i >= args.start_id]
    if args.end_id is not None:
        ids = [i for i in ids if i <= args.end_id]
    if not ids:
        raise RuntimeError("No common ids found")

    rng = np.random.default_rng(args.seed)

    global_event_index = 0
    for file_id in ids:
        fht = np.load(det_feat_dir / f"fht_pmt_{file_id}.npy").astype(np.float32)  # (5,N)
        npe = np.load(det_feat_dir / f"npe_pmt_{file_id}.npy").astype(np.float32)  # (5,N)
        y = np.load(y_dir / f"y_{file_id}.npy").astype(np.float32)                 # (5,15)

        if fht.shape != npe.shape or fht.ndim != 2 or fht.shape[0] != 5:
            raise ValueError(f"Bad shape id={file_id}: fht={fht.shape} npe={npe.shape}")
        if fht.shape[1] != NPmt:
            raise ValueError(f"Npmt mismatch id={file_id}: got {fht.shape[1]} expected {NPmt}")

        for local_evt in range(5):
            fht_e = fht[local_evt].copy()
            npe_e = npe[local_evt].copy()

            # e+ hits
            hit_e = (fht_e > 0) & (npe_e > 0)

            if args.clip_eplus_to_1000:
                over = fht_e > FHT_WINDOW_MAX
                if np.any(over):
                    fht_e[over] = 0.0
                    npe_e[over] = 0.0
                    hit_e = (fht_e > 0) & (npe_e > 0)

            key = (int(file_id), int(local_evt))
            if key not in stats_map:
                raise KeyError(f"Missing stats for file_id={file_id} local_event={local_evt}")
            st = stats_map[key]

            nh_noise = int(st.get("Nhits_le_1000", 0))
            npe_dist = st.get("npe_dist_le_1000_1_6", {str(k): 0.0 for k in range(1, 7)})
            fht_min_hit = st.get("fht_min_hit", 0.0)

            # e+ hit-list（src=0）
            pmt_e = np.flatnonzero(hit_e).astype(np.int32)
            t_e = fht_e[hit_e].astype(np.float32)
            q_e = npe_e[hit_e].astype(np.float32)
            src_e = np.zeros((pmt_e.size,), dtype=np.uint8)

            # noise hit-list（src=1）
            pmt_n, t_n, q_n = sample_noise_hits(
                rng=rng,
                hit_e_mask=hit_e,
                nhits_noise=nh_noise,
                fht_min_hit=fht_min_hit,
                npe_dist_1_6=npe_dist,
                n_pmt=NPmt,
                fht_hi=FHT_WINDOW_MAX,
            )
            src_n = np.ones((pmt_n.size,), dtype=np.uint8)

            # 合并 + 按 (pmt,t) 排序，保证每个 PMT 内时间有序
            pmt_all = np.concatenate([pmt_e, pmt_n], axis=0)
            t_all = np.concatenate([t_e, t_n], axis=0)
            q_all = np.concatenate([q_e, q_n], axis=0)
            src_all = np.concatenate([src_e, src_n], axis=0)

            if pmt_all.size > 0:
                pmt_all, t_all, q_all, src_all = sort_hits_within_pmt(pmt_all, t_all, q_all, src_all)

            # PMT 标签：是否出现过 e+/noise
            pmt_label = np.zeros((NPmt, 2), dtype=np.uint8)
            if pmt_e.size > 0:
                pmt_label[pmt_e, 0] = 1
            if pmt_n.size > 0:
                # 注意：pmt_n 可重复，unique 后设置即可
                pmt_label[np.unique(pmt_n), 1] = 1

            out_path = out_hits / f"event_{global_event_index:06d}.npz"
            np.savez_compressed(
                out_path,
                file_id=np.int64(file_id),
                local_event=np.int64(local_evt),
                y=y[local_evt],  # (15,)
                hit_pmt=pmt_all.astype(np.int32),
                hit_t=t_all.astype(np.float32),
                hit_q=q_all.astype(np.float32),
                hit_src=src_all.astype(np.uint8),     # 0=e+, 1=noise
                pmt_label=pmt_label.astype(np.uint8), # (NPmt,2)
            )

            global_event_index += 1

        if (global_event_index % 250) == 0:
            print(f"processed events: {global_event_index}")

    print(f"Done. total_events={global_event_index} -> {out_hits}")
    

if __name__ == "__main__":
    main()
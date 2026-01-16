import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np


NPmt = 17612
CHUNK_RE = re.compile(r"^(AP|ME)_chunk_(\d+)\.npy$")


def list_chunks(root: Path, kind: str) -> Dict[int, Path]:
    """
    Return mapping: chunk_index -> path for files like '{kind}_chunk_{i}.npy'.
    """
    out: Dict[int, Path] = {}
    for p in root.glob(f"{kind}_chunk_*.npy"):
        m = CHUNK_RE.match(p.name)
        if not m:
            continue
        idx = int(m.group(2))
        out[idx] = p
    return out


def load_chunk_events(path: Path) -> np.ndarray:
    """
    Each chunk is an object array: shape (N_events,), each element is (Nhits, 3) with dtype=object.
    Columns: [PMTID, Time, Charge]
    """
    arr = np.load(path, allow_pickle=True)
    if arr.dtype != object or arr.ndim != 1:
        raise ValueError(f"Bad chunk format: {path} got shape={arr.shape} dtype={arr.dtype}")
    return arr


def event_to_numpy_hits(ev) -> np.ndarray:
    """
    Convert one event (object) to ndarray shape (Nhits,3).
    """
    a = np.array(ev, dtype=object)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"Bad event hit table: shape={a.shape} (expected (Nhits,3))")
    return a


def sanitize_hits(
    hit_tab: np.ndarray,
    n_pmt: int = NPmt,
    time_max: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    hit_tab: (Nhits,3) object array [pmtid, time, charge]
    Return pmt(int32), t(float32), q(float32) after filtering invalid rows.
    """
    # Cast
    pmt = hit_tab[:, 0].astype(np.int64, copy=False)
    t = hit_tab[:, 1].astype(np.float32, copy=False)
    q = hit_tab[:, 2].astype(np.float32, copy=False)

    # Basic validity
    ok = np.isfinite(t) & np.isfinite(q)
    ok &= (q > 0)

    # PMT range filter (important because ME PMTID currently looks like raw addresses)
    ok &= (pmt >= 0) & (pmt < int(n_pmt))

    if time_max is not None:
        ok &= (t >= 0) & (t <= float(time_max))

    pmt = pmt[ok].astype(np.int32, copy=False)
    t = t[ok].astype(np.float32, copy=False)
    q = q[ok].astype(np.float32, copy=False)
    return pmt, t, q


def downsample_hits(
    rng: np.random.Generator,
    pmt: np.ndarray,
    t: np.ndarray,
    q: np.ndarray,
    max_hits: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_hits is None or pmt.size <= max_hits:
        return pmt, t, q
    idx = rng.choice(pmt.size, size=int(max_hits), replace=False)
    return pmt[idx], t[idx], q[idx]


def sort_hits(pmt: np.ndarray, t: np.ndarray, q: np.ndarray, lbl: np.ndarray):
    order = np.lexsort((t, pmt))
    return pmt[order], t[order], q[order], lbl[order]


def main():
    ap = argparse.ArgumentParser("Build AP/ME mixed hit-level dataset (per-hit binary classification).")
    ap.add_argument("--ap-dir", default="/disk_pool1/houyh/data/AP_ME_raw/AP_chunks", help="Directory containing AP_chunk_*.npy")
    ap.add_argument("--me-dir", default="/disk_pool1/houyh/data/AP_ME_raw/ME_chunks", help="Directory containing ME_chunk_*.npy")
    ap.add_argument("--out-dir", default="/disk_pool1/houyh/data/AP_ME", help="Output dataset root")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--time-max", type=float, default=None, help="Optional: keep hits with 0<=t<=time-max")
    ap.add_argument(
        "--max-hits-per-class",
        type=int,
        default=None,
        help="Optional: cap hits per class (AP and ME separately) by random subsampling",
    )
    ap.add_argument("--max-events", type=int, default=None, help="Optional cap on total output events (for debugging)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    ap_root = Path(args.ap_dir)
    me_root = Path(args.me_dir)
    out_root = Path(args.out_dir)
    out_hits = out_root / "hits"
    out_hits.mkdir(parents=True, exist_ok=True)

    ap_chunks = list_chunks(ap_root, "AP")
    me_chunks = list_chunks(me_root, "ME")

    common_idx = sorted(set(ap_chunks.keys()) & set(me_chunks.keys()))
    if not common_idx:
        raise RuntimeError(f"No matching chunk indices between {ap_root} and {me_root}")

    global_event_index = 0
    for ci in common_idx:
        ap_path = ap_chunks[ci]
        me_path = me_chunks[ci]

        ap_events = load_chunk_events(ap_path)
        me_events = load_chunk_events(me_path)

        n_evt = min(len(ap_events), len(me_events))
        for ei in range(n_evt):
            ap_tab = event_to_numpy_hits(ap_events[ei])
            me_tab = event_to_numpy_hits(me_events[ei])

            ap_pmt, ap_t, ap_q = sanitize_hits(ap_tab, n_pmt=NPmt, time_max=args.time_max)
            me_pmt, me_t, me_q = sanitize_hits(me_tab, n_pmt=NPmt, time_max=args.time_max)

            ap_pmt, ap_t, ap_q = downsample_hits(rng, ap_pmt, ap_t, ap_q, args.max_hits_per_class)
            me_pmt, me_t, me_q = downsample_hits(rng, me_pmt, me_t, me_q, args.max_hits_per_class)

            ap_lbl = np.zeros((ap_pmt.size,), dtype=np.uint8)  # 0=AP
            me_lbl = np.ones((me_pmt.size,), dtype=np.uint8)   # 1=ME

            hit_pmt = np.concatenate([ap_pmt, me_pmt], axis=0)
            hit_t = np.concatenate([ap_t, me_t], axis=0)
            hit_q = np.concatenate([ap_q, me_q], axis=0)
            hit_label = np.concatenate([ap_lbl, me_lbl], axis=0)

            if hit_pmt.size > 0:
                hit_pmt, hit_t, hit_q, hit_label = sort_hits(hit_pmt, hit_t, hit_q, hit_label)

            out_path = out_hits / f"event_{global_event_index:06d}.npz"
            np.savez_compressed(
                out_path,
                chunk_index=np.int64(ci),
                local_event=np.int64(ei),
                hit_pmt=hit_pmt.astype(np.int32),
                hit_t=hit_t.astype(np.float32),
                hit_q=hit_q.astype(np.float32),
                hit_label=hit_label.astype(np.uint8),
                ap_src=str(ap_path),
                me_src=str(me_path),
            )

            global_event_index += 1
            if args.max_events is not None and global_event_index >= int(args.max_events):
                print(f"Reached --max-events={args.max_events}, stop.")
                print(f"Done. total_events={global_event_index} -> {out_hits}")
                return

        print(f"chunk {ci}: wrote {n_evt} events (total={global_event_index})")

    print(f"Done. total_events={global_event_index} -> {out_hits}")


if __name__ == "__main__":
    main()
import argparse
from pathlib import Path
import re
import numpy as np

ID_RE = re.compile(r".*_(\d+)\.npy$")


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


def load_one(det_feat_dir: Path, y_dir: Path, i: int):
    fht_pix = np.load(det_feat_dir / f"fht_pix_{i}.npy")  # (5, ...)
    npe_pix = np.load(det_feat_dir / f"npe_pix_{i}.npy")
    fht_pmt = np.load(det_feat_dir / f"fht_pmt_{i}.npy")
    npe_pmt = np.load(det_feat_dir / f"npe_pmt_{i}.npy")
    y = np.load(y_dir / f"y_{i}.npy")                      # (5, 15)
    return fht_pix, npe_pix, fht_pmt, npe_pmt, y


def combine_fht_min_nonzero(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_ = np.where(a > 0, a, np.inf)
    b_ = np.where(b > 0, b, np.inf)
    m = np.minimum(a_, b_)
    return np.where(np.isinf(m), 0.0, m)


def combine_group_adjacent(
    fht_pix: np.ndarray, npe_pix: np.ndarray,
    fht_pmt: np.ndarray, npe_pmt: np.ndarray,
    y: np.ndarray
):
    """
    Inputs are concatenated arrays with shape (10, ...).
    Output: 5 combined events via adjacent pairing (0,1), (2,3), ..., (8,9).
    y: take the first event's y in each pair.
    """
    assert fht_pix.shape[0] == 10 and y.shape[0] == 10, "Expect 10 events per group"
    pairs = [(i, i + 1) for i in range(0, 10, 2)]  # 5 pairs
    out_fht_pix, out_npe_pix, out_fht_pmt, out_npe_pmt, out_y = [], [], [], [], []
    for i, j in pairs:
        out_fht_pix.append(combine_fht_min_nonzero(fht_pix[i], fht_pix[j]))
        out_npe_pix.append(npe_pix[i] + npe_pix[j])
        out_fht_pmt.append(combine_fht_min_nonzero(fht_pmt[i], fht_pmt[j]))
        out_npe_pmt.append(npe_pmt[i] + npe_pmt[j])
        out_y.append(y[i])  # take first event's y as target

    return (
        np.stack(out_fht_pix, axis=0),  # (5, ...)
        np.stack(out_npe_pix, axis=0),  # (5, ...)
        np.stack(out_fht_pmt, axis=0),  # (5, ...)
        np.stack(out_npe_pmt, axis=0),  # (5, ...)
        np.stack(out_y, axis=0),        # (5, 15)
    )


def main():
    ap = argparse.ArgumentParser(description="Combine adjacent events from two consecutive ids into 5 outputs.")
    ap.add_argument("--det-feat-dir", default="/disk_pool1/weijsh/e+/det_feat", help="Path to det_feat directory")
    ap.add_argument("--y-dir", default="/disk_pool1/weijsh/e+/y", help="Path to y directory")
    ap.add_argument("--out-det-feat-dir", default="/disk_pool1/houyh/data/mixed/det_feat", help="Output directory for combined det_feat")
    ap.add_argument("--out-y-dir", default="/disk_pool1/houyh/data/mixed/y", help="Output directory for combined y")
    ap.add_argument("--start-id", type=int, default=None, help="Optional start id")
    ap.add_argument("--end-id", type=int, default=None, help="Optional end id (inclusive)")
    args = ap.parse_args()

    det_feat_dir = Path(args.det_feat_dir)
    y_dir = Path(args.y_dir)
    out_det = Path(args.out_det_feat_dir)
    out_y = Path(args.out_y_dir)
    out_det.mkdir(parents=True, exist_ok=True)
    out_y.mkdir(parents=True, exist_ok=True)

    ids = find_common_ids(det_feat_dir, y_dir)
    if args.start_id is not None:
        ids = [i for i in ids if i >= args.start_id]
    if args.end_id is not None:
        ids = [i for i in ids if i <= args.end_id]

    if len(ids) < 2:
        raise RuntimeError("Need at least two ids to form a 10-event group.")
    # Pair consecutive ids: (id0,id1), (id2,id3), ...
    groups = [ids[i:i+2] for i in range(0, len(ids) - 1, 2)]

    for out_idx, g in enumerate(groups):
        if len(g) != 2:
            continue
        a, b = g
        fht_pix_a, npe_pix_a, fht_pmt_a, npe_pmt_a, y_a = load_one(det_feat_dir, y_dir, a)
        fht_pix_b, npe_pix_b, fht_pmt_b, npe_pmt_b, y_b = load_one(det_feat_dir, y_dir, b)

        # concat to 10 events
        fht_pix = np.concatenate([fht_pix_a, fht_pix_b], axis=0)
        npe_pix = np.concatenate([npe_pix_a, npe_pix_b], axis=0)
        fht_pmt = np.concatenate([fht_pmt_a, fht_pmt_b], axis=0)
        npe_pmt = np.concatenate([npe_pmt_a, npe_pmt_b], axis=0)
        y = np.concatenate([y_a, y_b], axis=0)

        Fp, Np, Fm, Nm, Y = combine_group_adjacent(fht_pix, npe_pix, fht_pmt, npe_pmt, y)
       
        base = f"{out_idx}" 
        # each output file contains 5 combined events
        np.save(out_det / f"fht_pix_{base}.npy", Fp)
        np.save(out_det / f"npe_pix_{base}.npy", Np)
        np.save(out_det / f"fht_pmt_{base}.npy", Fm)
        np.save(out_det / f"npe_pmt_{base}.npy", Nm)
        np.save(out_y / f"y_{base}.npy", Y)

        print(f"Group ({a},{b}) -> saved 5 combined events as {base}")

    print("Done.")


if __name__ == "__main__":
    main()
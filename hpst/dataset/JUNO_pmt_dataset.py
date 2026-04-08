import re
from pathlib import Path
from typing import List, Tuple, Optional,Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from hpst.utils.options import Options

ID_RE = re.compile(r".*_(\d+)\.npy$")

def parse_id(p: Path) -> int:
    m = ID_RE.match(p.name)
    if not m:
        raise ValueError(f"Bad filename: {p}")
    return int(m.group(1))

class JUNOTQPairHitDataset(Dataset):
    """
    Dataset for the legacy "tq_pair + target" format.

    Folder structure:
      root/tq_pair/tq_pair_*.npy    shape (5, NPmt, 2, 2)
      root/target/target_*.npy      shape (5, NPmt, 2)

    For one event:
      - choose a subset of PMTs (all positive + sampled negatives)
      - build hit-level features: each PMT produces 2 hits (e+ and c14)
      - model predicts per-hit logits, then trainer scatters/aggregates to per-PMT logits
    """

    def __init__(
        self,
        root_dir: str,
        options: Optional[Options] = None,
        coords_path: str = "/disk_pool1/houyh/data/whichPixel_nside32_LCDpmts.npy",
        vg_mm_per_ns: float = 190.0,
        stats_sample: int = 200,
        radius_mm: float = 19500.0,
        neg_ratio: int = 3,
        neg_cap: int = 8192,
        rng_seed: int = 12345,
    ):
        if options is not None:
            coords_path = getattr(options, "juno_coords_path", coords_path)
            vg_mm_per_ns = float(getattr(options, "juno_vg_mm_per_ns", vg_mm_per_ns))
            radius_mm = float(getattr(options, "juno_radius_mm", radius_mm))
            neg_ratio = int(getattr(options, "juno_neg_ratio", neg_ratio))
            neg_cap = int(getattr(options, "juno_neg_cap", neg_cap))
            rng_seed = int(getattr(options, "juno_rng_seed", rng_seed))

        self.root = Path(root_dir)
        self.tq_dir = self.root / "tq_pair"
        self.tgt_dir = self.root / "target"

        self.tq_files = sorted(self.tq_dir.glob("tq_pair_*.npy"), key=parse_id)
        self.tgt_files = sorted(self.tgt_dir.glob("target_*.npy"), key=parse_id)
        if len(self.tq_files) == 0 or len(self.tq_files) != len(self.tgt_files):
            raise RuntimeError(f"Dataset mismatch: tq={len(self.tq_files)} tgt={len(self.tgt_files)} in {self.root}")

        self.num_files = len(self.tq_files)
        self.num_events_per_file = 5

        # Load PMT xyz coords (mm). whichPixel_nside32_LCDpmts.npy uses columns [2,3,4] = xyz(mm)
        coord_data = np.load(coords_path)
        coordx = coord_data[:, 2].astype(np.float32)
        coordy = coord_data[:, 3].astype(np.float32)
        coordz = coord_data[:, 4].astype(np.float32)
        self.coords_mm = torch.from_numpy(np.stack([coordx, coordy, coordz], axis=-1)).float()  # (NPmt,3)

        self.vg_mm_per_ns = float(vg_mm_per_ns)
        self.radius_mm = float(radius_mm)
        self.neg_ratio = int(neg_ratio)
        self.neg_cap = int(neg_cap)
        self.rng = np.random.default_rng(int(rng_seed))

        # statistics for q normalization
        self.compute_statistics(stats_sample=stats_sample)

    def __len__(self):
        return self.num_files * self.num_events_per_file

    def _load_file(self, file_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        tq = np.load(self.tq_files[file_idx])    # (5, NPmt, 2, 2)
        tgt = np.load(self.tgt_files[file_idx])  # (5, NPmt, 2)
        return tq, tgt

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_idx = idx // self.num_events_per_file
        e = idx % self.num_events_per_file
        tq, tgt = self._load_file(file_idx)

        tgt_e = tgt[e].astype(np.int8)  # (NPmt,2)

        pos_mask = (tgt_e[:, 0] == 1) | (tgt_e[:, 1] == 1)
        neg_mask = (tgt_e[:, 0] == 0) & (tgt_e[:, 1] == 0)

        pos_pmts = np.nonzero(pos_mask)[0].astype(np.int64)
        neg_pmts = np.nonzero(neg_mask)[0].astype(np.int64)

        if pos_pmts.size == 0:
            kneg = min(1024, neg_pmts.size)
        else:
            kneg = min(self.neg_ratio * pos_pmts.size, self.neg_cap, neg_pmts.size)

        # guard: ensure at least 1 sampled PMT if possible
        if kneg == 0 and pos_pmts.size == 0 and neg_pmts.size > 0:
            kneg = 1

        if kneg > 0:
            neg_sample = self.rng.choice(neg_pmts, size=kneg, replace=False).astype(np.int64)
            pmt_ids = np.concatenate([pos_pmts, neg_sample], axis=0)
        else:
            pmt_ids = pos_pmts

        # stable unique + sort
        pmt_ids = np.unique(pmt_ids).astype(np.int64)  # (M,)
        M = int(pmt_ids.shape[0])

        # per-PMT labels: (M,2)
        pmt_labels = torch.from_numpy(tgt_e[pmt_ids].astype(np.float32))

        # pick the simplest per-PMT (t,q) for both channels
        t_eplus = tq[e, :, 0, 0].astype(np.float32)  # (NPmt,)
        q_eplus = tq[e, :, 1, 0].astype(np.float32)
        t_c14 = tq[e, :, 0, 1].astype(np.float32)
        q_c14 = tq[e, :, 1, 1].astype(np.float32)

        # Build hit-level: two hits per PMT: [e+, c14]
        # "hit_pmt_ids" indexes rows of pmt_labels (0..M-1)
        hit_pmt_ids = torch.repeat_interleave(torch.arange(M, dtype=torch.long), repeats=2)  # (2M,)

        xyz = (self.coords_mm[torch.from_numpy(pmt_ids).long()] / self.radius_mm)  # (M,3)

        vt_eplus = torch.from_numpy((t_eplus[pmt_ids] * self.vg_mm_per_ns).astype(np.float32)).unsqueeze(-1) / self.radius_mm
        vt_c14 = torch.from_numpy((t_c14[pmt_ids] * self.vg_mm_per_ns).astype(np.float32)).unsqueeze(-1) / self.radius_mm

        coords4_eplus = torch.cat([xyz, vt_eplus], dim=-1)  # (M,4)
        coords4_c14 = torch.cat([xyz, vt_c14], dim=-1)      # (M,4)
        coords4 = torch.stack([coords4_eplus, coords4_c14], dim=1).reshape(2 * M, 4)  # (2M,4)

        # Feature: normalized q
        q_e = ((q_eplus[pmt_ids] - self.q_mean) / self.q_std).astype(np.float32)
        q_c = ((q_c14[pmt_ids] - self.q_mean) / self.q_std).astype(np.float32)
        feats = torch.from_numpy(np.stack([q_e, q_c], axis=1).reshape(2 * M, 1)).float()  # (2M,1)

        hit_features = torch.cat([coords4, feats], dim=-1)  # (2M,5)

        return {
            "hit_features": hit_features,  # (Nhits=2M,5)
            "hit_pmt_ids": hit_pmt_ids,    # (2M,) local pmt row id [0..M-1]
            "pmt_labels": pmt_labels,      # (M,2)
            "unique_pmt_ids": torch.from_numpy(pmt_ids).long(),  # (M,) global pmt id
        }

    def compute_statistics(self, stats_sample: int = 200):
        n = min(stats_sample, len(self))
        q_list: List[np.ndarray] = []

        for i in range(n):
            file_idx = i // self.num_events_per_file
            e = i % self.num_events_per_file
            tq, _ = self._load_file(file_idx)

            q_eplus = tq[e, :, 1, 0].astype(np.float32)
            q_c14 = tq[e, :, 1, 1].astype(np.float32)

            # only consider hits with non-zero charge
            hit_any = (q_eplus > 0) | (q_c14 > 0)
            if not np.any(hit_any):
                continue

            q_obs = (q_eplus + q_c14)  # proxy
            q_list.append(q_obs[hit_any])

        if len(q_list) == 0:
            self.q_mean = 0.0
            self.q_std = 1.0
        else:
            q_cat = np.concatenate(q_list, axis=0)
            self.q_mean = float(q_cat.mean())
            self.q_std = float(max(q_cat.std(), 1e-6))

        mean = torch.tensor([self.q_mean], dtype=torch.float32)
        std = torch.tensor([self.q_std], dtype=torch.float32)
        return mean, std, torch.zeros(1), torch.ones(1), torch.zeros(1), torch.ones(1)


class JUNOHitListDataset(Dataset):
    """
    Legacy hit-list dataset for PMT-level labels (NOT the APME hit-level dataset).

    Folder structure:
      root/hits/event_*.npz
      Each npz must contain:
        hit_pmt: (N,) int
        hit_t:   (N,) float
        hit_q:   (N,) float
        pmt_label: (NPmt, 2) uint8/0-1
      Optional:
        unique_pmt_ids or other metadata (ignored)

    Output per sample (PMT-level training expects this):
      hit_features: (N',5) [x,y,z,vt,q_norm] for filtered hits
      hit_pmt_ids:  (N',) local PMT row indices [0..M-1]
      pmt_labels:   (M,2) labels for selected PMTs
      unique_pmt_ids: (M,) global pmt ids
    """

    def __init__(
        self,
        root_dir: str,
        options: Optional[Options] = None,
        coords_path: str = "/disk_pool1/houyh/data/whichPixel_nside32_LCDpmts.npy",
        vg_mm_per_ns: float = 190.0,
        stats_sample: int = 200,
        radius_mm: float = 19500.0,
        neg_ratio: int = 3,
        neg_cap: int = 8192,
        rng_seed: int = 12345,
    ):
        if options is not None:
            coords_path = getattr(options, "juno_coords_path", coords_path)
            vg_mm_per_ns = float(getattr(options, "juno_vg_mm_per_ns", vg_mm_per_ns))
            radius_mm = float(getattr(options, "juno_radius_mm", radius_mm))
            neg_ratio = int(getattr(options, "juno_neg_ratio", neg_ratio))
            neg_cap = int(getattr(options, "juno_neg_cap", neg_cap))
            rng_seed = int(getattr(options, "juno_rng_seed", rng_seed))

        self.root = Path(root_dir)
        self.hits_dir = self.root / "hits"
        self.hit_files = sorted(self.hits_dir.glob("event_*.npz"))
        if len(self.hit_files) == 0:
            raise RuntimeError(f"No event_*.npz found under {self.hits_dir}")

        coord_data = np.load(coords_path)
        coordx = coord_data[:, 2].astype(np.float32)
        coordy = coord_data[:, 3].astype(np.float32)
        coordz = coord_data[:, 4].astype(np.float32)
        self.coords_mm = torch.from_numpy(np.stack([coordx, coordy, coordz], axis=-1)).float()  # (NPmt,3)

        self.vg_mm_per_ns = float(vg_mm_per_ns)
        self.radius_mm = float(radius_mm)
        self.neg_ratio = int(neg_ratio)
        self.neg_cap = int(neg_cap)
        self.rng = np.random.default_rng(int(rng_seed))

        self.compute_statistics(stats_sample=stats_sample)

    def __len__(self):
        return len(self.hit_files)

    def _load_event(self, idx: int):
        return np.load(self.hit_files[idx], allow_pickle=False)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        d = self._load_event(idx)

        hit_pmt = d["hit_pmt"].astype(np.int64)         # (N,)
        hit_t = d["hit_t"].astype(np.float32)           # (N,)
        hit_q = d["hit_q"].astype(np.float32)           # (N,)
        pmt_label_full = d["pmt_label"].astype(np.uint8)  # (NPmt,2)

        # Sample PMTs: all positives + some negatives
        pos_mask = (pmt_label_full[:, 0] == 1) | (pmt_label_full[:, 1] == 1)
        neg_mask = ~pos_mask

        pos_pmts = np.nonzero(pos_mask)[0].astype(np.int64)
        neg_pmts = np.nonzero(neg_mask)[0].astype(np.int64)

        if pos_pmts.size == 0:
            kneg = min(1024, neg_pmts.size)
        else:
            kneg = min(self.neg_ratio * pos_pmts.size, self.neg_cap, neg_pmts.size)

        if kneg == 0 and pos_pmts.size == 0 and neg_pmts.size > 0:
            kneg = 1

        if kneg > 0:
            neg_sample = self.rng.choice(neg_pmts, size=kneg, replace=False).astype(np.int64)
            chosen_pmts = np.concatenate([pos_pmts, neg_sample], axis=0)
        else:
            chosen_pmts = pos_pmts

        chosen_pmts = np.unique(chosen_pmts).astype(np.int64)  # (M,)
        M = int(chosen_pmts.shape[0])

        # Filter hits: keep only hits whose global pmt is in chosen_pmts
        if hit_pmt.size > 0 and chosen_pmts.size > 0:
            chosen_set = np.zeros((pmt_label_full.shape[0],), dtype=bool)
            chosen_set[chosen_pmts] = True
            keep = chosen_set[hit_pmt]
        else:
            keep = np.zeros((hit_pmt.size,), dtype=bool)

        hit_pmt = hit_pmt[keep]
        hit_t = hit_t[keep]
        hit_q = hit_q[keep]

        # Guard: avoid N=0 (some models/collate cannot handle empty)
        if hit_pmt.size == 0 and chosen_pmts.size > 0:
            p0 = int(chosen_pmts[self.rng.integers(0, chosen_pmts.size)])
            hit_pmt = np.array([p0], dtype=np.int64)
            hit_t = np.array([0.0], dtype=np.float32)
            hit_q = np.array([0.0], dtype=np.float32)

        # Build mapping global PMT id -> local 0..M-1
        map_arr = np.full((pmt_label_full.shape[0],), -1, dtype=np.int64)
        map_arr[chosen_pmts] = np.arange(M, dtype=np.int64)
        hit_pmt_ids = map_arr[hit_pmt]
        if np.any(hit_pmt_ids < 0):
            raise RuntimeError("Internal mapping error: hit references PMT not in chosen_pmts")

        pmt_labels = torch.from_numpy(pmt_label_full[chosen_pmts].astype(np.float32))  # (M,2)
        unique_pmt_ids = torch.from_numpy(chosen_pmts).long()  # (M,)

        # coords/features:
        xyz = (self.coords_mm[unique_pmt_ids] / self.radius_mm)          # (M,3)
        xyz_hit = xyz[torch.from_numpy(hit_pmt_ids).long()]              # (N,3)
        vt = torch.from_numpy((hit_t * self.vg_mm_per_ns).astype(np.float32)).unsqueeze(-1) / self.radius_mm  # (N,1)
        coords4 = torch.cat([xyz_hit, vt], dim=-1)                       # (N,4)

        q_norm = ((hit_q - self.q_mean) / self.q_std).astype(np.float32)
        feats = torch.from_numpy(q_norm).unsqueeze(-1).float()           # (N,1)

        hit_features = torch.cat([coords4, feats], dim=-1)               # (N,5)

        return {
            "hit_features": hit_features,
            "hit_pmt_ids": torch.from_numpy(hit_pmt_ids).long(),
            "pmt_labels": pmt_labels,
            "unique_pmt_ids": unique_pmt_ids,
        }

    def compute_statistics(self, stats_sample: int = 200):
        n = min(stats_sample, len(self))
        q_list: List[np.ndarray] = []
        for i in range(n):
            d = self._load_event(i)
            hit_q = d["hit_q"].astype(np.float32)
            if hit_q.size == 0:
                continue
            q_list.append(hit_q)

        if len(q_list) == 0:
            self.q_mean = 0.0
            self.q_std = 1.0
        else:
            q_cat = np.concatenate(q_list, axis=0)
            self.q_mean = float(q_cat.mean())
            self.q_std = float(max(q_cat.std(), 1e-6))

        mean = torch.tensor([self.q_mean], dtype=torch.float32)
        std = torch.tensor([self.q_std], dtype=torch.float32)
        return mean, std, torch.zeros(1), torch.ones(1), torch.zeros(1), torch.ones(1)
    

class JUNOAPMEHitDataset(Dataset):
    """
    AP/ME mixed hit-level dataset:
      root/hits/event_*.npz
      each npz contains:
        hit_pmt (N,), hit_t (N,), hit_q (N,), hit_label (N,) where 0=AP, 1=ME
    Output per sample:
      hit_features: (N,5) [x,y,z,vt,q_norm]
      hit_labels:   (N,)  int64
      hit_pmt:      (N,)  int64  (global PMT id, for visualization)
    """

    def __init__(
        self,
        root_dir: str,
        options: Optional[Options] = None,
        coords_path: str = "/disk_pool1/houyh/data/whichPixel_nside32_LCDpmts.npy",
        vg_mm_per_ns: float = 190.0,
        stats_sample: int = 200,
        radius_mm: float = 19500.0,
        rng_seed: int = 12345,
    ):
        if options is not None:
            coords_path = getattr(options, "juno_coords_path", coords_path)
            vg_mm_per_ns = float(getattr(options, "juno_vg_mm_per_ns", vg_mm_per_ns))
            radius_mm = float(getattr(options, "juno_radius_mm", radius_mm))
            rng_seed = int(getattr(options, "juno_rng_seed", rng_seed))

        self.root = Path(root_dir)
        self.hits_dir = self.root / "hits"
        self.hit_files = sorted(self.hits_dir.glob("event_*.npz"))
        if len(self.hit_files) == 0:
            raise RuntimeError(f"No event_*.npz found under {self.hits_dir}")

        coord_data = np.load(coords_path)
        coordx = coord_data[:, 2].astype(np.float32)
        coordy = coord_data[:, 3].astype(np.float32)
        coordz = coord_data[:, 4].astype(np.float32)
        self.coords_mm = torch.from_numpy(np.stack([coordx, coordy, coordz], axis=-1)).float()  # (NPmt,3)

        self.vg_mm_per_ns = float(vg_mm_per_ns)
        self.radius_mm = float(radius_mm)
        self.rng = np.random.default_rng(int(rng_seed))

        self.compute_statistics(stats_sample=stats_sample)

    def __len__(self):
        return len(self.hit_files)

    def _load_event(self, idx: int):
        return np.load(self.hit_files[idx], allow_pickle=False)

    def __getitem__(self, idx: int):
        # d = self._load_event(idx)
        d = np.load(self.hit_files[idx], allow_pickle=False)

        hit_pmt = d["hit_pmt"].astype(np.int64)     # (N,)
        hit_t = d["hit_t"].astype(np.float32)       # (N,)
        hit_q = d["hit_q"].astype(np.float32)       # (N,)
        hit_label = d["hit_label"].astype(np.int64) # (N,) 0/1

        # Guard: avoid empty events crashing the model
        if hit_pmt.size == 0:
            # choose a dummy PMT 0
            hit_pmt = np.array([0], dtype=np.int64)
            hit_t = np.array([0.0], dtype=np.float32)
            hit_q = np.array([0.0], dtype=np.float32)
            hit_label = np.array([0], dtype=np.int64)

        # coords: xyz from pmt + vt from time
        xyz_hit = (self.coords_mm[torch.from_numpy(hit_pmt).long()] / self.radius_mm)  # (N,3)

        vt = torch.from_numpy((hit_t * self.vg_mm_per_ns).astype(np.float32)).unsqueeze(-1) / self.radius_mm  # (N,1)
        coords4 = torch.cat([xyz_hit, vt], dim=-1)  # (N,4)

        q_norm = ((hit_q - self.q_mean) / self.q_std).astype(np.float32)
        feats = torch.from_numpy(q_norm).unsqueeze(-1).float()  # (N,1)

        hit_features = torch.cat([coords4, feats], dim=-1)  # (N,5)
        hit_labels = torch.from_numpy(hit_label).long()  # (N,)
        chunk_index = int(np.asarray(d["chunk_index"]).item())
        local_event = int(np.asarray(d["local_event"]).item())

        return {
            "hit_features": hit_features,
            "hit_labels": hit_labels,
            # FIX: make it a Tensor
            "hit_pmt": torch.from_numpy(hit_pmt).long(),
            "chunk_index": torch.tensor(chunk_index, dtype=torch.long),
            "local_event": torch.tensor(local_event, dtype=torch.long),
        }
        # return {
        #     "hit_features": hit_features,
        #     "hit_labels": torch.from_numpy(hit_label).long(),
        #     "hit_pmt": torch.from_numpy(hit_pmt).long(),
        # }

    def compute_statistics(self, stats_sample: int = 200):
        n = min(stats_sample, len(self))
        q_list = []
        for i in range(n):
            d = self._load_event(i)
            hit_q = d["hit_q"].astype(np.float32)
            if hit_q.size == 0:
                continue
            q_list.append(hit_q)

        if len(q_list) == 0:
            self.q_mean = 0.0
            self.q_std = 1.0
        else:
            q_cat = np.concatenate(q_list, axis=0)
            self.q_mean = float(q_cat.mean())
            self.q_std = float(max(q_cat.std(), 1e-6))

        mean = torch.tensor([self.q_mean], dtype=torch.float32)
        std = torch.tensor([self.q_std], dtype=torch.float32)
        return mean, std, torch.zeros(1), torch.ones(1), torch.zeros(1), torch.ones(1)
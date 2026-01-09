import re
from pathlib import Path
from typing import List, Tuple, Optional

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

    def __init__(
        self,
        root_dir: str,
        options: Optional[Options] = None,  # 新增：options 放在这里
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

        # 加载坐标（whichPixel_nside32_LCDpmts.npy: 用第2/3/4列为 xyz(mm)）
        coord_data = np.load(coords_path)
        coordx = coord_data[:, 2].astype(np.float32)
        coordy = coord_data[:, 3].astype(np.float32)
        coordz = coord_data[:, 4].astype(np.float32)
        self.coords_mm = torch.from_numpy(np.stack([coordx, coordy, coordz], axis=-1)).float()

        self.vg_mm_per_ns = float(vg_mm_per_ns)
        self.radius_mm = float(radius_mm)
        self.neg_ratio = int(neg_ratio)
        self.neg_cap = int(neg_cap)
        self.rng = np.random.default_rng(int(rng_seed))

        # 统计 q 的均值方差
        self.compute_statistics(stats_sample=stats_sample)

    def __len__(self):
        return self.num_files * self.num_events_per_file

    def _load_file(self, file_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        tq = np.load(self.tq_files[file_idx])    # (5, Npmt, 2, 2)
        tgt = np.load(self.tgt_files[file_idx])  # (5, Npmt, 2)
        return tq, tgt

    def __getitem__(self, idx: int):
        file_idx = idx // self.num_events_per_file
        e = idx % self.num_events_per_file
        tq, tgt = self._load_file(file_idx)

        tgt_e = tgt[e].astype(np.int8)

        pos_mask = (tgt_e[:, 0] == 1) | (tgt_e[:, 1] == 1)
        neg_mask = (tgt_e[:, 0] == 0) & (tgt_e[:, 1] == 0)

        pos_pmts = np.nonzero(pos_mask)[0].astype(np.int64)
        neg_pmts = np.nonzero(neg_mask)[0].astype(np.int64)

        if pos_pmts.size == 0:
            kneg = min(1024, neg_pmts.size)
        else:
            kneg = min(self.neg_ratio * pos_pmts.size, self.neg_cap, neg_pmts.size)

        # --- add guard: ensure at least 1 sample if possible ---
        if kneg == 0 and pos_pmts.size == 0 and neg_pmts.size > 0:
            kneg = 1

        if kneg > 0:
            neg_sample = self.rng.choice(neg_pmts, size=kneg, replace=False).astype(np.int64)
            pmt_ids = np.concatenate([pos_pmts, neg_sample], axis=0)
        else:
            pmt_ids = pos_pmts

        # PMT labels (M,2)
        pmt_labels = torch.from_numpy(tgt_e[pmt_ids].astype(np.float32))

        # 取两路 “最简单场景”的 per-PMT 一对 t/q
        t_eplus = tq[e, :, 0, 0].astype(np.float32)  # (Npmt,)
        q_eplus = tq[e, :, 1, 0].astype(np.float32)
        t_c14   = tq[e, :, 0, 1].astype(np.float32)
        q_c14   = tq[e, :, 1, 1].astype(np.float32)

        # === 构造 hit-level：每个 PMT 两条 hit ===
        # hit order: [pmt0:e+, pmt0:c14, pmt1:e+, pmt1:c14, ...]
        M = pmt_ids.shape[0]
        hit_pmt_ids = torch.repeat_interleave(torch.arange(M, dtype=torch.long), repeats=2)  # (2M,)

        # coords: (x,y,z,vt) 其中 vt 用对应hit的 t * vg
        xyz = (self.coords_mm[pmt_ids] / self.radius_mm)  # (M,3)

        vt_eplus = torch.from_numpy((t_eplus[pmt_ids] * self.vg_mm_per_ns)).float().unsqueeze(-1) / self.radius_mm  # (M,1)
        vt_c14   = torch.from_numpy((t_c14[pmt_ids]   * self.vg_mm_per_ns)).float().unsqueeze(-1) / self.radius_mm  # (M,1)

        coords4_eplus = torch.cat([xyz, vt_eplus], dim=-1)  # (M,4)
        coords4_c14   = torch.cat([xyz, vt_c14],   dim=-1)  # (M,4)

        coords4 = torch.stack([coords4_eplus, coords4_c14], dim=1).reshape(2 * M, 4)  # (2M,4)

        # feature: q，分别取两路，并用同一组统计量做标准化
        q_e = torch.from_numpy(((q_eplus - self.q_mean) / self.q_std)[pmt_ids]).float().unsqueeze(-1)  # (M,1)
        q_c = torch.from_numpy(((q_c14   - self.q_mean) / self.q_std)[pmt_ids]).float().unsqueeze(-1)  # (M,1)
        feats = torch.stack([q_e, q_c], dim=1).reshape(2 * M, 1)  # (2M,1)

        hit_features = torch.cat([coords4, feats], dim=-1)  # (2M,5)

        return {
            "hit_features": hit_features,                 # (Nhits=2M,5)
            "hit_pmt_ids": hit_pmt_ids,                   # (2M,)
            "pmt_labels": pmt_labels,                     # (M,2)
            "unique_pmt_ids": torch.from_numpy(pmt_ids),  # (M,)
        }
    
    def compute_statistics(self, stats_sample: int = 200):
        # 这里只统计 q_obs（因为 coords 已用 radius_mm 归一化，不走 mean/std）
        n = min(stats_sample, len(self))
        q_list: List[np.ndarray] = []
        for i in range(n):
            file_idx = i // self.num_events_per_file
            e = i % self.num_events_per_file
            tq, _ = self._load_file(file_idx)

            t_eplus = tq[e, :, 0, 0]
            q_eplus = tq[e, :, 1, 0]
            t_c14   = tq[e, :, 0, 1]
            q_c14   = tq[e, :, 1, 1]

            hit_eplus = (t_eplus > 0) & (q_eplus > 0)
            hit_c14   = (t_c14 > 0) & (q_c14 > 0)
            hit_any = hit_eplus | hit_c14
            if not np.any(hit_any):
                continue

            q_obs = (q_eplus.astype(np.float32) + q_c14.astype(np.float32))
            q_list.append(q_obs[hit_any].astype(np.float32))

        if len(q_list) == 0:
            self.q_mean = 0.0
            self.q_std = 1.0
        else:
            q_cat = np.concatenate(q_list, axis=0)
            self.q_mean = float(q_cat.mean())
            self.q_std = float(max(q_cat.std(), 1e-6))

        # 给 NeutrinoBase 的返回：feature_dim=1 -> (1,)
        mean = torch.tensor([self.q_mean], dtype=torch.float32)
        std = torch.tensor([self.q_std], dtype=torch.float32)
        return mean, std, torch.zeros(1), torch.ones(1), torch.zeros(1), torch.ones(1)
    



class JUNOHitListDataset(Dataset):
    """
    Hit-list dataset:
      root/hits/event_*.npz
      each npz contains:
        hit_pmt (N,), hit_t (N,), hit_q (N,), hit_src(optional)
        pmt_label (NPmt,2)
        y (15,)
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
        self.coords_mm = torch.from_numpy(np.stack([coordx, coordy, coordz], axis=-1)).float()

        self.vg_mm_per_ns = float(vg_mm_per_ns)
        self.radius_mm = float(radius_mm)
        self.neg_ratio = int(neg_ratio)
        self.neg_cap = int(neg_cap)
        self.rng = np.random.default_rng(int(rng_seed))

        self.compute_statistics(stats_sample=stats_sample)

    def __len__(self):
        return len(self.hit_files)

    def _load_event(self, idx: int):
        d = np.load(self.hit_files[idx], allow_pickle=False)
        return d

    def __getitem__(self, idx: int):
        d = self._load_event(idx)

        hit_pmt = d["hit_pmt"].astype(np.int64)  # (N,)
        hit_t = d["hit_t"].astype(np.float32)    # (N,)
        hit_q = d["hit_q"].astype(np.float32)    # (N,)
        pmt_label_full = d["pmt_label"].astype(np.uint8)  # (NPmt,2)

        # 先基于 pmt_label 做 PMT 采样（正 + 部分负）
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

        # 去重并排序（稳定）
        chosen_pmts = np.unique(chosen_pmts)

        # 过滤 hit：只保留落在 chosen_pmts 的 hits
        if hit_pmt.size > 0 and chosen_pmts.size > 0:
            # 建一个快速 membership mask
            chosen_set = np.zeros((pmt_label_full.shape[0],), dtype=bool)
            chosen_set[chosen_pmts] = True
            keep = chosen_set[hit_pmt]
        else:
            keep = np.zeros((hit_pmt.size,), dtype=bool)

        hit_pmt = hit_pmt[keep]
        hit_t = hit_t[keep]
        hit_q = hit_q[keep]

        # --- guard: avoid empty hit list (PointSetTransformer likely can't handle N=0) ---
        if hit_pmt.size == 0 and chosen_pmts.size > 0:
            p0 = int(chosen_pmts[self.rng.integers(0, chosen_pmts.size)])
            hit_pmt = np.array([p0], dtype=np.int64)
            hit_t = np.array([0.0], dtype=np.float32)
            hit_q = np.array([0.0], dtype=np.float32)

        # unique pmts（事件内）
        unique_pmts = chosen_pmts.astype(np.int64)
        M = unique_pmts.shape[0]

        # 建 map：global pmt id -> local 0..M-1
        # 用数组映射（更快）
        map_arr = np.full((pmt_label_full.shape[0],), -1, dtype=np.int64)
        map_arr[unique_pmts] = np.arange(M, dtype=np.int64)

        hit_pmt_ids = map_arr[hit_pmt]
        if np.any(hit_pmt_ids < 0):
            raise RuntimeError("Internal mapping error: hit references PMT not in unique_pmts")

        # labels (M,2)
        pmt_labels = torch.from_numpy(pmt_label_full[unique_pmts].astype(np.float32))

        # coords/features
        xyz = (self.coords_mm[unique_pmts] / self.radius_mm)  # (M,3)

        # 每个 hit 的 xyz 由其 pmt 决定
        xyz_hit = xyz[torch.from_numpy(hit_pmt_ids).long()]  # (N,3)

        vt = torch.from_numpy((hit_t * self.vg_mm_per_ns).astype(np.float32)).unsqueeze(-1) / self.radius_mm  # (N,1)
        coords4 = torch.cat([xyz_hit, vt], dim=-1)  # (N,4)

        q_norm = ((hit_q - self.q_mean) / self.q_std).astype(np.float32)
        feats = torch.from_numpy(q_norm).unsqueeze(-1).float()  # (N,1)

        hit_features = torch.cat([coords4, feats], dim=-1)  # (N,5)

        return {
            "hit_features": hit_features,
            "hit_pmt_ids": torch.from_numpy(hit_pmt_ids).long(),
            "pmt_labels": pmt_labels,
            "unique_pmt_ids": torch.from_numpy(unique_pmts).long(),
        }

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
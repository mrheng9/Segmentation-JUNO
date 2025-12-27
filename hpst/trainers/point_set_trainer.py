def collate_fn(batch):
    """
    合并 batch（事件级别），输出 dict：
      hit_features: (total_N, 5) [x,y,z,t_norm,q_norm]
      hit_pmt_ids:  (total_N,)   指向全局 pmt_label 索引（batch 内偏移后的）
      batch:        (total_N,)   每个 hit 属于哪个 event
      pmt_labels:   (total_M, 2) 每个事件的 unique pmts 的标签拼接
      pmt_batch:    (total_M,)   每个 unique pmt 属于哪个 event
      unique_pmt_ids:(total_M,)  全局PMT id（可用于评估/回填）
    """
    hit_features_list = []
    hit_pmt_ids_list = []
    pmt_labels_list = []
    unique_pmt_ids_list = []
    batch_idx_list = []
    pmt_batch_idx_list = []

    pmt_offset = 0
    for i, sample in enumerate(batch):
        N_hits = sample["hit_features"].shape[0]
        M_pmts = sample["unique_pmt_ids"].shape[0]

        hit_features_list.append(sample["hit_features"])
        # hit_pmt_ids 是本事件局部 0..M-1，这里偏移成全局 0..total_M-1
        hit_pmt_ids_list.append(sample["hit_pmt_ids"] + pmt_offset)

        pmt_labels_list.append(sample["pmt_labels"])
        unique_pmt_ids_list.append(sample["unique_pmt_ids"])

        batch_idx_list.append(torch.full((N_hits,), i, dtype=torch.long))
        pmt_batch_idx_list.append(torch.full((M_pmts,), i, dtype=torch.long))

        pmt_offset += M_pmts

    return {
        "hit_features": torch.cat(hit_features_list, dim=0),          # (total_N,5)
        "hit_pmt_ids": torch.cat(hit_pmt_ids_list, dim=0),            # (total_N,)
        "batch": torch.cat(batch_idx_list, dim=0),                    # (total_N,)
        "pmt_labels": torch.cat(pmt_labels_list, dim=0),              # (total_M,2)
        "pmt_batch": torch.cat(pmt_batch_idx_list, dim=0),            # (total_M,)
        "unique_pmt_ids": torch.cat(unique_pmt_ids_list, dim=0),      # (total_M,)
    }

import numpy as np
import torch
from matplotlib import pyplot as plt
# from pytorch_lightning import metrics
import torchmetrics as metrics
from sklearn.metrics import ConfusionMatrixDisplay
from torch import Tensor, jit

from torch import Tensor
import torch.nn.functional as F
from typing import List, Tuple, Union
from torch_scatter import scatter
from scipy.optimize import linear_sum_assignment

from sklearn.metrics import f1_score
from hpst.utils.options import Options
from hpst.trainers.neutrino_base import NeutrinoBase
from hpst.dataset.JUNO_pmt_dataset import JUNOTQPairHitDataset
from hpst.models.point_set_transformer import PointSetTransformerInterface

import sys

TArray = np.ndarray





class PointSetTrainer(NeutrinoBase):
    def __init__(self, options: Options, train_perc=None):
        super(PointSetTrainer, self).__init__(options, train_perc=train_perc)

        self.num_classes = 2
        # 只要 seg head：输出维度=2
        self.network = PointSetTransformerInterface(
            options,
            feature_dim=1,     
            output_dim=2
        )

        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    @property
    def dataset(self):
        return JUNOTQPairHitDataset

    @property
    def dataloader_options(self):
        return {
            "drop_last": False,
            "batch_size": self.options.batch_size,
            "pin_memory": self.options.num_gpu > 0,
            "num_workers": self.options.num_dataloader_workers,
            "collate_fn": collate_fn,
        }

    def forward(self, coords: Tensor, feats: Tensor, batch: Tensor) -> Tensor:
        feats = feats.clone()
        if self.options.normalize_features and feats.numel() > 0:
            feats = (feats - self.mean) / self.std  # 去掉 [:1]
        logits = self.network((coords, feats, batch), None)
        return logits

    def training_step(self, batch, batch_idx):
        hit_features = batch["hit_features"]   # (N,5)
        hit_pmt_ids  = batch["hit_pmt_ids"]    # (N,) -> 指向 pmt_labels 的行
        pmt_labels   = batch["pmt_labels"]     # (M,2)

        coords = hit_features[:, :4]
        feats  = hit_features[:, 4:]
        ev_batch = batch["batch"]

        logits_hit = self.forward(coords, feats, ev_batch)  # (N,2)

        # 聚合到 PMT（同一 PMT 多个 hit 的情况用 mean；你当前实现每 PMT 1 个点，也兼容）
        # 注意：hit_pmt_ids 是 “全 batch 内 unique pmt”的索引
        logits_pmt = scatter(logits_hit, hit_pmt_ids, dim=0, reduce="mean", dim_size=pmt_labels.shape[0])

        loss = self.loss_fn(logits_pmt, pmt_labels)

        self.log("train_loss", loss, prog_bar=True, batch_size=self.options.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        hit_features = batch["hit_features"]
        hit_pmt_ids  = batch["hit_pmt_ids"]
        pmt_labels   = batch["pmt_labels"]

        coords = hit_features[:, :4]
        feats  = hit_features[:, 4:]
        ev_batch = batch["batch"]

        logits_hit = self.forward(coords, feats, ev_batch)
        logits_pmt = scatter(logits_hit, hit_pmt_ids, dim=0, reduce="mean", dim_size=pmt_labels.shape[0])

        loss = self.loss_fn(logits_pmt, pmt_labels)
        
        probs = torch.sigmoid(logits_pmt)
        pred = (probs > 0.5).float()
        
        # 分别计算 e+ (第0列) 和 C14 (第1列) 的准确率
        acc_eplus = (pred[:, 0] == pmt_labels[:, 0]).float().mean()
        acc_c14 = (pred[:, 1] == pmt_labels[:, 1]).float().mean()
        
        # 整体 exact match accuracy
        acc_exact = (pred == pmt_labels).all(dim=1).float().mean()
        
        # 逐元素准确率（你当前用的，但不推荐作为主要指标）
        acc_elementwise = (pred == pmt_labels).float().mean()

        # F1 for e+ (第0列) 和 C14 (第1列)
        
        pred_np = pred.cpu().numpy()
        label_np = pmt_labels.cpu().numpy()
        
        f1_eplus = f1_score(label_np[:, 0], pred_np[:, 0], zero_division=0)
        f1_c14 = f1_score(label_np[:, 1], pred_np[:, 1], zero_division=0)
        
        self.log("val_f1_eplus", f1_eplus, sync_dist=True)
        self.log("val_f1_c14", f1_c14, sync_dist=True)
        
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, batch_size=self.options.batch_size)
        self.log("val_acc_exact", acc_exact, prog_bar=True, sync_dist=True)
        self.log("val_acc_eplus", acc_eplus, sync_dist=True)
        self.log("val_acc_c14", acc_c14, sync_dist=True)
        self.log("val_acc_element", acc_elementwise, sync_dist=True)
        return loss
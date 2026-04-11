# Segmentation-JUNO-C

This repository is for **reconstruction/segmentation tasks on JUNO PMT data** as well as **PID (classification)**. The core network is **PST (Point Set Transformer / Point Set Attention family)**. This README first covers **training**; evaluation/plots will be supplemented later.

---

## 1. Task Overview (Training Scope)

The code currently supports **three task/data modalities** (corresponding to three Dataset types), all sharing the same PST backbone:

1) **PMT-level binary multi-label (pair)**  
- Labels are at the **PMT level**: each PMT has two-channel labels (e.g., `e+` and `C14`), typically shaped `(M, 2)`.  
- The model first outputs hit-level logits `(N, 2)`, then aggregates them to PMT logits `(M, 2)` via `scatter` using `hit_pmt_ids`, and trains with `BCEWithLogitsLoss`.

2) **PMT-level binary multi-label (hitlist)**  
- Same task definition as pair (PMT-level `(M, 2)` labels + hit→PMT aggregation).  
- The difference is the input format comes from “hit list” data in `event_*.npz`.

3) **Hit-level binary classification (apme)**  
- Labels are at the **hit level**: each hit has a class `0/1` (e.g., AP/ME), shaped `(N,)`.  
- The model outputs hit logits `(N, 2)` directly and trains with `CrossEntropyLoss` (no PMT aggregation).

> Summary: PST always takes **hit features (point features)** as input; the key differences are label granularity (PMT vs hit) and how loss/metrics are computed.

---

## 2. Training Pipeline (Simplified)

Training entry:

- `scripts/train_homogenous.py` → builds `PointSetTrainer` → `pl.Trainer.fit()`

Core chain (from data to loss):

- **Dataset** `hpst/dataset/JUNO_pmt_dataset.py`  
  Produces a per-event sample dict (hit_features + label)

- **Collate** `hpst/trainers/point_set_trainer.py`  
  Concatenates multiple events into one batch (and creates the `batch` index)

- **Trainer** `hpst/trainers/point_set_trainer.py`  
  Runs the PST forward pass, computes loss, and calls `self.log(...)`

- **Model** `hpst/models/point_set_transformer.py` + `hpst/layers/point_set_attention.py`  
  PST backbone network, outputs logits

---

## 3. Data & Feature Definition (Unified Convention)

Regardless of `pair/hitlist/apme`, the model always receives:

- `coords: (N, 4)`  (space/time: `[x, y, z, vt]`)
- `feats:  (N, 1)`  (point feature: here we use `q_norm`)
- `batch:  (N,)`    (which event in the batch each hit belongs to, values in `[0..B-1]`)

All Datasets ultimately provide `hit_features: (N, 5)`, which is split in the trainer into:
- `coords = hit_features[:, :4]`  → `(N, 4)`
- `feats  = hit_features[:, 4:]`  → `(N, 1)`

### 3.1 Meaning and shape of the 5D `hit_features`
`hit_features: (N, 5) = [x, y, z, vt, q_norm]`

- `x,y,z`: PMT coordinates (mm), then divided by `juno_radius_mm` for normalization  
- `vt`: `t * juno_vg_mm_per_ns` (map time to a length scale), then divided by `juno_radius_mm`  
- `q_norm`: `(q - q_mean) / q_std` (estimated by the dataset `compute_statistics()`)

### 3.2 Labels / collate / loss for the three tasks

#### (A) pair / hitlist (PMT-level two-channel labels)
Key fields from a single Dataset sample (per event):
- `hit_features: (N, 5)`
- `hit_pmt_ids:  (N,)` which selected PMT (local index `0..M-1`) each hit belongs to
- `pmt_labels:   (M, 2)` two-channel labels for the M selected PMTs in this event
- `unique_pmt_ids: (M,)` global PMT ids of those selected PMTs

After collating into a batch:
- `hit_features: (N_total, 5)`
- `hit_pmt_ids:  (N_total,)` (offset-adjusted to index the concatenated `pmt_labels`)
- `pmt_labels:   (M_total, 2)`

During training:
- Model outputs `logits_hit: (N_total, 2)`
- Aggregate to `logits_pmt: (M_total, 2)` (`scatter(..., reduce=juno_scatter_reduce)`)
- Loss: `BCEWithLogitsLoss(logits_pmt, pmt_labels)`

#### (B) apme (hit-level binary classification)
Single-sample output:
- `hit_features: (N, 5)`
- `hit_labels:   (N,)` (0/1)  
(optional) `hit_pmt: (N,)` only for visualization/tracking

After collating:
- `hit_features: (N_total, 5)`
- `hit_labels:   (N_total,)`

During training:
- Model outputs `logits_hit: (N_total, 2)`
- Loss: `CrossEntropyLoss(logits_hit, hit_labels)`

---

## 4. Training Configuration (Options + JSON)

Training is mainly driven by an `Options` object (`hpst/utils/options.py`) and can be overridden via JSON. Example config:

- `config/pst/pst_small_tune.json`

Key fields:

### 4.1 Training hyperparameters
- `epochs`
- `batch_size`
- `optimizer` (expected to be a class name under `torch.optim`, e.g., `AdamW`)
- `learning_rate`
- `learning_rate_cycles`
- `learning_rate_warmup_epochs`
- `l2_penalty`
- `gradient_clip`
- `num_dataloader_workers`
- `num_gpu`

### 4.2 Dataset and splits
- `training_file`: JUNO dataset root directory (a folder)
- `train_validation_split`: train/(train+val) ratio (after excluding test)
- `test_split`: if `testing_file` is not explicitly provided, split a portion from `training_file` as test
- `split_seed`: random seed for splitting
- `juno_dataset_type`: `pair` / `hitlist` / `apme`

### 4.3 JUNO geometry/sampling parameters
- `juno_coords_path`: PMT coordinate table (npy)
- `juno_vg_mm_per_ns`: group velocity
- `juno_radius_mm`: normalization radius for coords and vt
- `juno_rng_seed`: sampling RNG seed
- `juno_neg_ratio / juno_neg_cap`: negative sampling only for `pair/hitlist`
- `juno_scatter_reduce`: `scatter` reduction method (e.g., `sum`)

### 4.4 Hit-level threshold (for apme threshold-based metrics)
- `hit_threshold`: threshold applied to `softmax(logits)[:,1]`, default 0.5

---

## 5. Training

The training entry script is `scripts/train_homogenous.py` (Lightning). The repo root provides `train_JUNO.sh` as a convenience launcher.

### 5.1 Launch training

Run:

```
bash train_JUNO.sh
```

The default command in `train_JUNO.sh` (run in background + redirect output) is:
```
CUDA_VISIBLE_DEVICES=0 nohup python scripts/train_homogenous.py  --name "JUNO_run" --log_dir "results" --gpus 1 > JUNO.log 2>&1 &
```

Where:

- `--name`: subdirectory name for this experiment (to distinguish runs)
- `--log_dir`: output directory (TensorBoard logs and checkpoints)

- `--gpus`: number of GPUs to use (overrides `options.num_gpu`)

### 5.2 Output location and file structure

The training script uses `TensorBoardLogger`. All outputs are saved under `{log_dir}/{name}/version_{k}/`:

```text
{log_dir}/{name}/version_0/
├── events.out.tfevents*           # TensorBoard event files
├── options.json                   # Hyperparameter snapshot for this run
├── training_curves.png            # Training curves (auto-generated after training)
└── checkpoints/
    ├── last.ckpt                  # Weights from the last epoch
    ├── epoch=xx-step=yyyy.ckpt    # top-k best models (by monitored metric)
    └── ...
```

## 6. Testing & Plotting

This repo provides evaluation and plotting scripts for **JUNO (pair/hitlist; PMT-level, 2 channels)** and **APME (hit-level binary classification)**.

### 6.1 Overview of the main scripts

- `scripts/evaluation_JUNO.py`  
  Runs inference for **JUNO pair/hitlist** and exports prediction artifacts (typically a **single-event** `.npz` for visualization; optionally full-set predictions for metrics).

- `scripts/plots.py`  
  Reads the `.npz` produced by `evaluation_JUNO.py` and generates (or reproduces) plots such as:
  - confusion matrices / ROC-PR (some blocks are currently commented out in the script)
  - Mollweide (PMT sphere) projections for a single event (truth vs pred), including count/NPE overlays and scatter visualizations

- `scripts/evaluation_hits.py`  
  Runs inference for **APME hit-level binary classification** and exports:
  - `y_true`, `y_pred`, `y_prob` (where `y_prob` is `P(ME)` from `softmax(logits)[:,1]`)
  - per-hit event identifiers `chunk_index` and `local_event` (required by some downstream analyses)

- `scripts/plots_hits.py`  
  Reads the `.npz` exported by `evaluation_hits.py` and generates:
  - hit-level confusion matrices (counts + row-normalized)
  - ROC / PR curves
  - probability histograms by class
  - optional analyses that require AP raw events:
    - “AP raw event nhits vs event balanced accuracy”
    - “AP raw event nhits vs ME hit-accuracy (within event)”

---

### 6.2 JUNO (pair/hitlist): run evaluation and make plots

#### 6.2.1 Export a single-event artifact (for Mollweide plots)

`evaluation_JUNO.py` always exports a single-event `.npz` for visualization. If you set `--single_only`, it will skip exporting the (commented-out) full-set file.

Example:

```bash
python scripts/evaluation_JUNO.py 
  --training_file "...\data\scattered" 
  --options_file "config\pst\pst_small_tune.json" 
  --ckpt "...\results\Noise_run\version_2\checkpoints\last.ckpt" 
  --out_dir "...\results\Noise_run\version_2" 
  --device cuda 
  --seed 279 
  --single_only
```

This will produce:

- `{out_dir}/eval_single_event_{seed}.npz` (e.g. `eval_single_event_279.npz`)

Contents (key fields):

- `split` (`val` or `test`)
- `event_idx` (global index inside the underlying dataset)
- `y_true_bin`, `y_pred_bin`, `y_prob` at **PMT level** with shape `(M,2)`
- `unique_pmt_ids` (global PMT ids for the selected PMTs in this event)
- `pmt_npe` (PMT-level summed npe derived from hit features)

#### 6.2.2 Plot Mollweide projections (single event)

Use `scripts/plots.py` to load the `eval_single_event_*.npz` and draw Mollweide plots.

Example:

```bash
python scripts/plots.py 
  --eval_dir "...\results\Noise_run\version_2" 
  --mixed_root "...\data\scattered" 
  --options_file "config\pst\pst_small_tune.json" 
  --split auto
```

Notes:
- `--eval_dir` is where `eval_single_event_*.npz` lives and is also used as the output root.
- `--mixed_root` + `--options_file` are used to reconstruct PMT geometry so the script can map `unique_pmt_ids -> coords_mm` for Mollweide projection.
- Many plotting blocks (confusion matrices, ROC/PR, and additional analyses) are currently **commented out** in `plots.py`. Enable the blocks you need.

---

### 6.3 APME (hit-level): run evaluation and make plots

#### 6.3.1 Export hit-level predictions to NPZ

Example:

```bash
python scripts/evaluation_hits.py 
  --training_file "...\data\AP_ME" 
  --options_file "config\pst\pst_small_tune.json" 
  --ckpt "...\results\ME_AP_run_new\version_1\checkpoints\epoch=15-step=3000.ckpt" 
  --out_dir "...\results\ME_AP_run_new" 
  --device cuda 
  --out_name "eval_full_apme.npz"
```

Output: `{out_dir}/eval_full_apme.npz`

Key fields:
- `split`: `val` or `test`
- `y_true`: `(N,)` in `{0,1}` (0=AP, 1=ME)
- `y_prob`: `(N,)` float, predicted `P(ME)`
- `y_pred`: `(N,)` predicted class (threshold 0.5 by default)
- `chunk_index`, `local_event`: `(N,)` per-hit event ids (required for event-level grouping)

#### 6.3.2 Generate hit-level plots (CM/ROC/PR/hist)

Example:

```bash
python scripts/plots_hits.py 
  --eval_dir "...\results\ME_AP_run_new" 
  --npz "eval_full_apme.npz" 
  --out_subdir "plots_apme"
```

Outputs (in `{eval_dir}/{out_subdir}/`):
- `confusion_counts_{split}.png`
- `confusion_norm_{split}.png`
- `roc_me.png`, `pr_me.png`
- `prob_hist_by_class.png`
- `metrics.json`

---

### 6.4 Optional: AP raw event nhits vs accuracy analyses (APME)

Some analyses in `plots_hits.py` require the AP raw-event directory containing `AP_chunk_*.npy`:

- `--ap_raw_dir <dir>`: directory with files like `AP_chunk_0.npy`, `AP_chunk_1.npy`, ...

Two optional plots:

1) **AP event nhits vs event balanced accuracy**  
`balanced_acc = (acc_AP + acc_ME)/2` (averaged over classes present)

```bash
python scripts/plots_hits.py 
  --eval_dir "...\results\ME_AP_run_new" 
  --npz "eval_full_apme.npz" 
  --out_subdir "plots_apme" 
  --ap_raw_dir "...\data\AP_ME_raw\AP_chunks" 
  --plot_ap_nhits_vs_acc
```

2) **AP event nhits vs ME hit accuracy (within event)**  
Only includes events with at least `--min_me_hits` ME hits.

```bash
python scripts/plots_hits.py 
  --eval_dir "...\results\ME_AP_run_new" 
  --npz "eval_full_apme.npz" 
  --out_subdir "plots_apme" 
  --ap_raw_dir "...\data\AP_ME_raw\AP_chunks" 
  --plot_me_acc_vs_ap_nhits 
  --min_me_hits 1
```

If your evaluation `.npz` does not contain `chunk_index/local_event`, re-run `evaluation_hits.py` using the version that exports event ids (current script enforces this).


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from argparse import ArgumentParser
from typing import Optional
from os import getcwd, makedirs, environ
import json
import lightning.pytorch as pl
import torch
import datetime

from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
    RichProgressBar,
    RichModelSummary,
    EarlyStopping
)

from hpst.utils.options import Options


def plot_training_curves(log_dir: str):
    """
    训练结束后，从 TensorBoard event 文件读取指标并画图
    保存到 {log_dir}/training_curves.png
    """
    import matplotlib
    matplotlib.use('Agg')  # 无 GUI 后端
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import os
    
    # 找到最新的 events 文件
    event_files = [f for f in os.listdir(log_dir) if f.startswith('events.out.tfevents')]
    if not event_files:
        print(f"Warning: No TensorBoard event files found in {log_dir}")
        return
    
    event_path = os.path.join(log_dir, event_files[0])
    
    # 加载 TensorBoard 数据
    ea = EventAccumulator(event_path)
    ea.Reload()
    
    # 提取指标
    metrics_to_plot = {
        'train_loss': 'Train Loss',
        'val_loss': 'Val Loss',
        'val_acc_exact': 'Val Accuracy (Exact Match)',
        'val_acc_eplus': 'Val Acc (e+)',
        'val_acc_c14': 'Val Acc (C14)',
        'val_f1_eplus': 'Val F1 (e+)',
        'val_f1_c14': 'Val F1 (C14)'
    }
    
    data = {}
    for key in metrics_to_plot.keys():
        if key in ea.Tags()['scalars']:
            events = ea.Scalars(key)
            data[key] = {
                'steps': [e.step for e in events],
                'values': [e.value for e in events]
            }
    
    if not data:
        print("Warning: No scalar data found in TensorBoard logs")
        return
    
    # 创建 2x2 子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics', fontsize=16)
    
    # 子图 1: Train Loss vs Val Loss
    ax = axes[0, 0]
    if 'train_loss' in data:
        ax.plot(data['train_loss']['steps'], data['train_loss']['values'], 
                label='Train Loss', color='blue', alpha=0.7)
    if 'val_loss' in data:
        ax.plot(data['val_loss']['steps'], data['val_loss']['values'], 
                label='Val Loss', color='red', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 子图 2: Val Accuracy (Exact Match)
    ax = axes[0, 1]
    if 'val_acc_exact' in data:
        ax.plot(data['val_acc_exact']['steps'], data['val_acc_exact']['values'], 
                label='Val Acc (Exact)', color='green', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Accuracy')
    ax.set_title('Validation Accuracy (Exact Match)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # 子图 3: Per-Class Accuracy (e+ vs C14)
    ax = axes[1, 0]
    if 'val_acc_eplus' in data:
        ax.plot(data['val_acc_eplus']['steps'], data['val_acc_eplus']['values'], 
                label='Val Acc (e+)', color='orange', alpha=0.8)
    if 'val_acc_c14' in data:
        ax.plot(data['val_acc_c14']['steps'], data['val_acc_c14']['values'], 
                label='Val Acc (C14)', color='purple', alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Class Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # 子图 4: F1 Scores (e+ vs C14)
    ax = axes[1, 1]
    if 'val_f1_eplus' in data:
        ax.plot(data['val_f1_eplus']['steps'], data['val_f1_eplus']['values'], 
                label='Val F1 (e+)', color='cyan', alpha=0.8)
    if 'val_f1_c14' in data:
        ax.plot(data['val_f1_c14']['steps'], data['val_f1_c14']['values'], 
                label='Val F1 (C14)', color='magenta', alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-Class F1 Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(log_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Training curves saved to: {save_path}")
    
    # 同时打印最终指标
    print("\n" + "="*60)
    print("FINAL METRICS (Last Epoch)")
    print("="*60)
    for key, label in metrics_to_plot.items():
        if key in data and len(data[key]['values']) > 0:
            final_val = data[key]['values'][-1]
            print(f"{label:30s}: {final_val:.4f}")
    print("="*60 + "\n")


def main(
        log_dir: str,
        name: str,
        options_file: str,
        training_file: str,
        checkpoint: Optional[str],
        fp16: bool,
        graph: bool,
        verbose: bool,
        batch_size: Optional[int],
        gpus: Optional[int],
        threads: Optional[int],
        debug: bool,
        eval: int,
        **kwargs
):
    
    master = "NODE_RANK" not in environ
    from hpst.trainers.point_set_trainer import PointSetTrainer
    Network = PointSetTrainer

    options = Options(training_file)
    if options_file is not None:
        with open(options_file, 'r') as json_file:
            options.update_options(json.load(json_file))

    # Apply Command line overrides
    # ---------------------------------------------------------------------------------------------
    options.verbose_output = verbose

    if threads is not None:
        if master:
            print(f"Setting CPU count: {threads}")
        torch.set_num_threads(threads)
        environ["OMP_NUM_THREADS"] = str(threads)
        environ["MKL_NUM_THREADS"] = str(threads)

    if gpus is not None:
        if master:
            print(f"Overriding GPU count: {gpus}")
        options.num_gpu = gpus

    if batch_size is not None:
        if master:
            print(f"Overriding Batch Size: {batch_size}")
        options.batch_size = batch_size

    if debug:
        if master:
            print(f"Debug Mode: 1 GPU, 0 dataloader workers, Small Batch size")
        torch.jit.script = lambda x: x
        options.num_dataloader_workers = 0
        options.batch_size = 2

    # JUNO overrides from CLI
    # ---------------------------------------------------------------------------------------------
    if kwargs.get("juno_coords_path") is not None:
        options.juno_coords_path = kwargs["juno_coords_path"]
    if kwargs.get("juno_vg_mm_per_ns") is not None:
        options.juno_vg_mm_per_ns = float(kwargs["juno_vg_mm_per_ns"])
    if kwargs.get("juno_radius_mm") is not None:
        options.juno_radius_mm = float(kwargs["juno_radius_mm"])
    if kwargs.get("juno_neg_ratio") is not None:
        options.juno_neg_ratio = int(kwargs["juno_neg_ratio"])
    if kwargs.get("juno_neg_cap") is not None:
        options.juno_neg_cap = int(kwargs["juno_neg_cap"])
    if kwargs.get("juno_rng_seed") is not None:
        options.juno_rng_seed = int(kwargs["juno_rng_seed"])

    # Print hyperparameters
    # ---------------------------------------------------------------------------------------------
    if master:
        options.display()

    # Create model
    # ---------------------------------------------------------------------------------------------
    model = Network(options)

    # Create Loggers and Callbacks
    # ---------------------------------------------------------------------------------------------
    if log_dir is None:
        log_dir = "results/"
    
    if debug:
        logger = False
        callbacks = None
    else:
        logger = TensorBoardLogger(save_dir=log_dir, name=name, log_graph=graph)

        checkpoint_callback = ModelCheckpoint(
            verbose=options.verbose_output,
            every_n_train_steps=eval,
            monitor="val_acc_exact",
            mode="max",
            save_top_k=5,
            save_last=True
        )

        early_stop_callback = EarlyStopping(
            monitor=options.early_stop_monitor,
            patience=options.early_stop_patience,
            min_delta=options.early_stop_min_delta,
            mode=options.early_stop_mode,
            verbose=True,
            strict=False  
        )


        callbacks = [
            checkpoint_callback,
            early_stop_callback,
            LearningRateMonitor(),
            TQDMProgressBar(refresh_rate=10), 
            RichModelSummary(max_depth=3),
        ]

    distributed_backend = 'auto'
    if options.num_gpu > 1:
        distributed_backend = DDPStrategy(find_unused_parameters=True)

    # Create trainer
    # ---------------------------------------------------------------------------------------------
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=options.epochs,
        callbacks=callbacks,
        strategy=distributed_backend,
        accelerator='gpu' if options.num_gpu > 0 else None,
        devices=options.num_gpu if options.num_gpu > 0 else None,
        gradient_clip_val=options.gradient_clip,
        precision=16 if fp16 else 32,
        check_val_every_n_epoch=1,
        log_every_n_steps=3
    )

    if master and not debug:
        print(f"Training Version {trainer.logger.version}")
        makedirs(trainer.logger.log_dir, exist_ok=True)
        with open(trainer.logger.log_dir + "/options.json", 'w') as json_file:
            json.dump(options.__dict__, json_file, indent=4)

    # Train
    # ---------------------------------------------------------------------------------------------
    trainer.fit(model)

    # Plot training curves (only on master process)
    # ---------------------------------------------------------------------------------------------
    if master and not debug:
        try:
            plot_training_curves(trainer.logger.log_dir)
        except Exception as e:
            print(f"Warning: Failed to plot training curves: {e}")


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("-t", "--training_file", type=str, default="../data/training_prong_pixels_sparse_64.h5",
                        help="Input file containing training data.")

    parser.add_argument("-o", "--options_file", type=str, default="config/pst/pst_small_tune.json",
                        help="JSON file with option overloads.")

    parser.add_argument("-c", "--checkpoint", type=str, default=None,
                        help="Optional checkpoint to load from")

    parser.add_argument("-n", "--name", type=str, default="lightning_logs",
                        help="The sub-directory to create for this run.")

    parser.add_argument("-l", "--log_dir", type=str, default="/disk_pool1/houyh/results/",
                        help="Output directory for the checkpoints and tensorboard logs.")

    parser.add_argument("-fp16", action="store_true",
                        help="Use AMP for training.")

    parser.add_argument("-g", "--graph", action="store_true",
                        help="Log the computation graph.")

    parser.add_argument("-v", "--verbose", action='store_true',
                        help="Output additional information to console and log.")

    parser.add_argument("-b", "--batch_size", type=int, default=None,
                        help="Override batch size in hyperparameters.")

    parser.add_argument("-e", "--eval", type=int, default=500,
                        help="Number of steps before eval")

    parser.add_argument("--gpus", type=int, default=None,
                        help="Override GPU count in hyperparameters.")

    parser.add_argument("--threads", type=int, default=None,
                        help="Override CPU count in hyperparameters.")

    parser.add_argument("-p", "--pixels", action="store_true",
                        help="Use Pixel information in dataset.")

    parser.add_argument("-d", "--debug", action="store_true",
                        help="Debug options super-switch. ")
    
    # JUNO-specific dataset overrides
    parser.add_argument("--juno_coords_path", type=str, default=None, help="Path to whichPixel_nside32_LCDpmts.npy")
    parser.add_argument("--juno_vg_mm_per_ns", type=float, default=None, help="Group velocity in mm/ns (default 190)")
    parser.add_argument("--juno_radius_mm", type=float, default=None, help="Radius scale in mm for coord normalization")
    parser.add_argument("--juno_neg_ratio", type=int, default=None, help="Negative sampling ratio (neg = ratio * pos)")
    parser.add_argument("--juno_neg_cap", type=int, default=None, help="Maximum number of negatives per event")
    parser.add_argument("--juno_rng_seed", type=int, default=None, help="RNG seed for negative sampling")

    main(**parser.parse_args().__dict__)
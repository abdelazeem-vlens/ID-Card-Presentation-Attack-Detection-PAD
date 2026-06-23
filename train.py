"""
train.py
--------
Training entry point for the ID Card PAD project.

Usage:
    # Fresh training run with default config:
    python train.py

    # Fresh training with a custom config YAML:
    python train.py --config my_config.yaml

    # Resume from the last checkpoint of a previous run:
    python train.py --resume experiments/2026-06-23_14-32-05

Each run creates a timestamped directory under experiments/ containing:
    config_snapshot.yaml   — exact config used
    train.log              — human-readable training log
    metrics.csv            — machine-readable epoch metrics
    checkpoints/
        best.pth           — checkpoint with highest val AUC
        last.pth           — checkpoint from the most recent epoch
        epoch_XX.pth       — per-epoch checkpoints (if enabled in config)
    plots/                 — reserved for post-training plots
"""

import os
import sys
import argparse
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from config import Config
from dataset import get_dataloader
from model import build_model
from utils import MetricsAccumulator, ExperimentLogger


# --------------------------------------------------------------------------- #
#  Experiment setup helpers                                                     #
# --------------------------------------------------------------------------- #

def create_experiment_dir(cfg: Config) -> str:
    """Create a timestamped experiment directory and required subdirs."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join(cfg.paths.experiments_root, timestamp)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
    return exp_dir


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_auc: float,
    cfg: Config,
) -> None:
    """Save a full training checkpoint."""
    torch.save(
        {
            "epoch":      epoch,
            "best_hter":  best_auc,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "scheduler":  scheduler.state_dict() if scheduler else None,
            "config":     cfg.to_dict(),
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
):
    """
    Load a checkpoint into model, optimizer, scheduler.

    Returns:
        (start_epoch, best_auc)
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"] + 1, ckpt["best_hter"]


def build_scheduler(optimizer, cfg: Config, steps_after_warmup: int):
    """Build LR scheduler (excluding warmup, which is handled manually)."""
    stype = cfg.scheduler.scheduler_type.lower()

    if stype == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=max(cfg.training.num_epochs - cfg.scheduler.warmup_epochs, 1),
            eta_min=cfg.scheduler.eta_min,
        )
    elif stype == "step":
        return StepLR(
            optimizer,
            step_size=cfg.scheduler.step_size,
            gamma=cfg.scheduler.gamma,
        )
    elif stype == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler_type: '{cfg.scheduler.scheduler_type}'")


def get_warmup_lr(base_lr: float, warmup_factor: float, epoch: int, warmup_epochs: int) -> float:
    """Linear warmup: ramp LR from base_lr * warmup_factor to base_lr."""
    if warmup_epochs <= 0:
        return base_lr
    progress = min(epoch / warmup_epochs, 1.0)
    return base_lr * (warmup_factor + (1.0 - warmup_factor) * progress)


# --------------------------------------------------------------------------- #
#  Train / Validate one epoch                                                   #
# --------------------------------------------------------------------------- #

def run_epoch(
    phase: str,
    model: nn.Module,
    loader,
    optimizer: Optional[torch.optim.Optimizer],
    cfg: Config,
    device: torch.device,
    epoch: int,
    logger: ExperimentLogger,
    accumulator: MetricsAccumulator,
) -> dict:
    """
    Run one full epoch of training or validation.

    Args:
        phase:       "TRAIN" or "VAL"
        model:       The PAD model.
        loader:      DataLoader for the current split.
        optimizer:   Optimizer (None during validation).
        cfg:         Master config.
        device:      Compute device.
        epoch:       Current epoch number (1-indexed).
        logger:      ExperimentLogger instance.
        accumulator: MetricsAccumulator (will be reset and filled).

    Returns:
        Metrics dict from accumulator.compute().
    """
    is_train = phase == "TRAIN"
    model.train(is_train)
    accumulator.reset()

    cls_criterion = nn.BCEWithLogitsLoss()
    freq_criterion = nn.MSELoss()

    total_steps = len(loader)

    with torch.set_grad_enabled(is_train):
        for step, batch in enumerate(loader, start=1):
            images   = batch["image"].to(device, non_blocking=True)     # (B, 3, H, W)
            freq_gt  = batch["freq_map"].to(device, non_blocking=True)  # (B, 1, fH, fW)
            labels   = batch["label"].to(device, non_blocking=True)     # (B,)

            # Forward
            out = model(images, training=True)

            cls_logit = out["cls_logit"].squeeze(1)        # (B,)
            freq_pred = out.get("freq_pred")               # (B, 1, fH, fW) or None

            # Losses
            l_cls  = cls_criterion(cls_logit, labels.float())
            l_freq = freq_criterion(freq_pred, freq_gt) if freq_pred is not None else torch.tensor(0.0, device=device)
            loss   = cfg.training.lambda_cls * l_cls + cfg.training.lambda_freq * l_freq

            if is_train:
                optimizer.zero_grad()
                loss.backward()

                if cfg.training.grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)

                optimizer.step()

            # Accumulate scores and labels
            scores = torch.sigmoid(cls_logit).detach()
            accumulator.update(scores, labels, loss=loss.item())

            # Batch-level logging
            if (
                cfg.training.log_every_n_steps > 0
                and step % cfg.training.log_every_n_steps == 0
            ):
                logger.log_batch(
                    epoch=epoch,
                    phase=phase,
                    step=step,
                    total_steps=total_steps,
                    loss_cls=l_cls.item(),
                    loss_freq=l_freq.item(),
                    loss_total=loss.item(),
                )

    return accumulator.compute()


# --------------------------------------------------------------------------- #
#  Main training loop                                                           #
# --------------------------------------------------------------------------- #

def train(cfg: Config, exp_dir: str, resume_epoch: int = 1, best_auc: float = 0.0):
    """Full training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------ #
    #  Logger                                                              #
    # ------------------------------------------------------------------ #
    logger = ExperimentLogger(
        exp_dir=exp_dir,
        log_name="train",
        total_epochs=cfg.training.num_epochs,
    )
    logger.separator("=")
    logger.info(f"Experiment directory : {exp_dir}")
    logger.info(f"Device               : {device}")
    logger.info(f"Backbone             : {cfg.model.backbone}")
    logger.info(f"Aux branch stage     : {cfg.model.aux_branch_stage}")
    logger.info(f"Batch size           : {cfg.data.batch_size}")
    logger.info(f"Learning rate        : {cfg.training.learning_rate}")
    logger.separator("=")

    # ------------------------------------------------------------------ #
    #  Data                                                                #
    # ------------------------------------------------------------------ #
    train_loader = get_dataloader("train", cfg)
    val_loader   = get_dataloader("val",   cfg)

    logger.info(f"Train set: {train_loader.dataset}")
    logger.info(f"Val   set: {val_loader.dataset}")
    logger.separator()

    # ------------------------------------------------------------------ #
    #  Model, optimizer, scheduler                                         #
    # ------------------------------------------------------------------ #
    model = build_model(cfg).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = build_scheduler(optimizer, cfg, steps_after_warmup=0)

    # ------------------------------------------------------------------ #
    #  Resume if requested                                                 #
    # ------------------------------------------------------------------ #
    if resume_epoch > 1:
        ckpt_path = os.path.join(exp_dir, "checkpoints", "last.pth")
        resume_epoch, best_auc = load_checkpoint(ckpt_path, model, optimizer, scheduler, device)
        logger.log_resume(exp_dir, resume_epoch)

    # ------------------------------------------------------------------ #
    #  Metrics accumulators                                                #
    # ------------------------------------------------------------------ #
    train_acc = MetricsAccumulator()
    val_acc   = MetricsAccumulator()

    # ------------------------------------------------------------------ #
    #  Epoch loop                                                          #
    # ------------------------------------------------------------------ #
    no_improve_count = 0

    for epoch in range(resume_epoch, cfg.training.num_epochs + 1):

        # LR warmup — override scheduler during warmup phase
        if epoch <= cfg.scheduler.warmup_epochs:
            warmup_lr = get_warmup_lr(
                cfg.training.learning_rate,
                cfg.scheduler.warmup_factor,
                epoch,
                cfg.scheduler.warmup_epochs,
            )
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        # -------- Train --------
        train_metrics = run_epoch(
            phase="TRAIN",
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            cfg=cfg,
            device=device,
            epoch=epoch,
            logger=logger,
            accumulator=train_acc,
        )
        logger.log_metrics(epoch, "TRAIN", train_metrics)

        # -------- Validate --------
        val_metrics = run_epoch(
            phase="VAL",
            model=model,
            loader=val_loader,
            optimizer=None,
            cfg=cfg,
            device=device,
            epoch=epoch,
            logger=logger,
            accumulator=val_acc,
        )
        logger.log_metrics(epoch, "VAL", val_metrics)
        logger.separator()

        # -------- Scheduler step (after warmup) --------
        if scheduler is not None and epoch > cfg.scheduler.warmup_epochs:
            scheduler.step()

        # -------- Checkpointing --------
        val_auc = val_metrics["auc"]
        is_best  = val_auc > best_auc

        if is_best:
            best_auc = val_auc
            no_improve_count = 0
            best_path = os.path.join(exp_dir, "checkpoints", "best.pth")
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_auc, cfg)
            logger.log_checkpoint(epoch, best_path, is_best=True)
        else:
            no_improve_count += 1

        last_path = os.path.join(exp_dir, "checkpoints", "last.pth")
        save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_auc, cfg)
        logger.log_checkpoint(epoch, last_path, is_best=False)

        if cfg.training.save_every_epoch:
            epoch_path = os.path.join(exp_dir, "checkpoints", f"epoch_{epoch:03d}.pth")
            save_checkpoint(epoch_path, model, optimizer, scheduler, epoch, best_auc, cfg)

        # -------- Early stopping --------
        if no_improve_count >= cfg.training.early_stopping_patience:
            logger.log_early_stop(epoch, cfg.training.early_stopping_patience)
            break

    logger.separator("=")
    logger.info(f"Training complete. Best val AUC: {best_auc:.4f}")
    logger.separator("=")


# --------------------------------------------------------------------------- #
#  Entry point                                                                  #
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description="Train PAD model")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file. If not provided, defaults are used.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to an existing experiment directory to resume training from.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ---- Load or create config ----
    if args.resume:
        # Resume: reload config from the experiment's snapshot
        snap_path = os.path.join(args.resume, "config_snapshot.yaml")
        if not os.path.exists(snap_path):
            sys.exit(f"[ERROR] Config snapshot not found at: {snap_path}")
        cfg = Config.from_yaml(snap_path)
        exp_dir = args.resume
        resume_epoch = 2   # actual epoch will be read from checkpoint in train()
    else:
        cfg = Config.from_yaml(args.config) if args.config else Config()
        exp_dir = create_experiment_dir(cfg)
        cfg.save(os.path.join(exp_dir, "config_snapshot.yaml"))
        resume_epoch = 1

    train(cfg, exp_dir, resume_epoch=resume_epoch)

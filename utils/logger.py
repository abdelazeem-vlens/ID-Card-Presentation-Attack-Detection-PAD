"""
utils/logger.py
---------------
Structured logging utility for training and evaluation runs.

Each logger instance writes simultaneously to:
  1. stdout           — for live monitoring
  2. <exp_dir>/<name>.log  — human-readable timestamped log file
  3. <exp_dir>/metrics.csv — machine-readable metrics for plotting

Log line format:
  [2026-06-23 14:32:05] [Epoch 03/50] [TRAIN]  loss=0.312  eer=0.142  hter=0.142  auc=0.923

CSV line format (header written once on first call):
  epoch,phase,loss,eer,far,frr,hter,auc,eer_threshold
"""

import os
import csv
import logging
import sys
from datetime import datetime
from typing import Dict, Optional


class ExperimentLogger:
    """
    One logger per experiment run.

    Args:
        exp_dir:   Path to the experiment directory (must already exist).
        log_name:  Stem of the .log file (e.g. "train" → train.log).
        total_epochs: Used for zero-padded epoch formatting in log lines.
    """

    # CSV columns written to metrics.csv
    CSV_FIELDNAMES = [
        "timestamp", "epoch", "phase",
        "mean_loss", "eer", "far_at_eer", "frr_at_eer", "hter", "auc", "eer_threshold",
    ]

    def __init__(
        self,
        exp_dir: str,
        log_name: str = "train",
        total_epochs: int = 0,
    ) -> None:
        self.exp_dir = exp_dir
        self.total_epochs = total_epochs
        self._csv_initialized = False

        log_path = os.path.join(exp_dir, f"{log_name}.log")
        self._csv_path = os.path.join(exp_dir, "metrics.csv")

        # ---------------------------------------------------------------- #
        #  Python logger setup                                              #
        # ---------------------------------------------------------------- #
        logger_name = f"pad.{log_name}.{id(self)}"
        self._logger = logging.getLogger(logger_name)
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False  # don't bubble up to root logger

        fmt = logging.Formatter("%(message)s")   # we handle formatting ourselves

        # File handler
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        self._logger.addHandler(fh)

        # Stdout handler
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        self._logger.addHandler(sh)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def info(self, message: str) -> None:
        """Write a plain informational message."""
        ts = self._timestamp()
        self._logger.info(f"[{ts}]  {message}")

    def log_metrics(
        self,
        epoch: int,
        phase: str,
        metrics: Dict[str, float],
    ) -> None:
        """
        Write a full metrics dict for one epoch/phase combination.

        Args:
            epoch:   Current epoch number (1-indexed).
            phase:   "TRAIN" | "VAL" | "TEST"
            metrics: Dict as returned by MetricsAccumulator.compute()
        """
        ts = self._timestamp()
        epoch_str = self._fmt_epoch(epoch)
        phase_str = phase.upper().ljust(5)

        # Build human-readable metric string
        metric_parts = []
        for key in ["mean_loss", "eer", "far_at_eer", "frr_at_eer", "hter", "auc", "eer_threshold"]:
            if key in metrics:
                display_key = key.replace("_at_eer", "@eer").replace("mean_", "")
                metric_parts.append(f"{display_key}={metrics[key]:.4f}")

        metric_str = "  ".join(metric_parts)
        line = f"[{ts}] [{epoch_str}] [{phase_str}]  {metric_str}"
        self._logger.info(line)

        # Write to CSV
        self._write_csv(epoch, phase, metrics, ts)

    def log_batch(
        self,
        epoch: int,
        phase: str,
        step: int,
        total_steps: int,
        loss_cls: float,
        loss_freq: float,
        loss_total: float,
    ) -> None:
        """
        Write a batch-level loss line (called every N steps).

        Args:
            epoch, phase: Current epoch and phase name.
            step:         Current step index within the epoch.
            total_steps:  Total number of steps in the epoch.
            loss_cls:     Classification (BCE) loss value.
            loss_freq:    Frequency (MSE) loss value.
            loss_total:   Combined weighted loss value.
        """
        ts = self._timestamp()
        epoch_str = self._fmt_epoch(epoch)
        self._logger.debug(
            f"[{ts}] [{epoch_str}] [{phase.upper()}] "
            f"step={step}/{total_steps}  "
            f"loss_cls={loss_cls:.4f}  "
            f"loss_freq={loss_freq:.4f}  "
            f"loss_total={loss_total:.4f}"
        )

    def log_checkpoint(self, epoch: int, path: str, is_best: bool = False) -> None:
        """Log a checkpoint save event."""
        tag = " [BEST]" if is_best else ""
        self.info(f"Checkpoint saved{tag} → {path}  (epoch {epoch})")

    def log_early_stop(self, epoch: int, patience: int) -> None:
        """Log early stopping trigger."""
        self.info(
            f"Early stopping triggered at epoch {epoch} "
            f"(no improvement for {patience} epochs)."
        )

    def log_resume(self, exp_dir: str, epoch: int) -> None:
        """Log that training is resuming from a checkpoint."""
        self.info(f"Resuming from '{exp_dir}' at epoch {epoch}.")

    def separator(self, char: str = "-", width: int = 80) -> None:
        """Write a visual separator line."""
        self._logger.info(char * width)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _fmt_epoch(self, epoch: int) -> str:
        if self.total_epochs:
            pad = len(str(self.total_epochs))
            return f"Epoch {epoch:0{pad}d}/{self.total_epochs}"
        return f"Epoch {epoch}"

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _write_csv(
        self,
        epoch: int,
        phase: str,
        metrics: Dict[str, float],
        ts: Optional[str] = None,
    ) -> None:
        write_header = not os.path.exists(self._csv_path) or not self._csv_initialized
        self._csv_initialized = True

        row = {
            "timestamp":     ts or self._timestamp(),
            "epoch":         epoch,
            "phase":         phase.upper(),
            "mean_loss":     metrics.get("mean_loss", ""),
            "eer":           metrics.get("eer", ""),
            "far_at_eer":    metrics.get("far_at_eer", ""),
            "frr_at_eer":    metrics.get("frr_at_eer", ""),
            "hter":          metrics.get("hter", ""),
            "auc":           metrics.get("auc", ""),
            "eer_threshold": metrics.get("eer_threshold", ""),
        }

        with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

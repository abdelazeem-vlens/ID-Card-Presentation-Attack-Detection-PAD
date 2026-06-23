"""
utils/metrics.py
----------------
Accumulates per-batch predictions and computes PAD evaluation metrics
at the end of a full epoch or evaluation run.

Metrics reported:
  - AUC           Area Under the ROC Curve
  - EER           Equal Error Rate  (FAR == FRR point on the ROC curve)
  - FAR @ EER     False Acceptance Rate at the EER threshold
  - FRR @ EER     False Rejection Rate at the EER threshold
  - HTER @ EER    Half Total Error Rate = (FAR + FRR) / 2  [primary PAD metric]
  - EER threshold The decision threshold that achieves EER

All metrics are computed over the FULL epoch — never per-batch.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Union
from sklearn.metrics import roc_curve, auc


class MetricsAccumulator:
    """
    Accumulate raw scores and labels across batches, then compute metrics.

    Usage:
        acc = MetricsAccumulator()

        # Inside batch loop:
        acc.update(scores, labels)

        # At epoch end:
        metrics = acc.compute()
        acc.reset()
    """

    def __init__(self) -> None:
        self._scores: List[np.ndarray] = []   # predicted probabilities
        self._labels: List[np.ndarray] = []   # ground truth (0=real, 1=replay)
        self._losses: List[float] = []        # optional scalar losses

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def update(
        self,
        scores: Union[torch.Tensor, np.ndarray],
        labels: Union[torch.Tensor, np.ndarray],
        loss: Optional[float] = None,
    ) -> None:
        """
        Accumulate predictions from one batch.

        Args:
            scores: Predicted probabilities (after sigmoid), shape (B,) or (B, 1).
            labels: Ground truth labels (0 or 1), shape (B,).
            loss:   Optional scalar batch loss to track mean loss.
        """
        scores_np = self._to_numpy(scores).ravel()
        labels_np = self._to_numpy(labels).ravel()

        self._scores.append(scores_np)
        self._labels.append(labels_np)

        if loss is not None:
            self._losses.append(float(loss))

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics over accumulated predictions.

        Returns:
            dict with keys:
                auc, eer, far_at_eer, frr_at_eer, hter, eer_threshold,
                mean_loss (only if losses were provided)
        """
        if not self._scores:
            raise RuntimeError("No predictions accumulated. Call update() first.")

        scores = np.concatenate(self._scores)    # (N,)
        labels = np.concatenate(self._labels)    # (N,)

        # Validate we have both classes
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            raise ValueError(
                f"Only one class present in labels: {unique_labels}. "
                "Cannot compute ROC/EER metrics."
            )

        # ---------------------------------------------------------------- #
        #  ROC Curve                                                        #
        #  sklearn convention: fpr = FAR (impostor accepted as genuine)    #
        #                       tpr = 1 - FRR                              #
        # ---------------------------------------------------------------- #
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        roc_auc = auc(fpr, tpr)

        # ---------------------------------------------------------------- #
        #  EER — point where FAR ≈ FRR                                     #
        #  FRR = 1 - TPR                                                   #
        # ---------------------------------------------------------------- #
        frr = 1.0 - tpr
        far = fpr

        # Find index where |FAR - FRR| is minimized
        abs_diff = np.abs(far - frr)
        eer_idx = np.argmin(abs_diff)

        eer_threshold = float(thresholds[eer_idx])
        far_at_eer   = float(far[eer_idx])
        frr_at_eer   = float(frr[eer_idx])
        eer          = float((far_at_eer + frr_at_eer) / 2.0)   # average for numerical stability
        hter         = eer  # HTER at EER threshold == EER by definition

        results = {
            "auc":           round(roc_auc, 6),
            "eer":           round(eer, 6),
            "far_at_eer":    round(far_at_eer, 6),
            "frr_at_eer":    round(frr_at_eer, 6),
            "hter":          round(hter, 6),
            "eer_threshold": round(eer_threshold, 6),
        }

        if self._losses:
            results["mean_loss"] = round(float(np.mean(self._losses)), 6)

        return results

    def compute_roc(self):
        """
        Return the full ROC curve arrays for plotting.

        Returns:
            fpr (np.ndarray), tpr (np.ndarray), thresholds (np.ndarray)
        """
        scores = np.concatenate(self._scores)
        labels = np.concatenate(self._labels)
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        return fpr, tpr, thresholds

    def reset(self) -> None:
        """Clear all accumulated data. Call at the start of each epoch."""
        self._scores.clear()
        self._labels.clear()
        self._losses.clear()

    def __len__(self) -> int:
        """Return number of accumulated samples."""
        if not self._scores:
            return 0
        return sum(s.shape[0] for s in self._scores)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().float().numpy()
        return np.array(x, dtype=np.float32)

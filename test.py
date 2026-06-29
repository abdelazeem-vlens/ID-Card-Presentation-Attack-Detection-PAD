"""
test.py
-------
Standalone evaluation script for a trained PAD model.

Usage:
    # Evaluate best checkpoint of an experiment:
    python test.py --experiment experiments/2026-06-23_14-32-05

    # Evaluate a specific checkpoint:
    python test.py --experiment experiments/2026-06-23_14-32-05 --checkpoint last.pth

Results are saved inside the experiment directory:
    test.log            — human-readable test results
    test_metrics.csv    — machine-readable metrics row
    plots/roc_curve.png — ROC curve figure
"""

import os
import sys
import csv
import argparse
from tqdm import tqdm

import torch
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for servers
import matplotlib.pyplot as plt

from config import Config
from dataset import get_dataloader
from model import build_model
from utils import MetricsAccumulator, ExperimentLogger, append_global_metrics


# --------------------------------------------------------------------------- #
#  ROC curve plotting                                                           #
# --------------------------------------------------------------------------- #

def plot_roc_curve(
    fpr,
    tpr,
    auc: float,
    eer: float,
    eer_threshold: float,
    save_path: str,
) -> None:
    """
    Plot and save the ROC curve with the EER point marked.

    Args:
        fpr, tpr:      ROC curve arrays from sklearn.
        auc:           Area under the ROC curve.
        eer:           Equal Error Rate value.
        eer_threshold: Decision threshold at EER.
        save_path:     Full path to save the PNG figure.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    # ROC curve
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC (AUC = {auc:.4f})")

    # Diagonal (random classifier)
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")

    # EER point: FRR = 1 - TPR = FAR → on the ROC, FAR == FRR
    # FRR curve in ROC space: y = 1 - x (line from (0,1) to (1,0))
    # EER point is intersection of ROC and anti-diagonal
    eer_fpr = eer    # at EER, FAR ≈ FRR ≈ EER
    eer_tpr = 1.0 - eer
    ax.scatter(
        [eer_fpr], [eer_tpr],
        color="crimson", zorder=5, s=80,
        label=f"EER = {eer:.4f}  (thr={eer_threshold:.3f})",
    )

    # EER anti-diagonal reference
    ax.plot([0, 1], [1, 0], color="salmon", lw=1, linestyle=":", alpha=0.6)

    ax.set_xlabel("False Acceptance Rate (FAR)", fontsize=12)
    ax.set_ylabel("True Positive Rate (1 − FRR)", fontsize=12)
    ax.set_title("ROC Curve — ID Card PAD", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Evaluation loop                                                              #
# --------------------------------------------------------------------------- #

def evaluate(exp_dir: str, checkpoint_name: str = "best.pth") -> None:
    """
    Run full evaluation on the test set.

    Args:
        exp_dir:         Path to the experiment directory.
        checkpoint_name: Filename of the checkpoint inside checkpoints/.
    """
    # ------------------------------------------------------------------ #
    #  Setup                                                               #
    # ------------------------------------------------------------------ #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config snapshot from the experiment (guarantees identical settings)
    snap_path = os.path.join(exp_dir, "config_snapshot.yaml")
    if not os.path.exists(snap_path):
        sys.exit(f"[ERROR] config_snapshot.yaml not found in: {exp_dir}")
    cfg = Config.from_yaml(snap_path)

    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    logger = ExperimentLogger(
        exp_dir=exp_dir,
        log_name="test",
        total_epochs=0,
    )

    logger.separator("=")
    logger.info(f"Experiment : {exp_dir}")
    logger.info(f"Checkpoint : {checkpoint_name}")
    logger.info(f"Device     : {device}")
    logger.separator("=")

    # ------------------------------------------------------------------ #
    #  Model                                                               #
    # ------------------------------------------------------------------ #
    model = build_model(cfg).to(device)

    ckpt_path = os.path.join(exp_dir, "checkpoints", checkpoint_name)
    if not os.path.exists(ckpt_path):
        sys.exit(f"[ERROR] Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    trained_epoch = ckpt.get("epoch", "?")
    logger.info(f"Loaded checkpoint from epoch {trained_epoch}  ({ckpt_path})")
    logger.separator()

    model.eval()

    # ------------------------------------------------------------------ #
    #  Data                                                                #
    # ------------------------------------------------------------------ #
    test_loader = get_dataloader("test", cfg)
    logger.info(f"Test set: {test_loader.dataset}")
    logger.separator()

    # ------------------------------------------------------------------ #
    #  Inference loop                                                       #
    # ------------------------------------------------------------------ #
    accumulator = MetricsAccumulator()

    # Per-sample records for wrong_predictions.csv
    all_records = []   # list of dicts: {path, true_label, score}

    with torch.no_grad():
        for batch in tqdm(test_loader):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            # Inference mode: auxiliary branch disabled
            if cfg.training.use_tta:
                variants = [images, torch.flip(images, dims=[-1]), torch.clamp(images * 1.03, 0.0, 1.0)]
                scores_list = []
                for variant in variants:
                    out = model(variant.to(device), training=False)
                    cls_logit = out["cls_logit"].squeeze(1)
                    scores_list.append(torch.sigmoid(cls_logit).cpu())
                scores = torch.stack(scores_list).mean(dim=0)
            else:
                out = model(images, training=False)
                cls_logit = out["cls_logit"].squeeze(1)       # (B,)
                scores    = torch.sigmoid(cls_logit)          # (B,) probabilities

            accumulator.update(scores, labels)

            # Store per-sample info (move back to CPU for storage)
            for path, score, label in zip(
                batch["path"],
                scores.cpu().tolist(),
                labels.cpu().tolist(),
            ):
                all_records.append({
                    "image_path":  path,
                    "true_label":  "real" if label == 0 else "replay",
                    "score":       round(score, 6),
                })

    # ------------------------------------------------------------------ #
    #  Metrics                                                             #
    # ------------------------------------------------------------------ #
    metrics = accumulator.compute()

    logger.info("── Test Results ──────────────────────────────────────────────")
    logger.info(f"  AUC           : {metrics['auc']:.4f}")
    logger.info(f"  EER           : {metrics['eer']:.4f}  ({metrics['eer'] * 100:.2f}%)")
    logger.info(f"  FAR @ EER     : {metrics['far_at_eer']:.4f}")
    logger.info(f"  FRR @ EER     : {metrics['frr_at_eer']:.4f}")
    logger.info(f"  HTER          : {metrics['hter']:.4f}  ({metrics['hter'] * 100:.2f}%)")
    logger.info(f"  EER threshold : {metrics['eer_threshold']:.4f}")
    logger.info("──────────────────────────────────────────────────────────────")

    # Write to CSV (reuse logger._write_csv directly)
    logger._write_csv(epoch=trained_epoch, phase="TEST", metrics=metrics)
    append_global_metrics(
        exp_root=os.path.dirname(exp_dir),
        phase="test",
        exp_name=os.path.basename(exp_dir),
        epoch=trained_epoch,
        metrics=metrics,
    )

    # ------------------------------------------------------------------ #
    #  ROC curve plot                                                       #
    # ------------------------------------------------------------------ #
    fpr, tpr, thresholds = accumulator.compute_roc()
    roc_path = os.path.join(plots_dir, "roc_curve.png")
    plot_roc_curve(
        fpr=fpr,
        tpr=tpr,
        auc=metrics["auc"],
        eer=metrics["eer"],
        eer_threshold=metrics["eer_threshold"],
        save_path=roc_path,
    )
    logger.info(f"ROC curve saved → {roc_path}")

    # ------------------------------------------------------------------ #
    #  Wrong predictions CSV                                               #
    # ------------------------------------------------------------------ #
    eer_threshold = metrics["eer_threshold"]

    wrong = []
    for record in all_records:
        score       = record["score"]
        true_label  = record["true_label"]
        pred_label  = "replay" if score >= eer_threshold else "real"

        if pred_label != true_label:
            wrong.append({
                "image_path":       record["image_path"],
                "true_label":       true_label,
                "predicted_label":  pred_label,
                "score":            score,
                "eer_threshold":    round(eer_threshold, 6),
                "error_type":       "FAR" if true_label == "real" else "FRR",
            })

    wrong_csv_path = os.path.join(exp_dir, "wrong_predictions.csv")
    fieldnames = ["image_path", "true_label", "predicted_label", "score", "eer_threshold", "error_type"]

    with open(wrong_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(wrong)

    logger.info(f"Wrong predictions : {len(wrong)} / {len(all_records)} saved → {wrong_csv_path}")
    logger.separator("=")


# --------------------------------------------------------------------------- #
#  Entry point                                                                  #
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained PAD model on the test set.")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Path to the experiment directory (e.g. experiments/2026-06-23_14-32-05).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best.pth",
        help="Checkpoint filename inside <experiment>/checkpoints/. Default: best.pth",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(exp_dir=args.experiment, checkpoint_name=args.checkpoint)
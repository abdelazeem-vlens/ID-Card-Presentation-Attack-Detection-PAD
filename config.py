"""
config.py
---------
Central configuration for the ID Card PAD (Presentation Attack Detection) project.
All tunable parameters live here. This file is snapshotted into the experiment
directory at the start of every run, ensuring full reproducibility.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import yaml
import os


@dataclass
class PathConfig:
    """All filesystem paths."""
    dataset_root: str = "dataset"               # Root dir containing train/ val/ test/
    experiments_root: str = "experiments"       # Where experiment dirs are created


@dataclass
class ModelConfig:
    """Model architecture settings."""
    # Which model class to instantiate from model.py
    # Options: "convnext_freq"
    model_name: str = "convnext_freq"

    # ConvNeXt variant: "convnext_tiny" | "convnext_small" | "convnext_base"
    backbone: str = "convnext_tiny"

    # Whether to load ImageNet-pretrained backbone weights
    pretrained: bool = True

    # Freeze backbone weights (only train heads) — useful for small datasets
    freeze_backbone: bool = False

    # Stage index (1–4) from which the auxiliary branch taps its feature map
    # Stage 1 → early low-level features (highest spatial resolution)
    # Stage 2 → mid-level features
    # Stage 3 → deeper features
    # Stage 4 → deepest features (lowest spatial resolution)
    aux_branch_stage: int = 2


@dataclass
class FrequencyConfig:
    """FFT frequency map computation settings."""
    # Spatial size (H, W) to resize the FFT map to — should match the spatial
    # resolution of the auxiliary branch's tap point in the backbone.
    # ConvNeXt-tiny stage output spatial sizes (for 224x224 input):
    #   Stage 1: 56x56, Stage 2: 28x28, Stage 3: 14x14, Stage 4: 7x7
    freq_map_size: tuple = (28, 28)

    # Whether to apply per-image min-max normalization to the FFT map
    normalize: bool = True


@dataclass
class DataConfig:
    """Dataset and dataloader settings."""
    # Input image size fed to the model (H, W)
    image_size: tuple = (224, 224)

    # ImageNet normalization stats (change if you recompute on your dataset)
    norm_mean: tuple = (0.485, 0.456, 0.406)
    norm_std: tuple = (0.229, 0.224, 0.225)

    # DataLoader settings
    batch_size: int = 32
    num_workers: int = 1
    pin_memory: bool = True

    # Use WeightedRandomSampler to handle class imbalance in training set
    use_weighted_sampler: bool = True

    # Whether to compute and return FFT frequency maps from the dataset
    compute_freq_map: bool = True


@dataclass
class TrainingConfig:
    """Training loop hyperparameters."""
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    # Loss combination weights: total = lambda_cls * L_bce + lambda_freq * L_mse
    lambda_cls: float = 1.0
    lambda_freq: float = 0.5

    # Gradient clipping max norm (None to disable)
    grad_clip: Optional[float] = 1.0

    # Early stopping: stop if val HTER does not improve for this many epochs
    early_stopping_patience: int = 15

    # Save a checkpoint for every epoch (in addition to best.pth and last.pth)
    save_every_epoch: bool = False

    # Log batch-level loss every N steps (0 = disable batch logging)
    log_every_n_steps: int = 20


@dataclass
class SchedulerConfig:
    """Learning rate scheduler settings."""
    # Options: "cosine" | "step" | "none"
    scheduler_type: str = "cosine"

    # Number of warmup epochs (linear warmup from lr * warmup_factor → lr)
    warmup_epochs: int = 3
    warmup_factor: float = 0.1

    # For StepLR only
    step_size: int = 10
    gamma: float = 0.5

    # For CosineAnnealingLR: minimum LR at the end of the cycle
    eta_min: float = 1e-6


@dataclass
class Config:
    """
    Master config — composes all sub-configs.

    Usage:
        cfg = Config()                        # all defaults
        cfg = Config.from_yaml("cfg.yaml")    # load from file
        cfg.save("experiments/xxx/config_snapshot.yaml")
    """
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    frequency: FrequencyConfig = field(default_factory=FrequencyConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    # ------------------------------------------------------------------ #
    #  Serialisation helpers                                               #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        """Snapshot the full config to a YAML file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Reconstruct a Config from a previously saved YAML snapshot."""
        with open(path, "r") as f:
            d = yaml.safe_load(f)

        cfg = cls()
        cfg.paths = PathConfig(**d.get("paths", {}))
        cfg.model = ModelConfig(**d.get("model", {}))
        cfg.frequency = FrequencyConfig(**d.get("frequency", {}))
        cfg.data = DataConfig(**d.get("data", {}))
        cfg.training = TrainingConfig(**d.get("training", {}))
        cfg.scheduler = SchedulerConfig(**d.get("scheduler", {}))
        return cfg

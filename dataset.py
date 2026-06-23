"""
dataset.py
----------
Dataset class and dataloader factory for the ID Card PAD project.

Expected directory layout:
    dataset/
    ├── train/
    │   ├── real/       (label = 0)
    │   └── replay/     (label = 1)
    ├── val/
    │   ├── real/
    │   └── replay/
    └── test/
        ├── real/
        └── replay/

Each __getitem__ returns a dict:
    {
        "image":    FloatTensor (3, H, W)  — normalized RGB
        "freq_map": FloatTensor (1, fH, fW) — log FFT magnitude map
                    (all zeros if compute_freq_map=False)
        "label":    int  — 0 = real, 1 = replay
        "path":     str  — absolute path to the source image file
    }
"""

import os
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

from config import Config
from utils.frequency import compute_freq_map

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Class name → integer label mapping
CLASS_TO_LABEL: Dict[str, int] = {"real": 0, "replay": 1}


# --------------------------------------------------------------------------- #
#  Dataset class                                                                #
# --------------------------------------------------------------------------- #

class IDCardDataset(Dataset):
    """
    PyTorch Dataset for ID card Presentation Attack Detection.

    Args:
        root_dir:         Path to the split directory (e.g. "dataset/train").
        transform:        torchvision transform applied to the PIL image.
        compute_freq_map: Whether to compute and return the FFT frequency map.
        freq_map_size:    (H, W) target size for the frequency map tensor.
        freq_normalize:   Whether to apply per-image min-max normalization to
                          the frequency map.
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        compute_freq_map: bool = True,
        freq_map_size: Tuple[int, int] = (28, 28),
        freq_normalize: bool = True,
    ) -> None:
        super().__init__()

        self.root_dir = os.path.abspath(root_dir)
        self.transform = transform
        self.compute_freq_map_flag = compute_freq_map
        self.freq_map_size = freq_map_size
        self.freq_normalize = freq_normalize

        # Discover all samples
        self.samples: List[Tuple[str, int]] = []   # (abs_path, label)
        self.class_counts: Dict[int, int] = {0: 0, 1: 0}

        self._discover_samples()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found under '{self.root_dir}'. "
                "Check that real/ and replay/ subdirectories exist and contain images."
            )

    # ------------------------------------------------------------------ #
    #  Core Dataset interface                                              #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        img_path, label = self.samples[idx]

        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image '{img_path}': {e}") from e

        # Compute frequency map BEFORE applying transforms, on the raw PIL image.
        # This ensures the FFT reflects the original image statistics, not the
        # augmented/normalized version.
        if self.compute_freq_map_flag:
            freq_map = compute_freq_map(
                image,
                output_size=self.freq_map_size,
                normalize=self.freq_normalize,
            )
        else:
            freq_map = torch.zeros(1, *self.freq_map_size)

        # Apply image transforms
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return {
            "image":    image,
            "freq_map": freq_map,
            "label":    label,
            "path":     img_path,
        }

    # ------------------------------------------------------------------ #
    #  Class balance helpers                                               #
    # ------------------------------------------------------------------ #

    def get_sample_weights(self) -> List[float]:
        """
        Return a per-sample weight list for use with WeightedRandomSampler.
        Each sample is weighted inversely to its class frequency, so each
        class contributes equally to every training epoch on average.
        """
        total = len(self.samples)
        class_weight = {
            cls: total / (count + 1e-8)
            for cls, count in self.class_counts.items()
        }
        return [class_weight[label] for _, label in self.samples]

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _discover_samples(self) -> None:
        """Walk real/ and replay/ subdirs and populate self.samples."""
        for class_name, label in CLASS_TO_LABEL.items():
            class_dir = os.path.join(self.root_dir, class_name)

            if not os.path.isdir(class_dir):
                raise RuntimeError(
                    f"Expected class directory not found: '{class_dir}'"
                )

            count = 0
            for fname in sorted(os.listdir(class_dir)):
                ext = os.path.splitext(fname)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    self.samples.append(
                        (os.path.join(class_dir, fname), label)
                    )
                    count += 1

            self.class_counts[label] = count

    def __repr__(self) -> str:
        return (
            f"IDCardDataset(root='{self.root_dir}', "
            f"real={self.class_counts[0]}, "
            f"replay={self.class_counts[1]}, "
            f"total={len(self.samples)})"
        )


# --------------------------------------------------------------------------- #
#  Transform factories                                                          #
# --------------------------------------------------------------------------- #

def build_train_transforms(cfg: Config) -> transforms.Compose:
    """
    Augmentation pipeline for training.

    Augmentations chosen specifically for screen replay attack detection:
      - ColorJitter: screen color profiles vary across devices/settings
      - RandomPerspective: mild capture angle variation
      - GaussianBlur: lens variation across capture devices
      - RandomHorizontalFlip: mild geometric invariance (IDs are not vertically symmetric)
      - JPEG-quality simulation via RandomAdjustSharpness: simulates compression artifacts
        NOTE: For true JPEG quality simulation, consider using the albumentations library.
    """
    h, w = cfg.data.image_size

    return transforms.Compose([
        transforms.Resize((int(h * 1.1), int(w * 1.1))),   # slight oversize for crop
        transforms.RandomCrop((h, w)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05,
        ),
        transforms.RandomPerspective(distortion_scale=0.05, p=0.3),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))],
            p=0.3,
        ),
        transforms.RandomApply(
            [transforms.RandomAdjustSharpness(sharpness_factor=0)],  # simulates blur/compression
            p=0.2,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.data.norm_mean, std=cfg.data.norm_std),
    ])


def build_eval_transforms(cfg: Config) -> transforms.Compose:
    """
    Deterministic pipeline for validation and test — no augmentation.
    """
    h, w = cfg.data.image_size

    return transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.data.norm_mean, std=cfg.data.norm_std),
    ])


# --------------------------------------------------------------------------- #
#  DataLoader factory                                                           #
# --------------------------------------------------------------------------- #

def get_dataloader(split: str, cfg: Config) -> DataLoader:
    """
    Build and return a DataLoader for the given dataset split.

    Args:
        split: One of "train", "val", "test".
        cfg:   Master Config object.

    Returns:
        torch.utils.data.DataLoader
    """
    assert split in ("train", "val", "test"), \
        f"split must be 'train', 'val', or 'test'. Got: '{split}'"

    root_dir = os.path.join(cfg.paths.dataset_root, split)

    # Select transform pipeline
    if split == "train":
        transform = build_train_transforms(cfg)
    else:
        transform = build_eval_transforms(cfg)

    dataset = IDCardDataset(
        root_dir=root_dir,
        transform=transform,
        compute_freq_map=cfg.data.compute_freq_map,
        freq_map_size=cfg.frequency.freq_map_size,
        freq_normalize=cfg.frequency.normalize,
    )

    # WeightedRandomSampler only for training
    sampler = None
    shuffle = False

    if split == "train":
        if cfg.data.use_weighted_sampler:
            weights = dataset.get_sample_weights()
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True,
            )
        else:
            shuffle = True   # shuffle manually if not using sampler

    return DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=(split == "train" and len(dataset) >= cfg.data.batch_size),
        persistent_workers=(cfg.data.num_workers > 0),
    )

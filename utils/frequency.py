"""
utils/frequency.py
------------------
Computes the FFT-based frequency supervision map used as the regression
target for the auxiliary branch.

Pipeline:
  RGB image → Y (luminance) channel → 2D FFT → log-magnitude spectrum
  → per-image min-max normalization → resize to target spatial size
  → single-channel float tensor in [0, 1]

Why FFT on Y channel only:
  - Screen replay attacks introduce moiré patterns and pixel-grid regularity
    that appear as sharp periodic spikes in the frequency domain.
  - The luminance channel captures these structural artifacts most clearly,
    without the chrominance noise that would dilute the signal.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Tuple, Union


def compute_freq_map(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    output_size: Tuple[int, int] = (28, 28),
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute a log-scaled FFT magnitude map from an input image.

    Args:
        image:       PIL Image (RGB), numpy array (H, W, 3) uint8,
                     or float torch Tensor (3, H, W) in [0, 1].
        output_size: (H, W) to resize the final map to. Should match the
                     spatial resolution of the auxiliary branch tap point.
        normalize:   If True, apply per-image min-max normalization to [0, 1].

    Returns:
        Tensor of shape (1, H, W) with dtype float32, values in [0, 1].
    """
    # ------------------------------------------------------------------ #
    #  Step 1: Convert input to numpy float32 luminance (Y) channel       #
    # ------------------------------------------------------------------ #
    if isinstance(image, torch.Tensor):
        # Expect shape (3, H, W), values in [0, 1]
        img_np = image.permute(1, 2, 0).cpu().numpy()           # (H, W, 3)
        img_np = (img_np * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np, mode="RGB")
    elif isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image.astype(np.uint8), mode="RGB")
    elif isinstance(image, Image.Image):
        pil_img = image.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # Convert RGB → YCbCr and extract Y (luminance) channel
    ycbcr = pil_img.convert("YCbCr")
    y_channel = np.array(ycbcr)[:, :, 0].astype(np.float32)    # (H, W)

    # ------------------------------------------------------------------ #
    #  Step 2: 2D FFT → magnitude spectrum → log scaling                  #
    # ------------------------------------------------------------------ #
    fft = np.fft.fft2(y_channel)                 # complex (H, W)
    fft_shifted = np.fft.fftshift(fft)           # zero-freq component at center
    magnitude = np.abs(fft_shifted)              # magnitude spectrum (H, W)
    log_magnitude = np.log1p(magnitude)          # log(1 + |F|) — compress range

    # ------------------------------------------------------------------ #
    #  Step 3: Per-image min-max normalization to [0, 1]                  #
    # ------------------------------------------------------------------ #
    if normalize:
        min_val = log_magnitude.min()
        max_val = log_magnitude.max()
        denom = max_val - min_val
        if denom > 1e-8:
            log_magnitude = (log_magnitude - min_val) / denom
        else:
            log_magnitude = np.zeros_like(log_magnitude)

    # ------------------------------------------------------------------ #
    #  Step 4: Convert to tensor and resize to output_size                #
    # ------------------------------------------------------------------ #
    freq_tensor = torch.from_numpy(log_magnitude).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    freq_tensor = F.interpolate(
        freq_tensor,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )                                                                         # (1, 1, oH, oW)
    freq_tensor = freq_tensor.squeeze(0)                                      # (1, oH, oW)

    return freq_tensor.float()


def batch_compute_freq_maps(
    images: torch.Tensor,
    output_size: Tuple[int, int] = (28, 28),
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute frequency maps for a batch of images.

    Args:
        images:      Tensor of shape (B, 3, H, W), values in [0, 1].
        output_size: Target (H, W) for each map.
        normalize:   Per-image min-max normalization flag.

    Returns:
        Tensor of shape (B, 1, oH, oW).
    """
    maps = [
        compute_freq_map(images[i], output_size=output_size, normalize=normalize)
        for i in range(images.shape[0])
    ]
    return torch.stack(maps, dim=0)   # (B, 1, oH, oW)

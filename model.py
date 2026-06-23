"""
model.py
--------
Model definitions for the ID Card PAD project.

Classes:
  AuxFrequencyBranch     — small CNN head that regresses an FFT magnitude map
  ConvNeXtWithFreqBranch — ConvNeXt backbone + classification head +
                           auxiliary frequency branch (MinVision-inspired)

The auxiliary branch taps a feature map from a configurable ConvNeXt stage
via a forward hook and regresses it to the ground-truth FFT map using MSE loss.
The hook-based design keeps the backbone's forward() method untouched.

Loss is NOT computed inside the model — the model returns raw logits and
frequency predictions; the training loop combines them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    convnext_tiny, convnext_small, convnext_base,
    ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights, ConvNeXt_Base_Weights,
)
from typing import Dict, Optional, Tuple


# --------------------------------------------------------------------------- #
#  Backbone registry                                                            #
# --------------------------------------------------------------------------- #

_CONVNEXT_REGISTRY = {
    "convnext_tiny": (
        convnext_tiny,
        ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
        # Channel dims after each of the 4 stages
        [96, 192, 384, 768],
    ),
    "convnext_small": (
        convnext_small,
        ConvNeXt_Small_Weights.IMAGENET1K_V1,
        [96, 192, 384, 768],
    ),
    "convnext_base": (
        convnext_base,
        ConvNeXt_Base_Weights.IMAGENET1K_V1,
        [128, 256, 512, 1024],
    ),
}


# --------------------------------------------------------------------------- #
#  Auxiliary Frequency Branch                                                   #
# --------------------------------------------------------------------------- #

class AuxFrequencyBranch(nn.Module):
    """
    A lightweight convolutional branch that takes a ConvNeXt intermediate
    feature map and regresses it to the ground-truth FFT magnitude map.

    Architecture:
        1x1 conv (channel reduction → 64)
        → 3x3 depthwise conv (spatial processing, no cross-channel mixing)
        → 3x3 depthwise conv
        → 1x1 conv (64 → 1, output single-channel frequency map)
        → bilinear upsample to freq_map_size
        → sigmoid (output in [0, 1] to match normalized FFT target)

    Args:
        in_channels:   Number of input channels from the tapped feature map.
        freq_map_size: (H, W) of the ground-truth FFT map to match.
        hidden_dim:    Intermediate channel count inside the branch.
    """

    def __init__(
        self,
        in_channels: int,
        freq_map_size: Tuple[int, int] = (28, 28),
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        self.freq_map_size = freq_map_size

        self.branch = nn.Sequential(
            # Channel compression
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),

            # Spatial feature processing — depthwise separable convs
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1,
                      groups=hidden_dim, bias=False),          # depthwise
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),  # pointwise
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),

            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1,
                      groups=hidden_dim, bias=False),          # depthwise
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),  # pointwise
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),

            # Output projection to single channel
            nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=True),
        )

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_map: Tensor (B, C, H, W) from a ConvNeXt stage.

        Returns:
            freq_pred: Tensor (B, 1, fH, fW) with values in [0, 1].
        """
        x = self.branch(feature_map)   # (B, 1, h, w)

        # Upsample to match the ground-truth FFT map spatial size
        x = F.interpolate(
            x,
            size=self.freq_map_size,
            mode="bilinear",
            align_corners=False,
        )
        return torch.sigmoid(x)        # (B, 1, fH, fW)


# --------------------------------------------------------------------------- #
#  Main model                                                                   #
# --------------------------------------------------------------------------- #

class ConvNeXtWithFreqBranch(nn.Module):
    """
    ConvNeXt backbone enhanced with a MinVision-inspired auxiliary frequency branch.

    Forward pass returns a dict:
        Training:  {"cls_logit": Tensor(B,1), "freq_pred": Tensor(B,1,fH,fW)}
        Inference: {"cls_logit": Tensor(B,1)}

    The classification logit is raw (pre-sigmoid). Apply torch.sigmoid() to
    get a probability, or use BCEWithLogitsLoss directly in the training loop.

    Args:
        backbone_name:   One of "convnext_tiny", "convnext_small", "convnext_base".
        pretrained:      Load ImageNet-pretrained weights for the backbone.
        freeze_backbone: If True, freeze all backbone parameters (train heads only).
        aux_stage:       Stage index (1–4) to tap for the auxiliary branch.
                         Stage 1 = earliest (most spatial detail, fewer channels).
                         Stage 4 = latest (most semantic, least spatial).
        freq_map_size:   (H, W) of the FFT target map (must match dataset output).
        aux_hidden_dim:  Hidden channel width inside the auxiliary branch.
    """

    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        aux_stage: int = 2,
        freq_map_size: Tuple[int, int] = (28, 28),
        aux_hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        assert backbone_name in _CONVNEXT_REGISTRY, (
            f"Unknown backbone '{backbone_name}'. "
            f"Choose from: {list(_CONVNEXT_REGISTRY.keys())}"
        )
        assert 1 <= aux_stage <= 4, "aux_stage must be between 1 and 4."

        factory, weights, stage_channels = _CONVNEXT_REGISTRY[backbone_name]
        self.aux_stage = aux_stage
        self.freq_map_size = freq_map_size

        # ---------------------------------------------------------------- #
        #  Backbone                                                          #
        # ---------------------------------------------------------------- #
        if pretrained:
            backbone = factory(weights=weights)
        else:
            backbone = factory(weights=None)

        # ConvNeXt from torchvision has the structure:
        #   backbone.features[0]   → stem (downsampling patch embed)
        #   backbone.features[1]   → stage 1
        #   backbone.features[2]   → downsampling layer
        #   backbone.features[3]   → stage 2
        #   backbone.features[4]   → downsampling layer
        #   backbone.features[5]   → stage 3
        #   backbone.features[6]   → downsampling layer
        #   backbone.features[7]   → stage 4
        #
        # Stage N maps to features index:  stage_to_feat_idx[N]
        self._stage_to_feat_idx = {1: 1, 2: 3, 3: 5, 4: 7}
        tap_feat_idx = self._stage_to_feat_idx[aux_stage]

        # Keep only the feature extractor; discard the classifier head
        self.backbone_features = backbone.features   # nn.Sequential of 8 modules
        self.backbone_norm = backbone.avgpool        # adaptive avg pool
        # Final LayerNorm + flatten are inside the original classifier; replicate:
        final_channels = stage_channels[-1]
        self.backbone_final_norm = nn.LayerNorm(final_channels, eps=1e-6)

        if freeze_backbone:
            for param in self.backbone_features.parameters():
                param.requires_grad = False

        # ---------------------------------------------------------------- #
        #  Classification head                                              #
        # ---------------------------------------------------------------- #
        self.cls_head = nn.Sequential(
            nn.Linear(final_channels, 256),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1),   # raw logit; apply sigmoid externally
        )

        # ---------------------------------------------------------------- #
        #  Auxiliary frequency branch                                       #
        # ---------------------------------------------------------------- #
        tap_channels = stage_channels[aux_stage - 1]
        self.aux_branch = AuxFrequencyBranch(
            in_channels=tap_channels,
            freq_map_size=freq_map_size,
            hidden_dim=aux_hidden_dim,
        )

        # ---------------------------------------------------------------- #
        #  Forward hook to capture the intermediate feature map            #
        # ---------------------------------------------------------------- #
        self._aux_feature: Optional[torch.Tensor] = None
        self._hook_handle = self.backbone_features[tap_feat_idx].register_forward_hook(
            self._capture_aux_feature
        )

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor, training: bool = True) -> Dict[str, torch.Tensor]:
        """
        Args:
            x:        Input image tensor (B, 3, H, W).
            training: If True, also run the auxiliary branch and return freq_pred.
                      Set to False during inference to skip the aux branch.

        Returns:
            dict with "cls_logit" always present, "freq_pred" only when training=True.
        """
        # Reset captured feature
        self._aux_feature = None

        # Full forward through backbone (hook fires automatically at tap stage)
        features = self.backbone_features(x)         # (B, C_final, H', W')
        pooled = self.backbone_norm(features)        # (B, C_final, 1, 1)
        pooled = pooled.flatten(1)                   # (B, C_final)
        pooled = self.backbone_final_norm(pooled)    # (B, C_final)

        cls_logit = self.cls_head(pooled)            # (B, 1)

        result = {"cls_logit": cls_logit}

        if training and self._aux_feature is not None:
            freq_pred = self.aux_branch(self._aux_feature)   # (B, 1, fH, fW)
            result["freq_pred"] = freq_pred

        return result

    # ------------------------------------------------------------------ #
    #  Hook                                                                #
    # ------------------------------------------------------------------ #

    def _capture_aux_feature(
        self,
        module: nn.Module,
        input: tuple,
        output: torch.Tensor,
    ) -> None:
        """Forward hook: stores the stage output for the auxiliary branch."""
        self._aux_feature = output

    # ------------------------------------------------------------------ #
    #  Cleanup                                                             #
    # ------------------------------------------------------------------ #

    def remove_hook(self) -> None:
        """Remove the forward hook (call before saving if desired)."""
        self._hook_handle.remove()

    def __del__(self) -> None:
        try:
            self._hook_handle.remove()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
#  Model factory                                                                #
# --------------------------------------------------------------------------- #

def build_model(cfg) -> nn.Module:
    """
    Instantiate the model specified in cfg.model.

    Args:
        cfg: Master Config object.

    Returns:
        An nn.Module ready for training.
    """
    if cfg.model.model_name == "convnext_freq":
        return ConvNeXtWithFreqBranch(
            backbone_name=cfg.model.backbone,
            pretrained=cfg.model.pretrained,
            freeze_backbone=cfg.model.freeze_backbone,
            aux_stage=cfg.model.aux_branch_stage,
            freq_map_size=cfg.frequency.freq_map_size,
        )

    raise ValueError(
        f"Unknown model name: '{cfg.model.model_name}'. "
        "Add its class to model.py and register it in build_model()."
    )

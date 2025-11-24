

#!/usr/bin/env python
"""
Transform utilities for simple label-preserving augmentations.

Currently implemented:
    - RandomPairPermutation: randomly swap (i, j) → (j, i) with probability 0.5
    - RandomPairSubsampling: randomly drops a subset of pairs
    - CoVggtAug: convenience wrapper that bundles pair-aware and per-image augmentations

These augmentations work for both:
    - Pairwise mode: (images shape (2,3,H,W), label scalar)
    - Multiview mode: pairs tensor shape (P,2)

The transforms do NOT modify labels, because co-visibility(i,j) == co-visibility(j,i).
"""

import torch
import random
from typing import Dict, Any, Optional

from torchvision import transforms as T


class RandomPairPermutation:
    """
    Randomly permutes pair order (i, j) → (j, i) with p=0.5.
    Works for both pairwise and multiview datasets.

    Usage:
        transform = RandomPairPermutation()
        sample = transform(sample)

    The sample dict must contain:
        - "images" (pairwise): Tensor (2,3,H,W)
        - "label" or "labels" remain unchanged
        - "pairs" (multiview): Tensor (P,2)

    The transform inspects the keys and applies permutation appropriately.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # ----- Pairwise case -----
        if "images" in sample and sample["images"].dim() == 4 and sample["images"].shape[0] == 2:
            if random.random() < self.p:
                # Swap the two images
                img_i, img_j = sample["images"][0], sample["images"][1]
                sample["images"] = torch.stack([img_j, img_i], dim=0)
                # label unchanged
            return sample

        # ----- Multiview case -----
        if "pairs" in sample and isinstance(sample["pairs"], torch.Tensor):
            pairs = sample["pairs"]
            if pairs.dim() == 2 and pairs.shape[1] == 2:
                # For each pair independently
                mask = torch.rand(pairs.shape[0]) < self.p  # boolean mask
                # Swap indices for masked pairs
                swapped = pairs.clone()
                swapped[mask, 0] = pairs[mask, 1]
                swapped[mask, 1] = pairs[mask, 0]
                sample["pairs"] = swapped
            return sample

        # If neither structure matches, return unchanged
        return sample


# -------------------- RandomPairSubsampling --------------------
class RandomPairSubsampling:
    """
    Randomly subsamples pairs in both pairwise and multiview settings.
    
    For pairwise:
        - With keep_ratio < 1, the single pair may be dropped with probability (1 - keep_ratio).
        - If dropped, the sample is marked as empty by setting "pairs" or "images" accordingly.
          The caller must handle empty cases (as already done in multiview loops).

    For multiview:
        - Given pairs shape (P,2), keeps only a random subset of size floor(P * keep_ratio).
        - Labels and strengths are subsampled in the same order.

    The transform expects a dict containing:
        - Pairwise:  {"images": (2,3,H,W), "label": scalar, "strength": scalar}
        - Multiview: {"pairs": (P,2), "labels": (P,), "strengths": (P,)}

    NOTE:
        keep_ratio = 1.0 -> no subsampling.
        keep_ratio = 0.0 -> drop all pairs.
    """

    def __init__(self, keep_ratio: float = 1.0):
        assert 0.0 <= keep_ratio <= 1.0, "keep_ratio must be in [0,1]"
        self.keep_ratio = keep_ratio

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # ---- Pairwise case ----
        if "images" in sample and "label" in sample and sample["images"].dim() == 4 and sample["images"].shape[0] == 2:
            if random.random() > self.keep_ratio:
                # Drop the single pair
                sample["images"] = torch.empty(0)   # mark as empty
                sample["label"] = torch.empty(0)
                sample["strength"] = torch.empty(0)
            return sample

        # ---- Multiview case ----
        if "pairs" in sample and isinstance(sample["pairs"], torch.Tensor):
            pairs = sample["pairs"]
            if pairs.dim() == 2 and pairs.shape[1] == 2:
                P = pairs.shape[0]
                if P == 0 or self.keep_ratio >= 1.0:
                    return sample

                k = int(P * self.keep_ratio)
                if k <= 0:
                    # Drop all pairs
                    sample["pairs"] = torch.empty((0, 2), dtype=pairs.dtype)
                    if "labels" in sample:
                        sample["labels"] = torch.empty((0,), dtype=sample["labels"].dtype)
                    if "strengths" in sample:
                        sample["strengths"] = torch.empty((0,), dtype=sample["strengths"].dtype)
                    return sample

                # Sample a subset of k pairs
                idx = torch.randperm(P)[:k]

                sample["pairs"] = pairs[idx]
                if "labels" in sample:
                    sample["labels"] = sample["labels"][idx]
                if "strengths" in sample:
                    sample["strengths"] = sample["strengths"][idx]

            return sample

        # Otherwise, do nothing
        return sample


class CoVggtAug:
    """
    Bundle pair-aware and per-image augmentations used by the VGGT trainer.

    Operations:
        - Pair permutation (swap (i, j) -> (j, i))
        - Pair subsampling (multiview only)
        - Per-image color jitter, horizontal flip, and Gaussian noise
    """

    def __init__(
        self,
        pair_permutation_p: float = 0.0,
        pair_keep_ratio: float = 1.0,
        hflip_p: float = 0.0,
        color_jitter: float = 0.0,
        gaussian_noise_std: float = 0.0,
    ):
        self.pair_permutation = (
            RandomPairPermutation(p=pair_permutation_p) if pair_permutation_p > 0.0 else None
        )
        self.subsample_pairs = (
            RandomPairSubsampling(keep_ratio=pair_keep_ratio) if pair_keep_ratio < 1.0 else None
        )

        self.hflip_p = max(0.0, float(hflip_p))
        self.gaussian_noise_std = max(0.0, float(gaussian_noise_std))
        self.color_jitter = (
            T.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
                hue=min(0.5, color_jitter / 2),
            )
            if color_jitter > 0.0
            else None
        )

        self._has_image_ops = (
            self.hflip_p > 0.0 or self.color_jitter is not None or self.gaussian_noise_std > 0.0
        )
        self._has_pair_ops = self.pair_permutation is not None or self.subsample_pairs is not None
        self.is_noop = not (self._has_image_ops or self._has_pair_ops)

    def _augment_single_image(self, img: torch.Tensor) -> torch.Tensor:
        out = img
        if self.hflip_p > 0.0 and random.random() < self.hflip_p:
            out = torch.flip(out, dims=[2])
        if self.color_jitter is not None:
            out = self.color_jitter(out)
        if self.gaussian_noise_std > 0.0:
            noise = torch.randn_like(out) * self.gaussian_noise_std
            out = torch.clamp(out + noise, 0.0, 1.0)
        return out

    def _augment_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.numel() == 0 or images.shape[0] == 0:
            return images
        augmented = [self._augment_single_image(img) for img in images]
        return torch.stack(augmented, dim=0)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if self.is_noop:
            return sample

        out = dict(sample)
        if self._has_image_ops and "images" in out and isinstance(out["images"], torch.Tensor):
            out["images"] = self._augment_images(out["images"])

        if self.pair_permutation is not None:
            out = self.pair_permutation(out)

        # Subsampling is applied only when explicit pair indices are present (multiview/precomputed).
        if self.subsample_pairs is not None and "pairs" in out:
            out = self.subsample_pairs(out)

        return out

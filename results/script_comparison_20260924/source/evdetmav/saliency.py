"""Stage 1: density-aware saliency map from positive/negative event intersections."""

from __future__ import annotations

import argparse

import numpy as np
from scipy import ndimage


def binary_image_for_events(x: np.ndarray, y: np.ndarray, height: int, width: int) -> np.ndarray:
    image = np.zeros(height * width, dtype=bool)
    if x.size:
        image[y * width + x] = True
    return image.reshape(height, width)


def build_density_saliency(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    start_t: float,
    end_t: float,
    height: int,
    width: int,
    args: argparse.Namespace,
) -> np.ndarray:
    n_slices = max(int(args.saliency_slices), 1)
    duration = max(end_t - start_t, 1e-12)
    slice_ids = np.floor((t - start_t) / duration * n_slices).astype(np.int64)
    slice_ids = np.clip(slice_ids, 0, n_slices - 1)
    saliency = np.zeros((height, width), dtype=np.float32)
    structure = np.ones((3, 3), dtype=bool)

    for slice_id in range(n_slices):
        idx = slice_ids == slice_id
        if not np.any(idx):
            continue
        pos = idx & (p > 0)
        neg = idx & (p < 0)
        if not np.any(pos) or not np.any(neg):
            continue
        pos_img = binary_image_for_events(x[pos], y[pos], height, width)
        neg_img = binary_image_for_events(x[neg], y[neg], height, width)
        if args.intersection_radius > 0:
            for _ in range(int(args.intersection_radius)):
                pos_img = ndimage.binary_dilation(pos_img, structure=structure)
                neg_img = ndimage.binary_dilation(neg_img, structure=structure)
        saliency += (pos_img & neg_img).astype(np.float32)

    saliency = saliency * (255.0 / float(n_slices))
    if args.saliency_sigma > 0:
        saliency = ndimage.gaussian_filter(saliency, sigma=float(args.saliency_sigma), mode="nearest")
    return np.clip(saliency, 0.0, 255.0).astype(np.float32)

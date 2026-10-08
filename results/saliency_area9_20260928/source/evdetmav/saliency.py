"""Stage 1: density-aware saliency map from positive/negative event intersections."""

from __future__ import annotations

import argparse

import numpy as np
from scipy import ndimage


def remove_small_saliency_regions(
    saliency: np.ndarray, threshold: float, min_area: int,
) -> tuple[np.ndarray, dict]:
    """在候选膨胀/合并前按 8 邻域真实前景像素数过滤显著图。

    min_area=0 原样返回，用于复现旧版；正数启用显著性阈值与面积过滤。
    面积恰好等于 min_area 的区域保留。保留像素的显著值不重新归一化。
    本函数不删除原事件；周期性仍在原始事件中计算。
    """
    if min_area < 0:
        raise ValueError('saliency min_area must be nonnegative')
    if min_area == 0:
        return saliency, {'enabled': False, 'min_area_px': 0}
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('enabled saliency filtering requires a positive finite threshold')
    # foreground: 超过显著性阈值的有效像素；低于阈值的背景也不参与评分。
    foreground = saliency >= threshold
    labels, count = ndimage.label(foreground, structure=np.ones((3, 3), dtype=bool))
    # areas[label_id]: 实际像素数，不是外接框面积，也不是膨胀后的面积。
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    keep = areas >= min_area
    keep[0] = False  # 0 为背景标签，永不保留。
    filtered = np.where(keep[labels], saliency, 0).astype(saliency.dtype, copy=False)
    removed = (areas[1:] < min_area)
    return filtered, {
        'enabled': True, 'min_area_px': int(min_area), 'threshold': float(threshold),
        'connectivity': 8, 'components_before': int(count),
        'components_removed': int(removed.sum()), 'components_kept': int(keep[1:].sum()),
        'foreground_pixels_before': int(foreground.sum()),
        'foreground_pixels_removed': int(areas[1:][removed].sum()),
        'foreground_pixels_kept': int(areas[1:][~removed].sum()),
        'subthreshold_pixels_cleared': int(np.count_nonzero((saliency > 0) & ~foreground)),
    }


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

"""Stage 3: clustering-based coarse-to-fine propeller/MAV candidate extraction."""

from __future__ import annotations

import argparse

import numpy as np
from scipy import ndimage

from .models import Box, Candidate, RefinedCandidate, box_area, pad_box, rectangle_distance, union_boxes
from .saliency import remove_small_saliency_regions


def connected_component_boxes(mask: np.ndarray, min_area: int) -> list[tuple[Box, int]]:
    labels, count = ndimage.label(mask.astype(bool), structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return []
    objects = ndimage.find_objects(labels)
    boxes = []
    for label_id, sl in enumerate(objects, start=1):
        if sl is None:
            continue
        ys, xs = sl
        local = labels[sl] == label_id
        area = int(np.count_nonzero(local))
        if area < min_area:
            continue
        boxes.append(((xs.start, ys.start, xs.stop, ys.stop), area))
    return boxes


def merge_close_boxes(boxes: list[tuple[Box, int]], gap_px: float) -> list[tuple[Box, int]]:
    merged = [(box, area) for box, area in boxes]
    changed = True
    while changed:
        changed = False
        next_items: list[tuple[Box, int]] = []
        consumed = [False] * len(merged)
        for i, (box_i, area_i) in enumerate(merged):
            if consumed[i]:
                continue
            box = box_i
            area = area_i
            consumed[i] = True
            for j in range(i + 1, len(merged)):
                if consumed[j]:
                    continue
                box_j, area_j = merged[j]
                if rectangle_distance(box, box_j) <= gap_px:
                    box = union_boxes([box, box_j])
                    area += area_j
                    consumed[j] = True
                    changed = True
            next_items.append((box, area))
        merged = next_items
    return merged


def initialize_candidates(saliency_u8: np.ndarray, args: argparse.Namespace) -> list[Candidate]:
    height, width = saliency_u8.shape
    mask = saliency_u8 >= float(args.tau_s)
    if args.init_dilate_px > 0:
        structure = np.ones((3, 3), dtype=bool)
        for _ in range(int(args.init_dilate_px)):
            mask = ndimage.binary_dilation(mask, structure=structure)
    boxes = connected_component_boxes(mask, int(args.min_component_area))
    boxes = merge_close_boxes(boxes, float(args.cluster_gap_px))

    candidates = []
    for box, _ in boxes:
        box = pad_box(box, int(args.candidate_pad_px), width, height)
        if box_area(box) <= 0:
            continue
        crop_mask = mask[box[1] : box[3], box[0] : box[2]]
        area = int(np.count_nonzero(crop_mask))
        score = float(np.sum(saliency_u8[box[1] : box[3], box[0] : box[2]]))
        candidates.append(Candidate(box=box, area_px=area, saliency_score=score))

    candidates.sort(key=lambda c: (c.saliency_score, c.area_px), reverse=True)
    for rank, candidate in enumerate(candidates, start=1):
        candidate.rank = rank
    return candidates


def gaussian_consistency(values: np.ndarray, mask: np.ndarray, min_pixels: int) -> float:
    active = mask & (values > 0)
    ys, xs = np.nonzero(active)
    if ys.size < min_pixels:
        return 0.0
    weights = values[ys, xs].astype(np.float64)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-12:
        return 0.0
    points = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    mean = np.sum(points * weights[:, None], axis=0) / weight_sum
    centered = points - mean[None, :]
    cov = (centered * weights[:, None]).T @ centered / weight_sum
    cov += np.eye(2) * 1e-3
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return 0.0
    exponent = np.einsum("ij,jk,ik->i", centered, inv_cov, centered)
    pred = np.exp(-0.5 * np.clip(exponent, 0.0, 80.0))
    obs = weights / max(float(np.linalg.norm(weights)), 1e-12)
    pred = pred / max(float(np.linalg.norm(pred)), 1e-12)
    return float(np.clip(np.dot(obs, pred), 0.0, 1.0))


def refine_candidate(saliency_u8: np.ndarray, candidate: Candidate, args: argparse.Namespace) -> Optional[RefinedCandidate]:
    height, width = saliency_u8.shape
    x0, y0, x1, y1 = candidate.box
    crop = saliency_u8[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    fine_threshold = float(args.fine_threshold) if args.fine_threshold > 0 else float(args.tau_s)
    mask = crop >= fine_threshold
    # 裁剪可能把邻近大区域切成小碎片；同一面积门限再检查，避免精细阶段回退带回碎片。
    area_limit = int(getattr(args, 'saliency_min_area', 0))
    if area_limit > 0:
        filtered_crop, _ = remove_small_saliency_regions(crop, fine_threshold, area_limit)
        mask = filtered_crop > 0
    components = connected_component_boxes(mask, int(args.fine_min_area))
    kept_masks = []
    kept_scores = []
    kept_boxes: list[Box] = []
    for local_box, _ in components:
        lx0, ly0, lx1, ly1 = local_box
        local_mask = np.zeros_like(mask, dtype=bool)
        local_mask[ly0:ly1, lx0:lx1] = mask[ly0:ly1, lx0:lx1]
        score = gaussian_consistency(crop, local_mask, int(args.fine_min_area))
        if score >= float(args.gaussian_consistency_threshold):
            kept_masks.append(local_mask)
            kept_scores.append(score)
            kept_boxes.append((x0 + lx0, y0 + ly0, x0 + lx1, y0 + ly1))

    if not kept_masks and bool(args.fine_fallback_to_coarse):
        score = gaussian_consistency(crop, mask, int(args.fine_min_area))
        if np.count_nonzero(mask) > 0:
            kept_masks = [mask]
            kept_scores = [score]
            kept_boxes = [candidate.box]

    if not kept_masks:
        return None

    combined = np.zeros_like(mask, dtype=bool)
    for kept in kept_masks:
        combined |= kept
    refined_box = pad_box(union_boxes(kept_boxes), int(args.fine_box_pad_px), width, height)
    return RefinedCandidate(
        candidate=candidate,
        box=refined_box,
        mask=combined,
        offset_x=x0,
        offset_y=y0,
        segmentation_area_px=int(np.count_nonzero(combined)),
        gaussian_consistency=float(np.mean(kept_scores)) if kept_scores else 0.0,
    )

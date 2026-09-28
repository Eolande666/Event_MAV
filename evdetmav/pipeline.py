"""Three-stage detection pipeline: saliency -> periodicity -> clustering, per window."""

from __future__ import annotations

import argparse
import time

import numpy as np

from .models import (
    Box,
    Detection,
    RefinedCandidate,
    WindowResult,
    box_area,
    clipped_box,
    events_in_box,
    union_boxes,
)
from .saliency import build_density_saliency, remove_small_saliency_regions
from .periodicity import evaluate_periodicity
from .clustering import initialize_candidates, refine_candidate


def make_detection(
    source_file: str,
    window_id: int,
    start_t: float,
    end_t: float,
    label: str,
    box: Box,
    segmentation_area: int,
    event_count: int,
    refined: list[RefinedCandidate],
    latency_ms: float,
) -> Detection:
    periodicities = [r.candidate.periodicity for r in refined if r.candidate.periodicity is not None]
    saliency_score = float(sum(r.candidate.saliency_score for r in refined))
    periodicity_score = int(max((p.score for p in periodicities), default=0))
    gaussian = float(np.mean([r.gaussian_consistency for r in refined])) if refined else 0.0
    x0, y0, x1, y1 = box
    return Detection(
        source_file=source_file,
        window_id=window_id,
        t_start_sec=float(start_t),
        t_end_sec=float(end_t),
        label=label,
        bbox_x0=x0,
        bbox_y0=y0,
        bbox_x1=x1,
        bbox_y1=y1,
        bbox_w=max(0, x1 - x0),
        bbox_h=max(0, y1 - y0),
        bbox_area_px=box_area(box),
        segmentation_area_px=int(segmentation_area),
        event_count=int(event_count),
        saliency_score=saliency_score,
        periodicity_score=periodicity_score,
        density_has_peak=int(any(p.density_has_peak for p in periodicities)),
        density_has_valley=int(any(p.density_has_valley for p in periodicities)),
        structure_has_peak=int(any(p.structure_has_peak for p in periodicities)),
        structure_has_valley=int(any(p.structure_has_valley for p in periodicities)),
        direction_has_peak=int(any(p.direction_has_peak for p in periodicities)),
        direction_has_valley=int(any(p.direction_has_valley for p in periodicities)),
        gaussian_consistency=gaussian,
        propeller_component_count=len(refined),
        latency_ms=float(latency_ms),
    )


def process_window(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    start_t: float,
    end_t: float,
    height: int,
    width: int,
    source_file: str,
    window_id: int,
    args: argparse.Namespace,
) -> WindowResult:
    tic = time.perf_counter()
    saliency_u8 = build_density_saliency(x, y, t, p, start_t, end_t, height, width, args)
    raw_saliency_u8 = saliency_u8
    saliency_u8, filter_stats = remove_small_saliency_regions(
        saliency_u8, float(args.tau_s), int(getattr(args, 'saliency_min_area', 0)))
    candidates = initialize_candidates(saliency_u8, args)
    top_candidates = candidates[: max(int(args.top_k), 1)]

    accepted: list[RefinedCandidate] = []
    for candidate in top_candidates:
        periodicity = evaluate_periodicity(x, y, t, p, candidate, start_t, end_t, args)
        if periodicity.score < int(args.tau_p):
            continue
        refined = refine_candidate(saliency_u8, candidate, args)
        if refined is not None:
            accepted.append(refined)

    segmentation_mask = np.zeros((height, width), dtype=bool)
    for item in accepted:
        x0, y0 = item.offset_x, item.offset_y
        y1 = min(height, y0 + item.mask.shape[0])
        x1 = min(width, x0 + item.mask.shape[1])
        segmentation_mask[y0:y1, x0:x1] |= item.mask[: y1 - y0, : x1 - x0]

    detections: list[Detection] = []
    latency_ms = (time.perf_counter() - tic) * 1000.0
    if accepted and bool(args.merge_propellers):
        box = union_boxes([item.box for item in accepted])
        box = clipped_box(box, width, height)
        event_count = int(np.count_nonzero(events_in_box(x, y, box)))
        segmentation_area = int(np.count_nonzero(segmentation_mask[box[1] : box[3], box[0] : box[2]]))
        detections.append(make_detection(source_file, window_id, start_t, end_t, "mav", box, segmentation_area, event_count, accepted, latency_ms))
    elif accepted:
        for idx, item in enumerate(accepted, start=1):
            box = clipped_box(item.box, width, height)
            event_count = int(np.count_nonzero(events_in_box(x, y, box)))
            detections.append(make_detection(source_file, window_id, start_t, end_t, f"propeller_{idx}", box, item.segmentation_area_px, event_count, [item], latency_ms))

    if int(args.max_detections_per_window) > 0:
        detections.sort(key=lambda d: (d.periodicity_score, d.saliency_score, d.event_count), reverse=True)
        detections = detections[: int(args.max_detections_per_window)]

    return WindowResult(detections, saliency_u8, candidates, accepted, segmentation_mask,
                        filter_stats, raw_saliency_u8)

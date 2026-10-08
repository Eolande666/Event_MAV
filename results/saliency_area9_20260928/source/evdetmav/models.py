"""Core data types and box geometry utilities shared across the pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


Box = tuple[int, int, int, int]  # x0, y0, x1, y1 with x1/y1 exclusive.


FIELD_ALIASES = {
    "x": ("x", "xs", "col", "cols", "column", "x_addr", "xaddr"),
    "y": ("y", "ys", "row", "rows", "line", "y_addr", "yaddr"),
    "t": ("t", "ts", "time", "times", "timestamp", "timestamps"),
    "p": ("p", "ps", "pol", "pols", "polarity", "polarities"),
}


CSV_FIELDS = [
    "source_file",
    "window_id",
    "t_start_sec",
    "t_end_sec",
    "label",
    "bbox_x0",
    "bbox_y0",
    "bbox_x1",
    "bbox_y1",
    "bbox_w",
    "bbox_h",
    "bbox_area_px",
    "segmentation_area_px",
    "event_count",
    "saliency_score",
    "periodicity_score",
    "density_has_peak",
    "density_has_valley",
    "structure_has_peak",
    "structure_has_valley",
    "direction_has_peak",
    "direction_has_valley",
    "gaussian_consistency",
    "propeller_component_count",
    "latency_ms",
]


@dataclass
class Events:
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    p: Optional[np.ndarray]


@dataclass
class Candidate:
    box: Box
    area_px: int
    saliency_score: float
    rank: int = 0
    periodicity: Optional["PeriodicityScore"] = None


@dataclass
class PeriodicityScore:
    score: int
    density_has_peak: bool
    density_has_valley: bool
    structure_has_peak: bool
    structure_has_valley: bool
    direction_has_peak: bool
    direction_has_valley: bool


@dataclass
class RefinedCandidate:
    candidate: Candidate
    box: Box
    mask: np.ndarray
    offset_x: int
    offset_y: int
    segmentation_area_px: int
    gaussian_consistency: float


@dataclass
class Detection:
    source_file: str
    window_id: int
    t_start_sec: float
    t_end_sec: float
    label: str
    bbox_x0: int
    bbox_y0: int
    bbox_x1: int
    bbox_y1: int
    bbox_w: int
    bbox_h: int
    bbox_area_px: int
    segmentation_area_px: int
    event_count: int
    saliency_score: float
    periodicity_score: int
    density_has_peak: int
    density_has_valley: int
    structure_has_peak: int
    structure_has_valley: int
    direction_has_peak: int
    direction_has_valley: int
    gaussian_consistency: float
    propeller_component_count: int
    latency_ms: float


@dataclass
class WindowResult:
    detections: list[Detection]
    saliency_u8: np.ndarray
    candidates: list[Candidate]
    refined: list[RefinedCandidate]
    segmentation_mask: np.ndarray
    saliency_filter_stats: dict | None = None
    raw_saliency_u8: np.ndarray | None = None


def clipped_box(box: Box, width: int, height: int) -> Box:
    x0, y0, x1, y1 = box
    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)


def pad_box(box: Box, pad: int, width: int, height: int) -> Box:
    x0, y0, x1, y1 = box
    return clipped_box((x0 - pad, y0 - pad, x1 + pad, y1 + pad), width, height)


def union_boxes(boxes: list[Box]) -> Box:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def box_area(box: Box) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def rectangle_distance(a: Box, b: Box) -> float:
    dx = max(a[0] - b[2], b[0] - a[2], 0)
    dy = max(a[1] - b[3], b[1] - a[3], 0)
    return math.hypot(dx, dy)


def events_in_box(x: np.ndarray, y: np.ndarray, box: Box) -> np.ndarray:
    x0, y0, x1, y1 = box
    return (x >= x0) & (x < x1) & (y >= y0) & (y < y1)

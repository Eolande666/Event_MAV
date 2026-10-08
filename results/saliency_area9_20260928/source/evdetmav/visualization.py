"""Image rendering and visualization output: saliency maps, event frames, boxes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

from .models import Candidate, Detection, RefinedCandidate, WindowResult


def saliency_to_rgb(saliency_u8: np.ndarray) -> np.ndarray:
    v = np.clip(saliency_u8.astype(np.float32) / 255.0, 0.0, 1.0)
    rgb = np.zeros((*v.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(3.0 * v, 0.0, 1.0) * 255
    rgb[..., 1] = np.clip(3.0 * v - 1.0, 0.0, 1.0) * 255
    rgb[..., 2] = np.clip(3.0 * v - 2.0, 0.0, 1.0) * 255
    return rgb


def make_event_image(x: np.ndarray, y: np.ndarray, p: np.ndarray, height: int, width: int, percentile: float) -> Image.Image:
    pos = np.zeros((height, width), dtype=np.float32)
    neg = np.zeros((height, width), dtype=np.float32)
    if x.size:
        np.add.at(pos, (y[p > 0], x[p > 0]), 1.0)
        np.add.at(neg, (y[p < 0], x[p < 0]), 1.0)
    values = np.concatenate([pos[pos > 0], neg[neg > 0]])
    scale = float(np.percentile(values, percentile)) if values.size else 1.0
    scale = max(scale, 1.0)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(pos / scale, 0.0, 1.0) * 255
    rgb[..., 2] = np.clip(neg / scale, 0.0, 1.0) * 255
    return Image.fromarray(rgb, mode="RGB")


def draw_boxes(
    image: Image.Image,
    detections: list[Detection],
    candidates: Optional[list[Candidate]] = None,
    refined: Optional[list[RefinedCandidate]] = None,
) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    if candidates:
        for candidate in candidates:
            x0, y0, x1, y1 = candidate.box
            draw.rectangle((x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)), outline=(255, 200, 40), width=1)
    if refined:
        for item in refined:
            x0, y0, x1, y1 = item.box
            draw.rectangle((x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)), outline=(60, 220, 255), width=2)
    for det in detections:
        color = (80, 255, 120) if det.label == "mav" else (60, 220, 255)
        draw.rectangle((det.bbox_x0, det.bbox_y0, max(det.bbox_x0, det.bbox_x1 - 1), max(det.bbox_y0, det.bbox_y1 - 1)), outline=color, width=2)
        draw.text((det.bbox_x0 + 2, det.bbox_y0 + 2), f"{det.label} sp={det.periodicity_score}", fill=color)
    return out


def save_saliency_image(path: Path, result: WindowResult) -> None:
    image = Image.fromarray(saliency_to_rgb(result.saliency_u8), mode="RGB")
    image = draw_boxes(image, result.detections, result.candidates[:8], result.refined)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def save_event_frame(path: Path, x: np.ndarray, y: np.ndarray, p: np.ndarray, height: int, width: int, result: WindowResult, args: argparse.Namespace) -> None:
    image = make_event_image(x, y, p, height, width, float(args.event_vis_percentile))
    image = draw_boxes(image, result.detections, None, result.refined)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def save_segmentation_image(path: Path, x: np.ndarray, y: np.ndarray, p: np.ndarray, height: int, width: int, result: WindowResult, args: argparse.Namespace) -> None:
    base = np.asarray(make_event_image(x, y, p, height, width, float(args.event_vis_percentile))).astype(np.float32)
    mask = result.segmentation_mask
    base[mask, 1] = 255.0
    base[mask, 0] *= 0.35
    base[mask, 2] *= 0.35
    image = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")
    image = draw_boxes(image, result.detections)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)

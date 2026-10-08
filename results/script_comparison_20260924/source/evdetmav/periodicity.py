"""Stage 2: spatio-temporal periodicity features on top-K salient areas."""

from __future__ import annotations

import argparse
from typing import Optional

import numpy as np
from scipy.signal import find_peaks

from .models import Box, Candidate, PeriodicityScore, events_in_box


def slice_local_images(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    box: Box,
    start_t: float,
    end_t: float,
    n_slices: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    x0, y0, x1, y1 = box
    roi_h = max(1, y1 - y0)
    roi_w = max(1, x1 - x0)
    images = np.zeros((n_slices, roi_h, roi_w), dtype=np.float32)
    point_sets: list[list[tuple[float, float]]] = [[] for _ in range(n_slices)]
    duration = max(end_t - start_t, 1e-12)
    slice_ids = np.floor((t - start_t) / duration * n_slices).astype(np.int64)
    slice_ids = np.clip(slice_ids, 0, n_slices - 1)
    lx = x - x0
    ly = y - y0

    for slice_id in range(n_slices):
        idx = slice_ids == slice_id
        if not np.any(idx):
            continue
        signs = np.where(p[idx] > 0, 1.0, -1.0).astype(np.float32)
        np.add.at(images[slice_id], (ly[idx], lx[idx]), signs)
        point_sets[slice_id] = list(zip(lx[idx].astype(np.float64), ly[idx].astype(np.float64)))
    return images, [np.asarray(points, dtype=np.float64) for points in point_sets]


def normalized_vector(image: np.ndarray) -> Optional[np.ndarray]:
    vec = image.reshape(-1).astype(np.float64)
    std = float(np.std(vec))
    if std < 1e-12:
        return None
    return (vec - float(np.mean(vec))) / std


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def principal_direction(points: np.ndarray, min_points: int) -> Optional[np.ndarray]:
    if points.shape[0] < min_points:
        return None
    centered = points - np.mean(points, axis=0, keepdims=True)
    cov = centered.T @ centered / max(points.shape[0], 1)
    if not np.all(np.isfinite(cov)):
        return None
    vals, vecs = np.linalg.eigh(cov)
    if float(vals[-1]) <= 1e-9:
        return None
    return vecs[:, -1]


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return values
    filled = values.copy()
    filled[~finite] = np.nanmean(values[finite])
    window = max(int(window), 1)
    if window <= 1 or values.size < 3:
        return filled
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(filled, kernel, mode="same")


def has_peak_and_valley(values: np.ndarray, args: argparse.Namespace) -> tuple[bool, bool]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 3:
        return False, False
    smooth = moving_average(values, int(args.feature_ma_window))
    amp = float(np.max(smooth) - np.min(smooth))
    if amp < float(args.feature_min_amplitude):
        return False, False
    norm = (smooth - float(np.min(smooth))) / max(amp, 1e-12)
    peaks, _ = find_peaks(norm, prominence=float(args.feature_prominence))
    valleys, _ = find_peaks(1.0 - norm, prominence=float(args.feature_prominence))
    return bool(peaks.size), bool(valleys.size)


def evaluate_periodicity(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    candidate: Candidate,
    start_t: float,
    end_t: float,
    args: argparse.Namespace,
) -> PeriodicityScore:
    n_slices = max(int(args.periodicity_slices), 4)
    inside = events_in_box(x, y, candidate.box)
    if int(np.count_nonzero(inside)) < int(args.min_candidate_events):
        score = PeriodicityScore(0, False, False, False, False, False, False)
        candidate.periodicity = score
        return score

    xi, yi, ti, pi = x[inside], y[inside], t[inside], p[inside]
    images, point_sets = slice_local_images(xi, yi, ti, pi, candidate.box, start_t, end_t, n_slices)

    density = np.array([np.count_nonzero((pi > 0) & (np.floor((ti - start_t) / max(end_t - start_t, 1e-12) * n_slices).astype(np.int64).clip(0, n_slices - 1) == i)) for i in range(n_slices)], dtype=np.float64)

    structure = []
    for i in range(n_slices - 1):
        v0 = normalized_vector(images[i])
        v1 = normalized_vector(images[i + 1])
        structure.append(float("nan") if v0 is None or v1 is None else cosine_similarity(v0, v1))
    structure = np.asarray(structure, dtype=np.float64)

    directions = [principal_direction(points, int(args.principal_min_events)) for points in point_sets]
    direction_similarity = []
    for i in range(n_slices - 1):
        d0, d1 = directions[i], directions[i + 1]
        if d0 is None or d1 is None:
            direction_similarity.append(float("nan"))
        else:
            direction_similarity.append(abs(cosine_similarity(d0, d1)))
    direction_similarity = np.asarray(direction_similarity, dtype=np.float64)

    d_peak, d_valley = has_peak_and_valley(density, args)
    s_peak, s_valley = has_peak_and_valley(structure, args)
    p_peak, p_valley = has_peak_and_valley(direction_similarity, args)
    total = int(d_peak) + int(d_valley) + int(s_peak) + int(s_valley) + int(p_peak) + int(p_valley)
    score = PeriodicityScore(total, d_peak, d_valley, s_peak, s_valley, p_peak, p_valley)
    candidate.periodicity = score
    return score

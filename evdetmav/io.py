"""Event loading and preprocessing: H5/NPZ/CSV/TXT readers, time units, polarity."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import h5py
except ImportError:  # H5 inputs will report a clear error; NPZ/CSV still work.
    h5py = None

from .models import Events, FIELD_ALIASES


def canonical_field(name: str) -> Optional[str]:
    lower = name.lower()
    for canonical, aliases in FIELD_ALIASES.items():
        if lower in aliases:
            return canonical
    return None


def iter_h5_datasets(h5) -> list[tuple[str, object]]:
    datasets = []

    def visitor(name, obj):
        if h5py is not None and isinstance(obj, h5py.Dataset):
            datasets.append((name, obj))

    h5.visititems(visitor)
    return datasets


def find_compound_events(h5):
    for _, ds in iter_h5_datasets(h5):
        if ds.dtype.fields is None:
            continue
        mapping = {}
        for field_name in ds.dtype.fields:
            canonical = canonical_field(field_name)
            if canonical is not None and canonical not in mapping:
                mapping[canonical] = field_name
        if {"x", "y", "t"}.issubset(mapping):
            return ds, mapping
    return None


def find_array_events(h5):
    by_parent: dict[str, dict[str, object]] = {}
    for path, ds in iter_h5_datasets(h5):
        canonical = canonical_field(Path(path).name)
        if canonical is None:
            continue
        parent = str(Path(path).parent).replace("\\", "/")
        by_parent.setdefault(parent, {})
        by_parent[parent].setdefault(canonical, ds)
    for group in by_parent.values():
        if {"x", "y", "t"}.issubset(group):
            return group
    flat = {}
    for path, ds in iter_h5_datasets(h5):
        canonical = canonical_field(Path(path).name)
        if canonical is not None and canonical not in flat:
            flat[canonical] = ds
    if {"x", "y", "t"}.issubset(flat):
        return flat
    return None


def find_matrix_events(h5):
    candidates = []
    for path, ds in iter_h5_datasets(h5):
        if len(ds.shape) == 2 and 3 <= ds.shape[1] <= 5:
            candidates.append((path, ds))
    if not candidates:
        return None
    for preferred in ("data", "events", "event", "events/data"):
        for path, ds in candidates:
            if path.lower() == preferred or Path(path).name.lower() == preferred:
                return ds
    return candidates[0][1]


def looks_like_polarity(values: np.ndarray) -> bool:
    if values.size == 0:
        return False
    sample = values[: min(values.size, 100000)]
    unique = np.unique(sample)
    if unique.size > 8:
        return False
    rounded = np.round(unique).astype(np.int64)
    return np.all(np.isclose(unique, rounded)) and set(rounded.tolist()).issubset({-1, 0, 1})


def events_from_matrix(data: np.ndarray) -> Events:
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("Event matrix must have shape Nx3 or Nx4/Nx5.")
    if data.shape[1] == 3:
        return Events(np.asarray(data[:, 0]), np.asarray(data[:, 1]), np.asarray(data[:, 2]), None)

    col2 = np.asarray(data[:, 2])
    col3 = np.asarray(data[:, 3])
    col2_pol = looks_like_polarity(col2)
    col3_pol = looks_like_polarity(col3)
    col2_span = float(np.nanmax(col2) - np.nanmin(col2)) if col2.size else 0.0
    col3_span = float(np.nanmax(col3) - np.nanmin(col3)) if col3.size else 0.0

    if col2_pol and not col3_pol:
        return Events(np.asarray(data[:, 0]), np.asarray(data[:, 1]), col3, col2)
    if col3_pol and not col2_pol:
        return Events(np.asarray(data[:, 0]), np.asarray(data[:, 1]), col2, col3)
    if col2_pol and col3_span > col2_span:
        return Events(np.asarray(data[:, 0]), np.asarray(data[:, 1]), col3, col2)
    return Events(np.asarray(data[:, 0]), np.asarray(data[:, 1]), col2, col3)


def load_h5_events(path: Path, max_events: Optional[int]) -> Events:
    if h5py is None:
        raise ImportError("h5py is required for H5 input. Install with: python -m pip install h5py")
    with h5py.File(path, "r") as h5:
        compound = find_compound_events(h5)
        if compound is not None:
            ds, mapping = compound
            data = ds[:max_events] if max_events else ds[:]
            p = np.asarray(data[mapping["p"]]) if "p" in mapping else None
            return Events(np.asarray(data[mapping["x"]]), np.asarray(data[mapping["y"]]), np.asarray(data[mapping["t"]]), p)

        matrix = find_matrix_events(h5)
        if matrix is not None:
            data = matrix[:max_events] if max_events else matrix[:]
            return events_from_matrix(np.asarray(data))

        arrays = find_array_events(h5)
        if arrays is None:
            raise ValueError(f"Could not find event arrays in {path}")
        sl = slice(None, max_events)
        p = np.asarray(arrays["p"][sl]) if "p" in arrays else None
        return Events(np.asarray(arrays["x"][sl]), np.asarray(arrays["y"][sl]), np.asarray(arrays["t"][sl]), p)


def load_npz_events(path: Path, max_events: Optional[int]) -> Events:
    data = np.load(path)
    sl = slice(None, max_events)
    if "data" in data:
        return events_from_matrix(np.asarray(data["data"][sl]))
    p = data["p"] if "p" in data else data["polarity"] if "polarity" in data else None
    return Events(
        np.asarray(data["x"][sl]),
        np.asarray(data["y"][sl]),
        np.asarray(data["t"][sl]),
        None if p is None else np.asarray(p[sl]),
    )


def load_text_events(path: Path, max_events: Optional[int]) -> Events:
    delimiter = "," if path.suffix.lower() == ".csv" else None
    data = np.loadtxt(path, delimiter=delimiter, max_rows=max_events)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return events_from_matrix(data)


def load_events(path: Path, max_events: Optional[int] = None) -> Events:
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        return load_h5_events(path, max_events)
    if suffix == ".npz":
        return load_npz_events(path, max_events)
    if suffix in {".csv", ".txt"}:
        return load_text_events(path, max_events)
    raise ValueError(f"Unsupported input type: {path.suffix}. Use .h5/.hdf5/.npz/.csv/.txt.")


def convert_time_to_seconds(t: np.ndarray, unit: str) -> tuple[np.ndarray, str]:
    t = np.asarray(t, dtype=np.float64)
    if unit == "s":
        return t, "s"
    if unit == "ms":
        return t * 1e-3, "ms"
    if unit == "us":
        return t * 1e-6, "us"
    if unit == "ns":
        return t * 1e-9, "ns"
    if unit != "auto":
        raise ValueError(f"Unknown time unit: {unit}")
    if t.size == 0:
        return t, "s"
    span = float(np.nanmax(t) - np.nanmin(t))
    max_abs = float(np.nanmax(np.abs(t)))
    if span > 1e8 or max_abs > 1e12:
        return t * 1e-9, "ns"
    if span > 1e5 or max_abs > 1e5:
        return t * 1e-6, "us"
    if span > 1e2 or max_abs > 1e2:
        return t * 1e-3, "ms"
    return t, "s"


def normalize_polarity(p: Optional[np.ndarray], n: int) -> np.ndarray:
    if p is None:
        return np.ones(n, dtype=np.int8)
    p = np.asarray(p)
    return np.where(p > 0, 1, -1).astype(np.int8)


def prepare_events(events: Events, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, str]:
    x = np.asarray(events.x, dtype=np.int64)
    y = np.asarray(events.y, dtype=np.int64)
    t, used_unit = convert_time_to_seconds(np.asarray(events.t), args.time_unit)
    p = normalize_polarity(events.p, x.size)
    if x.size == 0:
        width = int(args.width) if args.width > 0 else 0
        height = int(args.height) if args.height > 0 else 0
        return x, y, t, p, width, height, used_unit

    order = np.argsort(t, kind="mergesort")
    x, y, t, p = x[order], y[order], t[order], p[order]

    width = int(args.width) if args.width > 0 else int(np.nanmax(x)) + 1
    height = int(args.height) if args.height > 0 else int(np.nanmax(y)) + 1
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height) & np.isfinite(t)
    return x[valid], y[valid], t[valid], p[valid], width, height, used_unit


def iter_input_files(input_path: Path, limit_files: int) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    files = [
        path
        for path in sorted(input_path.rglob("*"))
        if path.is_file() and path.suffix.lower() in {".h5", ".hdf5", ".npz", ".csv", ".txt"}
    ]
    if limit_files > 0:
        files = files[:limit_files]
    return files

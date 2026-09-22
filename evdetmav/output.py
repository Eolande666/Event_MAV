"""CSV output: per-file detections, batch detections, and the source manifest."""

from __future__ import annotations

import csv
from pathlib import Path

from .models import CSV_FIELDS, Detection


def write_csv(path: Path, rows: list[Detection]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in CSV_FIELDS})


def write_manifest(path: Path, rows: list[tuple[str, Path]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["output_prefix", "source_file"])
        writer.writeheader()
        for prefix, source in rows:
            writer.writerow({"output_prefix": prefix, "source_file": str(source)})

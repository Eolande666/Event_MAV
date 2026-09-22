"""Command-line interface and per-file driver: windows, processing loop, argparse."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .io import iter_input_files, load_events, prepare_events
from .models import Detection
from .output import write_csv, write_manifest
from .pipeline import process_window
from .visualization import save_event_frame, save_saliency_image, save_segmentation_image


def build_windows(t: np.ndarray, args: argparse.Namespace) -> list[tuple[int, float, float]]:
    if t.size == 0:
        return []
    start = float(t[0])
    end = float(t[-1])
    window_ms = float(args.window_ms)
    if window_ms <= 0:
        return [(0, start, end)]
    window_s = window_ms * 1e-3
    step_ms = float(args.step_ms) if args.step_ms > 0 else window_ms
    step_s = step_ms * 1e-3
    windows = []
    current = start
    window_id = 0
    while current <= end:
        win_end = min(current + window_s, end)
        if win_end - current >= max(float(args.min_window_ms) * 1e-3, 1e-9):
            windows.append((window_id, current, win_end))
            window_id += 1
        if current + step_s > end:
            break
        current += step_s
    return windows


def process_file(input_file: Path, out_root: Path, output_prefix: str, args: argparse.Namespace) -> list[Detection]:
    events = load_events(input_file, args.max_events)
    x, y, t, p, width, height, used_unit = prepare_events(events, args)
    rows: list[Detection] = []
    if x.size == 0 or width <= 0 or height <= 0:
        print(f"{input_file}: no valid events")
        return rows

    windows = build_windows(t, args)
    if int(args.max_windows) > 0:
        windows = windows[: int(args.max_windows)]
    print(
        f"{input_file.name}: events={x.size} size={width}x{height} time_unit={used_unit} "
        f"windows={len(windows)}"
    )

    saliency_dir = out_root / "saliency"
    event_dir = out_root / "event_boxes_evdetmav"
    segmentation_dir = out_root / "segmentation"

    for local_window_id, start_t, end_t in windows:
        start_idx = int(np.searchsorted(t, start_t, side="left"))
        end_idx = int(np.searchsorted(t, end_t, side="right"))
        xw, yw, tw, pw = x[start_idx:end_idx], y[start_idx:end_idx], t[start_idx:end_idx], p[start_idx:end_idx]
        if xw.size < int(args.min_window_events):
            continue
        result = process_window(xw, yw, tw, pw, start_t, end_t, height, width, str(input_file), local_window_id, args)
        rows.extend(result.detections)

        if local_window_id % int(args.save_every) == 0:
            image_stem = f"{output_prefix}__win_{local_window_id:06d}"
            if args.save_saliency:
                save_saliency_image(saliency_dir / f"{image_stem}.png", result)
            if args.save_event_frames:
                save_event_frame(event_dir / f"{image_stem}.png", xw, yw, pw, height, width, result, args)
            if args.save_segmentation:
                save_segmentation_image(segmentation_dir / f"{image_stem}.png", xw, yw, pw, height, width, result, args)

        if args.progress_every > 0 and local_window_id % int(args.progress_every) == 0:
            print(
                f"  window={local_window_id:06d} t=[{start_t:.6f},{end_t:.6f}] "
                f"events={xw.size} candidates={len(result.candidates)} detections={len(result.detections)}"
            )

    if args.save_per_file_csv:
        per_file_dir = out_root / "per_file"
        write_csv(per_file_dir / f"{output_prefix}_detections.csv", rows)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce EvDetMAV: density-aware saliency + spatio-temporal periodicity + clustering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, type=Path, help="Input event file or directory of .h5/.hdf5/.npz/.csv/.txt files.")
    parser.add_argument("--out", type=Path, default=Path("outputs/evdetmav"), help="Output directory.")
    parser.add_argument("--width", type=int, default=0, help="Sensor width. 0 infers from events.")
    parser.add_argument("--height", type=int, default=0, help="Sensor height. 0 infers from events.")
    parser.add_argument("--time-unit", choices=("auto", "s", "ms", "us", "ns"), default="auto", help="Timestamp unit.")
    parser.add_argument("--max-events", type=int, default=None, help="Load at most this many events from each file.")
    parser.add_argument("--limit-files", type=int, default=0, help="Process at most this many files from a directory. 0 keeps all.")
    parser.add_argument("--max-windows", type=int, default=0, help="Process at most this many windows per file. 0 keeps all.")
    parser.add_argument("--window-ms", type=float, default=0.0, help="Sliding window duration. 0 treats each file as one EventMAV period.")
    parser.add_argument("--step-ms", type=float, default=0.0, help="Sliding window step. 0 uses --window-ms.")
    parser.add_argument("--min-window-ms", type=float, default=1.0, help="Drop windows shorter than this duration.")
    parser.add_argument("--min-window-events", type=int, default=20, help="Drop windows with fewer events.")

    parser.add_argument("--saliency-slices", type=int, default=10, help="n in the density-aware saliency stage.")
    parser.add_argument("--intersection-radius", type=int, default=1, help="Spatial tolerance for positive/negative intersection. 0 is the strict paper rule.")
    parser.add_argument("--saliency-sigma", type=float, default=0.0, help="Optional Gaussian smoothing on saliency map.")
    parser.add_argument("--tau-s", type=float, default=50.0, help="Paper saliency threshold tau_s.")
    parser.add_argument("--tau-p", type=int, default=3, help="Paper periodicity threshold tau_p.")
    parser.add_argument("--top-k", type=int, default=4, help="K top salient areas evaluated with periodicity features.")
    parser.add_argument("--init-dilate-px", type=int, default=0, help="Optional dilation before initial connected components.")
    parser.add_argument("--min-component-area", type=int, default=3, help="Minimum saliency component area.")
    parser.add_argument("--cluster-gap-px", type=float, default=8.0, help="Merge saliency rectangles whose minimum rectangle distance is below this.")
    parser.add_argument("--candidate-pad-px", type=int, default=4, help="Padding around initialized clustering areas.")

    parser.add_argument("--periodicity-slices", type=int, default=12, help="m temporal slices for local spatio-temporal features.")
    parser.add_argument("--min-candidate-events", type=int, default=10, help="Minimum events in candidate before periodicity scoring.")
    parser.add_argument("--feature-ma-window", type=int, default=3, help="Moving-average window for fd, fs and fp.")
    parser.add_argument("--feature-prominence", type=float, default=0.15, help="Prominence threshold after feature normalization.")
    parser.add_argument("--feature-min-amplitude", type=float, default=0.05, help="Minimum normalized-feature amplitude needed for extrema.")
    parser.add_argument("--principal-min-events", type=int, default=4, help="Minimum slice events for principal direction extraction.")

    parser.add_argument("--fine-threshold", type=float, default=0.0, help="Fine saliency threshold. 0 reuses --tau-s.")
    parser.add_argument("--fine-min-area", type=int, default=3, help="Minimum fine-stage connected area.")
    parser.add_argument("--gaussian-consistency-threshold", type=float, default=0.25, help="Minimum cosine match to a fitted 2D Gaussian.")
    parser.add_argument("--fine-box-pad-px", type=int, default=0, help="Padding around refined boxes.")
    parser.add_argument("--fine-fallback-to-coarse", action=argparse.BooleanOptionalAction, default=True, help="Use coarse mask if no component passes Gaussian consistency.")
    parser.add_argument("--merge-propellers", action=argparse.BooleanOptionalAction, default=False, help="Merge accepted propeller regions into one MAV detection box. Default keeps refined cyan boxes as final results.")
    parser.add_argument("--max-detections-per-window", type=int, default=0, help="Keep top N detections. 0 keeps all refined final boxes.")

    parser.add_argument("--save-saliency", action=argparse.BooleanOptionalAction, default=True, help="Save saliency map visualizations.")
    parser.add_argument("--save-event-frames", action=argparse.BooleanOptionalAction, default=True, help="Save event-frame visualizations with boxes.")
    parser.add_argument("--save-segmentation", action=argparse.BooleanOptionalAction, default=True, help="Save fine segmentation overlays.")
    parser.add_argument("--save-per-file-csv", action=argparse.BooleanOptionalAction, default=True, help="Save one detection CSV for every input file in addition to the batch CSV.")
    parser.add_argument("--event-vis-percentile", type=float, default=99.0, help="Event frame normalization percentile.")
    parser.add_argument("--save-every", type=int, default=1, help="Save every Nth processed window.")
    parser.add_argument("--progress-every", type=int, default=20, help="Print progress every N windows. 0 disables.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    files = iter_input_files(input_path, int(args.limit_files))
    if not files:
        raise FileNotFoundError(f"No supported event files found under {input_path}")

    all_rows: list[Detection] = []
    manifest_rows: list[tuple[str, Path]] = []
    print(f"EvDetMAV files: {len(files)}")
    for file_index, path in enumerate(files):
        prefix = f"{file_index:06d}__{path.stem}" if len(files) > 1 else path.stem
        manifest_rows.append((prefix, path))
        rows = process_file(path, out_root, prefix, args)
        all_rows.extend(rows)

    write_csv(out_root / "evdetmav_detections_batch.csv", all_rows)
    write_manifest(out_root / "source_manifest.csv", manifest_rows)
    print(f"\nDone. Batch detections: {len(all_rows)}")
    print(f"Outputs: {out_root}")


run = main

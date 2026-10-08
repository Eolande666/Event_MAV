"""Standalone command line entry point. No imports from Vo_Event required."""
import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
from time import perf_counter
import numpy as np
from config import Config
from detector import FixedEventDetector
from event_io import iter_windows, list_inputs


FIELDS = ['source_file', 'window', 'start_s', 'end_s', 'track_id',
          'bbox_x0', 'bbox_y0', 'bbox_x1', 'bbox_y1', 'state', 'published',
          'events', 'saliency', 'occupied_bins',
          'observable', 'valid_pixels', 'rotation_blocks', 'rotation_q',
          'rotation_significance', 'rotation_omega', 'rotation_fit',
          'rotation_rigidity', 'rotation_residual', 'rotation_polarity_sign',
          'rotation_omega_positive', 'rotation_omega_negative', 'switch_pixels', 'repeat_pixels',
          'repeat_mass', 'switch_time_coverage', 'repetition_score', 'spatial_score',
          'joint_score', 'joint_threshold', 'joint_pass',
          'repetition_base', 'persistent_switch_support', 'background_contrast', 'foreground_density', 'background_density']
TIMING = ['window', 'start_s', 'end_s', 'events', 'proposals', 'candidates', 'rotation_calls',
          'detections', 'proposal_ms', 'verification_ms', 'tracking_ms', 'total_ms']


def row(candidate, source, timing):
    result = {key: candidate[key] for key in FIELDS if key in candidate}
    result.update(source_file=source, window=timing['window'], start_s=timing['start_s'], end_s=timing['end_s'])
    result.update(zip(('bbox_x0', 'bbox_y0', 'bbox_x1', 'bbox_y1'), candidate['box']))
    return result


def main():
    parser = argparse.ArgumentParser(description='Fixed-camera lightweight bipolar/rotation event detector')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--time-unit', choices=['s', 'ms', 'us', 'ns'], required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--window-ms', type=float)
    parser.add_argument('--rotation-window-ms', type=float,
                        help='trailing interval used by the single IOC call; default 9 ms')
    parser.add_argument('--joint-threshold', type=float, help='override the base joint score threshold; larger is stricter')
    parser.add_argument('--score-weights', type=float, nargs=3, metavar=('ROTATION', 'REPETITION', 'STRUCTURE'), help='positive relative weights for rotation, repetition and spatial structure; normalized automatically')
    parser.add_argument('--file-windows', action='store_true', help='one independent period per file, explicitly rebase timestamps')
    parser.add_argument('--overlap-policy', choices=['error', 'trim'], default='error', help='trim explicitly discards out-of-order file prefixes; equal timestamps are kept')
    parser.add_argument('--limit-files', type=int, default=0)
    parser.add_argument('--start-file', type=int, default=0, help='skip this many files before running; creates a fresh detector')
    parser.add_argument('--log-every', type=int, default=100, help='print progress every N windows; 0 disables')
    parser.add_argument('--max-windows', type=int, default=0)
    parser.add_argument('--save-frames', action='store_true', help='save event images and final detection boxes to frames/*.png')
    parser.add_argument('--show-candidates', action='store_true', help='also draw pending/rejected candidates; requires --save-frames')
    parser.add_argument('--frame-stride', type=int, default=1, help='save every Nth processed window, starting with the first')
    args = parser.parse_args()
    if min(args.limit_files, args.max_windows, args.start_file, args.log_every) < 0:
        parser.error('limits must be nonnegative')
    if args.frame_stride < 1:
        parser.error('--frame-stride must be positive')
    if args.show_candidates and not args.save_frames:
        parser.error('--show-candidates requires --save-frames')
    if args.save_frames:
        from visualization import save_frame
    settings = json.loads(args.config.read_text(encoding='utf-8-sig')) if args.config else {}
    if args.window_ms is not None:
        settings['window_ms'] = args.window_ms
    if args.rotation_window_ms is not None:
        settings['rotation_window_ms'] = args.rotation_window_ms
    if args.joint_threshold is not None:
        settings['joint_threshold'] = args.joint_threshold
    if args.score_weights is not None:
        if any(not math.isfinite(w) or w <= 0 for w in args.score_weights):
            parser.error('--score-weights requires three finite positive numbers')
        largest = max(args.score_weights)
        scaled = [w/largest for w in args.score_weights]
        total = sum(scaled)
        for name, weight in zip(('rotation_weight', 'repetition_weight', 'structure_weight'), scaled):
            settings[name] = weight/total
    cfg = Config(**settings)
    detector = FixedEventDetector(args.width, args.height, cfg)
    files = list_inputs(args.input)[args.start_file:]
    if args.limit_files:
        files = files[:args.limit_files]
    if not files:
        parser.error('--start-file is beyond the input file list')
    # Refuse to mix files from different runs.
    args.out.mkdir(parents=True, exist_ok=False)
    metadata = {'config': asdict(cfg), 'width': args.width, 'height': args.height,
                'time_unit': args.time_unit, 'file_windows': args.file_windows, 'overlap_policy': args.overlap_policy,
                'inputs': [str(p.resolve()) for p in files], 'algorithm_version': '1.0-shared-rigid',
                'visualization': {'save_frames': args.save_frames, 'show_candidates': args.show_candidates, 'frame_stride': args.frame_stride},
                'boxes': 'x0,y0 inclusive; x1,y1 exclusive', 'status': 'running'}
    (args.out/'configuration.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    latencies, total_detections, windows = [], 0, 0
    visualization_ms, frames_saved = 0.0, 0
    begin = perf_counter()
    try:
        with (args.out/'detections.csv').open('w', newline='', encoding='utf-8') as fd, \
             (args.out/'candidates.csv').open('w', newline='', encoding='utf-8') as fc, \
             (args.out/'timing.csv').open('w', newline='', encoding='utf-8') as ft:
            detections, candidates, timing_writer = csv.DictWriter(fd, FIELDS), csv.DictWriter(fc, FIELDS), csv.DictWriter(ft, TIMING)
            for writer in (detections, candidates, timing_writer):
                writer.writeheader()
            for source, events, start, end in iter_windows(files, args.time_unit, args.width, args.height, cfg.window_ms, args.file_windows, args.overlap_policy):
                outputs, diagnostics, timing = detector.process(events, start, end)
                detections.writerows(row(c, source, timing) for c in outputs)
                candidates.writerows(row(c, source, timing) for c in diagnostics)
                timing_writer.writerow(timing)
                latencies.append(timing['total_ms'])
                total_detections += len(outputs)
                windows += 1
                if args.log_every and (windows == 1 or windows % args.log_every == 0):
                    print(f"[{windows} windows] {source} candidates={len(diagnostics)} detections={len(outputs)} detector={timing['total_ms']:.1f}ms", flush=True)
                if args.save_frames and (windows-1) % args.frame_stride == 0:
                    render_begin = perf_counter()
                    save_frame(args.out/'frames'/f"{timing['window']:06d}.png", events, outputs,
                               args.width, args.height, candidates=diagnostics,
                               show_candidates=args.show_candidates, window=timing['window'],
                               start_s=start, end_s=end)
                    visualization_ms += (perf_counter()-render_begin)*1000
                    frames_saved += 1
                if args.max_windows and windows >= args.max_windows:
                    break
        summary = {'status': 'complete', 'windows': windows, 'detections': total_detections,
                   'mean_detector_ms': float(np.mean(latencies)) if latencies else 0,
                   'p95_detector_ms': float(np.percentile(latencies, 95)) if latencies else 0,
                   'wall_seconds_including_io': perf_counter()-begin,
                   'frames_saved': frames_saved, 'visualization_total_ms': visualization_ms,
                   'note': 'Detector timing excludes IO and visualization; wall time includes both.'}
    except Exception as exc:
        metadata.update(status='failed', error=str(exc))
        (args.out/'configuration.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        (args.out/'summary.json').write_text(json.dumps({'status': 'failed', 'error': str(exc)}, indent=2), encoding='utf-8')
        raise
    metadata.update(status='complete', windows=windows)
    (args.out/'configuration.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

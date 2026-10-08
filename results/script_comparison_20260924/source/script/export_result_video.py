"""Render saved detections at acquisition speed, one frame per event window."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import warnings

import imageio_ffmpeg
import numpy as np
from event_io import iter_windows
from visualization import render_frame, HEADER


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True,
                        help='configuration.json from the original event window run')
    parser.add_argument('--detections', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output already exists')
    meta = json.loads(args.metadata.read_text(encoding='utf-8-sig'))
    expected_frames = meta.get('windows')
    if expected_frames is None:
        summary_path = args.metadata.with_name('summary.json')
        if summary_path.exists():
            expected_frames = json.loads(summary_path.read_text(encoding='utf-8-sig')).get('windows')
    duration = meta['config']['window_ms'] / 1000
    width, height = meta['width'], meta['height']
    rows = defaultdict(list)
    with args.detections.open(newline='', encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            if row.get('published', 'True').lower() not in ('true', '1'):
                continue
            window = int(row['window'])
            if (abs(float(row['start_s'])-(window-1)*duration) > 1e-6
                    or abs(float(row['end_s'])-window*duration) > 1e-6):
                raise ValueError('detection timestamps do not match metadata windows')
            candidate = dict(row)
            candidate['box'] = [int(float(row['bbox_'+key])) for key in ('x0', 'y0', 'x1', 'y1')]
            for key in ('joint_score', 'rotation_q', 'rotation_omega',
                        'rotation_fit', 'rotation_rigidity'):
                candidate[key] = float(row[key])
            candidate['observable'] = row.get('observable', '').lower() in ('true', '1')
            rows[window].append(candidate)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(args.out), (width, height+HEADER), fps=1/duration,
        codec='libx264', pix_fmt_in='rgb24', pix_fmt_out='yuv420p',
        macro_block_size=1, ffmpeg_log_level='error',
        output_params=['-crf', '18', '-preset', 'fast', '-movflags', '+faststart'])
    writer.send(None)
    frames, boxes = 0, 0
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            windows = iter_windows([Path(p) for p in meta['inputs']],
                                   meta['time_unit'], width, height,
                                   meta['config']['window_ms'], meta['file_windows'],
                                   meta['overlap_policy'])
            for frame_index, (_, events, start, end) in enumerate(windows, 1):
                if expected_frames is not None and frame_index > expected_frames:
                    break
                frames = frame_index
                detections = rows.pop(frames, [])
                frame = render_frame(events, detections, width, height,
                                     window=frames, start_s=start, end_s=end)
                writer.send(np.asarray(frame))
                boxes += len(detections)
                if frames == 1 or frames % 100 == 0:
                    print(f'{frames} frames | {frames*duration:.2f}s | {boxes} boxes', flush=True)
            warning_count = len(caught)
        if rows:
            raise ValueError('detections extend beyond event stream')
    finally:
        writer.close()
    result = dict(video=str(args.out.resolve()), frames=frames, detections=boxes,
                  fps=1/duration, duration_s=frames*duration, width=width,
                  height=height+HEADER, window_ms=duration*1000,
                  input_warnings=warning_count, metadata=str(args.metadata.resolve()),
                  source_detections=str(args.detections.resolve()))
    args.out.with_suffix('.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()

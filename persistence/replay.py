"""Lossless baseline CSV adapter and per-window persistence log writer."""
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from .types import Detection
from .tracker import Tracker


def load_baseline(directory):
    directory=Path(directory)
    windows=list(csv.DictReader((directory/'video_windows.csv').open()))
    detections=defaultdict(list)
    for row in csv.DictReader((directory/'evdetmav_detections_batch.csv').open()):
        detections[int(row['window_id'])].append(row)
    return windows,detections


def adapt(row,timestamp):
    return Detection(timestamp,tuple(float(row['bbox_'+k]) for k in ('x0','y0','x1','y1')),
                     float(row['saliency_score']),float(row['periodicity_score']),int(row['event_count']))


def dump_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields or ['timestamp']);writer.writeheader()
        for row in rows:
            # Serialize nested fields without NaN/Inf. Check scalar finiteness too.
            json.dumps(row,allow_nan=False)
            writer.writerow({k:json.dumps(v,allow_nan=False) if isinstance(v,(list,tuple,dict)) else v for k,v in row.items()})


def replay(windows, baseline, config, output=None):
    if not config.use_periodicity:
        raise ValueError('periodicity-off requires raw H5 recomputation, not frozen CSV replay')
    tracker=Tracker(config);rows=[];window_rows=[];transitions=[];pairs=[]
    for window in windows:
        i=int(window['frame']);stamp=float(window['end_sec'])
        raw=baseline.get(i,[])
        detections=[adapt(d,stamp) for d in raw]
        tic=time.perf_counter()
        current,changes,debug=tracker.step(stamp,detections)
        elapsed=(time.perf_counter()-tic)*1000
        for r in current:
            r.update(window_id=i,baseline_label=raw[r['detection_index']]['label'])
        rows.extend(current)
        transitions.extend(dict(timestamp=stamp,window_id=i,**x) for x in changes)
        pairs.extend(dict(timestamp=stamp,window_id=i,**x) for x in debug)
        window_rows.append(dict(window_id=i,timestamp=stamp,baseline_detections=len(raw),
                                accepted=sum(r['accepted'] for r in current),active_tracks=len(tracker.tracks),
                                persistence_ms=elapsed))
    if output:
        dest=Path(output);dest.mkdir(parents=True,exist_ok=True);config.save(dest/'config.json')
        dump_csv(dest/'tracks.csv',rows);dump_csv(dest/'windows.csv',window_rows)
        for name,items in [('debug',rows),('track_events',transitions),('association',pairs)]:
            with (dest/(name+'.jsonl')).open('w') as f:
                for item in items:f.write(json.dumps(item,allow_nan=False)+'\n')
        accepted=[baseline[r['window_id']][r['detection_index']] for r in rows if r['accepted']]
        dump_csv(dest/'accepted_baseline_rows.csv',accepted)
    return rows,window_rows


def boxes_by_window(baseline, rows=None):
    output=defaultdict(list)
    if rows is None:
        for wid,ds in baseline.items():
            output[wid]=[adapt(d,float(d['t_end_sec'])).bbox for d in ds]
    else:
        for row in rows:
            if row['accepted']:output[row['window_id']].append(row['bbox'])
    return output

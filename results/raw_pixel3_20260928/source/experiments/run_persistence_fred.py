"""Calibration-only parameter selection followed by frozen local holdout evaluation."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse,csv,json,zipfile,re,hashlib,time
from dataclasses import replace,asdict
from collections import defaultdict
import numpy as np
from persistence.config import Config
from persistence.replay import load_baseline,replay,boxes_by_window,dump_csv
from persistence.evaluation import align_frames,evaluate,metrics,match_boxes

SEQUENCES=['8','20','51','65','93','114']
CALIBRATION=['8','20']
HOLDOUT=['51','65','93','114']
OUT=ROOT/'results/persistence_mvp/fred_v1'


def write(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def dataset(seq):
    windows,baseline=load_baseline(ROOT/'comparison'/seq)
    annotations=defaultdict(list)
    with zipfile.ZipFile(ROOT/'FRED'/f'{seq}.zip') as archive:
        names=archive.namelist()
        text=archive.read(f'{seq}/coordinates.txt').decode()
        frame_times=sorted(int(re.search(r'_(\d+)\.png$',n)[1]) for n in names if '/Event/Frames/' in n and n.endswith('.png'))
    for line in text.splitlines():
        timestamp,rest=line.split(':',1);fields=rest.split(',')
        raw=list(map(float,fields[:4]))
        # Same clipping for every method: extended GT is restricted to sensor FOV.
        box=(max(0.,raw[0]),max(0.,raw[1]),min(1280.,raw[2]),min(720.,raw[3]))
        if box[2]>box[0] and box[3]>box[1]:
            annotations[round(float(timestamp)*1e6)].append(dict(bbox=box,identity=int(fields[4])))
    return windows,baseline,align_frames(windows,frame_times,annotations)


def totals(results):
    return metrics(*(sum(r[k] for r in results) for k in ('tp','fp','fn')))


def chosen_rows(rows,threshold):
    return [dict(r,accepted=r['persistence_score'] is not None and r['persistence_score']>=threshold) for r in rows]


def calibrate(data):
    # Primary calibration objective is declared before opening holdout labels.
    # It is a candidate-localization diagnostic, NOT standard box detection F1.
    base=Config(enabled=True,cold_start='fixed',sigma_position=1.,sigma_direction=1.,
                sigma_acceleration=1.,threshold=0.,radius=20.)
    trials=[];best=None;scale_records=[]
    for radius in (10.,20.,40.):
        provisional=replace(base,radius=radius,use_trajectory=False)
        features=[]
        for seq in CALIBRATION:
            windows,baseline,aligned=data[seq]
            rows,_=replay(windows,baseline,provisional)
            by_window=defaultdict(list)
            for r in rows:by_window[r['window_id']].append(r)
            # Estimate scales only from calibration observations whose centers
            # match an annotated drone; no holdout values enter this calculation.
            for frame in aligned:
                rs=by_window[frame['window_id']]
                pairs=match_boxes([r['bbox'] for r in rs],[g['bbox'] for g in frame['gt']],'center')
                for i,j in pairs:features.append(rs[i])
        scales={};counts={}
        for key,field in [('position','normalized_error'),('direction','direction_change'),('acceleration','acceleration_norm')]:
            values=[r[field] for r in features if r[field] is not None and r[field]>base.epsilon]
            counts[key]=len(values)
            if not values:
                raise RuntimeError(f'Cannot calibrate {key}: no positive valid calibration values')
            scales[key]=float(np.quantile(values,.9))
        config=replace(base,radius=radius,sigma_position=scales['position'],sigma_direction=scales['direction'],sigma_acceleration=scales['acceleration'])
        scale_records.append(dict(radius=radius,quantile=.9,scales=scales,sample_counts=counts))
        for mode,flags in [('neighborhood',dict(use_neighborhood=True,use_trajectory=False)),
                           ('trajectory',dict(use_neighborhood=False,use_trajectory=True)),
                           ('combined',dict(use_neighborhood=True,use_trajectory=True))]:
            c=replace(config,**flags)
            replayed={seq:replay(data[seq][0],data[seq][1],c)[0] for seq in CALIBRATION}
            for threshold in np.linspace(0,1,21):
                results=[]
                for seq in CALIBRATION:
                    predictions=boxes_by_window(data[seq][1],chosen_rows(replayed[seq],float(threshold)))
                    result,_=evaluate(data[seq][2],predictions,'center');results.append(result)
                total=totals(results)
                trials.append(dict(mode=mode,radius=radius,threshold=float(threshold),**total))
        print('CALIBRATED RADIUS',radius,flush=True)
    configs={}
    for mode in ('neighborhood','trajectory','combined'):
        # Tie break favors recall, then a lower threshold and smaller gate.
        winner=max((t for t in trials if t['mode']==mode),key=lambda t:(t['f1'] or 0,t['recall'] or 0,-t['threshold'],-t['radius']))
        scale=next(s for s in scale_records if s['radius']==winner['radius'])['scales']
        configs[mode]=replace(base,radius=winner['radius'],threshold=winner['threshold'],
                             sigma_position=scale['position'],sigma_direction=scale['direction'],sigma_acceleration=scale['acceleration'],
                             use_neighborhood=mode!='trajectory',use_trajectory=mode!='neighborhood')
    dump_csv(OUT/'calibration_search.csv',trials);write(OUT/'calibration_scales.json',scale_records)
    return configs


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    # Verify cached baseline code before reading detections.
    old=json.loads((ROOT/'comparison/code_sha256.json').read_text())
    for name,digest in old.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Baseline code changed: '+name)
    write(OUT/'baseline_verified_hashes.json',old)
    protocol=dict(calibration=CALIBRATION,holdout=HOLDOUT,official_membership='All six are official test sequences; this is a local exploratory split, not official benchmark reporting',
                  alignment='latest completed baseline window at official Event frame timestamp, age <=30.001 ms',
                  primary_calibration_objective='one-to-one center containment F1 (diagnostic, not box detection)',
                  strict_protocols=['IoU >= 0.4','IoU >= 0.5'],gt='coordinates.txt; clipped to 1280x720 sensor FOV',
                  empty_frame_policy='Existing official frames without annotation have empty GT, as in official loader',
                  cold_start='fixed L; include current baseline hit; unobserved history is zero',
                  trajectory_warmup='third valid observation starts residual scoring; combined falls back to Sn; trajectory-only rejects until available',
                  history='all valid baseline candidates, including persistence-rejected candidates',
                  candidate_granularity='unchanged final baseline propeller boxes; no GT-informed enlargement or merging',
                  radius_grid=[10,20,40],threshold_grid='0 to 1 inclusive, step .05',sigma_calibration='90th percentile of positive motion residuals of center-matched calibration candidates')
    write(OUT/'protocol.json',protocol)
    data={seq:dataset(seq) for seq in CALIBRATION}
    configs=calibrate(data)
    for mode,config in configs.items():config.save(OUT/(mode+'_config.json'))
    # Holdout data are loaded only after calibration/config selection is complete.
    data.update({seq:dataset(seq) for seq in HOLDOUT})
    summary=[];timings=[]
    for seq in SEQUENCES:
        windows,baseline,aligned=data[seq]
        seqout=OUT/seq;seqout.mkdir(exist_ok=True)
        dump_csv(seqout/'evaluation_alignment.csv',[dict(timestamp=f['timestamp'],window_id=f['window_id'],lag_sec=f['lag_sec'],gt=f['gt']) for f in aligned])
        rawboxes=boxes_by_window(baseline)
        for mode in ('baseline','neighborhood','trajectory','combined'):
            if mode=='baseline':predictions=rawboxes
            else:
                rows,window_rows=replay(windows,baseline,configs[mode],seqout/mode)
                predictions=boxes_by_window(baseline,rows)
                times=[w['persistence_ms'] for w in window_rows]
                timings.append(dict(sequence=seq,mode=mode,windows=len(windows),mean_ms=float(np.mean(times)),p95_ms=float(np.quantile(times,.95)),
                                    baseline_candidates=sum(len(v) for v in baseline.values()),accepted=sum(r['accepted'] for r in rows)))
            for p in ('iou40','iou50','center'):
                result,frames=evaluate(aligned,predictions,p)
                summary.append(dict(sequence=seq,split='calibration' if seq in CALIBRATION else 'holdout',mode=mode,protocol=p,**result))
                dump_csv(seqout/(mode+'_'+p+'_frames.csv'),frames)
        print('EVALUATED',seq,flush=True)
    aggregate=[]
    for split in ('calibration','holdout'):
        for mode in ('baseline','neighborhood','trajectory','combined'):
            for protocol in ('iou40','iou50','center'):
                group=[r for r in summary if r['split']==split and r['mode']==mode and r['protocol']==protocol]
                aggregate.append(dict(split=split,mode=mode,protocol=protocol,frames=sum(r['frames'] for r in group),**totals(group)))
    dump_csv(OUT/'metrics_by_sequence.csv',summary);dump_csv(OUT/'metrics_aggregate.csv',aggregate);dump_csv(OUT/'runtime.csv',timings)
    write(OUT/'metrics_aggregate.json',aggregate)
    write(OUT/'status.json',dict(stage='evaluation_complete',sequences=SEQUENCES))
    print(json.dumps(aggregate,indent=2),flush=True)


if __name__=='__main__':main()

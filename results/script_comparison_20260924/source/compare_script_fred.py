"""Fresh detectors on identical raw FRED H5 events; GT is evaluation-only."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import sys,json,csv,time,hashlib,argparse,subprocess
from pathlib import Path
from dataclasses import asdict
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'script'))
import numpy as np
import imageio_ffmpeg
from PIL import Image,ImageDraw
from config import Config as ScriptConfig
from detector import FixedEventDetector
from evdetmav.cli import build_parser
from persistence.events import stream_windows
from persistence.baseline_adapter import detect
from persistence.config import Config as PersistenceConfig
from persistence.tracker import Tracker
from persistence.replay import adapt,dump_csv
from persistence.visualization import event_image,overlay_baseline,overlay_persistence,with_header,label
from persistence.evaluation import evaluate,metrics
from experiments.run_persistence_fred import dataset
OUT=ROOT/'results/script_comparison_20260924'

def write(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))
def writer(path,size):
    w=imageio_ffmpeg.write_frames(str(path),size,fps=100/3,macro_block_size=2,codec='libx264',output_params=['-crf','18','-preset','veryfast','-threads','2','-movflags','+faststart']);w.send(None);return w

def run(seq,limit):
    dest=OUT/seq;dest.mkdir(parents=True,exist_ok=False)
    cfg=ScriptConfig(**json.loads((ROOT/'script/config.default.json').read_text()));cfg.window_ms=30.
    pcfg=PersistenceConfig(**json.loads((ROOT/'results/persistence_mvp/fred_v1/combined_config.json').read_text()))
    detector=FixedEventDetector(1280,720,cfg);tracker=Tracker(pcfg)
    args=build_parser().parse_args(['--input',str(ROOT/'FRED'/f'{seq}.zip'),'--window-ms','30','--step-ms','30'])
    args.max_windows=limit
    oldwindows,oldbaseline,aligned=dataset(seq)
    write(dest/'configuration.json',dict(script=asdict(cfg),persistence=asdict(pcfg),baseline={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},input=str(args.input),event_policy='identical [start,end) events for all fresh methods; absolute H5 timestamp in seconds',default_change='script window_ms 33.333 -> 30 for common-window comparison',status='running'))
    predictions={k:defaultdict(list) for k in ('baseline','persistence','script')}
    stats=[];allbaseline=[];allscript=[];candcount=observable=accepted_observable=deferred=boundary=0
    t0=time.time();best_rank=-1;best=None
    sw=writer(dest/'Script.partial.mp4',(1280,784));cw=writer(dest/'Comparison.partial.mp4',(3840,784))
    with (dest/'script_candidates.jsonl').open('w') as cf,(dest/'persistence_debug.jsonl').open('w') as pf:
      try:
       for wid,start,end,raw in stream_windows(args.input,args):
        assert abs(start-float(oldwindows[wid]['start_sec']))<1e-9 and abs(end-float(oldwindows[wid]['end_sec']))<1e-9
        times=raw['t']*1e-6;keep=times<end;boundary+=int((~keep).sum());raw=raw[keep];times=times[keep]
        matrix=np.column_stack((raw['x'],raw['y'],times,np.where(raw['p']>0,1,-1)))
        tic=time.perf_counter();bs=[asdict(d) for d in detect(raw,start,end,1280,720,args.input,wid,args)];bms=(time.perf_counter()-tic)*1000
        tic=time.perf_counter();pr,changes,pairs=tracker.step(end,[adapt(d,end) for d in bs]);pms=(time.perf_counter()-tic)*1000
        tic=time.perf_counter();outs,cands,timing=detector.process(matrix,start,end);sms=(time.perf_counter()-tic)*1000
        for c in cands:
            record=dict(window_id=wid,start_sec=start,end_sec=end,**c);cf.write(json.dumps(record,allow_nan=False)+'\n')
        for r in pr:pf.write(json.dumps(dict(window_id=wid,**r),allow_nan=False)+'\n')
        for b in bs:predictions['baseline'][wid].append([b['bbox_'+k] for k in ('x0','y0','x1','y1')])
        predictions['persistence'][wid]=[r['bbox'] for r in pr if r['accepted']]
        predictions['script'][wid]=[c['box'] for c in outs]
        allbaseline.extend(bs);allscript.extend(dict(window_id=wid,start_sec=start,end_sec=end,**c) for c in outs)
        candcount+=len(cands);observable+=sum(c['observable'] for c in cands);accepted_observable+=sum(c['observable'] for c in outs);deferred+=sum(c['state']=='budget_deferred' for c in cands)
        stats.append(dict(window_id=wid,start_sec=start,end_sec=end,events=len(raw),baseline=len(bs),persistence=sum(r['accepted'] for r in pr),script=len(outs),baseline_ms=bms,persistence_ms=pms,script_ms=sms,**{'script_'+k:v for k,v in timing.items() if k.endswith('_ms')}))
        im=event_image(raw);sim=im.copy();draw=ImageDraw.Draw(sim);occupied=[]
        for c in outs:
            x0,y0,x1,y1=c['box'];color=(255,210,50);draw.rectangle((x0,y0,x1-1,y1-1),outline=color,width=3)
            rot=f"{c['rotation_q']:.2f}" if c['observable'] else 'N/A'
            label(draw,c['box'],f"T{c['track_id']} Q={c['joint_score']:.2f} Rot={rot}",color,occupied)
        sframe=with_header(sim,'script 算法检测',f'Seq {seq} | t={end:.3f}s | Q >= 0.34 | N={len(outs)}')
        left=with_header(overlay_baseline(im,bs),'原始 EvDetMAV',f'Seq {seq} | t={end:.3f}s | N={len(bs)}')
        middle=with_header(overlay_persistence(im,pr),'EvDetMAV 加持续性检验',f'Seq {seq} | t={end:.3f}s | N={len(predictions["persistence"][wid])}')
        panel=Image.new('RGB',(3840,784));panel.paste(left,(0,0));panel.paste(middle,(1280,0));panel.paste(sframe,(2560,0))
        sw.send(np.asarray(sframe));cw.send(np.asarray(panel))
        rank=len(outs)+min(len(bs),4)
        if wid>50 and rank>best_rank:best_rank=rank;best=(panel.copy(),wid)
        if wid%100==0:
            write(dest/'progress.json',dict(window=wid+1,total=len(oldwindows),elapsed_s=time.time()-t0,script_detections=len(allscript)))
            print(seq,wid+1,'/',len(oldwindows),'elapsed',round(time.time()-t0),flush=True)
      finally:sw.close();cw.close()
    count=len(stats)
    if not limit:assert count==len(oldwindows)
    aligned=[f for f in aligned if f['window_id']<count]
    scores=[]
    for method,boxes in predictions.items():
        for protocol in ('iou40','iou50','center'):
            result,frames=evaluate(aligned,boxes,protocol);scores.append(dict(sequence=seq,method=method,protocol=protocol,**result));dump_csv(dest/f'{method}_{protocol}_frames.csv',frames)
    dump_csv(dest/'metrics.csv',scores);write(dest/'metrics.json',scores);dump_csv(dest/'timing.csv',stats);dump_csv(dest/'baseline_detections.csv',allbaseline);dump_csv(dest/'script_detections.csv',allscript)
    for name in ('Script','Comparison'):(dest/(name+'.partial.mp4')).replace(dest/(name+'.mp4'))
    if best:best[0].save(dest/'preview.png')
    summary=dict(status='complete',sequence=seq,windows=count,video_duration_s=count*.03,boundary_events_excluded=boundary,script_detections=len(allscript),baseline_detections=len(allbaseline),persistence_detections=sum(s['persistence'] for s in stats),candidates=candcount,observable_candidates=observable,accepted_observable=accepted_observable,budget_deferred=deferred,elapsed_s=time.time()-t0,preview_window=best[1] if best else None)
    for key in ('baseline_ms','persistence_ms','script_ms'):summary[key]=dict(mean=float(np.mean([s[key] for s in stats])),p95=float(np.percentile([s[key] for s in stats],95)))
    write(dest/'summary.json',summary);print('COMPLETE',seq,json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sequences',nargs='+',required=True);p.add_argument('--max-windows',type=int,default=0);a=p.parse_args()
    OUT.mkdir(exist_ok=True,parents=True)
    for seq in a.sequences:run(seq,a.max_windows)

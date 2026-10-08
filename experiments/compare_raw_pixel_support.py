"""Original persistence vs saliency built from genuine spatial event support.

Only Stage 1 input changes. Periodicity and persistence retain their original
rules and raw candidate-event inputs. Original and area-9 evidence stay intact.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,csv,json,time,hashlib,shutil,subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
import numpy as np
import imageio_ffmpeg
from persistence.config import Config
from persistence.tracker import Tracker
from persistence.events import stream_windows
from persistence.replay import adapt,dump_csv
from persistence.evaluation import evaluate,metrics
from persistence.baseline_adapter import detect
from persistence.visualization import event_image,overlay_persistence,with_header
from evdetmav.cli import build_parser
from evdetmav.pipeline import process_window
from experiments.run_persistence_fred import dataset,CALIBRATION,HOLDOUT
from experiments.compare_saliency_area import write,writer,panel,normalized


def run_sequence(seq,out,minimum,limit):
    dest=Path(out)/seq;dest.mkdir()
    config=Config.load(ROOT/'configs/persistence_original.json')
    trackers={name:Tracker(config) for name in ['original','raw_support']}
    args=build_parser().parse_args(['--input',str(ROOT/'FRED'/f'{seq}.zip'),'--width','1280','--height','720',
        '--time-unit','us','--window-ms','30','--step-ms','30','--min-raw-component-pixels',str(minimum),'--saliency-min-area','0'])
    args.max_windows=limit;oldargs=argparse.Namespace(**vars(args));oldargs.min_raw_component_pixels=0
    windows,frozen,aligned=dataset(seq)
    predictions={name:defaultdict(list) for name in [*trackers,'before_persistence']}
    stats=[];detections=[];start_clock=time.monotonic();checked=0;best_rank=-1
    write(dest/'config.json',dict(min_raw_component_pixels=minimum,saliency_min_area=0,persistence=asdict(config),
        detector={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        change='Only pre-dilation saliency support filtering; periodicity reads unchanged raw events; original inclusive windows'))
    video=writer(dest/'Detection_compare.partial.mp4')
    try:
      with (dest/'persistence_debug.jsonl').open('w') as debug:
        for wid,start,end,raw in stream_windows(args.input,args):
            assert abs(start-float(windows[wid]['start_sec']))<1e-9 and abs(end-float(windows[wid]['end_sec']))<1e-9
            original=frozen.get(wid,[])
            if wid<12:
                fresh=[asdict(d) for d in detect(raw,start,end,1280,720,args.input,wid,oldargs)]
                assert normalized(fresh)==normalized(original),(seq,wid,'original cache mismatch');checked+=1
            tic=time.perf_counter()
            if len(raw)>=args.min_window_events:
                result=process_window(raw['x'].astype(np.int64),raw['y'].astype(np.int64),raw['t']*1e-6,
                    np.where(raw['p']>0,1,-1),start,end,720,1280,str(args.input),wid,args)
                filtered=[asdict(d) for d in result.detections];filter_stats=result.saliency_filter_stats
            else:
                filtered=[];filter_stats=dict(skipped_low_events=True,raw_support_min_pixels=minimum)
            detection_ms=(time.perf_counter()-tic)*1000
            records={}
            for name,raw_dets in [('original',original),('raw_support',filtered)]:
                records[name],changes,pairs=trackers[name].step(end,[adapt(d,end) for d in raw_dets])
                predictions[name][wid]=[r['bbox'] for r in records[name] if r['accepted']]
                if name=='raw_support':debug.write(json.dumps(dict(window_id=wid,timestamp=end,rows=records[name],changes=changes,association=pairs),allow_nan=False)+'\n')
            predictions['before_persistence'][wid]=[[d['bbox_'+k] for k in ['x0','y0','x1','y1']] for d in filtered]
            detections.extend(filtered)
            stats.append(dict(window_id=wid,start_sec=start,end_sec=end,event_count=len(raw),original_candidates=len(original),
                filtered_candidates=len(filtered),original_accepted=len(predictions['original'][wid]),filtered_accepted=len(predictions['raw_support'][wid]),
                detection_ms=detection_ms,**filter_stats))
            events=event_image(raw)
            comparison=panel(with_header(overlay_persistence(events,records['original']),'原版：含持续性检验',f'Seq {seq} | t={end:.3f}s | N={len(predictions["original"][wid])}'),
                with_header(overlay_persistence(events,records['raw_support']),'新版：先检验真实像素覆盖',f'Seq {seq} | t={end:.3f}s | Raw pixels >= {minimum} | N={len(predictions["raw_support"][wid])}'))
            video.send(np.asarray(comparison))
            rank=abs(len(predictions['original'][wid])-len(predictions['raw_support'][wid]))
            if rank>best_rank:
                best_rank=rank;comparison.save(dest/'preview.png');write(dest/'preview.json',dict(window_id=wid,selection='maximum absolute detection-count difference; no GT selection'))
            if seq=='20' and wid==817:comparison.save(Path(out)/'T174_regression_comparison.png')
            if wid%500==0:
                write(dest/'progress.json',dict(window=wid+1,total=min(limit,len(windows)) if limit else len(windows),elapsed_s=time.monotonic()-start_clock))
                print(seq,wid+1,'/',min(limit,len(windows)) if limit else len(windows),flush=True)
    finally:video.close()
    count=len(stats);assert count==(min(limit,len(windows)) if limit else len(windows))
    selected=[frame for frame in aligned if frame['window_id']<count];scores=[]
    for method,boxes in predictions.items():
        score,frames=evaluate(selected,boxes,'center');scores.append(dict(sequence=seq,method=method,protocol='center',**score))
        dump_csv(dest/(method+'_center_frames.csv'),frames)
    dump_csv(dest/'metrics.csv',scores);dump_csv(dest/'raw_support_windows.csv',stats);dump_csv(dest/'filtered_detections.csv',detections)
    (dest/'Detection_compare.partial.mp4').replace(dest/'Detection_compare.mp4')
    summary=dict(sequence=seq,status='complete',windows=count,fresh_original_windows_checked=checked,elapsed_s=time.monotonic()-start_clock,
        raw_components_removed=sum(s.get('raw_components_removed',0) for s in stats),
        raw_unique_pixels_removed=sum(s.get('raw_unique_pixels_removed',0) for s in stats),raw_events_removed=sum(s.get('raw_events_removed',0) for s in stats))
    write(dest/'summary.json',summary);print('COMPLETE',json.dumps(summary),flush=True)


def finalize(out,sequences,minimum,limit):
    rows=[]
    for seq in sequences:
        with (out/seq/'metrics.csv').open() as f:
            for row in csv.DictReader(f):rows.append(dict(sequence=seq,method=row['method'],**{k:int(row[k]) for k in ['tp','fp','fn','frames']}))
    aggregate=[]
    for split,seqs in [('calibration',CALIBRATION),('holdout',HOLDOUT),('all',sequences)]:
        for method in ['original','raw_support','before_persistence']:
            subset=[r for r in rows if r['sequence'] in seqs and r['method']==method]
            if subset:aggregate.append(dict(split=split,method=method,protocol='center',frames=sum(r['frames'] for r in subset),**metrics(*(sum(r[k] for r in subset) for k in ['tp','fp','fn']))))
    dump_csv(out/'metrics_aggregate.csv',aggregate)
    listing=out/'concat.txt';listing.write_text(''.join(f"file '{seq}/Detection_compare.mp4'\n" for seq in sequences))
    with (out/'concat.log').open('w') as log:
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','concat','-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart',str(out/'Detection_compare_all.mp4')],stdout=log,stderr=log,check=True)
    percent=lambda v:'N/A' if v is None else f'{v*100:.2f}%'
    lines=['# 扩张前真实像素覆盖过滤对比','',f'新门限：每个 30 ms 窗口中，8 邻域连通区域至少覆盖 {minimum} 个不同的原始事件坐标。次数和极性均不重复计数，先剔除不合格区域，再构造正负事件图并扩张。',
        '上一版显著图面积过滤关闭（saliency_min_area=0）；周期性继续读取原始候选事件，持续性配置不变。仅改变显著图的输入支持，未改变后两级评分规则。',
        '原版使用已验证的冻结检测，每段前 12 窗重新从 H5 核验；两侧持续性独立从第 0 窗重放。评价采用原有一对一中心匹配，不使用整机 IoU 排名。',
        f'素材顺序：{", ".join(sequences)}；'+(f'每段前 {limit} 窗，仅为试运行。' if limit else '全部时间窗，无截断。'),
        '','| 范围 | 方法 | Precision | Recall | F1 | TP | FP | FN |','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in aggregate:lines.append(f'| {r["split"]} | {r["method"]} | {percent(r["precision"])} | {percent(r["recall"])} | {percent(r["f1"])} | {r["tp"]} | {r["fp"]} | {r["fn"]} |')
    lines+=['','original：原版含持续性；raw_support：新过滤含持续性；before_persistence：新过滤后、持续性前。',
        '门限在运行前固定，未按保留测试集调参。中心匹配基于已有整机标注，不是逐旋翼标注精度。使用缓存的原版耗时不参与速度比较。',
        '逐窗原始像素与事件剔除统计见 raw_support_windows.csv；周期性结果见 filtered_detections.csv；持续性与关联记录见 persistence_debug.jsonl。',
        '完整对比视频左原版、右新过滤；原始事件画面仍显示全部事件，以便观察小点仍在输入中、但不再被判定。中文宋体、英文 Times New Roman。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(out/'status.json',dict(status='complete',sequences=sequences,min_raw_component_pixels=minimum,max_windows=limit))


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--minimum',type=int,default=3);p.add_argument('--sequences',nargs='+',default=['8','20','51','65','93']);p.add_argument('--workers',type=int,default=4);p.add_argument('--max-windows',type=int,default=0);a=p.parse_args()
    assert a.minimum>0;a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
    write(a.out/'status.json',dict(status='running',sequences=a.sequences,min_raw_component_pixels=a.minimum,max_windows=a.max_windows))
    (a.out/'source').mkdir()
    files={}
    for directory in ['evdetmav','persistence','configs','experiments']:
        shutil.copytree(ROOT/directory,a.out/'source'/directory,ignore=shutil.ignore_patterns('__pycache__'))
        for file in (ROOT/directory).glob('*.py'):files[str(file.relative_to(ROOT))]=hashlib.sha256(file.read_bytes()).hexdigest()
    write(a.out/'source_sha256.json',files)
    try:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            scheduled=sorted(a.sequences,key=lambda seq:sum(1 for _ in (ROOT/'comparison'/seq/'video_windows.csv').open()),reverse=True)
            futures=[pool.submit(run_sequence,seq,str(a.out),a.minimum,a.max_windows) for seq in scheduled]
            for future in as_completed(futures):future.result()
        finalize(a.out,a.sequences,a.minimum,a.max_windows)
    except BaseException as error:write(a.out/'status.json',dict(status='failed',error=repr(error)));raise

if __name__=='__main__':main()

"""Frozen original vs area-filtered saliency on identical full FRED ZIP/H5 windows.

No RGB input. GT is used exclusively for the existing one-to-one center metric.
Original detections are frozen CSVs validated by fresh disabled-path checks.
Both persistence trackers start at window 0 with exactly the same parameters.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,csv,json,time,hashlib,subprocess
from dataclasses import asdict,replace
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from PIL import Image
import imageio_ffmpeg
from evdetmav.cli import build_parser
from evdetmav.pipeline import process_window
from evdetmav.models import WindowResult
from evdetmav.visualization import saliency_to_rgb
from persistence.config import Config
from persistence.events import stream_windows
from persistence.tracker import Tracker
from persistence.replay import adapt,dump_csv
from persistence.baseline_adapter import detect
from persistence.evaluation import evaluate,metrics
from persistence.visualization import event_image,overlay_persistence,with_header
from experiments.run_persistence_fred import dataset,CALIBRATION,HOLDOUT


def write(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def writer(path):
    output=imageio_ffmpeg.write_frames(str(path),(2560,784),fps=100/3,macro_block_size=2,codec='libx264',
        output_params=['-crf','18','-preset','veryfast','-threads','1','-movflags','+faststart'])
    output.send(None);return output


def panel(left,right):
    image=Image.new('RGB',(2560,784));image.paste(left,(0,0));image.paste(right,(1280,0));return image


def normalized(rows):
    return [{k:(v if k=='label' else float(v)) for k,v in r.items() if k not in ('source_file','latency_ms')} for r in rows]


def run_sequence(seq,out,area,limit):
    dest=Path(out)/seq;dest.mkdir(parents=True,exist_ok=False)
    config=Config.load(ROOT/'configs/persistence_original.json')
    trackers={name:Tracker(config) for name in ['original','filtered']}
    args=build_parser().parse_args(['--input',str(ROOT/'FRED'/f'{seq}.zip'),'--width','1280','--height','720',
        '--time-unit','us','--window-ms','30','--step-ms','30','--saliency-min-area',str(area)])
    args.max_windows=limit
    oldargs=argparse.Namespace(**vars(args));oldargs.saliency_min_area=0
    windows,frozen,aligned=dataset(seq)
    predictions={name:defaultdict(list) for name in trackers};pre_predictions=defaultdict(list)
    stats=[];detections=[];best_rank=-1;start_clock=time.monotonic();checked=0
    write(dest/'config.json',dict(area_px=area,persistence=asdict(config),detector={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        windows='original inclusive [start,end] H5 windows, 30 ms',original='frozen CSV, fresh first 12 windows verified',status='running'))
    videos={name:writer(dest/(name+'.partial.mp4')) for name in ['Detection_compare','Saliency_compare']}
    try:
      with (dest/'persistence_debug.jsonl').open('w') as debug:
        for wid,start,end,raw in stream_windows(args.input,args):
            assert abs(start-float(windows[wid]['start_sec']))<1e-9 and abs(end-float(windows[wid]['end_sec']))<1e-9
            original=frozen.get(wid,[])
            if wid<12:
                fresh=[asdict(d) for d in detect(raw,start,end,1280,720,args.input,wid,oldargs)]
                assert normalized(fresh)==normalized(original),f'{seq}/{wid}: frozen original does not match fresh H5'
                checked+=1
            tic=time.perf_counter()
            if len(raw)>=args.min_window_events:
                result=process_window(raw['x'].astype(np.int64),raw['y'].astype(np.int64),raw['t']*1e-6,
                    np.where(raw['p']>0,1,-1),start,end,720,1280,str(args.input),wid,args)
                filtered=[asdict(d) for d in result.detections]
            else:
                blank=np.zeros((720,1280),np.float32)
                result=WindowResult([],blank,[],[],blank.astype(bool),dict(enabled=True,min_area_px=area,skipped_low_events=True),blank)
                filtered=[]
            detect_ms=(time.perf_counter()-tic)*1000
            records={}
            for name,raw_dets in [('original',original),('filtered',filtered)]:
                records[name],changes,pairs=trackers[name].step(end,[adapt(d,end) for d in raw_dets])
                predictions[name][wid]=[r['bbox'] for r in records[name] if r['accepted']]
                if name=='filtered':
                    debug.write(json.dumps(dict(window_id=wid,timestamp=end,rows=records[name],changes=changes,association=pairs),allow_nan=False)+'\n')
            pre_predictions[wid]=[[d['bbox_'+k] for k in ('x0','y0','x1','y1')] for d in filtered]
            detections.extend(filtered)
            stats.append(dict(window_id=wid,start_sec=start,end_sec=end,event_count=len(raw),original_candidates=len(original),
                filtered_candidates=len(filtered),original_accepted=len(predictions['original'][wid]),filtered_accepted=len(predictions['filtered'][wid]),
                filtered_detection_ms=detect_ms,**result.saliency_filter_stats))
            events=event_image(raw)
            comparison=panel(with_header(overlay_persistence(events,records['original']),'修改前：原版持续性检测',f'Seq {seq} | t={end:.3f}s | N={len(predictions["original"][wid])}'),
                with_header(overlay_persistence(events,records['filtered']),'修改后：显著性小区域过滤',f'Seq {seq} | t={end:.3f}s | A >= {area} px | N={len(predictions["filtered"][wid])}'))
            saliency=panel(with_header(Image.fromarray(saliency_to_rgb(result.raw_saliency_u8)),'修改前：显著图',f'Seq {seq} | t={end:.3f}s'),
                with_header(Image.fromarray(saliency_to_rgb(result.saliency_u8)),'修改后：显著图',f'A >= {area} px | Removed regions = {result.saliency_filter_stats.get("components_removed",0)}'))
            videos['Detection_compare'].send(np.asarray(comparison));videos['Saliency_compare'].send(np.asarray(saliency))
            rank=result.saliency_filter_stats.get('foreground_pixels_removed',0)
            if rank>best_rank:
                best_rank=rank;comparison.save(dest/'Detection_preview.png');saliency.save(dest/'Saliency_preview.png')
                np.savez_compressed(dest/'preview_saliency.npz',raw=result.raw_saliency_u8,filtered=result.saliency_u8,window_id=wid)
                write(dest/'preview.json',dict(window_id=wid,timestamp=end,selection='maximum removed foreground pixels; no GT selection',**result.saliency_filter_stats))
            if wid%250==0:
                write(dest/'progress.json',dict(window=wid+1,total=min(limit,len(windows)) if limit else len(windows),elapsed_s=time.monotonic()-start_clock))
                print(f'{seq}: {wid+1}/{min(limit,len(windows)) if limit else len(windows)}',flush=True)
    finally:
        for video in videos.values():video.close()
    count=len(stats);assert count==(min(limit,len(windows)) if limit else len(windows))
    selected=[frame for frame in aligned if frame['window_id']<count]
    scores=[]
    for name,pred in {**predictions,'filtered_before_persistence':pre_predictions}.items():
        score,frames=evaluate(selected,pred,'center');scores.append(dict(sequence=seq,method=name,protocol='center',**score))
        dump_csv(dest/(name+'_center_frames.csv'),frames)
    dump_csv(dest/'metrics.csv',scores);dump_csv(dest/'saliency_windows.csv',stats);dump_csv(dest/'filtered_detections.csv',detections)
    for name in videos:(dest/(name+'.partial.mp4')).replace(dest/(name+'.mp4'))
    summary=dict(sequence=seq,status='complete',windows=count,fresh_original_windows_checked=checked,elapsed_s=time.monotonic()-start_clock,
        removed_components=sum(s.get('components_removed',0) for s in stats),removed_pixels=sum(s.get('foreground_pixels_removed',0) for s in stats),
        original_detections=sum(s['original_candidates'] for s in stats),filtered_detections=len(detections),
        mean_filtered_detection_ms=float(np.mean([s['filtered_detection_ms'] for s in stats])))
    write(dest/'summary.json',summary);print('COMPLETE',json.dumps(summary),flush=True)
    return summary


def finalize(out,sequences,area,limit):
    rows=[]
    for seq in sequences:
        with (out/seq/'metrics.csv').open() as f:
            for row in csv.DictReader(f):
                rows.append(dict(sequence=seq,method=row['method'],protocol='center',**{key:int(row[key]) for key in ['tp','fp','fn','frames']}))
    aggregate=[]
    for split,seqs in [('calibration',CALIBRATION),('holdout',HOLDOUT),('all',sequences)]:
        for method in ['original','filtered','filtered_before_persistence']:
            subset=[r for r in rows if r['sequence'] in seqs and r['method']==method]
            if not subset:continue
            aggregate.append(dict(split=split,method=method,protocol='center',frames=sum(r['frames'] for r in subset),
                **metrics(*(sum(r[k] for r in subset) for k in ['tp','fp','fn']))))
    dump_csv(out/'metrics_aggregate.csv',aggregate)
    for name in ['Detection_compare','Saliency_compare']:
        listing=out/(name+'_concat.txt');listing.write_text(''.join(f"file '{seq}/{name}.mp4'\n" for seq in sequences))
        with (out/(name+'_concat.log')).open('w') as log:
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','concat','-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart',str(out/(name+'_all.mp4'))],stdout=log,stderr=log,check=True)
    def percent(v):return 'N/A' if v is None else f'{v*100:.2f}%'
    lines=['# 显著性小区域过滤对比','',f'有效前景阈值为 50，8 邻域连通面积小于 {area} 像素的区域在候选膨胀与合并前清零；低于显著性阈值的背景也清零。面积等于阈值保留。',
        '周期性、持续性与原始事件不变；原版结果使用已验证的冻结 CSV，并逐序列对前 12 窗进行 H5 重算核验。',
        f'序列顺序：{", ".join(sequences)}。'+('本次是每段前 '+str(limit)+' 窗预览，不是完整评测。' if limit else '每段从首窗至末窗完整处理，持续性历史不截断。'),
        '中心匹配采用既有一对一规则：预测框中心位于标注目标框内；GT 不参与检测。统计是该匹配规则下的结果，不能当作逐旋翼标注的精度。',
        '面积参数在本轮测试前固定，未按测试结果调参。','', '| 范围 | 方法 | Precision | Recall | F1 | TP | FP | FN |','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in aggregate:
        lines.append(f'| {r["split"]} | {r["method"]} | {percent(r["precision"])} | {percent(r["recall"])} | {percent(r["f1"])} | {r["tp"]} | {r["fp"]} | {r["fn"]} |')
    lines+=['','original：修改前原版含持续性；filtered：新显著性过滤＋同参数持续性；filtered_before_persistence：新过滤但尚未经过持续性。',
        '','视频左侧修改前、右侧修改后；原生分辨率每侧 1280×720，加独立标题区，33.333 fps。Detection_compare_all.mp4 为最终检测对比，Saliency_compare_all.mp4 为显著图对比。',
        '逐窗面积清理统计见各序列 saliency_windows.csv；后两级日志见 filtered_detections.csv 和 persistence_debug.jsonl；逐帧评价见 *_center_frames.csv。',
        '预览图选取清除前景像素最多的一窗，用于展示过滤行为，不用来代表平均检测效果。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(out/'status.json',dict(status='complete',sequences=sequences,area_px=area,max_windows=limit))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--area',type=int,default=9)
    parser.add_argument('--sequences',nargs='+',default=['8','20','51','65','93','114']);parser.add_argument('--workers',type=int,default=2);parser.add_argument('--max-windows',type=int,default=0)
    a=parser.parse_args();assert a.area>0;a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
    write(a.out/'status.json',dict(status='running',sequences=a.sequences,area_px=a.area,max_windows=a.max_windows))
    write(a.out/'source_sha256.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for directory in ['evdetmav','persistence'] for p in (ROOT/directory).glob('*.py')})
    (a.out/'source').mkdir()
    import shutil
    for directory in ['evdetmav','persistence','configs']:
        shutil.copytree(ROOT/directory,a.out/'source'/directory,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(__file__,a.out/'source/compare_saliency_area.py')
    try:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            # 先调度较长序列，合并顺序仍严格按用户素材编号递增。
            scheduled=sorted(a.sequences,key=lambda seq:sum(1 for _ in (ROOT/'comparison'/seq/'video_windows.csv').open()),reverse=True)
            futures=[pool.submit(run_sequence,seq,str(a.out),a.area,a.max_windows) for seq in scheduled]
            for future in as_completed(futures):future.result()
        finalize(a.out,a.sequences,a.area,a.max_windows)
    except BaseException as error:
        write(a.out/'status.json',dict(status='failed',error=repr(error)));raise

if __name__=='__main__':main()

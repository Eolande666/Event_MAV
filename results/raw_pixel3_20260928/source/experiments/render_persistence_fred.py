"""Render exact original H5 event windows; no RGB or GT is supplied to rendering."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json,csv,subprocess,time,hashlib,zipfile
from collections import defaultdict
import numpy as np
import imageio_ffmpeg
from PIL import Image
from evdetmav.cli import build_parser
from persistence.events import stream_windows
from persistence.replay import load_baseline
from persistence.visualization import event_image,overlay_baseline,overlay_persistence,with_header,debug_panel

OUT=ROOT/'results/persistence_mvp/fred_v1'
SEQUENCES=['8','20','51','65','93','114']
FF=imageio_ffmpeg.get_ffmpeg_exe()


def progress(**data):
    temp=OUT/'video_progress.tmp';temp.write_text(json.dumps(data,ensure_ascii=False,indent=2));temp.replace(OUT/'video_progress.json')


def writer(path,size):
    generator=imageio_ffmpeg.write_frames(str(path),size,fps=100/3,macro_block_size=2,codec='libx264',
        output_params=['-r','100/3','-crf','17','-preset','veryfast','-threads','2','-movflags','+faststart'])
    generator.send(None);return generator


def render_signature(seq):
    digest=hashlib.sha256()
    paths=[ROOT/'comparison'/seq/'video_windows.csv',ROOT/'comparison'/seq/'evdetmav_detections_batch.csv',
           OUT/seq/'combined/debug.jsonl',OUT/seq/'combined/track_events.jsonl',
           ROOT/'persistence/visualization.py',ROOT/'persistence/events.py',Path(__file__)]
    for path in paths:digest.update(path.read_bytes())
    with zipfile.ZipFile(ROOT/'FRED'/f'{seq}.zip') as archive:
        info=archive.getinfo(f'{seq}/Event/events.hdf5')
        digest.update(f'{info.CRC}:{info.file_size}'.encode())
    return digest.hexdigest()


def main():
    started=time.time()
    for seq in SEQUENCES:
        dest=OUT/seq/'videos';dest.mkdir(exist_ok=True)
        signature=render_signature(seq)
        if (dest/'complete.json').exists() and json.loads((dest/'complete.json').read_text()).get('render_signature')==signature:continue
        windows,baseline=load_baseline(ROOT/'comparison'/seq)
        rows=defaultdict(list);lost=defaultdict(list)
        for line in (OUT/seq/'combined/debug.jsonl').open():
            row=json.loads(line);rows[row['window_id']].append(row)
        for line in (OUT/seq/'combined/track_events.jsonl').open():
            row=json.loads(line);lost[row['window_id']].append(row)
        args=build_parser().parse_args(['--input',str(ROOT/'FRED'/f'{seq}.zip'),'--window-ms','30','--step-ms','30'])
        writers={'Persistence':writer(dest/'Persistence.partial.mp4',(1280,784)),
                 'Debug':writer(dest/'Debug.partial.mp4',(1280,928)),
                 'Comparison':writer(dest/'Comparison.partial.mp4',(2560,784))}
        best=None;best_rank=-1
        count=0
        try:
            for wid,start,end,events in stream_windows(args.input,args):
                cached=windows[wid]
                assert abs(start-float(cached['start_sec']))<1e-9 and abs(end-float(cached['end_sec']))<1e-9
                assert len(events)==int(cached['events']),(seq,wid,len(events),cached['events'])
                image=event_image(events)
                left=with_header(overlay_baseline(image,baseline.get(wid,[])),'原始检测',f'Baseline | Sequence {seq} | t={end:.3f} s | N={len(baseline.get(wid,[]))}')
                accepted=sum(r['accepted'] for r in rows[wid]);rejected=len(rows[wid])-accepted
                right=with_header(overlay_persistence(image,rows[wid]),'加入第三级持续性检验',f'Persistence | Sequence {seq} | t={end:.3f} s | Accept={accepted} Reject={rejected}')
                debug=debug_panel(with_header(overlay_persistence(image,rows[wid],debug=True,lost=lost[wid]),'持续性检验中间结果',f'Sequence {seq} | t={end:.3f} s | Green: ACCEPT  Orange: REJECT  Cross: PRED'),rows[wid])
                pair=Image.new('RGB',(2560,784));pair.paste(left,(0,0));pair.paste(right,(1280,0))
                writers['Persistence'].send(np.asarray(right));writers['Debug'].send(np.asarray(debug));writers['Comparison'].send(np.asarray(pair))
                rank=10*min(accepted,rejected)+accepted
                if rank>best_rank and wid>50:
                    best=(pair.copy(),debug.copy(),wid);best_rank=rank
                count+=1
                if wid%100==0:
                    progress(sequence=seq,frame=wid+1,total=len(windows),elapsed_sec=time.time()-started)
                    print(seq,wid+1,'/',len(windows),flush=True)
        finally:
            for wr in writers.values():wr.close()
        assert count==len(windows)
        for kind in writers:
            target=dest/(kind+'.mp4');(dest/(kind+'.partial.mp4')).replace(target)
            with (dest/(kind+'_verify.log')).open('w') as log:
                subprocess.run([FF,'-v','error','-i',str(target),'-f','null','-'],stdout=log,stderr=log,check=True)
            assert (dest/(kind+'_verify.log')).stat().st_size==0
        if best:
            best[0].save(dest/'comparison_preview.png');best[1].save(dest/'debug_preview.png')
        (dest/'complete.json').write_text(json.dumps(dict(frames=count,render_signature=signature,preview_window=best[2] if best else None,source='original ZIP H5, exact baseline windows, no RGB'),indent=2))
        print('COMPLETE',seq,flush=True)
    combined=OUT/'videos';combined.mkdir(exist_ok=True)
    for kind in ('Persistence','Debug','Comparison'):
        listing=combined/(kind+'_concat.txt')
        listing.write_text('\n'.join("file '"+str(OUT/seq/'videos'/(kind+'.mp4'))+"'" for seq in SEQUENCES))
        target=combined/(kind+'_all.mp4')
        with (combined/(kind+'_concat.log')).open('w') as log:
            subprocess.run([FF,'-y','-f','concat','-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart',str(target)],stdout=log,stderr=log,check=True)
        with (combined/(kind+'_verify.log')).open('w') as log:
            subprocess.run([FF,'-v','error','-i',str(target),'-f','null','-'],stdout=log,stderr=log,check=True)
        assert (combined/(kind+'_verify.log')).stat().st_size==0
    progress(stage='complete',sequences=SEQUENCES,elapsed_sec=time.time()-started)


if __name__=='__main__':main()

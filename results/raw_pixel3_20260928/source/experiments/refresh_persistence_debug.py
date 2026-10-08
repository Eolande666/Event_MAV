"""Re-render debug video after moving scores outside the sensor image."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json,subprocess,time
from collections import defaultdict
import numpy as np
from evdetmav.cli import build_parser
from persistence.events import stream_windows
from persistence.visualization import event_image,overlay_persistence,with_header,debug_panel
from experiments.render_persistence_fred import writer,FF,OUT,SEQUENCES,progress


def main():
    started=time.time()
    for seq in SEQUENCES:
        dest=OUT/seq/'videos';rows=defaultdict(list);lost=defaultdict(list)
        for line in (OUT/seq/'combined/debug.jsonl').open():
            r=json.loads(line);rows[r['window_id']].append(r)
        for line in (OUT/seq/'combined/track_events.jsonl').open():
            r=json.loads(line);lost[r['window_id']].append(r)
        info=json.loads((dest/'complete.json').read_text());preview=info['preview_window']
        args=build_parser().parse_args(['--input',str(ROOT/'FRED'/f'{seq}.zip'),'--window-ms','30','--step-ms','30'])
        wr=writer(dest/'Debug.revised.mp4',(1280,928));count=0
        try:
            for wid,start,end,events in stream_windows(args.input,args):
                image=event_image(events)
                debug=debug_panel(with_header(overlay_persistence(image,rows[wid],debug=True,lost=lost[wid]),
                    '持续性检验中间结果',f'Sequence {seq} | t={end:.3f} s | Green: ACCEPT  Orange: REJECT  Cross: PRED'),rows[wid])
                wr.send(np.asarray(debug));count+=1
                if wid==preview:debug.save(dest/'debug_preview.png')
                if wid%500==0:progress(stage='debug_layout_revision',sequence=seq,frame=wid+1,total=info['frames'],elapsed_sec=time.time()-started)
        finally:wr.close()
        assert count==info['frames']
        (dest/'Debug.revised.mp4').replace(dest/'Debug.mp4')
        with (dest/'Debug_verify.log').open('w') as log:
            subprocess.run([FF,'-v','error','-i',str(dest/'Debug.mp4'),'-f','null','-'],stdout=log,stderr=log,check=True)
        assert (dest/'Debug_verify.log').stat().st_size==0
        print('DEBUG COMPLETE',seq,flush=True)
    target=OUT/'videos/Debug_all.mp4'
    with (OUT/'videos/Debug_concat.log').open('w') as log:
        subprocess.run([FF,'-y','-f','concat','-safe','0','-i',str(OUT/'videos/Debug_concat.txt'),'-c','copy','-movflags','+faststart',str(target)],stdout=log,stderr=log,check=True)
    with (OUT/'videos/Debug_verify.log').open('w') as log:
        subprocess.run([FF,'-v','error','-i',str(target),'-f','null','-'],stdout=log,stderr=log,check=True)
    assert (OUT/'videos/Debug_verify.log').stat().st_size==0
    progress(stage='complete',sequences=SEQUENCES,debug_revision_elapsed_sec=time.time()-started)


if __name__=='__main__':main()

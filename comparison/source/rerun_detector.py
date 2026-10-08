import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['VECLIB_MAXIMUM_THREADS']='1'
import sys,time,json,zipfile,shutil,ctypes,subprocess,csv,math,hashlib,traceback
from pathlib import Path
from collections import defaultdict
import numpy as np,h5py,imageio_ffmpeg
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evdetmav.cli import build_parser,build_windows
from drawing import event_image,overlay
OUT=ROOT/'comparison';CACHE=ROOT/'tmp/comparison_inputs';CACHE.mkdir(parents=True,exist_ok=True)
PYTHON=sys.executable;FF=imageio_ffmpeg.get_ffmpeg_exe();started=time.time()
lib=ctypes.CDLL(str(ROOT/'tmp/fred_figures/libecf_decode.dylib'));lib.decode.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p];lib.decode.restype=ctypes.c_size_t

def atomic(p,d):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2));q.replace(p)
def progress(seq,stage,**kw):
 obj=dict(sequence=seq,stage=stage,elapsed_seconds=time.time()-started,**kw);atomic(OUT/'progress.json',obj)
 (OUT/'progress.md').write_text('# evdetmav_detector.py 重新检测\n\n'+f'序列：{seq}；阶段：{stage}\n\n'+'\n'.join(f'- {k}: {v}' for k,v in kw.items())+'\n')
def render(seq,hp,dest):
 ds=defaultdict(list)
 with open(dest/'evdetmav_detections_batch.csv') as f:
  for d in csv.DictReader(f):
   for k in ['window_id','periodicity_score','bbox_x0','bbox_y0','bbox_x1','bbox_y1']:d[k]=int(d[k])
   ds[d['window_id']].append(d)
 args=build_parser().parse_args(['--input',str(hp),'--window-ms','30','--step-ms','30'])
 with h5py.File(hp) as f:
  data=f['events'];assert len(data)>0;first=int(data[0]['t']);last=int(data[-1]['t']);windows=build_windows(np.array([first,last],dtype=np.float64)*1e-6,args)
  step=data.chunks[0];buf=data[:step];offset=step
  video=dest/'Detection.partial.mp4';wr=imageio_ffmpeg.write_frames(str(video),(1280,720),fps=100/3,macro_block_size=2,codec='libx264',output_params=['-r','100/3','-crf','16','-preset','veryfast','-threads','2','-movflags','+faststart']);wr.send(None)
  rows=[]
  try:
   for i,s,e in windows:
    # Match the CLI's seconds conversion and inclusive right window boundary exactly.
    parts=[]
    while True:
     times=buf['t']*1e-6;cut=np.searchsorted(times,e,side='right');parts.append(buf[:cut]);buf=buf[cut:]
     if len(buf) or offset>=len(data):break
     buf=data[offset:offset+step];offset+=step
    v=np.concatenate(parts) if len(parts)>1 else parts[0];v=v[v['t']*1e-6>=s]
    # Keep events at the shared boundary for the next CLI window too.
    if len(v):
     boundary=v[v['t']*1e-6>=e]
     if len(boundary):buf=np.concatenate((boundary,buf))
    im=overlay(event_image(v,1280,720),ds[i]);wr.send(np.asarray(im));rows.append((i,s,e,len(v),len(ds[i])))
    if ds[i] and i>=100 and not (dest/'preview.png').exists():im.save(dest/'preview.png')
    if i%100==0:progress(seq,'生成本轮检测视频',frame=i+1,total=len(windows))
  finally:wr.close()
  with open(dest/'video_windows.csv','w') as f:w=csv.writer(f);w.writerow(['frame','start_sec','end_sec','events','detections']);w.writerows(rows)
 video.replace(dest/'Detection.mp4')
 with open(dest/'video_verify.log','w') as log:subprocess.run([FF,'-v','error','-i',str(dest/'Detection.mp4'),'-f','null','-'],stdout=log,stderr=log,check=True)
 assert (dest/'video_verify.log').stat().st_size==0
 return len(windows)

def main():
 hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'evdetmav_detector.py',*sorted((ROOT/'evdetmav').glob('*.py'))]};atomic(OUT/'code_sha256.json',hashes)
 sources=sorted((ROOT/'FRED').glob('*.zip'),key=lambda p:int(p.stem));assert sources
 for src in sources:
  seq=src.stem;dest=OUT/seq;dest.mkdir(exist_ok=True)
  if (dest/'status.json').exists() and json.loads((dest/'status.json').read_text()).get('state')=='complete':continue
  atomic(dest/'status.json',dict(state='processing',source=str(src)));progress(seq,'无损解码原始事件')
  raw=CACHE/(seq+'_ecf.h5');decoded=CACHE/(seq+'_events.h5')
  if not decoded.exists():
   if not raw.exists():
    with zipfile.ZipFile(src) as z:
     name=next(n for n in z.namelist() if n.endswith('events.hdf5'))
     with z.open(name) as fi,open(raw,'wb') as fo:shutil.copyfileobj(fi,fo,16*1024*1024)
   with h5py.File(raw) as hf,h5py.File(decoded.with_suffix('.partial'),'w') as dst:
    data=hf['CD/events'];step=data.chunks[0];target=dst.create_dataset('events',shape=data.shape,dtype=data.dtype,chunks=(step,))
    for off in range(0,len(data),step):
     mask,compressed=data.id.read_direct_chunk((off,));n=int.from_bytes(compressed[:4],'little')>>2;v=np.empty(n,data.dtype);assert lib.decode(compressed,len(compressed),v.ctypes.data)==v.nbytes;count=min(len(v),len(data)-off);target[off:off+count]=v[:count]
    dst.attrs['source_zip']=str(src);dst.attrs['time_unit']='us';dst.attrs['geometry']='1280x720'
   decoded.with_suffix('.partial').replace(decoded);raw.unlink()
  cmd=[PYTHON,'-u',str(ROOT/'evdetmav_detector.py'),'--input',str(decoded),'--out',str(dest),'--width','1280','--height','720','--time-unit','us','--window-ms','30','--step-ms','30','--no-save-saliency','--no-save-event-frames','--no-save-segmentation','--progress-every','50']
  atomic(dest/'command.json',dict(argv=cmd,source_zip=str(src),reused_detections=False))
  progress(seq,'运行 evdetmav_detector.py')
  with open(dest/'detector.log','w') as log:
   proc=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
   for line in proc.stdout:
    log.write(line);log.flush()
    if 'window=' in line or 'windows=' in line:progress(seq,'运行 evdetmav_detector.py',latest_output=line.strip())
   code=proc.wait()
   if code:raise RuntimeError(f'{seq}: detector exit {code}, see {dest}/detector.log')
  count=render(seq,decoded,dest);atomic(dest/'status.json',dict(state='complete',source_zip=str(src),entrypoint='evdetmav_detector.py',reused_detections=False,video_frames=count));decoded.unlink();print('COMPLETE',seq,flush=True)
 progress('全部','拼接检测视频');lst=OUT/'concat.txt';lst.write_text('\n'.join("file '"+str(OUT/p.stem/'Detection.mp4')+"'" for p in sources))
 with open(OUT/'concat.log','w') as log:subprocess.run([FF,'-y','-f','concat','-safe','0','-i',str(lst),'-c','copy','-movflags','+faststart',str(OUT/'Detection_all.mp4')],stdout=log,stderr=log,check=True)
 with open(OUT/'video_verify.log','w') as log:subprocess.run([FF,'-v','error','-i',str(OUT/'Detection_all.mp4'),'-f','null','-'],stdout=log,stderr=log,check=True)
 assert (OUT/'video_verify.log').stat().st_size==0
 progress('全部','完成',sequences=[p.stem for p in sources]);print('ALL COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  progress('异常','失败',error=traceback.format_exc());raise

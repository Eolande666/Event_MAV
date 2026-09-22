import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['VECLIB_MAXIMUM_THREADS']='1'
import sys,json,time,zipfile,shutil,ctypes,csv,math,subprocess,re,io
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np,h5py,imageio_ffmpeg
from PIL import Image
from evdetmav.cli import build_parser
from evdetmav.pipeline import process_window
from drawing import event_image,overlay
OUT=ROOT/'output/FRED_current';CACHE=ROOT/'tmp/fred_current';CACHE.mkdir(exist_ok=True)
lib=ctypes.CDLL(str(ROOT/'tmp/fred_figures/libecf_decode.dylib'));lib.decode.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p];lib.decode.restype=ctypes.c_size_t
ff=imageio_ffmpeg.get_ffmpeg_exe();started=time.time()
def write(p,d):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2,default=str));q.replace(p)
def progress(seq,stage,**kw):
 write(OUT/'progress.json',dict(sequence=seq,stage=stage,elapsed_sec=time.time()-started,**kw))
 (OUT/'progress.md').write_text('# 当前素材处理进度\n\n'+f'序列：{seq}；阶段：{stage}\n\n'+ '\n'.join(f'- {k}: {v}' for k,v in kw.items())+'\n')
def writer(path,size,fps='100/3'):
 wr=imageio_ffmpeg.write_frames(str(path),size,fps=eval(fps),macro_block_size=2,codec='libx264',output_params=['-crf','16','-preset','veryfast','-threads','2','-r',fps,'-movflags','+faststart']);wr.send(None);return wr
sources=sorted((ROOT/'FRED').glob('*.zip'),key=lambda p:int(p.stem));assert [int(p.stem) for p in sources]==[8,20,51,65,93,114]
for src in sources:
 seq=src.stem;dest=OUT/seq;dest.mkdir(exist_ok=True);stfile=dest/'status.json'
 if stfile.exists() and json.loads(stfile.read_text()).get('state')=='complete':continue
 progress(seq,'读取素材');write(stfile,dict(state='processing',source=str(src)))
 with zipfile.ZipFile(src) as z:
  # Preserve and sort original RGB frame times.
  names=[n for n in z.namelist() if '/RGB/' in n and n.lower().endswith(('.jpg','.png','.jpeg'))]
  def stamp(n):
   match=re.search(r'_(\d{2})_(\d{2})_(\d{2}\.\d+)\.',n);assert match,n;hh,mm,ss=match.groups();return int(hh)*3600+int(mm)*60+float(ss)
  names.sort(key=stamp);assert names
  if not (dest/'RGB_original.mp4').exists():
   rgbw=None;timeline=[]
   try:
    for j,n in enumerate(names):
     im=Image.open(io.BytesIO(z.read(n))).convert('RGB')
     if rgbw is None:rgbsize=im.size;rgbw=writer(dest/'RGB_original.partial.mp4',rgbsize,'30')
     assert im.size==rgbsize;rgbw.send(np.asarray(im));timeline.append([j,stamp(n),n])
     if j%250==0:progress(seq,'RGB 原视频',frame=j+1,total=len(names))
   finally:
    if rgbw:rgbw.close()
   (dest/'RGB_original.partial.mp4').replace(dest/'RGB_original.mp4')
   with open(dest/'RGB_timestamps.csv','w') as f:w=csv.writer(f);w.writerow(['frame','source_time_of_day_sec','source_filename']);w.writerows(timeline)
  hname=next(n for n in z.namelist() if n.endswith('events.hdf5'));hp=CACHE/(seq+'.h5')
  if not hp.exists():
   with z.open(hname) as fi,open(hp.with_suffix('.partial'),'wb') as fo:shutil.copyfileobj(fi,fo,16*1024*1024)
   hp.with_suffix('.partial').replace(hp)
  for n in z.namelist():
   if n.endswith('/coordinates.txt'):(dest/'annotations.txt').write_bytes(z.read(n))
 args=build_parser().parse_args(['--input',str(src)]);write(dest/'parameters.json',vars(args)|{'window_ms':30,'width':1280,'height':720})
 saved=defaultdict(list);old=dest/'previous_detection';use_saved=(old/'detections.jsonl').exists()
 if use_saved:
  for ln in open(old/'detections.jsonl'):
   d=json.loads(ln);saved[d['window_id']].append(d)
  savedrows=list(csv.DictReader(open(old/'windows.csv')))
 progress(seq,'事件原视频与检测视频')
 with h5py.File(hp) as hf,open(dest/'detections.jsonl','w') as det,open(dest/'windows.csv','w') as win:
  data=hf['CD/events'];step=data.chunks[0]
  def chunk(offset):
   mask,raw=data.id.read_direct_chunk((offset,));n=int.from_bytes(raw[:4],'little')>>2;v=np.empty(n,data.dtype);assert lib.decode(raw,len(raw),v.ctypes.data)==v.nbytes;return v
  buf=chunk(0);last=chunk((len(data)-1)//step*step);t0=int(buf['t'][0]);tend=int(last['t'][-1])+1;nframes=math.ceil((tend-t0)/30000);offset=step
  geom=hf.attrs.get('geometry','1280x720');geom=geom.decode() if isinstance(geom,bytes) else geom;width,height=map(int,str(geom).split('x'));assert (width,height)==(1280,720)
  if use_saved:assert len(savedrows)==nframes
  raww=writer(dest/'Events_original.partial.mp4',(width,height));detw=writer(dest/'Detection.partial.mp4',(width,height));cw=csv.writer(win);cw.writerow(['frame','start_sec','end_sec','events','detections'])
  try:
   for i in range(nframes):
    lo=t0+i*30000;hi=min(lo+30000,tend);parts=[]
    while True:
     cut=np.searchsorted(buf['t'],hi);parts.append(buf[:cut]);buf=buf[cut:]
     if len(buf) or offset>=len(data):break
     buf=chunk(offset);offset+=step
    v=np.concatenate(parts) if len(parts)>1 else parts[0];s=lo*1e-6;e=hi*1e-6
    if use_saved:
     row=savedrows[i];assert len(v)==int(row['events']) and abs(s-float(row['start_sec']))<1e-9 and abs(e-float(row['end_sec']))<1e-9;ds=saved[i]
     for d in ds:d['source_file']=str(src)
    else:
     r=process_window(v['x'].astype(np.int64),v['y'].astype(np.int64),v['t']*1e-6,np.where(v['p']>0,1,-1),s,e,height,width,str(src),i,args);ds=[asdict(d) for d in r.detections]
    im=event_image(v,width,height);raww.send(np.asarray(im));di=overlay(im.copy(),ds);detw.send(np.asarray(di));cw.writerow([i,s,e,len(v),len(ds)])
    for d in ds:det.write(json.dumps(d)+'\n')
    if ds and i>=100 and not (dest/'detection_preview.png').exists():di.save(dest/'detection_preview.png');im.save(dest/'event_preview.png')
    if i%50==0 or i==nframes-1:win.flush();det.flush();progress(seq,'事件原视频与检测视频',frame=i+1,total=nframes,reused_verified_detections=use_saved)
  finally:raww.close();detw.close()
 for name in ['Events_original','Detection']:(dest/(name+'.partial.mp4')).replace(dest/(name+'.mp4'))
 for name in ['RGB_original','Events_original','Detection']:
  with open(dest/(name+'_verify.log'),'w') as log:subprocess.run([ff,'-v','error','-i',str(dest/(name+'.mp4')),'-f','null','-'],stdout=log,stderr=log,check=True)
  assert (dest/(name+'_verify.log')).stat().st_size==0
 write(stfile,dict(state='complete',source=str(src),rgb_frames=len(names),event_frames=nframes,event_start_sec=t0/1e6,event_end_sec=tend/1e6,event_fps='100/3',rgb_fps=30,reused_verified_detections=use_saved));hp.unlink();print('COMPLETE',seq,flush=True)
 if old.exists():shutil.rmtree(old)
combined=OUT/'combined';combined.mkdir(exist_ok=True)
for kind in ['RGB_original','Events_original','Detection']:
 progress('全部','拼接 '+kind);parts=[OUT/p.stem/(kind+'.mp4') for p in sources];lst=combined/(kind+'_concat.txt');lst.write_text('\n'.join("file '"+str(p)+"'" for p in parts));target=combined/(kind+'_all.mp4')
 with open(combined/(kind+'_concat.log'),'w') as log:subprocess.run([ff,'-y','-f','concat','-safe','0','-i',str(lst),'-c','copy','-movflags','+faststart',str(target)],stdout=log,stderr=log,check=True)
 with open(combined/(kind+'_verify.log'),'w') as log:subprocess.run([ff,'-v','error','-i',str(target),'-f','null','-'],stdout=log,stderr=log,check=True)
 assert (combined/(kind+'_verify.log')).stat().st_size==0
progress('全部','完成',sequences=[p.stem for p in sources]);print('ALL COMPLETE',flush=True)

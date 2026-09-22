import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,csv,zipfile,shutil,ctypes,time,subprocess
from collections import defaultdict
import numpy as np,h5py,imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[4];BASE=ROOT/'figures/all_processed';OUT=BASE/'clear_1280';CACHE=ROOT/'tmp/fred_clear';CACHE.mkdir(exist_ok=True)
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman.ttf',22)
lib=ctypes.CDLL(str(ROOT/'tmp/fred_figures/libecf_decode.dylib'));lib.decode.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p];lib.decode.restype=ctypes.c_size_t
ff=imageio_ffmpeg.get_ffmpeg_exe()
def status(d):
 p=OUT/'progress.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2));q.replace(p)
def event_image(v,w,h):
 idx=v['y'].astype(np.int64)*w+v['x'];pol=v['p']>0
 counts=np.bincount(np.concatenate((idx[pol],idx[~pol]+w*h)),minlength=w*h*2).reshape(2,h,w)
 values=counts[counts>0];scale=max(float(np.percentile(values,99)),1) if values.size else 1
 rgb=np.zeros((h,w,3),np.uint8);rgb[:,:,0]=(np.clip(counts[0]/scale,0,1)*255).astype(np.uint8);rgb[:,:,2]=(np.clip(counts[1]/scale,0,1)*255).astype(np.uint8)
 return Image.fromarray(rgb)
def overlay(im,ds):
 draw=ImageDraw.Draw(im);occupied=[];w,h=im.size
 for d in ds:
  x0,y0,x1,y1=[int(d[k]) for k in ['bbox_x0','bbox_y0','bbox_x1','bbox_y1']];color=(60,220,255)
  draw.rectangle((x0,y0,max(x0,x1-1),max(y0,y1-1)),outline=color,width=3)
  name=d['label'].replace('propeller_','P');label=f"{name}  {d['periodicity_score']}/6";bounds=draw.textbbox((0,0),label,font=font);tw=bounds[2]-bounds[0]+10;th=28
  options=[(x0,y0-th-3),(x1+4,y0),(x0,y1+4),(x0-tw-4,y0)]
  best=None;bestover=1e20
  for xx,yy in options:
   xx=max(0,min(w-tw,xx));yy=max(0,min(h-th,yy));rect=(xx,yy,xx+tw,yy+th);over=sum(max(0,min(rect[2],b[2])-max(rect[0],b[0]))*max(0,min(rect[3],b[3])-max(rect[1],b[1])) for b in occupied)
   if over<bestover:best=rect;bestover=over
  xx,yy,xx1,yy1=best;draw.rectangle(best,fill=(0,0,0));draw.text((xx+5,yy+2-bounds[1]),label,font=font,fill=color);occupied.append(best)
 return im
items=[]
for p in (BASE/'sequences').glob('*/status.json'):
 st=json.loads(p.read_text())
 if st['state']=='complete':items.append((int(p.parent.name.split('_')[-1]),p.parent,st))
items.sort();videos=[];total=sum(s['frames'] for _,_,s in items);done=0;started=time.time()
for num,seq,st in items:
 dst=OUT/(seq.name+'_1280.mp4');videos.append(dst);meta=dst.with_suffix('.json')
 if dst.exists() and meta.exists():done+=st['frames'];continue
 status(dict(state='extracting',sequence=seq.name,completed_frames=done,total_frames=total,elapsed_sec=time.time()-started))
 src=Path(st['source']);hp=CACHE/(seq.name+'.h5')
 if not hp.exists():
  with zipfile.ZipFile(src) as z:
   name=next(n for n in z.namelist() if n.endswith('events.hdf5'))
   with z.open(name) as fi,open(hp.with_suffix('.partial'),'wb') as fo:shutil.copyfileobj(fi,fo,16*1024*1024)
   hp.with_suffix('.partial').replace(hp)
 rows=list(csv.DictReader(open(seq/'windows.csv')));ds=defaultdict(list)
 for ln in open(seq/'detections.jsonl'):
  d=json.loads(ln);ds[d['window_id']].append(d)
 with h5py.File(hp) as f:
  data=f['CD/events'];step=data.chunks[0];geometry=f.attrs.get('geometry','1280x720');geometry=geometry.decode() if isinstance(geometry,bytes) else geometry;w,h=map(int,str(geometry).split('x'));assert (w,h)==(1280,720)
  def chunk(offset):
   mask,raw=data.id.read_direct_chunk((offset,));n=int.from_bytes(raw[:4],'little')>>2;ar=np.empty(n,data.dtype);assert lib.decode(raw,len(raw),ar.ctypes.data)==ar.nbytes;return ar
  buf=chunk(0);offset=step
  temp=dst.with_suffix('.partial.mp4');writer=imageio_ffmpeg.write_frames(str(temp),(w,h),fps=100/3,macro_block_size=2,codec='libx264',output_params=['-crf','16','-preset','veryfast','-threads','3','-r','100/3','-movflags','+faststart']);writer.send(None)
  try:
   for row in rows:
    i=int(row['frame']);hi=round(float(row['end_sec'])*1e6);parts=[]
    while True:
     cut=np.searchsorted(buf['t'],hi);parts.append(buf[:cut]);buf=buf[cut:]
     if len(buf) or offset>=len(data):break
     buf=chunk(offset);offset+=step
    v=np.concatenate(parts) if len(parts)>1 else parts[0];assert len(v)==int(row['events']),(seq.name,i,len(v),row['events'])
    im=overlay(event_image(v,w,h),ds[i]);writer.send(np.asarray(im))
    if ds[i] and i>=100 and not (OUT/(seq.name+'_preview.png')).exists():im.save(OUT/(seq.name+'_preview.png'))
    if i%100==0:status(dict(state='rendering',sequence=seq.name,frame=i+1,sequence_frames=len(rows),completed_frames=done+i+1,total_frames=total,elapsed_sec=time.time()-started))
  finally:writer.close()
 temp.replace(dst);meta.write_text(json.dumps(dict(sequence=seq.name,frames=len(rows),source=str(src),resolution=[w,h],fps='100/3',box_width=3,label_font='Times New Roman 22',method='original events plus saved detections; no detector rerun'),indent=2));done+=len(rows);hp.unlink();print('DONE',seq.name,done,'/',total,flush=True)
status(dict(state='merging',completed_frames=done,total_frames=total))
concat=OUT/'concat.txt';concat.write_text('\n'.join("file '"+str(p)+"'" for p in videos));merged=OUT/'FRED_completed_16_clear_1280.mp4'
with open(OUT/'concat.log','w') as log:subprocess.run([ff,'-y','-f','concat','-safe','0','-i',str(concat),'-i',str(BASE/'completed_chapters.ffmeta'),'-map','0:v:0','-map_chapters','1','-c','copy','-movflags','+faststart',str(merged)],stdout=log,stderr=log,check=True)
with open(OUT/'verify.log','w') as log:subprocess.run([ff,'-v','error','-i',str(merged),'-f','null','-'],stdout=log,stderr=log,check=True)
assert (OUT/'verify.log').stat().st_size==0
status(dict(state='complete',completed_frames=done,total_frames=total,elapsed_sec=time.time()-started,video=str(merged)));print('COMPLETE',merged,flush=True)

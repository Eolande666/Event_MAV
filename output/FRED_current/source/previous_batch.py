import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['VECLIB_MAXIMUM_THREADS']='1'
import sys,json,time,zipfile,ctypes,csv,math,subprocess,traceback,concurrent.futures,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np,h5py,imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
from evdetmav.cli import build_parser
from evdetmav.pipeline import process_window
from evdetmav.visualization import make_event_image,draw_boxes,saliency_to_rgb
from dataclasses import asdict
OUT=ROOT/'figures/all_processed';CACHE=ROOT/'tmp/fred_batch';CACHE.mkdir(exist_ok=True)
FONT='/System/Library/Fonts/Supplemental/Times New Roman.ttf'
def atomic(p,d):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2));tmp.replace(p)
def run(item):
 src=ROOT/item['path'];key=src.parent.name+'_'+src.stem;dest=OUT/'sequences'/key;dest.mkdir(parents=True,exist_ok=True);status=dest/'status.json';final=dest/(key+'.mp4')
 if status.exists() and json.loads(status.read_text()).get('state')=='complete' and final.exists():return key
 start=time.time();hpath=CACHE/(key+'.h5');writer=None
 try:
  atomic(status,dict(state='extracting',source=str(src),started=start))
  with zipfile.ZipFile(src) as z:
   if not hpath.exists():
    with z.open(item['h5']) as fi,open(hpath.with_suffix('.partial'),'wb') as fo:shutil.copyfileobj(fi,fo,16*1024*1024)
    hpath.with_suffix('.partial').replace(hpath)
   ann=[]
   for n in z.namelist():
    if n.endswith('/coordinates.txt'):
     for ln in z.read(n).decode().splitlines():
      try:ts,rest=ln.split(':',1);ann.append((float(ts),[float(v) for v in rest.split(',')[:4]]))
      except ValueError:pass
   ann.sort();ats=np.array([r[0] for r in ann])
  lib=ctypes.CDLL(str(ROOT/'tmp/fred_figures/libecf_decode.dylib'));lib.decode.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p];lib.decode.restype=ctypes.c_size_t
  args=build_parser().parse_args(['--input',str(src)]);font=ImageFont.truetype(FONT,21)
  with h5py.File(hpath) as hf,open(dest/'detections.jsonl','w') as det,open(dest/'windows.csv','w') as win:
   d=hf['CD/events'];step=d.chunks[0]
   def chunk(offset):
    mask,raw=d.id.read_direct_chunk((offset,));n=int.from_bytes(raw[:4],'little')>>2;ar=np.empty(n,dtype=d.dtype);size=lib.decode(raw,len(raw),ar.ctypes.data);assert size==ar.nbytes;return ar
   first=chunk(0);last=chunk((len(d)-1)//step*step);t0=int(first['t'][0]);tend=int(last['t'][-1])+1;nframes=math.ceil((tend-t0)/30000)
   geom=hf.attrs.get('geometry','1280x720');geom=geom.decode() if isinstance(geom,bytes) else geom;width,height=map(int,str(geom).split('x'))
   writer=imageio_ffmpeg.write_frames(str(dest/(key+'.partial.mp4')),(1280,760),fps=100/3,macro_block_size=2,codec='libx264',output_params=['-crf','20','-preset','veryfast','-threads','1']);writer.send(None)
   cw=csv.writer(win);cw.writerow(['frame','start_sec','end_sec','events','candidates','detections','processing_ms'])
   buf=first;offset=step;prev=t0
   for i in range(nframes):
    lo=t0+i*30000;hi=min(lo+30000,tend);parts=[]
    while True:
     cut=np.searchsorted(buf['t'],hi,side='left');parts.append(buf[:cut]);buf=buf[cut:]
     if len(buf) or offset>=len(d):break
     nxt=chunk(offset);offset+=step
     if len(nxt) and int(nxt['t'][0])<prev:raise ValueError('Nonmonotonic timestamps')
     if len(nxt):prev=int(nxt['t'][-1])
     buf=nxt
    v=np.concatenate(parts) if len(parts)>1 else parts[0];x=v['x'].astype(np.int64);y=v['y'].astype(np.int64);t=v['t']*1e-6;p=np.where(v['p']>0,1,-1);s=lo*1e-6;e=hi*1e-6
    tick=time.time();r=process_window(x,y,t,p,s,e,height,width,str(src),i,args)
    ev=draw_boxes(make_event_image(x,y,p,height,width,99),r.detections,r.candidates[:4],r.refined);sal=draw_boxes(Image.fromarray(saliency_to_rgb(r.saliency_u8)),r.detections,r.candidates[:4],r.refined)
    frame=Image.new('RGB',(1280,760),'white');frame.paste(ev.resize((640,360)),(0,40));frame.paste(sal.resize((640,360)),(640,40));dr=ImageDraw.Draw(frame);dr.text((12,8),f'FRED {key} | {s:.3f}-{e:.3f} s | Events / Saliency | frame {i}',font=font,fill='black')
    if ann:
     ai=int(np.clip(np.searchsorted(ats,(s+e)/2),0,len(ats)-1))
     if ai and abs(ats[ai-1]-(s+e)/2)<abs(ats[ai]-(s+e)/2):ai-=1
     gt=ann[ai][1];cx=(gt[0]+gt[2])/2;cy=(gt[1]+gt[3])/2
    else:cx=width/2;cy=height/2
    bx=max(0,min(width-280,int(cx-140)));by=max(0,min(height-160,int(cy-80)));crop=(bx,by,bx+280,by+160);frame.paste(ev.crop(crop).resize((630,360)),(5,400));frame.paste(sal.crop(crop).resize((630,360)),(645,400));writer.send(np.asarray(frame))
    for dd in r.detections:det.write(json.dumps(asdict(dd))+'\n')
    cw.writerow([i,s,e,len(v),len(r.candidates),len(r.detections),round((time.time()-tick)*1000,3)])
    if i%25==0 or i==nframes-1:
     det.flush();win.flush();atomic(status,dict(state='processing',source=str(src),frame=i+1,total_frames=nframes,elapsed_sec=time.time()-start,started=start))
    if i==min(100,nframes-1):frame.save(dest/'preview.jpg')
   writer.close();writer=None
  # Read every encoded frame to ensure finalized video is decodable.
  reader=imageio_ffmpeg.read_frames(str(dest/(key+'.partial.mp4')));meta=next(reader);count=sum(1 for _ in reader);assert count==nframes,(count,nframes)
  (dest/(key+'.partial.mp4')).replace(final);atomic(status,dict(state='complete',source=str(src),frames=nframes,duration_sec=nframes*.03,elapsed_sec=time.time()-start,video=str(final)));hpath.unlink(missing_ok=True);return key
 except Exception:
  if writer:
   try:writer.close()
   except Exception:pass
  atomic(status,dict(state='failed',source=str(src),error=traceback.format_exc(),elapsed_sec=time.time()-start));raise

def main():
 items=json.loads((ROOT/'tmp/fred_figures/inventory.json').read_text());items.sort(key=lambda d:(int(Path(d['path']).stem),Path(d['path']).parent.name));atomic(OUT/'inventory.json',items)
 errors=[]
 for it in items:
  try:print('COMPLETE',run(it),flush=True)
  except Exception as exc:
   errors.append(it['path']);print('FAILED',it['path'],str(exc),flush=True)
   break
 if errors:atomic(OUT/'batch_status.json',dict(state='failed',failed=errors));return
 videos=[];chapters=[];elapsed=0
 for it in items:
  p=Path(it['path']);key=p.parent.name+'_'+p.stem;dest=OUT/'sequences'/key;st=json.loads((dest/'status.json').read_text());videos.append(dest/(key+'.mp4'));chapters.append(dict(sequence=key,start_sec=elapsed,duration_sec=st['duration_sec']));elapsed+=st['duration_sec']
 concat=OUT/'concat.txt';concat.write_text('\n'.join("file '"+str(p)+"'" for p in videos));atomic(OUT/'chapters.json',chapters)
 ff=imageio_ffmpeg.get_ffmpeg_exe();full=OUT/'FRED_all_complete.mp4';subprocess.run([ff,'-y','-f','concat','-safe','0','-i',str(concat),'-c','copy','-movflags','+faststart',str(full)],check=True,stdout=subprocess.DEVNULL,stderr=open(OUT/'concat.log','w'))
 reader=imageio_ffmpeg.read_frames(str(full));meta=next(reader);reader.close();atomic(OUT/'batch_status.json',dict(state='complete',sequences=len(videos),duration_sec=elapsed,video=str(full),metadata=meta));print('ALL COMPLETE',full,flush=True)
if __name__=='__main__':main()

from pathlib import Path
import sys,pickle,json,csv
import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from evdetmav.models import events_in_box
from evdetmav.periodicity import slice_local_images,normalized_vector,cosine_similarity,principal_direction,moving_average
from evdetmav.visualization import make_event_image,draw_boxes,saliency_to_rgb
OUT=ROOT/'figures/processed_video';PDF=ROOT/'output/pdf';TMP=ROOT/'tmp/fred_figures'
args,rr=pickle.load(open(TMP/'results.pkl','rb'));a=np.load(TMP/'events_129_34.5_36.2.npy')
pdfmetrics.registerFont(TTFont('Songti',str(ROOT/'figures/fonts/SongtiSC-Regular.ttf')));pdfmetrics.registerFont(TTFont('TNR','/System/Library/Fonts/Supplemental/Times New Roman.ttf'))
W,H=900,620

def txt(x,y,s,size=13,color=(.13,.18,.24)):
 c.setFillColorRGB(*color)
 for ch in s:
  f='Songti' if ord(ch)>=0x2e80 else 'TNR';c.setFont(f,size);c.drawString(x,H-y,ch);x+=pdfmetrics.stringWidth(ch,f,size)
def begin(name,title,sub):
 global c
 c=canvas.Canvas(str(PDF/name),pagesize=(W,H));c.setTitle(title);txt(30,35,title,22);txt(30,60,sub,12)
def end():
 txt(30,594,'数据来源：FRED [1]；第 129 段测试序列。图元按处理视频的对应窗口数据矢量重绘。',10)
 txt(30,609,'中文：宋体（宋体-简）；英文及数字：Times New Roman。未修改检测算法。',9);c.save()
def box(x,y,w,h,color=(.8,.83,.86)):
 c.setStrokeColorRGB(*color);c.setLineWidth(.6);c.rect(x,H-y-h,w,h,fill=0,stroke=1)
def pix(rgb,x,y,w,h):
 # Exact RGB cell representation, no raster embedded in PDF.
 hh,ww=rgb.shape[:2];sc=min(w/ww,h/hh);x+=(w-ww*sc)/2;y+=(h-hh*sc)/2
 c.setFillColorRGB(0,0,0);c.rect(x,H-y-hh*sc,ww*sc,hh*sc,fill=1,stroke=0)
 flat=rgb.reshape(-1,3);colors=np.unique(flat,axis=0)
 for col in colors:
  if not np.any(col):continue
  yy,xx=np.where(np.all(rgb==col,axis=2));p=c.beginPath()
  for xx1,yy1 in zip(xx,yy):p.rect(x+xx1*sc,H-y-(yy1+1)*sc,sc,sc)
  c.setFillColorRGB(*(col/255));c.drawPath(p,fill=1,stroke=0)
 return x,y,sc

def window(i):
 s,e,r=rr[i];v=a[(a['t']>=round(s*1e6))&(a['t']<round(e*1e6))];return v['x'].astype(int),v['y'].astype(int),v['t']*1e-6,np.where(v['p']>0,1,-1),s,e,r

def features(i,ci):
 x,y,t,p,s,e,r=window(i);cand=r.candidates[ci];inside=events_in_box(x,y,cand.box);x,y,t,p=x[inside],y[inside],t[inside],p[inside];im,pts=slice_local_images(x,y,t,p,cand.box,s,e,12);ids=np.clip(np.floor((t-s)/(e-s)*12).astype(int),0,11);density=np.array([sum((ids==j)&(p>0)) for j in range(12)]);vs=[normalized_vector(z) for z in im];ds=[principal_direction(z,4) for z in pts]
 structure=np.array([cosine_similarity(vs[j],vs[j+1]) if vs[j] is not None and vs[j+1] is not None else np.nan for j in range(11)]);direction=np.array([abs(cosine_similarity(ds[j],ds[j+1])) if ds[j] is not None and ds[j+1] is not None else np.nan for j in range(11)])
 return cand,[density,structure,direction]

def curve(v,x,y,w,h,normalized=False):
 vals=v[np.isfinite(v)];sm=moving_average(vals,3);amp=np.ptp(sm);z=(sm-sm.min())/max(amp,1e-12) if normalized else vals;lo,hi=(0,1) if normalized else (float(z.min()),float(z.max()));hi=max(hi,lo+1e-6);box(x,y,w,h)
 def coord(j,val):return x+j*w/max(len(z)-1,1),H-y-h+(val-lo)/(hi-lo)*h
 c.setLineWidth(1.3);c.setStrokeColorRGB(.14,.36,.6);path=c.beginPath()
 for j,val in enumerate(z):
  xx,yy=coord(j,val)
  if j==0:path.moveTo(xx,yy)
  else:path.lineTo(xx,yy)
 c.drawPath(path);txt(x-24,y+8,f'{hi:.5g}',8);txt(x-24,y+h,f'{lo:.5g}',8);txt(x,y+h+12,'1',8);txt(x+w-10,y+h+12,str(len(z)),8)
 peaks=find_peaks(z,prominence=.15)[0] if normalized and amp>=.05 else [];valleys=find_peaks(1-z,prominence=.15)[0] if normalized and amp>=.05 else []
 for indexes,color in [(peaks,(.85,.15,.12)),(valleys,(.08,.6,.4))]:
  c.setFillColorRGB(*color)
  for j in indexes:xx,yy=coord(j,z[j]);c.circle(xx,yy,3,fill=1,stroke=0)
 return int(len(peaks)>0),int(len(valleys)>0),float(amp)

# Processing video uses every original event, original output boxes and saliency.
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman.ttf',22)
writer=imageio_ffmpeg.write_frames(str(OUT/'materials/FRED129_processed.mp4'),(1280,760),fps=100/3,macro_block_size=2,codec='libx264',pix_fmt_in='rgb24',output_params=['-crf','18']);writer.send(None)
all_det=[]
for i,(s,e,r) in enumerate(rr):
 x,y,t,p,_,_,_=window(i);ev=draw_boxes(make_event_image(x,y,p,720,1280,99),r.detections,r.candidates[:4],r.refined);sal=draw_boxes(Image.fromarray(saliency_to_rgb(r.saliency_u8)),r.detections,r.candidates[:4],r.refined)
 frame=Image.new('RGB',(1280,760),'white');frame.paste(ev.resize((640,360)),(0,40));frame.paste(sal.resize((640,360)),(640,40));dd=ImageDraw.Draw(frame);dd.text((15,8),f'FRED 129 | frame {i} | {s:.3f}-{e:.3f} s | events / saliency',font=font,fill='black')
 # Fixed crop tracks drone vicinity in chosen segment; all events remain visible above.
 anns=[]
 for line in (TMP/'129/coordinates.txt').read_text().splitlines():
  try:ts,rest=line.split(':',1);coords=[float(z.strip()) for z in rest.split(',')[:4]];anns.append((abs(float(ts)-(s+e)/2),coords))
  except ValueError:pass
 gt=min(anns)[1];cx=(gt[0]+gt[2])/2;cy=(gt[1]+gt[3])/2;bx=max(0,min(1000,int(cx-140)));by=max(0,min(560,int(cy-80)));crop=(bx,by,bx+280,by+160)
 frame.paste(ev.crop(crop).resize((630,360)),(5,400));frame.paste(sal.crop(crop).resize((630,360)),(645,400));writer.send(np.asarray(frame))
 if i in (29,30):frame.save(OUT/f'materials/frame_{i:03d}.png')
 for d in r.detections:all_det.append(vars(d))
writer.close()
with open(OUT/'materials/detections.json','w') as f:json.dump(all_det,f,indent=2)
# Main example is an actual retained candidate on UAV.
x,y,t,p,s,e,r=window(29);cand,ff=features(29,0);b=cand.box;crop=(b[0]-12,b[1]-18,b[2]+12,b[3]+18);x0,y0,x1,y1=crop
rgb=np.asarray(make_event_image(x,y,p,720,1280,99));roi=rgb[y0:y1,x0:x1];sal=r.saliency_u8[y0:y1,x0:x1]
np.savez_compressed(OUT/'materials/selected_window.npz',x=x,y=y,t=t,p=p,saliency=r.saliency_u8,segmentation=r.segmentation_mask,candidate_box=b)
begin('fig5_real_events.pdf','真实无人机区域的红蓝事件','视频第 29 帧（从 0 计数） | 35.370–35.400 s | 时间窗口 30 ms')
# Downsample by binning original pixel colors only for overview vector size.
small=np.asarray(Image.fromarray(rgb).resize((640,360),resample=Image.Resampling.NEAREST));ox,oy,sc=pix(small,30,90,480,270)
c.setStrokeColorRGB(.94,.62,.05);c.setLineWidth(1.5);c.rect(ox+x0/2*sc,H-oy-y1/2*sc,(x1-x0)/2*sc,(y1-y0)/2*sc)
pix(roi,530,90,340,270);txt(30,390,'（a）处理事件帧及局部位置',15);txt(530,390,'（b）无人机候选区域放大',15)
txt(30,430,'红色：正极性事件；蓝色：负极性事件；同像素两者叠加呈紫色。',14)
txt(30,457,'放大区域保留原始像素，不添加旋翼轨迹或人造事件。',14)
txt(30,484,'可见事件来自机体、旋翼及运动背景；数据集未提供单片桨叶的真值。',14)
txt(30,530,f'候选框：{b}；周期性峰谷分数：{cand.periodicity.score}/6。',12);end()
# Ten intersections exactly matching saliency implementation, full-frame dilation before crop.
ids=np.clip(np.floor((t-s)/(e-s)*10).astype(int),0,9);inter=[]
for j in range(10):
 pos=np.zeros((720,1280),bool);neg=pos.copy();ii=(ids==j)&(p>0);pos[y[ii],x[ii]]=1;ii=(ids==j)&(p<0);neg[y[ii],x[ii]]=1;inter.append((ndimage.binary_dilation(pos,structure=np.ones((3,3)))&ndimage.binary_dilation(neg,structure=np.ones((3,3))))[y0:y1,x0:x1])
assert np.allclose(np.sum(inter,axis=0)*25.5,sal)
begin('fig6_saliency_accumulation.pdf','显著图：逐片求交与累积','同一视频窗口 | 10 个时间片，每片 3 ms | 先对正、负事件各膨胀 1 次，再求交')
for j,z in enumerate(inter):
 xx=30+(j%5)*174;yy=90+(j//5)*126;pix(np.repeat((z*255).astype('uint8')[...,None],3,axis=2),xx,yy,158,85);txt(xx,yy+104,f'第 {j+1} 片：{int(z.sum())} 像素',11)
for k,xx in [(1,30),(5,320),(10,610)]:
 accum=np.sum(inter[:k],axis=0)*25.5;pix(saliency_to_rgb(accum),xx,361,260,138);txt(xx,521,f'累计前 {k} 片：最大值 {accum.max():.1f}',12)
txt(30,560,'S(x,y) = (255/10) × Σ Ij(x,y)；白色表示交集，热色表示累计显著性。',13);end()
begin('fig7_periodicity_comparison.pdf','周期性证据对比：通过与未通过','真实相邻窗口中的无人机候选；代码检测峰谷，不等同于严格的周期存在性检验。')
for col,(ii,ci) in enumerate([(29,0),(30,1)]):
 cc,fs=features(ii,ci);xx=50+col*440;txt(xx,94,f'第 {ii} 帧 | {rr[ii][0]:.3f}–{rr[ii][1]:.3f} s',14);txt(xx,120,f'峰谷分数 {cc.periodicity.score}/6：'+('通过阈值 3' if cc.periodicity.score>=3 else '未通过阈值 3'),14)
 for j,(name,v) in enumerate(zip(['正事件数量','相邻片结构相似度','相邻片主方向相似度'],fs)):
  yy=152+j*130;txt(xx,yy,name,12);curve(v,xx+25,yy+15,330,75)
 txt(xx,558,'横轴：时间片或相邻片对；纵轴：原始特征值。',10)
end()
begin('fig8_periodicity_scoring.pdf','周期性评分：从特征序列到峰谷计分','第 29 帧的同一候选 | 平滑窗口 3 | 最小幅度 0.05 | 峰谷突出度 0.15')
rows=[]
for j,(name,v) in enumerate(zip(['正事件数量','结构相似度','主方向相似度'],ff)):
 yy=104+j*132;txt(30,yy,name,14);peak,valley,amp=curve(v,170,yy-12,460,82,True);txt(662,yy+3,f'幅度：{amp:.3g}',12);txt(662,yy+27,f'有峰 {peak} + 有谷 {valley} = {peak+valley}',14);rows.append((name,peak,valley,amp))
 for k,value in enumerate(v):pass
total=sum(z[1]+z[2] for z in rows);assert total==cand.periodicity.score
txt(30,505,'去除非有限值 → 三点滑动平均（边界补零）→ 幅度检查 → 归一化 → 找峰谷',14)
txt(30,536,f'总分 = '+ ' + '.join(str(z[1]+z[2]) for z in rows)+f' = {total}；总分 ≥ 3，进入精细定位。',16)
txt(30,563,'红点：峰；绿点：谷。同一特征有多个峰或谷时，对应项仍只计 1 分。',12);end()
with open(OUT/'materials/features.csv','w') as f:
 w=csv.writer(f);w.writerow(['frame','feature','index','raw_value'])
 for ii,ci in [(29,0),(30,1)]:
  cc,fs=features(ii,ci)
  for name,v in zip(['density','structure','direction'],fs):
   for j,val in enumerate(v):w.writerow([ii,name,j+1,val])
print('finished',rows,flush=True)

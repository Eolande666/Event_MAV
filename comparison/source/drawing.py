import numpy as np
from PIL import Image,ImageDraw,ImageFont
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman.ttf',22)
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

"""Native-resolution event rendering; Songti Chinese and Times New Roman Latin."""
from pathlib import Path
import os
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
def load_font(size, chinese=False):
    """Use installed fonts instead of requiring a previous author's font files."""
    candidates = ([os.environ.get('EVDETMAV_CHINESE_FONT'),
                   str(ROOT/'figures/fonts/SongtiSC-Regular.ttf'),
                   '/System/Library/Fonts/Supplemental/Songti.ttc',
                   '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']
                  if chinese else
                  [os.environ.get('EVDETMAV_LATIN_FONT'),
                   '/System/Library/Fonts/Supplemental/Times New Roman.ttf',
                   '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'])
    for candidate in candidates:
        if candidate:
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                pass
    return ImageFont.load_default(size=size)

LATIN=load_font(20)
TITLE=load_font(24)
CHINESE=load_font(24, chinese=True)


def event_image(events,width=1280,height=720):
    indices=events['y'].astype(np.int64)*width+events['x'];positive=events['p']>0
    counts=np.bincount(np.concatenate((indices[positive],indices[~positive]+width*height)),minlength=2*width*height).reshape(2,height,width)
    nonzero=counts[counts>0];scale=max(float(np.percentile(nonzero,99)),1) if nonzero.size else 1
    rgb=np.zeros((height,width,3),np.uint8)
    rgb[:,:,0]=(np.minimum(counts[0]/scale,1)*255).astype(np.uint8)
    rgb[:,:,2]=(np.minimum(counts[1]/scale,1)*255).astype(np.uint8)
    return Image.fromarray(rgb)


def box_from_raw(raw):
    return tuple(int(raw['bbox_'+k]) for k in ('x0','y0','x1','y1'))


def label(draw,box,text,color,occupied):
    x0,y0,x1,y1=box
    width=int(draw.textlength(text,font=LATIN))+8;height=26
    options=[(x0,y0-height-3),(x1+4,y0),(x0,y1+3),(x0-width-4,y0)]
    candidates=[]
    for x,y in options:
        x=max(0,min(1280-width,x));y=max(0,min(720-height,y))
        rect=(x,y,x+width,y+height)
        overlap=sum(max(0,min(rect[2],b[2])-max(rect[0],b[0]))*max(0,min(rect[3],b[3])-max(rect[1],b[1])) for b in occupied)
        candidates.append((overlap,rect))
    _,rect=min(candidates,key=lambda v:v[0]);occupied.append(rect)
    draw.rectangle(rect,fill='black');draw.text((rect[0]+4,rect[1]+1),text,font=LATIN,fill=color)


def overlay_baseline(image,raws):
    image=image.copy();draw=ImageDraw.Draw(image);occupied=[]
    for raw in raws:
        box=box_from_raw(raw);x0,y0,x1,y1=box;color=(60,220,255)
        draw.rectangle((x0,y0,x1-1,y1-1),outline=color,width=2)
        label(draw,box,f"{raw['label'].replace('propeller_','P')} Sp={int(raw['periodicity_score'])}/6",color,occupied)
    return image


def value(x):return 'N/A' if x is None else f'{x:.2f}'


def overlay_persistence(image,rows,debug=False,lost=()):
    image=image.copy();draw=ImageDraw.Draw(image);occupied=[]
    for row in rows:
        if not debug and not row['accepted']:continue
        box=tuple(int(v) for v in row['bbox']);x0,y0,x1,y1=box
        color=(60,255,130) if row['accepted'] else (255,170,60)
        if debug:
            history=[tuple(v) for v in row['history']]
            if len(history)>1:draw.line(history,fill=color,width=1)
            pred=row['predicted_center'] or row['association_prediction']
            if pred is not None:
                px,py=pred
                if 0<=px<1280 and 0<=py<720:
                    draw.line((px-4,py,px+4,py),fill='white',width=1)
                    draw.line((px,py-4,px,py+4),fill='white',width=1)
                    draw.text((px+5,py+4),'PRED',font=LATIN,fill='white')
        draw.rectangle((x0,y0,x1-1,y1-1),outline=color,width=2)
        text=f"T{row['track_id']} {'ACCEPT' if row['accepted'] else 'REJECT'} Sc={value(row['persistence_score'])}"
        label(draw,box,text,color,occupied)
    if debug:
        for entry in lost:
            if entry['state']!='Lost':continue
            px,py=entry['predicted_center']
            if 0<=px<1280 and 0<=py<720:
                draw.line((px-4,py,px+4,py),fill=(180,180,180),width=1)
                draw.line((px,py-4,px,py+4),fill=(180,180,180),width=1)
                draw.text((px+5,py),f"T{entry['track_id']} LOST PRED",font=LATIN,fill=(180,180,180))
    return image


def with_header(image,chinese,english):
    out=Image.new('RGB',(image.width,image.height+64),(12,12,12));out.paste(image,(0,64))
    draw=ImageDraw.Draw(out);draw.text((12,2),chinese,font=CHINESE,fill='white')
    draw.text((12,32),english,font=TITLE,fill='white')
    return out


def debug_panel(image,rows):
    """Score table outside sensor image: never erase events or target boxes."""
    panel=Image.new('RGB',(image.width,image.height+144),(12,12,12))
    panel.paste(image,(0,0));draw=ImageDraw.Draw(panel)
    for i,row in enumerate(rows):
        text=f"T{row['track_id']} {row['track_state']}  Sp={int(row['periodicity_score'])}/6  Sn={value(row['neighborhood_score'])}  St={value(row['trajectory_score'])}  Sc={value(row['persistence_score'])}"
        draw.text((8,image.height+6+28*i),text,font=LATIN,fill=(60,255,130) if row['accepted'] else (255,170,60))
    return panel

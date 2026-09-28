"""Reproduce screenshot T174 from actual ZIP/H5, never a generated fixture."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json
import numpy as np
from PIL import Image,ImageDraw
from evdetmav.cli import build_parser
from evdetmav.pipeline import process_window
from persistence.events import stream_windows
from persistence.visualization import CHINESE,LATIN
from evdetmav.visualization import saliency_to_rgb


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);a=parser.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    args=build_parser().parse_args(['--input',str(ROOT/'FRED/20.zip'),'--window-ms','30','--step-ms','30','--max-windows','818'])
    details=[];maps=[]
    for wid,start,end,events in stream_windows(args.input,args):
        if wid!=817:continue
        x=events['x'].astype(np.int64);y=events['y'].astype(np.int64);t=events['t']*1e-6;p=np.where(events['p']>0,1,-1)
        inside=(x>=20)&(x<23)&(y>=409)&(y<412);pixels=np.unique(np.column_stack((x[inside],y[inside])),axis=0)
        assert len(pixels)==1 and len(x[inside])==23 and pixels.tolist()==[[21,410]]
        for mode,area,minimum in [('original',0,0),('expanded_area9',9,0),('raw_pixels3',0,3)]:
            config=argparse.Namespace(**vars(args));config.saliency_min_area=area;config.min_raw_component_pixels=minimum
            result=process_window(x,y,t,p,start,end,720,1280,str(args.input),wid,config)
            covering=[d for d in result.detections if d.bbox_x0<=21<d.bbox_x1 and d.bbox_y0<=410<d.bbox_y1]
            patch=result.saliency_u8[407:414,18:25].copy();maps.append(patch)
            details.append(dict(mode=mode,raw_unique_pixels=1,raw_events=23,salient_pixels_in_original_box=int(np.count_nonzero(result.saliency_u8[409:412,20:23])),
                target_point_detected=bool(covering),periodicity_scores=[d.periodicity_score for d in covering],saliency_stats=result.saliency_filter_stats))
        assert details[0]['target_point_detected'] and details[1]['target_point_detected']
        assert not details[2]['target_point_detected'] and details[2]['salient_pixels_in_original_box']==0
        np.savez_compressed(a.out/'T174_saliency_arrays.npz',original=maps[0],area9=maps[1],raw3=maps[2],original_events=events[inside])
        # 每个原始图像像素显示为 36×36 方块，最近邻放大，无生成或插值补点。
        canvas=Image.new('RGB',(840,360),'white');draw=ImageDraw.Draw(canvas)
        for i,(patch,title) in enumerate(zip(maps,['原版显著图','上一版面积过滤','真实像素覆盖过滤'])):
            draw.text((i*280+8,8),title,font=CHINESE,fill='black')
            draw.text((i*280+8,42),'1 raw pixel / 23 events',font=LATIN,fill='black')
            image=Image.fromarray(np.repeat(np.repeat(saliency_to_rgb(patch),36,axis=0),36,axis=1))
            canvas.paste(image,(i*280+8,80))
        canvas.save(a.out/'T174_saliency_comparison.png')
        record=dict(status='passed',sequence='20',window_id=817,time_end=end,roi=[18,407,25,414],pixel_scale=36,variants=details)
        (a.out/'T174_regression.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps(record,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

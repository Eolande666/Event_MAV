"""Evaluate timestamp-named YOLO GT with one-to-one matching.

OCC convention: window n (1-based) -> label (n+frame_offset)*period_us.
This is an offline evaluator and is never imported by the detector.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path


def load_gt(folder, index, period_us, width, height):
    # Prefix is deliberately not tied to sequence 28.
    paths=list(folder.glob(f'*_{index*period_us}.txt'))
    if len(paths)!=1:
        raise ValueError(f'Expected one GT file for frame {index}: found {len(paths)}')
    result=[]
    for line in paths[0].read_text(encoding='utf-8-sig').splitlines():
        if not line.strip(): continue
        _,x,y,w,h=map(float,line.split())
        result.append(((x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height))
    return result


def maximum_matching(edges):
    owners={}
    def visit(i,seen):
        for j in edges[i]:
            if j in seen: continue
            seen.add(j)
            if j not in owners or visit(owners[j],seen):
                owners[j]=i
                return True
        return False
    return sum(visit(i,set()) for i in range(len(edges)))


def evaluate(predictions, labels, frames, width, height, frame_offset=0, period_us=33333):
    predicted=defaultdict(list)
    with predictions.open(encoding='utf-8-sig') as handle:
        for r in csv.DictReader(handle):
            predicted[int(r['window'])].append(tuple(float(r[k]) for k in ('bbox_x0','bbox_y0','bbox_x1','bbox_y1')))
    counts={name:dict(tp=0,fp=0,fn=0) for name in ('iou_0.3','iou_0.5','ioa_0.5')}
    gt_count=positive_frames=0
    for index in range(1,frames+1):
        truth=load_gt(labels,index+frame_offset,period_us,width,height)
        gt_count+=len(truth);positive_frames+=bool(truth)
        boxes=predicted[index]
        edges={name:[] for name in counts}
        for a in boxes:
            matched={name:[] for name in counts}
            area=max(0,a[2]-a[0])*max(0,a[3]-a[1])
            for j,b in enumerate(truth):
                intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
                gt_area=max(0,b[2]-b[0])*max(0,b[3]-b[1])
                iou=intersection/max(area+gt_area-intersection,1e-12)
                ioa=intersection/max(area,1e-12)
                for name,value,threshold in (('iou_0.3',iou,.3),('iou_0.5',iou,.5),('ioa_0.5',ioa,.5)):
                    if value>=threshold: matched[name].append(j)
            for name in counts: edges[name].append(matched[name])
        for name,result in counts.items():
            tp=maximum_matching(edges[name])
            result['tp']+=tp;result['fp']+=len(boxes)-tp;result['fn']+=len(truth)-tp
    for result in counts.values():
        tp,fp,fn=result['tp'],result['fp'],result['fn']
        result.update(precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),
                      f1=2*tp/max(2*tp+fp+fn,1),fppi=fp/frames)
    return {'frames':frames,'gt_boxes':gt_count,'positive_frames':positive_frames,
            'prediction_boxes':sum(len(predicted[i]) for i in range(1,frames+1)),
            'predictions_outside_evaluation_range':sum(len(v) for i,v in predicted.items() if not 1<=i<=frames),
            'alignment':f'window n -> YOLO suffix (n+{frame_offset})*{period_us}',
            'box_convention':'continuous/exclusive edges, no +1; YOLO boxes retained as annotated',
            'matching':'maximum-cardinality one-to-one; IoA denominator is predicted area',
            'note':'IoA measures target-region hits, not full-object localization. Unmatched boxes are FP, not confirmed insect identities.',
            **counts}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--labels',type=Path,required=True)
    parser.add_argument('--frames',type=int,required=True)
    parser.add_argument('--width',type=int,default=1280)
    parser.add_argument('--height',type=int,default=720)
    parser.add_argument('--frame-offset',type=int,default=0)
    parser.add_argument('--period-us',type=int,default=33333)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if min(args.frames,args.width,args.height,args.period_us)<=0 or args.frame_offset<0:
        parser.error('positive dimensions/count/period and nonnegative offset required')
    result=evaluate(args.predictions,args.labels,args.frames,args.width,args.height,args.frame_offset,args.period_us)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

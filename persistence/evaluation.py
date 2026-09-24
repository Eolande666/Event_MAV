"""Frame-level one-to-one evaluation, kept separate from the detector and tracker.

Strict IoU metrics use unmodified candidate boxes. Center containment is a
separate localization diagnostic, NOT standard box detection precision.
"""
import math
from bisect import bisect_right


def iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union>0 else 0.


def match_boxes(predictions, ground_truth, protocol='iou50'):
    """Maximum-cardinality bipartite matching; removing a box cannot increase TP.

    Used only for evaluation, not temporal association. No prediction scores or
    ground truth are passed to the tracker.
    """
    edges=[]
    for box in predictions:
        candidates=[]
        for j,gt in enumerate(ground_truth):
            overlap=iou(box,gt)
            if protocol=='center':
                x,y=(box[0]+box[2])/2,(box[1]+box[3])/2
                valid=gt[0]<=x<gt[2] and gt[1]<=y<gt[3]
            elif protocol in ('iou40','iou50'):
                valid=overlap >= (.4 if protocol=='iou40' else .5)
            else:
                raise ValueError('unknown evaluation protocol')
            if valid:candidates.append((-overlap,j))
        edges.append([j for _,j in sorted(candidates)])
    matched={}
    def augment(i,seen):
        for j in edges[i]:
            if j in seen:continue
            seen.add(j)
            if j not in matched or augment(matched[j],seen):
                matched[j]=i
                return True
        return False
    for i in range(len(predictions)):
        augment(i,set())
    return sorted((i,j) for j,i in matched.items())


def metrics(tp,fp,fn):
    precision=tp/(tp+fp) if tp+fp else None
    recall=tp/(tp+fn) if tp+fn else None
    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None
    return dict(tp=tp,fp=fp,fn=fn,precision=precision,recall=recall,f1=f1)


def align_frames(windows, frame_times, annotations, *, max_age=.030001):
    """Causal latest completed 30 ms detector window at each official frame time.

    No nearest future window, interpolation or per-method alignment changes.
    Missing labels on an existing official frame mean no visible annotated UAV,
    matching the official loader's empty-box convention.
    """
    ends=[float(w['end_sec']) for w in windows]
    aligned=[]
    for time_us in frame_times:
        timestamp=time_us*1e-6
        index=bisect_right(ends,timestamp+1e-9)-1
        if index<0 or timestamp-ends[index] > max_age:
            continue
        aligned.append(dict(timestamp=timestamp,window_id=int(windows[index]['frame']),
                            lag_sec=timestamp-ends[index],gt=annotations.get(time_us,[])))
    return aligned


def evaluate(aligned, predictions, protocol):
    tp=fp=fn=0; rows=[]
    for frame in aligned:
        boxes=predictions.get(frame['window_id'],[])
        gt=[item['bbox'] for item in frame['gt']]
        matches=match_boxes(boxes,gt,protocol)
        a=len(matches);b=len(boxes)-a;c=len(gt)-a
        tp+=a;fp+=b;fn+=c
        rows.append(dict(timestamp=frame['timestamp'],window_id=frame['window_id'],tp=a,fp=b,fn=c,
                         matches=matches,lag_sec=frame['lag_sec']))
    result=metrics(tp,fp,fn)
    result['frames']=len(aligned)
    return result,rows

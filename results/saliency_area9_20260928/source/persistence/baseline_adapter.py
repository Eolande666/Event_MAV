"""Original process_window by default; explicitly named no-periodicity ablation."""
import time
import numpy as np
from evdetmav.pipeline import process_window,make_detection
from evdetmav.saliency import build_density_saliency, remove_small_saliency_regions
from evdetmav.clustering import initialize_candidates,refine_candidate
from evdetmav.models import clipped_box,events_in_box


def detect(events,start,end,width,height,source,window_id,args,use_periodicity=True,saliency_stats=None):
    if len(events)<args.min_window_events:
        if saliency_stats is not None:saliency_stats.update(skipped_low_events=True)
        return []
    x=events['x'].astype(np.int64);y=events['y'].astype(np.int64)
    t=events['t']*1e-6;p=np.where(events['p']>0,1,-1)
    if use_periodicity:
        result=process_window(x,y,t,p,start,end,height,width,str(source),window_id,args)
        if saliency_stats is not None:saliency_stats.update(result.saliency_filter_stats or {})
        return result.detections
    if args.merge_propellers:
        raise ValueError('periodicity-off ablation currently requires separate candidate outputs')
    tic=time.perf_counter()
    saliency=build_density_saliency(x,y,t,p,start,end,height,width,args)
    saliency,stats=remove_small_saliency_regions(saliency,float(args.tau_s),int(getattr(args,'saliency_min_area',0)))
    if saliency_stats is not None:saliency_stats.update(stats)
    candidates=initialize_candidates(saliency,args)[:max(int(args.top_k),1)]
    refined=[r for c in candidates if (r:=refine_candidate(saliency,c,args)) is not None]
    latency=(time.perf_counter()-tic)*1000
    rows=[]
    for i,item in enumerate(refined,1):
        box=clipped_box(item.box,width,height)
        rows.append(make_detection(str(source),window_id,start,end,f'propeller_{i}',box,item.segmentation_area_px,
                                   int(np.count_nonzero(events_in_box(x,y,box))),[item],latency))
    if args.max_detections_per_window>0:
        rows.sort(key=lambda d:(d.periodicity_score,d.saliency_score,d.event_count),reverse=True)
        rows=rows[:args.max_detections_per_window]
    return rows

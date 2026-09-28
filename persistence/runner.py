"""Direct ZIP/H5 baseline-to-persistence execution; no RGB reads."""
from dataclasses import asdict
from pathlib import Path
import json
from evdetmav.cli import build_parser
from .config import Config
from .events import stream_windows
from .baseline_adapter import detect
from .replay import replay,dump_csv


def run(config,argv):
    args=build_parser().parse_args(argv)
    if args.time_unit!='us':
        raise ValueError('FRED streaming runner requires explicit --time-unit us')
    if args.width<=0 or args.height<=0:
        raise ValueError('provide explicit --width and --height')
    dest=Path(args.out);dest.mkdir(parents=True,exist_ok=True)
    windows=[];baseline={};saliency_logs=[]
    for wid,start,end,events in stream_windows(args.input,args):
        stats={}
        raw=detect(events,start,end,args.width,args.height,args.input,wid,args,config.use_periodicity,saliency_stats=stats)
        if args.saliency_min_area>0 or args.min_raw_component_pixels>0:saliency_logs.append(dict(window_id=wid,start_sec=start,end_sec=end,**stats))
        baseline[wid]=[asdict(d) for d in raw]
        windows.append(dict(frame=wid,start_sec=start,end_sec=end,events=len(events),detections=len(raw)))
        if args.progress_every>0 and wid%args.progress_every==0:
            print(f'window={wid} events={len(events)} baseline={len(raw)}',flush=True)
    dump_csv(dest/'baseline_detections.csv',[d for ds in baseline.values() for d in ds])
    dump_csv(dest/'video_windows.csv',windows)
    if args.saliency_min_area>0 or args.min_raw_component_pixels>0:dump_csv(dest/'saliency_filter_windows.csv',saliency_logs)
    rows,timing=replay(windows,baseline,config,dest) if config.use_periodicity else _ablation_replay(windows,baseline,config,dest)
    (dest/'source.json').write_text(json.dumps(dict(source=str(args.input),use_periodicity=config.use_periodicity,
        baseline_label='repository baseline' if config.use_periodicity else 'periodicity-off ablation',
        saliency_min_area_px=args.saliency_min_area,
        min_raw_component_pixels=args.min_raw_component_pixels,
        detector_args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}),indent=2))
    return rows


def _ablation_replay(windows,baseline,config,dest):
    # Data were freshly computed with periodicity disabled; use the same tracker
    # without pretending an existing periodicity-filtered CSV can be unfiltered.
    from dataclasses import replace
    rows,timing=replay(windows,baseline,replace(config,use_periodicity=True),dest)
    config.save(dest/'config.json')
    return rows,timing

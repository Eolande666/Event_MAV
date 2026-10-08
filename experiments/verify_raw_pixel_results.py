"""Audit full raw-pixel-support experiment and the exact single-pixel regression."""
from pathlib import Path
import sys,argparse,csv,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.verify_saliency_area_results import verify_video,csv_rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);a=parser.parse_args();out=a.out
    state=json.loads((out/'status.json').read_text());assert state['status']=='complete'
    hashes=json.loads((out/'source_sha256.json').read_text())
    for name,digest in hashes.items():assert hashlib.sha256((out/'source'/name).read_bytes()).hexdigest()==digest,name
    total=0;checks=[]
    for seq in state['sequences']:
        dest=out/seq;summary=json.loads((dest/'summary.json').read_text());n=summary['windows'];total+=n
        assert summary['fresh_original_windows_checked']==min(n,12)
        rows=csv_rows(dest/'raw_support_windows.csv');assert len(rows)==n
        assert [int(r['window_id']) for r in rows]==list(range(n))
        for row in rows:
            if row.get('skipped_low_events')=='True':continue
            assert row['raw_support_enabled']=='True' and int(row['raw_support_min_pixels'])==state['min_raw_component_pixels']
            assert int(row['raw_unique_pixels_removed'])<=int(row['raw_unique_pixels_before'])
            assert int(row['raw_events_removed'])<=int(row['raw_events_before'])
        for method in ['original','raw_support','before_persistence']:
            frames=csv_rows(dest/(method+'_center_frames.csv'));metric=next(r for r in csv_rows(dest/'metrics.csv') if r['method']==method)
            for k in ['tp','fp','fn']:assert sum(int(r[k]) for r in frames)==int(metric[k])
            assert len(frames)==int(metric['frames'])
        checks.append(dict(sequence=seq,windows=n,frames_and_filter_logs_verified=True))
    historical=[]
    if not state['max_windows'] and set(state['sequences'])=={'8','20','51','65','93'}:
        for split in ['calibration','holdout']:
            old=next(r for r in csv_rows(ROOT/'results/persistence_mvp/fred_v1/metrics_aggregate.csv') if r['split']==split and r['mode']=='combined' and r['protocol']=='center')
            new=next(r for r in csv_rows(out/'metrics_aggregate.csv') if r['split']==split and r['method']=='original')
            for k in ['tp','fp','fn','frames']:assert int(old[k])==int(new[k]),(split,k)
            historical.append(split)
    regression=False
    if '20' in state['sequences'] and (not state['max_windows'] or state['max_windows']>817):
        with (out/'20/persistence_debug.jsonl').open() as f:
            row=next(json.loads(s) for s in f if json.loads(s)['window_id']==817)
        assert not any(r['accepted'] and r['bbox'][0]<=21<r['bbox'][2] and r['bbox'][1]<=410<r['bbox'][3] for r in row['rows'])
        regression=True
    video=verify_video(out/'Detection_compare_all.mp4',total)
    result=dict(status='passed',total_windows=total,source_hashes_verified=len(hashes),original_metrics_match_historical=historical,
        screenshot_false_positive_absent_with_full_history=regression,sequences=checks,video=video)
    (out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

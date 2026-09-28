"""Audit complete area-filter runs: counts, legacy reproduction, pixel maps and video frames."""
from pathlib import Path
import sys,json,csv,argparse,subprocess
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy import ndimage
import imageio_ffmpeg


def csv_rows(path):
    with path.open() as f:return list(csv.DictReader(f))


def verify_video(path,expected):
    result=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-threads','2','-i',str(path),
        '-map','0:v:0','-f','null','-','-progress','pipe:1','-nostats'],text=True,capture_output=True,check=True)
    frames=[int(line.split('=',1)[1]) for line in result.stdout.splitlines() if line.startswith('frame=')]
    assert frames and frames[-1]==expected,(str(path),frames[-1] if frames else None,expected)
    assert not result.stderr.strip(),result.stderr
    return dict(file=path.name,decoded_frames=frames[-1],decode_errors=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);a=parser.parse_args();out=a.out
    status=json.loads((out/'status.json').read_text());assert status['status']=='complete'
    sequences=status['sequences'];area=status['area_px'];count=0;checks=[]
    for seq in sequences:
        dest=out/seq;summary=json.loads((dest/'summary.json').read_text());assert summary['status']=='complete'
        count+=summary['windows'];assert summary['fresh_original_windows_checked']==min(summary['windows'],12)
        stats=csv_rows(dest/'saliency_windows.csv');assert len(stats)==summary['windows']
        assert [int(r['window_id']) for r in stats]==list(range(summary['windows']))
        for r in stats:
            if r.get('skipped_low_events')=='True':continue
            assert int(r['components_before'])==int(r['components_removed'])+int(r['components_kept'])
            assert int(r['foreground_pixels_before'])==int(r['foreground_pixels_removed'])+int(r['foreground_pixels_kept'])
        with np.load(dest/'preview_saliency.npz') as sample:
            before=sample['raw'];after=sample['filtered'];active=after>0
            np.testing.assert_array_equal(after[active],before[active])
            labels,_=ndimage.label(active,structure=np.ones((3,3)))
            assert np.all(np.bincount(labels.ravel())[1:]>=area)
            assert np.all(after[active]>=50)
        # Recompute totals from per-frame logs, independent of stored metric totals.
        for method in ['original','filtered','filtered_before_persistence']:
            frames=csv_rows(dest/(method+'_center_frames.csv'))
            metric=next(r for r in csv_rows(dest/'metrics.csv') if r['method']==method)
            for field in ['tp','fp','fn']:assert sum(int(r[field]) for r in frames)==int(metric[field])
            assert len(frames)==int(metric['frames'])
        checks.append(dict(sequence=seq,windows=summary['windows'],metrics_from_frames=True,preview_area_check=True))
    historical_matches=[]
    if not status['max_windows'] and set(sequences)=={'8','20','51','65','93','114'}:
        historical=csv_rows(ROOT/'results/persistence_mvp/fred_v1/metrics_aggregate.csv')
        actual=csv_rows(out/'metrics_aggregate.csv')
        for split in ['calibration','holdout']:
            expected=next(r for r in historical if r['split']==split and r['mode']=='combined' and r['protocol']=='center')
            restored=next(r for r in actual if r['split']==split and r['method']=='original')
            for key in ['frames','tp','fp','fn']:assert int(expected[key])==int(restored[key]),(split,key,expected[key],restored[key])
            historical_matches.append(split)
    with ThreadPoolExecutor(max_workers=2) as pool:
        videos=list(pool.map(lambda name:verify_video(out/name,count),['Detection_compare_all.mp4','Saliency_compare_all.mp4']))
    result=dict(status='passed',total_windows=count,historical_original_metrics_exact=historical_matches,sequences=checks,videos=videos)
    (out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

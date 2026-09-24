"""Verify direct ZIP integration and disabled-entry baseline equivalence."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import csv,json,subprocess,tempfile,hashlib
import h5py
from evdetmav.cli import build_parser
from persistence.events import stream_windows

OUT=ROOT/'results/persistence_mvp/fred_v1'


def rows(path):return list(csv.DictReader(Path(path).open()))


def normalize(records):
    return [{k:v for k,v in row.items() if k not in ('latency_ms','source_file')} for row in records]


def main():
    fresh=rows(OUT/'integration_h5/baseline_detections.csv')
    cache=[r for r in rows(ROOT/'comparison/8/evdetmav_detections_batch.csv') if int(r['window_id'])<12]
    assert normalize(fresh)==normalize(cache),'direct ZIP baseline differs from frozen CSV'
    fresh_scores=[json.loads(s) for s in (OUT/'integration_h5/debug.jsonl').read_text().splitlines()]
    cached_scores=[json.loads(s) for s in (OUT/'8/combined/debug.jsonl').read_text().splitlines() if json.loads(s)['window_id']<12]
    assert fresh_scores==cached_scores,'direct ZIP persistence differs from full replay'
    with tempfile.TemporaryDirectory(prefix='persistence_contract_') as temp:
        temp=Path(temp)
        args=build_parser().parse_args(['--input',str(ROOT/'FRED/8.zip'),'--window-ms','30','--step-ms','30','--max-windows','1'])
        _,_,_,events=next(stream_windows(args.input,args))
        path=temp/'fixture.h5'
        with h5py.File(path,'w') as hf:hf.create_dataset('events',data=events)
        common=['--input',str(path),'--width','1280','--height','720','--time-unit','us','--window-ms','30','--step-ms','30',
                '--no-save-event-frames','--no-save-saliency','--no-save-segmentation']
        for name,entry in [('original',['evdetmav_detector.py']),('disabled',['evdetmav_persistence.py','--config','configs/baseline.json'])]:
            with (OUT/(name+'_contract.log')).open('w') as log:
                subprocess.run([sys.executable,*entry,*common,'--out',str(temp/name)],cwd=ROOT,stdout=log,stderr=log,check=True)
        a=rows(temp/'original/evdetmav_detections_batch.csv');b=rows(temp/'disabled/evdetmav_detections_batch.csv')
        assert normalize(a)==normalize(b),'disabled wrapper changes baseline results'
        fixture_hash=hashlib.sha256(events.tobytes()).hexdigest()
    result=dict(direct_zip_windows=12,direct_zip_baseline_rows=len(fresh),direct_zip_matches_cached_baseline=True,
                direct_zip_persistence_matches_replay=True,disabled_cli_matches_original=True,fixture_sha256=fixture_hash,
                ignored_fields=['latency_ms','source_file'])
    (OUT/'integration_contract.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

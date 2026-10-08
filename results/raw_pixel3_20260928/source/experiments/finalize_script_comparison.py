"""Finalize this bounded six-sequence run once every sequence is complete."""
import json,time,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/script_comparison_20260924'
SEQS=['8','20','51','65','93','114'];TOTAL={'8':3491,'20':3757,'51':3579,'65':3816,'93':3708,'114':3737}
while True:
 states=[]
 for seq in SEQS:
  d=OUT/seq;complete=d/'summary.json';progress=d/'progress.json'
  if complete.exists():
   s=json.loads(complete.read_text());states.append(dict(sequence=seq,status=s['status'],windows=s['windows'],total=TOTAL[seq]))
  else:
   try:p=json.loads(progress.read_text()) if progress.exists() else {}
   except json.JSONDecodeError:p={}
   states.append(dict(sequence=seq,status='running' if d.exists() else 'pending',windows=p.get('window',0),total=TOTAL[seq]))
 tmp=OUT/'progress.tmp';tmp.write_text(json.dumps(dict(sequences=states,processed=sum(s['windows'] for s in states),total=sum(TOTAL.values())),indent=2));tmp.replace(OUT/'progress.json')
 if all(s['status']=='complete' for s in states):break
 time.sleep(5)
subprocess.run([sys.executable,str(ROOT/'experiments/report_script_comparison.py')],check=True)

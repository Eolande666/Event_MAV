import sys,json,pickle
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path.cwd()))
from evdetmav.cli import build_parser
from evdetmav.pipeline import process_window
from dataclasses import asdict
root=Path('tmp/fred_figures');a=np.load(root/'events_129_34.5_36.2.npy');args=build_parser().parse_args(['--input','FRED/test/129.zip']);results=[]
for i in range(56):
 s=34.5+i*.03;e=s+.03;v=a[(a['t']>=round(s*1e6))&(a['t']<round(e*1e6))];r=process_window(v['x'].astype(np.int64),v['y'].astype(np.int64),v['t']*1e-6,np.where(v['p']>0,1,-1),s,e,720,1280,'FRED/test/129.zip',i,args);results.append((s,e,r));print(i,round(s,3),len(v),[(c.box,c.periodicity.score if c.periodicity else None) for c in r.candidates[:4]],flush=True)
with open(root/'results.pkl','wb') as f:pickle.dump((args,results),f)

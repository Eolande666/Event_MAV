import ctypes,h5py,numpy as np
from pathlib import Path
root=Path(__file__).resolve().parents[3]/'tmp/fred_figures'
lib=ctypes.CDLL(str(root/'libecf_decode.dylib'));lib.decode.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p];lib.decode.restype=ctypes.c_size_t
with h5py.File(root/'129/events.hdf5') as f:
 d=f['CD/events']; idx=f['CD/indexes'][:]; print('chunks',d.chunks,flush=True)
 lo=int(idx['id'][np.searchsorted(idx['ts'],34500000)-1]);hi=int(idx['id'][np.searchsorted(idx['ts'],36200000)+1]); step=d.chunks[0]; arr=[]
 for start in range(lo//step*step,hi+step,step):
  if start>=len(d):break
  mask,raw=d.id.read_direct_chunk((start,)); n=int.from_bytes(raw[:4],'little')>>2; out=np.empty(n,dtype=d.dtype); size=lib.decode(raw,len(raw),out.ctypes.data); assert size==out.nbytes,(size,out.nbytes);arr.append(out)
 a=np.concatenate(arr);a=a[(a['t']>=34500000)&(a['t']<36200000)];np.save(root/'events_129_34.5_36.2.npy',a);print(len(a),a[:3],a[-3:])

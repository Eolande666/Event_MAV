"""Stream original FRED structured H5/ZIP events, preserving baseline inclusive windows."""
from contextlib import contextmanager
from pathlib import Path
import ctypes, tempfile, zipfile, shutil
import numpy as np
import h5py
from evdetmav.cli import build_windows


@contextmanager
def open_event_h5(path):
    path=Path(path)
    if path.suffix.lower()=='.zip':
        with tempfile.TemporaryDirectory(prefix='persistence_h5_') as temp:
            target=Path(temp)/'events.h5'
            with zipfile.ZipFile(path) as archive:
                name=next(n for n in archive.namelist() if n.endswith('/Event/events.hdf5'))
                with archive.open(name) as src,target.open('wb') as dst:
                    shutil.copyfileobj(src,dst,16*1024*1024)
            with h5py.File(target,'r') as hf:yield hf
    else:
        with h5py.File(path,'r') as hf:yield hf


def stream_windows(path,args):
    with open_event_h5(path) as hf:
        data=hf['CD/events'] if 'CD/events' in hf else hf['events']
        if len(data)==0:return
        if not data.dtype.names or not {'x','y','t','p'}<=set(data.dtype.names):
            raise ValueError('streaming adapter requires structured FRED x/y/t/p fields')
        step=data.chunks[0] if data.chunks else min(len(data),65536)
        plist=data.id.get_create_plist()
        ecf=any(plist.get_filter(i)[0]==36559 for i in range(plist.get_nfilters()))
        lib=None
        if ecf:
            root=Path(__file__).resolve().parents[1]
            lib=ctypes.CDLL(str(root/'tmp/fred_figures/libecf_decode.dylib'))
            lib.decode.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p]
            lib.decode.restype=ctypes.c_size_t
        def chunk(offset):
            count=min(step,len(data)-offset)
            if not ecf:return data[offset:offset+count]
            mask,raw=data.id.read_direct_chunk((offset,))
            if mask:raise ValueError('unsupported skipped filter mask')
            n=int.from_bytes(raw[:4],'little')>>2
            result=np.empty(n,data.dtype)
            if lib.decode(raw,len(raw),result.ctypes.data)!=result.nbytes:raise ValueError('ECF decode failed')
            return result[:count]
        first=chunk(0);last=chunk((len(data)-1)//step*step)
        windows=build_windows(np.array([first['t'][0],last['t'][-1]],dtype=np.float64)*1e-6,args)
        if args.max_windows>0:windows=windows[:args.max_windows]
        buf=first;offset=len(first)
        for wid,start,end in windows:
            # Keep enough data to include the right endpoint exactly, as original CLI.
            while (not len(buf) or buf['t'][-1]*1e-6<=end) and offset<len(data):
                next_chunk=chunk(offset);offset+=len(next_chunk)
                buf=np.concatenate((buf,next_chunk))
            times=buf['t']*1e-6
            lo=np.searchsorted(times,start,side='left');hi=np.searchsorted(times,end,side='right')
            yield wid,start,end,buf[lo:hi]
            # Supports overlapping windows; never discard before the next start.
            next_start=windows[wid+1][1] if wid+1<len(windows) else end
            buf=buf[np.searchsorted(times,next_start,side='left'):]

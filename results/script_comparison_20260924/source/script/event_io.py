"""Small explicit-format reader; timestamps require an explicit unit.

H5: data/events Nx4 (x,y,t,p), compound fields, or x/y/t/p datasets.
NPZ: x/y/t/p or Nx4 'events'/'data'. CSV: named x,y,t,p columns.
"""
from pathlib import Path
import re
import warnings
import numpy as np


def natural_key(path):
    return [int(v) if v.isdigit() else v.lower() for v in re.split(r'(\d+)', path.name)]


def list_inputs(path, limit=0):
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise ValueError(f'input does not exist: {path}')
    files = sorted((p for p in path.iterdir() if p.suffix.lower() in ('.h5', '.hdf5', '.npz', '.csv')), key=natural_key)
    if not files:
        raise ValueError(f'no event files in {path}')
    return files[:limit] if limit else files


def as_matrix(value):
    if value.dtype.names:
        return np.column_stack([value[k] for k in ('x', 'y', 't', 'p')])
    if value.ndim != 2 or value.shape[1] != 4:
        raise ValueError('matrix must have four columns x,y,t,p')
    return value


def load_events(path, unit, width, height):
    suffix = path.suffix.lower()
    if suffix in ('.h5', '.hdf5'):
        import h5py
        with h5py.File(path, 'r') as f:
            groups = [f]
            if 'events' in f and isinstance(f['events'], h5py.Group):
                groups.insert(0, f['events'])
            value = None
            for group in groups:
                if all(k in group for k in ('x', 'y', 't', 'p')):
                    value = np.column_stack([group[k][:] for k in ('x', 'y', 't', 'p')])
                    break
            if value is None:
                for name in ('data', 'events'):
                    if name in f and isinstance(f[name], h5py.Dataset):
                        value = as_matrix(f[name][:])
                        break
            if value is None:
                raise ValueError(f'{path}: no supported event datasets')
    elif suffix == '.npz':
        with np.load(path, allow_pickle=False) as f:
            if all(k in f for k in ('x', 'y', 't', 'p')):
                value = np.column_stack([f[k] for k in ('x', 'y', 't', 'p')])
            else:
                value = as_matrix(f['events'] if 'events' in f else f['data'])
    elif suffix == '.csv':
        raw = np.atleast_1d(np.genfromtxt(path, delimiter=',', names=True))
        value = as_matrix(raw)
    else:
        raise ValueError(f'unsupported input: {path}')
    value = np.asarray(value, dtype=np.float64).reshape(-1, 4)
    if not np.isfinite(value).all():
        raise ValueError(f'{path}: non-finite event values')
    if not np.isin(value[:, 3], [-1, 0, 1]).all():
        raise ValueError(f'{path}: polarity must be 0/1 or -1/+1')
    if not np.equal(value[:, :2], np.floor(value[:, :2])).all():
        raise ValueError(f'{path}: pixel coordinates must be integers')
    value[:, 2] *= {'s': 1, 'ms': 1e-3, 'us': 1e-6, 'ns': 1e-9}[unit]
    value[:, 3] = np.where(value[:, 3] > 0, 1, -1)
    inside = ((value[:, 0] >= 0) & (value[:, 0] < width)
              & (value[:, 1] >= 0) & (value[:, 1] < height))
    if not inside.all():
        raise ValueError(f'{path}: coordinates outside configured sensor size')
    return value[np.argsort(value[:, 2], kind='stable')]


def iter_windows(files, unit, width, height, window_ms, file_mode=False, overlap_policy='error'):
    """Continuous absolute timestamps by default; reject resets/overlap.

    file_mode explicitly treats each file as a separate acquisition period;
    local timestamps are rebased, including for datasets resetting each H5.
    Continuous mode keeps partial windows across file boundaries.
    """
    duration = window_ms/1000
    pending = np.empty((0, 4))
    origin, tick, previous = None, 0, None
    for file_index, path in enumerate(files):
        events = load_events(path, unit, width, height)
        if file_mode:
            start = file_index*duration
            if len(events):
                span = events[-1, 2]-events[0, 2]
                if span >= duration:
                    raise ValueError(f'{path}: file spans longer than window_ms; use continuous mode')
                events[:, 2] += start-events[0, 2]
            end = start+duration
            yield path.name, events, start, end
            continue
        if not len(events):
            continue
        if previous is not None and events[0, 2] < previous:
            if events[-1, 2] <= previous or overlap_policy != 'trim':
                raise ValueError(f'{path}: timestamp reset/overlap; use --file-windows for independent periods, or --overlap-policy trim for partial overlaps')
            cut = np.searchsorted(events[:, 2], previous, side='left')
            warnings.warn(f'{path.name}: trimmed {cut} events older than preceding file end', stacklevel=2)
            events = events[cut:]
        previous = events[-1, 2]
        if origin is None:
            origin = events[0, 2]
        events[:, 2] -= origin
        pending = np.concatenate((pending, events))
        while len(pending) and pending[-1, 2] >= (tick+1)*duration:
            start, end = tick*duration, (tick+1)*duration
            split = np.searchsorted(pending[:, 2], end, side='left')
            yield path.name, pending[:split], start, end
            pending = pending[split:]
            tick += 1
    if len(pending):
        yield files[-1].name, pending, tick*duration, (tick+1)*duration

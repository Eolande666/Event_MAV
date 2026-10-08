"""Build switch count and switch-time coverage ONCE for all candidate features."""
from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi

BIT_COUNTS = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


@dataclass
class SwitchMaps:
    flips: np.ndarray
    time_bits: np.ndarray
    activity: np.ndarray
    score: np.ndarray


def build_maps(events, start, end, width, height, cfg):
    size = width*height
    flips = np.zeros(size, np.int32)
    bits = np.zeros(size, np.uint16)
    activity = np.zeros(size, np.int32)
    if len(events):
        ids = events[:, 1].astype(np.int64)*width+events[:, 0].astype(np.int64)
        activity = np.bincount(ids, minlength=size).astype(np.int32)
        order = np.lexsort((events[:, 2], ids))
        pixel, t, p = ids[order], events[order, 2], events[order, 3]
        valid = (pixel[1:] == pixel[:-1]) & (t[1:] > t[:-1]) & (p[1:] != p[:-1])
        locations = pixel[1:][valid]
        flips = np.bincount(locations, minlength=size).astype(np.int32)
        bins = np.clip(((t[1:][valid]-start)/(end-start)*cfg.temporal_bins).astype(int), 0, cfg.temporal_bins-1)
        np.bitwise_or.at(bits, locations, np.left_shift(np.uint16(1), bins.astype(np.uint16)))
    # NumPy-version-independent 16-bit population count (constant-size table).
    lookup = BIT_COUNTS
    coverage = (lookup[bits & 255]+lookup[bits >> 8])/cfg.temporal_bins
    score = (flips/(flips+cfg.switch_saturation)*coverage).astype(np.float32)
    shape = (height, width)
    return SwitchMaps(flips.reshape(shape), bits.reshape(shape), activity.reshape(shape), score.reshape(shape))


def propose(maps, width, height, cfg):
    # One spatial aggregation instead of separate polarity images per slice.
    aggregated = ndi.maximum_filter(maps.score, size=3, mode='constant')
    labels, count = ndi.label(aggregated >= cfg.switch_map_threshold, np.ones((3, 3)))
    if not count:
        return []
    areas = np.bincount(labels.ravel())
    masses = np.bincount(labels.ravel(), weights=maps.score.ravel())
    rois = []
    for i, bounds in enumerate(ndi.find_objects(labels), 1):
        if bounds is None or areas[i] < cfg.min_component_pixels:
            continue
        ys, xs = bounds
        pad = cfg.candidate_pad
        box = (max(0, xs.start-pad), max(0, ys.start-pad), min(width, xs.stop+pad), min(height, ys.stop+pad))
        if (box[2]-box[0])*(box[3]-box[1]) <= cfg.max_roi_area:
            rois.append({'box': box, 'saliency': float(masses[i])})
    rois.sort(key=lambda c: c['saliency'], reverse=True)
    selected = []
    for roi in rois:
        a = roi['box']
        duplicate = False
        for previous in selected:
            b = previous['box']
            inter = max(0, min(a[2], b[2])-max(a[0], b[0]))*max(0, min(a[3], b[3])-max(a[1], b[1]))
            smaller = min((a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1]))
            if inter >= 0.8*smaller:
                duplicate = True
                break
        if not duplicate:
            selected.append(roi)
        if len(selected) == cfg.max_candidates:
            break
    return selected


def features(maps, box, cfg):
    x0, y0, x1, y1 = box
    flips = maps.flips[y0:y1, x0:x1]
    bits = maps.time_bits[y0:y1, x0:x1]
    active = flips > 0
    # One-off +/- edges produce proposals but provide zero repetition mass.
    mass = float(np.minimum(np.maximum(flips-1, 0)/2, 1).sum())
    union = int(np.bitwise_or.reduce(bits.ravel(), initial=np.uint16(0)))
    occupied = bin(union).count('1')
    coverage = occupied/cfg.temporal_bins
    support = mass/(mass+cfg.switch_support_scale)
    repetition_base = float(np.sqrt(coverage)*support)
    # Each pixel must itself switch in multiple time bins, rather than relying
    # on the union of disjoint pixels activated at different times.
    per_pixel_bins = BIT_COUNTS[bits & 255] + BIT_COUNTS[bits >> 8]
    persistent_support = np.minimum(np.maximum(per_pixel_bins.astype(float)-1, 0)/2, 1)
    persistence = float(persistent_support[active].mean()) if active.any() else 0.0
    # Compare event activity per pixel, not total count. The exterior ring
    # excludes the entire proposal, including its padding, and clips at edges.
    height, width = maps.activity.shape
    pad = cfg.background_ring_px
    rx0, ry0, rx1, ry1 = max(0,x0-pad), max(0,y0-pad), min(width,x1+pad), min(height,y1+pad)
    ring_area = (rx1-rx0)*(ry1-ry0)-(x1-x0)*(y1-y0)
    roi_sum = float(maps.activity[y0:y1,x0:x1].sum())
    ring_sum = float(maps.activity[ry0:ry1,rx0:rx1].sum())-roi_sum
    background_density = max(ring_sum, 0)/max(ring_area, 1)
    foreground_density = roi_sum/max((x1-x0)*(y1-y0), 1)
    if active.any():
        yy, xx = np.nonzero(active)
        core = maps.activity[y0+yy.min():y0+yy.max()+1,x0+xx.min():x0+xx.max()+1]
        foreground_density = float(core.mean())
    # No exterior pixels means no background evidence; do not invent a veto.
    contrast = float(np.clip((foreground_density-background_density)/max(foreground_density+background_density,1e-12),0,1)) if ring_area else 1.0
    repetition = repetition_base*(1-cfg.persistence_penalty*(1-persistence))*(1-cfg.background_penalty*(1-contrast))
    # Spatial continuity and fill use binary support, not switch counts again.
    spatial = 0.0
    if active.any():
        yy, xx = np.nonzero(active)
        fill = len(xx)/((xx.max()-xx.min()+1)*(yy.max()-yy.min()+1))
        labels, _ = ndi.label(ndi.maximum_filter(active, size=3), np.ones((3, 3)))
        sizes = np.bincount(labels[active])
        connected = sizes[1:].max()/max(active.sum(), 1)
        spatial = float(np.sqrt(fill*connected))
    return {'switch_pixels': int(active.sum()), 'repeat_pixels': int((flips >= 3).sum()),
            'repeat_mass': mass, 'switch_time_coverage': coverage,
            'occupied_bins': occupied, 'repetition_score': float(repetition), 'spatial_score': spatial,
            'repetition_base': repetition_base, 'persistent_switch_support': persistence,
            'background_contrast': contrast, 'foreground_density': foreground_density,
            'background_density': background_density}

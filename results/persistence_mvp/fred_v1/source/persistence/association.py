"""Deterministic, gated, one-to-one greedy nearest-neighbor association."""
import math


def predict(track, timestamp):
    last = track.detections[-1]
    dt = timestamp-last.timestamp
    if dt <= 0:
        raise ValueError('prediction timestamp must follow last observation')
    velocity = (0.,0.)
    if len(track.detections) >= 2:
        prior = track.detections[-2]
        elapsed = last.timestamp-prior.timestamp
        velocity = tuple((a-b)/elapsed for a,b in zip(last.center,prior.center))
    center = tuple(x+v*dt for x,v in zip(last.center,velocity))
    return center, velocity, dt


def associate(tracks, detections, timestamp, config):
    """Returns matches and every tested pair for intermediate logging."""
    pairs, debug, predictions = [], [], {}
    for track in tracks:
        center, velocity, dt = predict(track,timestamp)
        box = track.detections[-1].bbox
        base = config.radius if config.radius_mode == 'fixed' else config.radius_scale*max(box[2]-box[0],box[3]-box[1])
        radius = base+config.velocity_scale*math.hypot(*velocity)*dt
        predictions[track.track_id] = (center,radius)
        for index, d in enumerate(detections):
            distance = math.dist(center,d.center)
            inside = distance < radius
            debug.append(dict(track_id=track.track_id,detection_index=index,predicted_center=center,
                              distance=distance,radius=radius,inside=inside))
            if inside:
                pairs.append((distance,track.track_id,d.bbox,index))
    used_tracks, used_detections, matches = set(),set(),{}
    for distance,track_id,box,index in sorted(pairs):
        if track_id not in used_tracks and index not in used_detections:
            matches[index] = (track_id,distance)
            used_tracks.add(track_id);used_detections.add(index)
    return matches, predictions, debug

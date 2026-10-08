"""Bounded histories; rejected baseline candidates can maintain tracks, predictions cannot become detections."""
from collections import deque
from dataclasses import dataclass, field
from .association import associate
from .neighborhood import neighborhood_score
from .trajectory import motion_features
from .scoring import trajectory_score, fuse_scores


@dataclass
class Track:
    track_id: int
    detections: deque
    hits: deque
    age: int = 0
    hit_count: int = 0
    miss_count: int = 0
    state: str = 'Tentative'


class Tracker:
    def __init__(self, config):
        config.validate()
        self.config=config
        self.tracks={}
        self.next_id=1
        self.last_timestamp=None

    def step(self, timestamp, detections):
        import math
        if not math.isfinite(timestamp) or (self.last_timestamp is not None and timestamp <= self.last_timestamp):
            raise ValueError('windows must have finite strictly increasing timestamps')
        if any(not d.valid or d.timestamp != timestamp for d in detections):
            raise ValueError('all current observations must be valid and have this window timestamp')
        if not self.config.enabled:
            self.last_timestamp=timestamp
            return [dict(detection_index=i,accepted=True,reason='baseline') for i in range(len(detections))], [], []
        self.last_timestamp=timestamp
        cfg=self.config
        matches,predictions,debug=associate(list(self.tracks.values()),detections,timestamp,cfg)
        matched_ids={v[0] for v in matches.values()}
        transitions=[]
        for track_id,track in list(self.tracks.items()):
            track.age+=1
            if track_id not in matched_ids:
                track.hits.append(0);track.miss_count+=1
                old=track.state
                track.state='Deleted' if track.miss_count > cfg.max_missed_windows else 'Lost'
                transitions.append(dict(track_id=track_id,previous_state=old,state=track.state,
                                        predicted_center=predictions[track_id][0],miss_count=track.miss_count))
                if track.state=='Deleted':
                    del self.tracks[track_id]
        rows=[]
        # Canonical geometry ordering makes new track IDs independent of candidate ranking.
        for index in sorted(range(len(detections)),key=lambda i:(detections[i].bbox,i)):
            d=detections[index]
            if index in matches:
                track_id,distance=matches[index];track=self.tracks[track_id]
                predicted,radius=predictions[track_id]
            else:
                track_id=self.next_id;self.next_id+=1
                track=Track(track_id,deque(maxlen=cfg.history_length),deque(maxlen=cfg.history_length),age=1)
                self.tracks[track_id]=track
                distance=predicted=radius=None
            old=track.state
            m=motion_features(list(track.detections),d,min_speed=cfg.min_speed,epsilon=cfg.epsilon)
            track.hits.append(1)
            sn=neighborhood_score(track.hits,length=cfg.history_length,mode=cfg.neighborhood_mode,
                                  decay=cfg.temporal_decay,cold_start=cfg.cold_start) if cfg.use_neighborhood else None
            st,weights=trajectory_score(m,weights=cfg.weights,sigmas=cfg.sigmas,
                                       normalize_position=cfg.normalize_position) if cfg.use_trajectory else (None,{})
            sc=fuse_scores(sn,st,alpha=cfg.alpha) if cfg.use_neighborhood and cfg.use_trajectory else (sn if cfg.use_neighborhood else st)
            # Trajectory-only warm-up cannot validate a candidate; it stays tentative.
            accepted=sc is not None and sc >= cfg.threshold
            track.state='Confirmed' if accepted else 'Tentative'
            track.detections.append(d);track.hit_count+=1;track.miss_count=0
            row=dict(timestamp=timestamp,detection_index=index,track_id=track_id,center=d.center,bbox=d.bbox,
                     saliency_score=d.saliency_score,periodicity_score=d.periodicity_score,event_count=d.event_count,
                     neighborhood_score=sn,trajectory_score=st,persistence_score=sc,accepted=accepted,
                     reason='score_pass' if accepted else ('warmup_unavailable' if sc is None else 'below_threshold'),
                     track_state=track.state,age=track.age,hit_count=track.hit_count,miss_count=0,
                     association_distance=distance,association_prediction=predicted,gate_radius=radius,
                     effective_weights=weights,trajectory_valid=st is not None,neighborhood_valid=sn is not None,
                     history=[x.center for x in track.detections],**m.to_dict())
            rows.append(row)
            if old!=track.state:
                transitions.append(dict(track_id=track_id,previous_state=old,state=track.state,miss_count=0))
        return sorted(rows,key=lambda r:r['detection_index']), transitions, debug

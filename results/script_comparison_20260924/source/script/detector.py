"""Shared candidates -> cheap features -> budgeted ROI IOC -> output."""
from time import perf_counter
import numpy as np
from candidates import localize
from switch_frontend import build_maps, propose, features
from joint_score import score_candidate
from tracking import Tracker
from rotation import evaluate


class FixedEventDetector:
    def __init__(self, width, height, cfg):
        cfg.validate()
        if width <= 0 or height <= 0:
            raise ValueError('width and height must be positive')
        self.width, self.height, self.cfg = width, height, cfg
        self.tracker = Tracker(cfg)
        self.window = 0
        self.last_end = None

    def process(self, events, start, end):
        if not start < end or (self.last_end is not None and start < self.last_end-1e-9):
            raise ValueError('windows must be positive, non-overlapping and monotonic')
        events = np.asarray(events, dtype=float).reshape(-1, 4)
        if len(events) and (not np.isfinite(events).all() or np.any(events[:, 2] < start-1e-9)
                            or np.any(events[:, 2] >= end)):
            raise ValueError('events must be finite and belong to [start, end)')
        if len(events) and (np.any(events[:, 0] < 0) or np.any(events[:, 0] >= self.width)
                            or np.any(events[:, 1] < 0) or np.any(events[:, 1] >= self.height)
                            or not np.isin(events[:, 3], [-1, 1]).all()):
            raise ValueError('events must have valid pixel coordinates and +/-1 polarities')
        if self.last_end is not None:
            self.window += max(0, int(round((start-self.last_end)/(self.cfg.window_ms/1000))))
        self.last_end = end
        self.window += 1
        begin = perf_counter()
        maps = build_maps(events, start, end, self.width, self.height, self.cfg)
        proposals = propose(maps, self.width, self.height, self.cfg)
        proposed = perf_counter()
        candidates, calls = [], 0
        for proposal in proposals:
            x0, y0, x1, y1 = proposal['box']
            inside = ((events[:, 0] >= x0) & (events[:, 0] < x1)
                      & (events[:, 1] >= y0) & (events[:, 1] < y1))
            local = events[inside]
            if len(local) < self.cfg.min_events:
                continue
            c = dict(proposal, **features(maps, proposal['box'], self.cfg), events=len(local))
            c.update(observable=False, valid_pixels=0, rotation_blocks=0,
                     rotation_q=0.0, rotation_significance=0.0,
                     rotation_omega=0.0, rotation_fit=0.0, rotation_rigidity=0.0,
                     rotation_residual=0.0, rotation_polarity_sign=0,
                     rotation_omega_positive=0.0, rotation_omega_negative=0.0,
                     )
            # Cheap necessary condition; estimate observability independently of Q.
            pixels = []
            for sign in (1, -1):
                xy = local[local[:, 3] == sign, :2].astype(int)
                pixels.append(np.unique(xy[:, 1]*self.width+xy[:, 0]).size)
            deferred = False
            if min(pixels) >= self.cfg.ioc_min_support:
                if calls < self.cfg.rotation_max_calls:
                    c.update(evaluate(local, proposal['box'], end, self.cfg, start=start))
                    calls += 1
                else:
                    deferred = True
            c.update(score_candidate(c, c, self.cfg, deferred=deferred))
            c['box'] = localize(local, self.width, self.height)
            candidates.append(c)
        verified = perf_counter()
        outputs = self.tracker.update(candidates, self.window)
        done = perf_counter()
        timing = {'window': self.window, 'start_s': start, 'end_s': end,
                  'events': len(events), 'proposals': len(proposals), 'candidates': len(candidates),
                  'rotation_calls': calls, 'detections': len(outputs),
                  'proposal_ms': (proposed-begin)*1000, 'verification_ms': (verified-proposed)*1000,
                  'tracking_ms': (done-verified)*1000, 'total_ms': (done-begin)*1000}
        return outputs, candidates, timing

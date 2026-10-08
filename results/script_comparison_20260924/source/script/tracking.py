"""Bounded nearest-center association for stable detection IDs."""
from dataclasses import dataclass
import math


@dataclass
class Track:
    id: int
    box: tuple
    last_window: int


class Tracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tracks = []
        self.next_id = 1

    def update(self, candidates, window):
        cfg = self.cfg
        self.tracks = [t for t in self.tracks
                       if window-t.last_window <= cfg.max_misses+1]
        edges = []
        for candidate_index, candidate in enumerate(candidates):
            a = candidate['box']
            area = (a[2]-a[0])*(a[3]-a[1])
            for track_index, track in enumerate(self.tracks):
                b = track.box
                other_area = (b[2]-b[0])*(b[3]-b[1])
                distance = math.hypot(
                    (a[0]+a[2]-b[0]-b[2])/2,
                    (a[1]+a[3]-b[1]-b[3])/2)
                gap = window-track.last_window
                area_ratio = max(area, other_area)/max(min(area, other_area), 1)
                if (distance <= cfg.association_px*gap
                        and area_ratio <= cfg.max_area_ratio):
                    edges.append((distance, candidate_index, track_index))

        matched, used_tracks = {}, set()
        for _, candidate_index, track_index in sorted(edges):
            if candidate_index not in matched and track_index not in used_tracks:
                matched[candidate_index] = self.tracks[track_index]
                used_tracks.add(track_index)

        outputs = []
        for index, candidate in enumerate(candidates):
            track = matched.get(index)
            if track is None:
                track = Track(self.next_id, candidate['box'], window)
                self.next_id += 1
                self.tracks.append(track)
            track.box, track.last_window = candidate['box'], window
            publish = (candidate.get('joint_pass', False)
                       and candidate.get('state') == 'joint_candidate')
            candidate.update(track_id=track.id, published=publish)
            if publish:
                outputs.append(dict(candidate))
        return outputs

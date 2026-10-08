"""Validated immutable observations. Times are seconds; box ends are exclusive."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Detection:
    timestamp: float
    bbox: tuple[float, float, float, float]
    saliency_score: float
    periodicity_score: float
    event_count: int
    valid: bool = True

    def __post_init__(self):
        if len(self.bbox) != 4 or not all(math.isfinite(x) for x in (
            self.timestamp, *self.bbox, self.saliency_score, self.periodicity_score
        )):
            raise ValueError("observation values must be finite")
        x0, y0, x1, y1 = self.bbox
        if x1 <= x0 or y1 <= y0:
            raise ValueError("bbox must have positive width and height")
        if isinstance(self.event_count, bool) or not isinstance(self.event_count, int) or self.event_count < 0:
            raise ValueError("event_count must be a nonnegative integer")
        object.__setattr__(self, "bbox", tuple(self.bbox))

    @property
    def center(self):
        x0, y0, x1, y1 = self.bbox
        return ((x0+x1)/2, (y0+y1)/2)

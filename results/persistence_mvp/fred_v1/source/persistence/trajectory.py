"""Technical document sections 10–13. Current observation never predicts itself."""
from dataclasses import dataclass, asdict
import math
from .types import Detection


@dataclass(frozen=True)
class Motion:
    velocity: tuple | None = None
    speed: float | None = None
    predicted_center: tuple | None = None
    prediction_error: float | None = None
    normalized_error: float | None = None
    direction_change: float | None = None
    acceleration: tuple | None = None
    acceleration_norm: float | None = None

    def to_dict(self):
        return asdict(self)


def motion_features(history: list[Detection], current: Detection, *, min_speed, epsilon):
    if not math.isfinite(min_speed) or min_speed < 0 or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("min_speed >= 0 and epsilon > 0 are required")
    observations = [*history, current]
    if any(not d.valid for d in observations):
        raise ValueError("motion history must contain valid observations")
    if any(b.timestamp <= a.timestamp for a, b in zip(observations, observations[1:])):
        raise ValueError("observation timestamps must strictly increase")
    if not history:
        return Motion()
    prev = history[-1]
    dt = current.timestamp-prev.timestamp
    velocity = tuple((x-y)/dt for x, y in zip(current.center, prev.center))
    speed = math.hypot(*velocity)
    if not all(math.isfinite(v) for v in (*velocity, speed)):
        raise ValueError("velocity calculation overflow")
    if len(history) < 2:
        return Motion(velocity=velocity, speed=speed)
    older = history[-2]
    prior_dt = prev.timestamp-older.timestamp
    prior = tuple((x-y)/prior_dt for x, y in zip(prev.center, older.center))
    prior_speed = math.hypot(*prior)
    predicted = tuple(x+v*dt for x, v in zip(prev.center, prior))
    error = math.dist(current.center, predicted)
    x0, y0, x1, y1 = prev.bbox
    normalized = error/(math.sqrt((x1-x0)*(y1-y0))+epsilon)
    angle = None
    if speed > 0 and prior_speed > 0 and min(speed, prior_speed) >= min_speed:
        cosine = sum(x*y for x, y in zip(velocity, prior))/(speed*prior_speed+epsilon)
        angle = math.acos(max(-1.0, min(1.0, cosine)))
    acceleration = tuple((x-y)/dt for x, y in zip(velocity, prior))
    result = Motion(velocity, speed, predicted, error, normalized, angle,
                    acceleration, math.hypot(*acceleration))
    for value in result.to_dict().values():
        values = value if isinstance(value, tuple) else (value,)
        if any(v is not None and not math.isfinite(v) for v in values):
            raise ValueError("motion calculation overflow")
    return result

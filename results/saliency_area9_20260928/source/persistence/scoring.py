"""Explicit scales and missing-term masks; no inferred experimental thresholds."""
import math


def trajectory_score(motion, *, weights, sigmas, normalize_position):
    terms = {
        "position": motion.normalized_error if normalize_position else motion.prediction_error,
        "direction": motion.direction_change,
        "acceleration": motion.acceleration_norm,
    }
    if set(weights) != set(terms) or set(sigmas) != set(terms):
        raise ValueError("weights and sigmas require position, direction, acceleration")
    if any(not math.isfinite(w) or w < 0 for w in weights.values()):
        raise ValueError("weights must be finite and nonnegative")
    active = {}
    for name, value in terms.items():
        if weights[name] == 0:
            continue
        sigma = sigmas[name]
        if sigma is None or not math.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"explicit positive sigma required: {name}")
        if value is not None:
            if not math.isfinite(value) or value < 0:
                raise ValueError("motion terms must be finite and nonnegative")
            active[name] = value
    denominator = sum(weights[k] for k in active)
    if not denominator:
        return None, {}
    effective = {k: weights[k]/denominator for k in active}
    exponent = 0.0
    for k, value in active.items():
        ratio = value/sigmas[k]
        exponent += effective[k]*ratio*ratio
    return math.exp(-exponent), effective


def fuse_scores(neighborhood, trajectory, *, alpha):
    """Document section 14: unavailable trajectory falls back to neighborhood."""
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0,1]")
    for value in (neighborhood, trajectory):
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("scores must be in [0,1] or None")
    if trajectory is None:
        return neighborhood
    if neighborhood is None:
        return trajectory
    return alpha*neighborhood+(1-alpha)*trajectory

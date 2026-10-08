"""Technical document sections 6 and 8. Input contains one hit per window."""
import math


def neighborhood_score(hits, *, length, mode, decay, cold_start):
    """Chronological hits; explicit cold-start policy avoids a hidden decision."""
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise ValueError("length must be a positive integer")
    if mode not in ("binary", "weighted") or cold_start not in ("fixed", "observed"):
        raise ValueError("unknown mode or cold-start policy")
    if not math.isfinite(decay) or not 0 < decay <= 1:
        raise ValueError("decay must be in (0,1]")
    hits = list(hits)
    if any(h not in (0, 1) for h in hits):
        raise ValueError("hits must be binary")
    hits = hits[-length:]
    if not hits:
        return None
    count = length if cold_start == "fixed" else len(hits)
    rho = decay if mode == "weighted" else 1.0
    weights = [rho**j for j in range(count)]
    return sum(h*w for h, w in zip(reversed(hits), weights)) / sum(weights)

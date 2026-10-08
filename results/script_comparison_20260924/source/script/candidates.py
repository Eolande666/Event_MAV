"""Localize a candidate from its current-window events."""
import numpy as np


def localize(events, width, height):
    lo = np.floor(np.quantile(events[:, :2], 0.02, axis=0)).astype(int)
    hi = np.ceil(np.quantile(events[:, :2], 0.98, axis=0)).astype(int)+1
    return (max(0, int(lo[0])), max(0, int(lo[1])),
            min(width, int(hi[0])), min(height, int(hi[1])))

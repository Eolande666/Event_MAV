"""
EvDetMAV-style MAV detector reproduced from:

    "EvDetMAV: Generalized MAV Detection from Moving Event Cameras"

The implementation follows the paper's three-stage pipeline:

    event period
    -> density-aware saliency map from positive/negative intersections
    -> spatio-temporal periodicity features on top-K salient areas
    -> clustering-based coarse-to-fine propeller/MAV box detection

The paper leaves several engineering thresholds unspecified. They are exposed
as command-line options while keeping the reported values tau_s=50, tau_p=3
and K=4 as defaults.
"""

from .models import (
    Box,
    Candidate,
    Detection,
    Events,
    PeriodicityScore,
    RefinedCandidate,
    WindowResult,
)
from .io import iter_input_files, load_events, prepare_events
from .saliency import build_density_saliency
from .periodicity import evaluate_periodicity
from .clustering import initialize_candidates, refine_candidate
from .pipeline import process_window
from .cli import build_parser, main, run

__all__ = [
    "Box",
    "Candidate",
    "Detection",
    "Events",
    "PeriodicityScore",
    "RefinedCandidate",
    "WindowResult",
    "iter_input_files",
    "load_events",
    "prepare_events",
    "build_density_saliency",
    "evaluate_periodicity",
    "initialize_candidates",
    "refine_candidate",
    "process_window",
    "build_parser",
    "main",
    "run",
]

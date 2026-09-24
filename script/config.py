"""Fixed-camera detector settings."""
from dataclasses import dataclass


@dataclass
class Config:
    window_ms: float = 33.333
    min_component_pixels: int = 2
    candidate_pad: int = 3
    max_candidates: int = 24
    max_roi_area: int = 65536
    min_events: int = 12
    temporal_bins: int = 8
    rotation_max_calls: int = 4
    ioc_size: int = 5
    ioc_min_support: int = 6
    ioc_condition_ratio: float = 0.04
    max_speed_px_s: float = 20000.0
    rotation_window_ms: float = 9.0
    max_misses: int = 2
    association_px: float = 24.0
    max_area_ratio: float = 4.0
    switch_saturation: float = 2.0
    switch_map_threshold: float = 0.025
    switch_support_scale: float = 8.0
    rotation_weight: float = 0.55
    repetition_weight: float = 0.30
    structure_weight: float = 0.15
    joint_threshold: float = 0.34
    persistence_penalty: float = 0.025
    background_penalty: float = 0.10
    background_ring_px: int = 5

    def validate(self):
        import math
        for name in ('persistence_penalty', 'background_penalty'):
            if not math.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'{name} must be finite and in [0, 1]')
        if isinstance(self.background_ring_px, bool) or not isinstance(self.background_ring_px, int) or self.background_ring_px < 1:
            raise ValueError('background_ring_px must be a positive integer')
        for name in ('window_ms', 'rotation_window_ms', 'min_component_pixels', 'max_candidates', 'max_roi_area',
                     'min_events', 'temporal_bins',
                     'ioc_size', 'ioc_min_support',
                     'max_speed_px_s', 'association_px', 'max_area_ratio',
                     'switch_saturation', 'switch_support_scale', 'rotation_weight',
                     'repetition_weight', 'structure_weight'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        for name in ('candidate_pad', 'rotation_max_calls', 'max_misses'):
            if getattr(self, name) < 0:
                raise ValueError(f'{name} must be nonnegative')
        if not 1 <= self.temporal_bins <= 16:
            raise ValueError('temporal_bins must be in [1,16] for switch bit masks')
        if not 0 < self.joint_threshold <= 1:
            raise ValueError('joint_threshold must be in (0, 1]')
        if self.rotation_window_ms > self.window_ms:
            raise ValueError('rotation_window_ms cannot exceed window_ms')
        if self.ioc_size % 2 != 1:
            raise ValueError('ioc_size must be odd')
        for name in ('ioc_condition_ratio', 'switch_map_threshold'):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'{name} must be in [0, 1]')

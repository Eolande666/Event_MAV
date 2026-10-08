"""Explicit, serializable MVP configuration; empirical values are not paper defaults."""
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Config:
    enabled: bool = False
    history_length: int = 8
    max_missed_windows: int = 2
    cold_start: str | None = None
    sigma_position: float | None = None
    sigma_direction: float | None = None
    sigma_acceleration: float | None = None
    threshold: float | None = None
    use_periodicity: bool = True
    use_neighborhood: bool = True
    use_trajectory: bool = True
    neighborhood_mode: str = 'weighted'
    temporal_decay: float = .85
    radius_mode: str = 'fixed'
    radius: float = 20.
    radius_scale: float = 1.5
    velocity_scale: float = 0.
    normalize_position: bool = True
    min_speed: float = .5
    epsilon: float = 1e-9
    lambda_position: float = .5
    lambda_direction: float = .25
    lambda_acceleration: float = .25
    alpha: float = .5
    decision_mode: str = 'rejection'

    @classmethod
    def load(cls, path):
        value = cls(**json.loads(Path(path).read_text()))
        value.validate()
        return value

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2, allow_nan=False)+'\n')

    def validate(self):
        for name in ('enabled','use_periodicity','use_neighborhood','use_trajectory','normalize_position'):
            if type(getattr(self,name)) is not bool:
                raise ValueError(f'{name} must be boolean')
        if not self.enabled:
            if not self.use_periodicity:
                raise ValueError('disabled persistence must preserve the original baseline periodicity')
            return
        if type(self.history_length) is not int or self.history_length < 3:
            raise ValueError('history_length must be an integer >= 3')
        if type(self.max_missed_windows) is not int or self.max_missed_windows < 0:
            raise ValueError('max_missed_windows must be a nonnegative integer')
        if self.cold_start not in ('fixed', 'observed'):
            raise ValueError('cold_start must be explicitly selected')
        if self.neighborhood_mode not in ('binary','weighted') or self.radius_mode not in ('fixed','scale'):
            raise ValueError('unsupported neighborhood mode')
        if self.decision_mode != 'rejection':
            raise ValueError('MVP supports rejection only')
        if not (self.use_neighborhood or self.use_trajectory):
            raise ValueError('disable persistence for baseline mode')
        for name in ('radius','radius_scale','epsilon'):
            self._number(name, positive=True)
        for name in ('velocity_scale','min_speed','lambda_position','lambda_direction','lambda_acceleration'):
            self._number(name)
        if not 0 < self.temporal_decay <= 1 or not math.isfinite(self.temporal_decay):
            raise ValueError('temporal_decay must be in (0,1]')
        if not math.isfinite(self.alpha) or not 0 <= self.alpha <= 1:
            raise ValueError('alpha must be in [0,1]')
        if self.use_trajectory:
            if sum(self.weights.values()) <= 0:
                raise ValueError('trajectory requires at least one positive weight')
            for name, weight in self.weights.items():
                if weight > 0:
                    self._number('sigma_'+name, positive=True)
        if self.threshold is None or not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError('threshold requires an explicit value in [0,1]')

    def _number(self,name,positive=False):
        v=getattr(self,name)
        if v is None or isinstance(v,bool) or not math.isfinite(v) or (v <= 0 if positive else v < 0):
            raise ValueError(f'{name} must be finite and '+('positive' if positive else 'nonnegative'))

    @property
    def weights(self):
        return dict(position=self.lambda_position,direction=self.lambda_direction,acceleration=self.lambda_acceleration)

    @property
    def sigmas(self):
        return dict(position=self.sigma_position,direction=self.sigma_direction,acceleration=self.sigma_acceleration)

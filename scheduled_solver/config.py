"""Configuration for the independent scheduled solver architecture."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SchedulerConfig:
    """Tuning knobs for full-horizon prioritized scheduling."""

    beam_width: int = 8
    reservation_padding: int = 1
    path_horizon: int = 512
    max_path_expansions: int = 250_000
    max_beam_depth: int = 64
    require_24_columns: bool = True

    def __post_init__(self) -> None:
        if self.beam_width <= 0:
            raise ValueError("beam_width must be positive")
        if self.reservation_padding < 0:
            raise ValueError("reservation_padding must be nonnegative")
        if self.path_horizon <= 0:
            raise ValueError("path_horizon must be positive")
        if self.max_path_expansions <= 0:
            raise ValueError("max_path_expansions must be positive")
        if self.max_beam_depth <= 0:
            raise ValueError("max_beam_depth must be positive")

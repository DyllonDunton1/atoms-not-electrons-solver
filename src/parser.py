"""Parsing utilities for BIG_ORDER.txt."""

from pathlib import Path

from .models import Order, Pallet, ProblemInstance, Robot


def parse_problem(path: str | Path) -> ProblemInstance:
    """Parse a challenge worklist into a :class:`ProblemInstance`.

    Implementation will be added in the first solver milestone.
    """
    raise NotImplementedError

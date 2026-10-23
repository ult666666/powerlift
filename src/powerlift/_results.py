"""Base class for result objects, the ConfidenceInterval type and formatting helpers."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Result(ABC):
    """Base class of every result object returned by powerlift.

    Results are immutable (frozen dataclasses). ``to_dict()`` converts a result
    into plain Python types that ``json.dumps`` accepts, and ``summary()`` (also
    used by ``str()``) renders a short human-readable report.
    """

    @abstractmethod
    def summary(self) -> str:
        """Return a human-readable, multi-line summary of the result."""

    def to_dict(self) -> dict[str, Any]:
        """Return the result as a dictionary of plain Python types.

        Nested results become nested dictionaries and tuples become lists,
        so the output can be passed directly to :func:`json.dumps`.
        """
        return {
            field.name: _to_builtin(getattr(self, field.name))
            for field in dataclasses.fields(self)
        }

    def __str__(self) -> str:
        return self.summary()


def _to_builtin(value: object) -> Any:
    if isinstance(value, Result):
        return value.to_dict()
    if isinstance(value, tuple | list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class ConfidenceInterval(Result):
    """A two-sided interval estimate.

    Used for frequentist confidence intervals and Bayesian credible intervals.

    Attributes
    ----------
    lower, upper : float
        Interval end points.
    level : float
        Nominal coverage, e.g. ``0.95``.
    """

    lower: float
    upper: float
    level: float

    @property
    def width(self) -> float:
        """Length of the interval, ``upper - lower``."""
        return self.upper - self.lower

    def contains(self, value: float) -> bool:
        """Return True if ``lower <= value <= upper``."""
        return self.lower <= value <= self.upper

    def format(self, spec: str = ".4g") -> str:
        """Format the end points with a format spec, e.g. ``ci.format(".2%")``."""
        return f"[{self.lower:{spec}}, {self.upper:{spec}}]"

    def summary(self) -> str:
        """Return e.g. ``'95% interval [0.012, 0.034]'``."""
        return f"{self.level:.0%} interval {self.format()}"


def format_p_value(p_value: float) -> str:
    """Format a p-value with four decimals, using ``< 0.0001`` for tiny values."""
    return "< 0.0001" if p_value < 1e-4 else f"{p_value:.4f}"


def format_level(level: float) -> str:
    """Format a confidence level such as 0.95 as ``'95%'`` (or ``'97.5%'``)."""
    return f"{level * 100:.4g}%"


def verdict(p_value: float, alpha: float) -> str:
    """Return the decision text used by summaries."""
    return (
        f"significant at alpha = {alpha:g}"
        if p_value <= alpha
        else f"not significant at alpha = {alpha:g}"
    )

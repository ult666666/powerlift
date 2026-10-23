"""Input validation shared by the public API.

Public functions validate their arguments up front, so that a mistake surfaces
as an :class:`~powerlift.exceptions.InvalidInputError` that names the offending
argument, instead of a NaN or an obscure NumPy error deep inside a formula.
"""

from __future__ import annotations

import math
import numbers

import numpy as np
import numpy.typing as npt

from powerlift._stats import ALTERNATIVES, Alternative
from powerlift.exceptions import InvalidInputError

FloatArray = npt.NDArray[np.float64]


def check_real(name: str, value: object) -> float:
    """Return ``value`` as a finite float."""
    # bool is a subclass of int; accepting True as "1.0" would hide bugs.
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise InvalidInputError(f"{name} must be a real number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise InvalidInputError(f"{name} must be finite, got {value!r}")
    return result


def check_positive(name: str, value: object) -> float:
    """Return ``value`` as a float, requiring it to be finite and > 0."""
    result = check_real(name, value)
    if result <= 0.0:
        raise InvalidInputError(f"{name} must be > 0, got {result:g}")
    return result


def check_open_unit(name: str, value: object) -> float:
    """Return ``value`` as a float, requiring 0 < value < 1."""
    result = check_real(name, value)
    if not 0.0 < result < 1.0:
        raise InvalidInputError(
            f"{name} must be strictly between 0 and 1, got {result:g}"
        )
    return result


def check_count(name: str, value: object, *, minimum: int = 0) -> int:
    """Return ``value`` as an int, requiring an integer >= ``minimum``."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise InvalidInputError(f"{name} must be an integer, got {value!r}")
    result = int(value)
    if result < minimum:
        raise InvalidInputError(f"{name} must be >= {minimum}, got {result}")
    return result


def check_binomial(arm: str, successes: object, trials: object) -> tuple[int, int]:
    """Validate a (conversions, n) pair for one experiment arm."""
    n = check_count(f"n_{arm}", trials, minimum=1)
    x = check_count(f"conversions_{arm}", successes)
    if x > n:
        raise InvalidInputError(f"conversions_{arm} ({x}) cannot exceed n_{arm} ({n})")
    return x, n


def check_alternative(value: object) -> Alternative:
    """Validate the ``alternative`` argument of a hypothesis test."""
    if value not in ALTERNATIVES:
        raise InvalidInputError(
            f"alternative must be one of {ALTERNATIVES}, got {value!r}"
        )
    return value


def as_float_array(
    name: str, values: npt.ArrayLike, *, min_size: int = 2
) -> FloatArray:
    """Convert ``values`` to a finite 1-D float array of at least ``min_size`` items."""
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidInputError(f"{name} must be a 1-D sequence of numbers") from exc
    if array.ndim != 1:
        raise InvalidInputError(
            f"{name} must be one-dimensional, got shape {array.shape}"
        )
    if array.size < min_size:
        raise InvalidInputError(
            f"{name} needs at least {min_size} observations, got {array.size}"
        )
    if not np.all(np.isfinite(array)):
        raise InvalidInputError(f"{name} contains NaN or infinite values")
    return array

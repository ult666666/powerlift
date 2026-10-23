"""powerlift: statistics for A/B tests done right."""

from importlib.metadata import version

from powerlift._results import ConfidenceInterval
from powerlift.exceptions import (
    ConvergenceError,
    InsufficientDataError,
    InvalidInputError,
    PowerliftError,
)

__version__ = version("powerlift")

__all__ = [
    "ConfidenceInterval",
    "ConvergenceError",
    "InsufficientDataError",
    "InvalidInputError",
    "PowerliftError",
    "__version__",
]

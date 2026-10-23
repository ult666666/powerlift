"""powerlift: statistics for A/B tests done right."""

from importlib.metadata import version

from powerlift._results import ConfidenceInterval
from powerlift.design import (
    SampleSizeResult,
    experiment_duration,
    mde_means,
    mde_proportions,
    power_means,
    power_proportions,
    sample_size_means,
    sample_size_proportions,
)
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
    "SampleSizeResult",
    "__version__",
    "experiment_duration",
    "mde_means",
    "mde_proportions",
    "power_means",
    "power_proportions",
    "sample_size_means",
    "sample_size_proportions",
]

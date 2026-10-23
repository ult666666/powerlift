"""powerlift: statistics for A/B tests done right."""

from importlib.metadata import version

from powerlift._results import ConfidenceInterval
from powerlift.analysis import (
    ComparisonResult,
    ProportionComparison,
    compare_means,
    compare_means_from_stats,
    compare_proportions,
    compare_ratio_metrics,
    wilson_interval,
)
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
    "ComparisonResult",
    "ConfidenceInterval",
    "ConvergenceError",
    "InsufficientDataError",
    "InvalidInputError",
    "PowerliftError",
    "ProportionComparison",
    "SampleSizeResult",
    "__version__",
    "compare_means",
    "compare_means_from_stats",
    "compare_proportions",
    "compare_ratio_metrics",
    "experiment_duration",
    "mde_means",
    "mde_proportions",
    "power_means",
    "power_proportions",
    "sample_size_means",
    "sample_size_proportions",
    "wilson_interval",
]

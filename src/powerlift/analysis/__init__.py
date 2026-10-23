"""Frequentist analysis of conversion, continuous and ratio metrics.

* :func:`compare_proportions` - pooled two-proportion z-test with Newcombe (or
  Wald) intervals for the difference, a log-ratio interval for the relative
  lift and Wilson intervals per arm.
* :func:`compare_means` / :func:`compare_means_from_stats` - Welch's t-test.
* :func:`compare_ratio_metrics` - delta-method test for ratio metrics.
"""

from powerlift.analysis._common import ComparisonResult
from powerlift.analysis.means import compare_means, compare_means_from_stats
from powerlift.analysis.proportions import (
    ProportionComparison,
    compare_proportions,
    wilson_interval,
)
from powerlift.analysis.ratio import compare_ratio_metrics

__all__ = [
    "ComparisonResult",
    "ProportionComparison",
    "compare_means",
    "compare_means_from_stats",
    "compare_proportions",
    "compare_ratio_metrics",
    "wilson_interval",
]

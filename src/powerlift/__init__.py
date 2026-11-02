"""powerlift: statistics for A/B tests done right."""

from importlib.metadata import version

from powerlift._results import ConfidenceInterval
from powerlift.analysis import (
    ArmComparison,
    ComparisonResult,
    ManyToOneComparison,
    ProportionComparison,
    compare_means,
    compare_means_from_stats,
    compare_means_to_control,
    compare_proportions,
    compare_proportions_to_control,
    compare_ratio_metrics,
    dunnett_critical_value,
    wilson_interval,
)
from powerlift.bayes import (
    BayesianComparison,
    BetaPosterior,
    compare_proportions_bayes,
    prob_beta_greater,
)
from powerlift.cuped import CupedResult, compare_means_cuped
from powerlift.design import (
    MultiArmSampleSize,
    SampleSizeResult,
    experiment_duration,
    mde_means,
    mde_proportions,
    power_means,
    power_proportions,
    sample_size_means,
    sample_size_multi_arm_means,
    sample_size_multi_arm_proportions,
    sample_size_proportions,
)
from powerlift.exceptions import (
    ConvergenceError,
    InsufficientDataError,
    InvalidInputError,
    PowerliftError,
)
from powerlift.multiple_testing import MultipleTestingResult, adjust_p_values
from powerlift.sanity import AASimulationResult, SRMResult, simulate_aa, srm_test
from powerlift.sequential import (
    AlwaysValidResult,
    GroupSequentialBoundaries,
    always_valid_inference,
    crossing_probability,
    group_sequential_boundaries,
)

__version__ = version("powerlift")

__all__ = [
    "AASimulationResult",
    "AlwaysValidResult",
    "ArmComparison",
    "BayesianComparison",
    "BetaPosterior",
    "ComparisonResult",
    "ConfidenceInterval",
    "ConvergenceError",
    "CupedResult",
    "GroupSequentialBoundaries",
    "InsufficientDataError",
    "InvalidInputError",
    "ManyToOneComparison",
    "MultiArmSampleSize",
    "MultipleTestingResult",
    "PowerliftError",
    "ProportionComparison",
    "SRMResult",
    "SampleSizeResult",
    "__version__",
    "adjust_p_values",
    "always_valid_inference",
    "compare_means",
    "compare_means_cuped",
    "compare_means_from_stats",
    "compare_means_to_control",
    "compare_proportions",
    "compare_proportions_bayes",
    "compare_proportions_to_control",
    "compare_ratio_metrics",
    "crossing_probability",
    "dunnett_critical_value",
    "experiment_duration",
    "group_sequential_boundaries",
    "mde_means",
    "mde_proportions",
    "power_means",
    "power_proportions",
    "prob_beta_greater",
    "sample_size_means",
    "sample_size_multi_arm_means",
    "sample_size_multi_arm_proportions",
    "sample_size_proportions",
    "simulate_aa",
    "srm_test",
    "wilson_interval",
]

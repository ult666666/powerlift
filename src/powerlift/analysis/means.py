"""Welch's t-test for a difference in means between two arms."""

from __future__ import annotations

import numpy.typing as npt

from powerlift._stats import Alternative
from powerlift._validation import (
    as_float_array,
    check_alternative,
    check_count,
    check_open_unit,
    check_real,
)
from powerlift.analysis._common import ComparisonResult, compare_estimates
from powerlift.exceptions import InsufficientDataError, InvalidInputError


def compare_means(
    control: npt.ArrayLike,
    treatment: npt.ArrayLike,
    *,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
) -> ComparisonResult:
    """Compare two means with Welch's unequal-variance t-test.

    Parameters
    ----------
    control, treatment : array_like
        One value per randomization unit (e.g. revenue per user).
    alpha : float, default 0.05
        Significance level; intervals have confidence ``1 - alpha``.
    alternative : {"two-sided", "greater", "less"}, default "two-sided"
        Alternative hypothesis for ``mean(treatment) - mean(control)``.

    Returns
    -------
    ComparisonResult
        Difference in means with a t-interval and a delta-method interval for
        the relative lift. Intervals are always two-sided.

    Notes
    -----
    The statistic is ``t = (mean_t - mean_c) / sqrt(s_c^2/n_c + s_t^2/n_t)``
    with Welch-Satterthwaite degrees of freedom::

        df = (s_c^2/n_c + s_t^2/n_t)^2
             / ((s_c^2/n_c)^2 / (n_c - 1) + (s_t^2/n_t)^2 / (n_t - 1))

    It matches ``scipy.stats.ttest_ind(treatment, control, equal_var=False)``.
    Welch's test does not assume equal variances and is accurate for skewed
    metrics once samples are large (central limit theorem).

    References
    ----------
    Welch, B. L. (1947). The generalization of "Student's" problem when several
    different population variances are involved. *Biometrika*, 34, 28-35.
    """
    control_values = as_float_array("control", control)
    treatment_values = as_float_array("treatment", treatment)
    return compare_means_from_stats(
        mean_control=float(control_values.mean()),
        sd_control=float(control_values.std(ddof=1)),
        n_control=control_values.size,
        mean_treatment=float(treatment_values.mean()),
        sd_treatment=float(treatment_values.std(ddof=1)),
        n_treatment=treatment_values.size,
        alpha=alpha,
        alternative=alternative,
    )


def compare_means_from_stats(
    *,
    mean_control: float,
    sd_control: float,
    n_control: int,
    mean_treatment: float,
    sd_treatment: float,
    n_treatment: int,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
) -> ComparisonResult:
    """Welch's t-test from summary statistics (means, sample SDs, sizes).

    ``sd_*`` are sample standard deviations (``ddof=1``). See
    :func:`compare_means` for details; this variant is useful when only
    aggregates are available, e.g. from a metrics warehouse.
    """
    n_c = check_count("n_control", n_control, minimum=2)
    n_t = check_count("n_treatment", n_treatment, minimum=2)
    sd_c, sd_t = (
        check_real("sd_control", sd_control),
        check_real("sd_treatment", sd_treatment),
    )
    if sd_c < 0 or sd_t < 0:
        raise InvalidInputError("standard deviations must be non-negative")
    var_c, var_t = sd_c**2 / n_c, sd_t**2 / n_t
    if var_c + var_t == 0.0:
        raise InsufficientDataError(
            "both arms have zero variance; the t-test is undefined"
        )
    df = (var_c + var_t) ** 2 / (var_c**2 / (n_c - 1) + var_t**2 / (n_t - 1))
    return compare_estimates(
        method="Welch's t-test",
        n_control=n_c,
        n_treatment=n_t,
        estimate_control=check_real("mean_control", mean_control),
        estimate_treatment=check_real("mean_treatment", mean_treatment),
        variance_control=var_c,
        variance_treatment=var_t,
        df=df,
        alpha=check_open_unit("alpha", alpha),
        alternative=check_alternative(alternative),
    )

"""Delta-method analysis for ratio metrics such as revenue per session."""

from __future__ import annotations

import numpy.typing as npt

from powerlift._stats import Alternative
from powerlift._validation import (
    FloatArray,
    as_float_array,
    check_alternative,
    check_open_unit,
)
from powerlift.analysis._common import ComparisonResult, compare_estimates
from powerlift.exceptions import InvalidInputError


def compare_ratio_metrics(
    numerator_control: npt.ArrayLike,
    denominator_control: npt.ArrayLike,
    numerator_treatment: npt.ArrayLike,
    denominator_treatment: npt.ArrayLike,
    *,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
) -> ComparisonResult:
    """Compare a ratio metric whose analysis unit differs from the randomization unit.

    Typical example: users are randomized, but the metric is revenue per
    *session*, ``R = sum(revenue) / sum(sessions)``. Sessions of the same
    user are correlated, so treating sessions as independent observations
    underestimates the variance. Instead each randomized unit ``i``
    contributes a numerator ``Y_i`` and a denominator ``X_i``, and the
    variance of ``R = mean(Y) / mean(X)`` comes from the delta method.

    Parameters
    ----------
    numerator_control, denominator_control : array_like
        Per-unit numerators and denominators in the control arm (same length).
    numerator_treatment, denominator_treatment : array_like
        Per-unit numerators and denominators in the treatment arm.
    alpha : float, default 0.05
        Significance level; intervals have confidence ``1 - alpha``.
    alternative : {"two-sided", "greater", "less"}, default "two-sided"
        Alternative hypothesis for ``R_treatment - R_control``.

    Returns
    -------
    ComparisonResult
        Ratios per arm, their difference with a z-test and normal interval,
        and a delta-method interval for the relative lift.

    Notes
    -----
    For one arm with ``n`` units, a first-order Taylor expansion of
    ``mean(Y) / mean(X)`` around the population means gives::

        Var(R) ~= (s_Y^2 - 2 R s_XY + R^2 s_X^2) / (n * mean(X)^2)
               =  Var(Y - R X) / (n * mean(X)^2)

    where ``s`` are sample (co)variances. Arms are independent, so the
    variance of the difference is the sum of the two. The approximation needs
    many units per arm and a denominator mean well away from zero.

    References
    ----------
    Deng, A., Knoblich, U., & Lu, J. (2018). Applying the delta method in
    metric analytics: a practical guide with novel ideas. *KDD 2018*.
    """
    alpha = check_open_unit("alpha", alpha)
    alternative = check_alternative(alternative)
    ratio_c, variance_c, n_c = _ratio_and_variance(
        "control", numerator_control, denominator_control
    )
    ratio_t, variance_t, n_t = _ratio_and_variance(
        "treatment", numerator_treatment, denominator_treatment
    )
    return compare_estimates(
        method="Delta-method z-test (ratio metric)",
        n_control=n_c,
        n_treatment=n_t,
        estimate_control=ratio_c,
        estimate_treatment=ratio_t,
        variance_control=variance_c,
        variance_treatment=variance_t,
        df=None,
        alpha=alpha,
        alternative=alternative,
    )


def _ratio_and_variance(
    arm: str, numerator: npt.ArrayLike, denominator: npt.ArrayLike
) -> tuple[float, float, int]:
    y: FloatArray = as_float_array(f"numerator_{arm}", numerator)
    x: FloatArray = as_float_array(f"denominator_{arm}", denominator)
    if y.size != x.size:
        raise InvalidInputError(
            f"numerator_{arm} and denominator_{arm} must have the same length "
            f"({y.size} != {x.size})"
        )
    x_mean = float(x.mean())
    if x_mean <= 0.0 or x_mean * x_mean == 0.0:  # the square is used below
        raise InvalidInputError(
            f"denominator_{arm} must have a positive mean, got {x_mean:g}"
        )
    ratio = float(y.mean()) / x_mean
    # Var(Y - R X) with ddof=1 equals s_Y^2 - 2 R s_XY + R^2 s_X^2 exactly.
    variance = float((y - ratio * x).var(ddof=1)) / (y.size * x_mean**2)
    return ratio, variance, y.size

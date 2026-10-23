"""CUPED variance reduction with a pre-experiment covariate.

CUPED (Controlled-experiment Using Pre-Experiment Data) replaces the metric
``Y`` with ``Y_cuped = Y - theta * (X - mean(X))``, where ``X`` is a covariate
measured *before* the experiment (typically the same metric in a pre-period).
Because ``X`` cannot be affected by the treatment, both arms have the same
expected ``X``, so the adjusted difference in means is still an unbiased
estimate of the treatment effect, while its variance shrinks by the factor
``1 - corr(Y, X)^2`` when ``theta = cov(Y, X) / var(X)``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from powerlift._results import Result, format_level, format_p_value
from powerlift._stats import Alternative
from powerlift._validation import FloatArray, as_float_array
from powerlift.analysis import ComparisonResult, compare_means
from powerlift.exceptions import InsufficientDataError, InvalidInputError


@dataclass(frozen=True)
class CupedResult(Result):
    """Result of a CUPED-adjusted comparison of means.

    Attributes
    ----------
    theta : float
        Adjustment coefficient ``cov(Y, X) / var(X)`` estimated on the pooled
        data of both arms.
    correlation : float
        Pooled Pearson correlation between metric and covariate.
    variance_reduction : float
        ``1 - Var(adjusted difference) / Var(unadjusted difference)``, the
        share of variance removed; close to ``correlation ** 2``.
    adjusted : ComparisonResult
        Welch's t-test on the CUPED-adjusted metric.
    unadjusted : ComparisonResult
        Welch's t-test on the raw metric, for comparison.
    """

    theta: float
    correlation: float
    variance_reduction: float
    adjusted: ComparisonResult
    unadjusted: ComparisonResult

    def summary(self) -> str:
        """Return a human-readable summary."""
        level = format_level(1 - self.adjusted.alpha)

        def row(label: str, result: ComparisonResult) -> str:
            return (
                f"  {label:<11} difference {result.difference:+.6g}   {level} CI "
                f"{result.ci_difference.format('+.6g')}   "
                f"p-value = {format_p_value(result.p_value)}"
            )

        return "\n".join(
            [
                f"CUPED (Deng et al. 2013): theta = {self.theta:.4g}, "
                f"corr(metric, covariate) = {self.correlation:.3f}",
                f"  variance of the estimate reduced by {self.variance_reduction:.1%} "
                f"(corr^2 = {self.correlation**2:.1%})",
                row("unadjusted", self.unadjusted),
                row("adjusted", self.adjusted),
            ]
        )


def compare_means_cuped(
    control: npt.ArrayLike,
    control_covariate: npt.ArrayLike,
    treatment: npt.ArrayLike,
    treatment_covariate: npt.ArrayLike,
    *,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
) -> CupedResult:
    """Compare means after CUPED adjustment with a pre-experiment covariate.

    Parameters
    ----------
    control, treatment : array_like
        Metric value per unit in each arm.
    control_covariate, treatment_covariate : array_like
        Pre-experiment covariate per unit, aligned with the metric arrays.
        It must not be influenced by the treatment.
    alpha : float, default 0.05
        Significance level; intervals have confidence ``1 - alpha``.
    alternative : {"two-sided", "greater", "less"}, default "two-sided"
        Alternative hypothesis for the difference in means.

    Returns
    -------
    CupedResult
        ``theta``, the variance reduction and Welch's t-tests on the adjusted
        and on the raw metric.

    Notes
    -----
    ``theta`` is the slope of the pooled regression of ``Y`` on ``X``, and the
    covariate is centred at its pooled mean so the adjusted metric keeps the
    scale of the original one. The adjusted values are analysed with Welch's
    t-test, treating ``theta`` as fixed; estimating it from the same data adds
    variance of order ``1/n``, negligible at experiment sample sizes.

    References
    ----------
    Deng, A., Xu, Y., Kohavi, R., & Walker, T. (2013). Improving the
    sensitivity of online controlled experiments by utilizing pre-experiment
    data. *WSDM 2013*.
    """
    y_c, x_c = _aligned("control", control, control_covariate)
    y_t, x_t = _aligned("treatment", treatment, treatment_covariate)
    y = np.concatenate([y_c, y_t])
    x = np.concatenate([x_c, x_t])
    covariance = np.cov(x, y, ddof=1)
    if covariance[0, 0] == 0.0:
        raise InsufficientDataError(
            "the covariate is constant, so it cannot explain variance"
        )
    theta = float(covariance[0, 1] / covariance[0, 0])
    x_mean = float(x.mean())

    unadjusted = compare_means(y_c, y_t, alpha=alpha, alternative=alternative)
    adjusted = compare_means(
        y_c - theta * (x_c - x_mean),
        y_t - theta * (x_t - x_mean),
        alpha=alpha,
        alternative=alternative,
    )
    adjusted = dataclasses.replace(
        adjusted, method="Welch's t-test on CUPED-adjusted metric"
    )
    correlation = float(covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1]))
    return CupedResult(
        theta=theta,
        correlation=correlation,
        variance_reduction=1.0
        - adjusted.se_difference**2 / unadjusted.se_difference**2,
        adjusted=adjusted,
        unadjusted=unadjusted,
    )


def _aligned(
    arm: str, metric: npt.ArrayLike, covariate: npt.ArrayLike
) -> tuple[FloatArray, FloatArray]:
    y = as_float_array(arm, metric)
    x = as_float_array(f"{arm}_covariate", covariate)
    if y.size != x.size:
        raise InvalidInputError(
            f"{arm} and {arm}_covariate must have the same length "
            f"({y.size} != {x.size})"
        )
    return y, x

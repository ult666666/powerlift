"""The result type shared by mean, ratio and CUPED comparisons."""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats

from powerlift._results import (
    ConfidenceInterval,
    Result,
    format_level,
    format_p_value,
    verdict,
)
from powerlift._stats import (
    Alternative,
    normal_critical_value,
    normal_p_value,
    t_p_value,
)
from powerlift.exceptions import InsufficientDataError


@dataclass(frozen=True)
class ComparisonResult(Result):
    """Comparison of a metric between a control and a treatment arm.

    Attributes
    ----------
    method : str
        Name of the test that produced the result.
    n_control, n_treatment : int
        Number of randomization units in each arm.
    estimate_control, estimate_treatment : float
        Per-arm estimates (means, or ratios for ratio metrics).
    difference : float
        ``estimate_treatment - estimate_control``.
    se_difference : float
        Standard error of ``difference``.
    relative_lift : float or None
        ``estimate_treatment / estimate_control - 1``; ``None`` when the
        control estimate is zero.
    statistic : float
        ``difference / se_difference`` (a t- or z-statistic).
    df : float or None
        Degrees of freedom of the t reference distribution, or ``None`` when
        the statistic is compared with a standard normal.
    p_value : float
        P-value for ``alternative``.
    alternative : {"two-sided", "greater", "less"}
        Alternative hypothesis for ``treatment - control``.
    alpha : float
        Significance level; intervals have confidence ``1 - alpha``.
    significant : bool
        ``p_value <= alpha``.
    ci_difference : ConfidenceInterval
        Two-sided interval for ``difference``.
    ci_relative_lift : ConfidenceInterval or None
        Two-sided delta-method interval for ``relative_lift``.
    """

    method: str
    n_control: int
    n_treatment: int
    estimate_control: float
    estimate_treatment: float
    difference: float
    se_difference: float
    relative_lift: float | None
    statistic: float
    df: float | None
    p_value: float
    alternative: Alternative
    alpha: float
    significant: bool
    ci_difference: ConfidenceInterval
    ci_relative_lift: ConfidenceInterval | None

    def summary(self) -> str:
        """Return a human-readable summary."""
        level = format_level(1 - self.alpha)
        lift = (
            "n/a (control estimate is zero)"
            if self.relative_lift is None or self.ci_relative_lift is None
            else f"{self.relative_lift:+.2%}   {level} CI "
            f"{self.ci_relative_lift.format('+.2%')}"
        )
        statistic = (
            f"z = {self.statistic:.3f}"
            if self.df is None
            else f"t = {self.statistic:.3f} (df = {self.df:.1f})"
        )
        return "\n".join(
            [
                f"{self.method}, {self.alternative}, alpha = {self.alpha:g}",
                f"  control      n = {self.n_control:<8,} estimate = "
                f"{self.estimate_control:.6g}",
                f"  treatment    n = {self.n_treatment:<8,} estimate = "
                f"{self.estimate_treatment:.6g}",
                f"  difference   {self.difference:+.6g}   {level} CI "
                f"{self.ci_difference.format('+.6g')}",
                f"  lift         {lift}",
                f"  {statistic}, p-value = {format_p_value(self.p_value)} "
                f"({verdict(self.p_value, self.alpha)})",
            ]
        )


def compare_estimates(
    *,
    method: str,
    n_control: int,
    n_treatment: int,
    estimate_control: float,
    estimate_treatment: float,
    variance_control: float,
    variance_treatment: float,
    df: float | None,
    alpha: float,
    alternative: Alternative,
) -> ComparisonResult:
    """Build a :class:`ComparisonResult` from two independent estimates.

    ``variance_*`` are the sampling variances of the per-arm estimates
    (squared standard errors). With ``df=None`` the statistic is referred to
    a standard normal distribution, otherwise to Student's t with ``df``.
    """
    se = math.sqrt(variance_control + variance_treatment)
    if se == 0.0:
        raise InsufficientDataError(
            "the standard error of the difference is zero; the data show no variation"
        )
    difference = estimate_treatment - estimate_control
    statistic = difference / se
    if df is None:
        p_value = normal_p_value(statistic, alternative)
        critical = normal_critical_value(alpha)
    else:
        p_value = t_p_value(statistic, df, alternative)
        critical = float(stats.t.isf(alpha / 2.0, df))
    lift, ci_lift = relative_lift_interval(
        estimate_control,
        estimate_treatment,
        variance_control,
        variance_treatment,
        alpha,
    )
    return ComparisonResult(
        method=method,
        n_control=n_control,
        n_treatment=n_treatment,
        estimate_control=estimate_control,
        estimate_treatment=estimate_treatment,
        difference=difference,
        se_difference=se,
        relative_lift=lift,
        statistic=statistic,
        df=df,
        p_value=p_value,
        alternative=alternative,
        alpha=alpha,
        significant=p_value <= alpha,
        ci_difference=ConfidenceInterval(
            difference - critical * se, difference + critical * se, 1.0 - alpha
        ),
        ci_relative_lift=ci_lift,
    )


def relative_lift_interval(
    estimate_control: float,
    estimate_treatment: float,
    variance_control: float,
    variance_treatment: float,
    alpha: float,
) -> tuple[float | None, ConfidenceInterval | None]:
    """Delta-method interval for the relative lift ``T / C - 1``.

    For independent arms, a first-order Taylor expansion of ``g(C, T) = T / C``
    gives ``Var(T / C) ~= Var(T) / C^2 + T^2 Var(C) / C^4`` (Deng, Knoblich &
    Lu 2018; Kohavi et al. 2020, ch. 18). The interval is symmetric on the
    lift scale and uses the normal critical value.
    """
    if estimate_control == 0.0:
        return None, None
    lift = estimate_treatment / estimate_control - 1.0
    variance = (
        variance_treatment / estimate_control**2
        + estimate_treatment**2 * variance_control / estimate_control**4
    )
    half_width = normal_critical_value(alpha) * math.sqrt(variance)
    return lift, ConfidenceInterval(lift - half_width, lift + half_width, 1.0 - alpha)

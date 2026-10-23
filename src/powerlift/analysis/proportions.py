"""Two-proportion z-test and confidence intervals for conversion metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from powerlift._results import (
    ConfidenceInterval,
    Result,
    format_level,
    format_p_value,
    verdict,
)
from powerlift._stats import Alternative, normal_critical_value, normal_p_value
from powerlift._validation import check_alternative, check_binomial, check_open_unit
from powerlift.exceptions import InsufficientDataError, InvalidInputError

DifferenceCI = Literal["newcombe", "wald"]
_DIFFERENCE_CIS: tuple[DifferenceCI, ...] = ("newcombe", "wald")


@dataclass(frozen=True)
class ProportionComparison(Result):
    """Result of comparing two conversion rates.

    Attributes
    ----------
    conversions_control, n_control, conversions_treatment, n_treatment : int
        The input counts.
    rate_control, rate_treatment : float
        Observed conversion rates.
    difference : float
        ``rate_treatment - rate_control`` (absolute difference).
    relative_lift : float or None
        ``rate_treatment / rate_control - 1``; ``None`` if either arm has zero
        conversions (the log-ratio interval is undefined there).
    z_statistic, p_value : float
        Pooled two-proportion z-test.
    alternative : {"two-sided", "greater", "less"}
        Alternative hypothesis for ``treatment - control``.
    alpha : float
        Significance level; all intervals have confidence ``1 - alpha``.
    significant : bool
        ``p_value <= alpha``.
    ci_method : {"newcombe", "wald"}
        Method used for ``ci_difference``.
    ci_difference : ConfidenceInterval
        Interval for the absolute difference.
    ci_relative_lift : ConfidenceInterval or None
        Delta-method interval for the relative lift, computed on the log ratio.
    ci_control, ci_treatment : ConfidenceInterval
        Wilson score intervals for each arm's rate.
    """

    conversions_control: int
    n_control: int
    conversions_treatment: int
    n_treatment: int
    rate_control: float
    rate_treatment: float
    difference: float
    relative_lift: float | None
    z_statistic: float
    p_value: float
    alternative: Alternative
    alpha: float
    significant: bool
    ci_method: DifferenceCI
    ci_difference: ConfidenceInterval
    ci_relative_lift: ConfidenceInterval | None
    ci_control: ConfidenceInterval
    ci_treatment: ConfidenceInterval

    def summary(self) -> str:
        """Return a human-readable summary."""
        level = format_level(1 - self.alpha)
        diff_ci = (
            f"[{self.ci_difference.lower * 100:+.2f} pp, "
            f"{self.ci_difference.upper * 100:+.2f} pp]"
        )
        lift = (
            "n/a (an arm has zero conversions)"
            if self.relative_lift is None or self.ci_relative_lift is None
            else f"{self.relative_lift:+.2%}     {level} CI "
            f"{self.ci_relative_lift.format('+.2%')} (log-ratio delta method)"
        )
        method = "Newcombe" if self.ci_method == "newcombe" else "Wald"
        p_value = format_p_value(self.p_value)
        return "\n".join(
            [
                "Two-proportion z-test (pooled), "
                f"{self.alternative}, alpha = {self.alpha:g}",
                self._arm_line("control", self.conversions_control, self.n_control),
                self._arm_line(
                    "treatment", self.conversions_treatment, self.n_treatment
                ),
                f"  difference   {self.difference * 100:+.2f} pp   "
                f"{level} CI {diff_ci} ({method})",
                f"  lift         {lift}",
                f"  z = {self.z_statistic:.3f}, p-value = {p_value} "
                f"({verdict(self.p_value, self.alpha)})",
            ]
        )

    def _arm_line(self, arm: str, conversions: int, n: int) -> str:
        ci = self.ci_control if arm == "control" else self.ci_treatment
        return (
            f"  {arm:<12} {conversions:,} / {n:,} = {conversions / n:.2%}   "
            f"{format_level(1 - self.alpha)} Wilson CI {ci.format('.2%')}"
        )


def wilson_interval(
    successes: int, trials: int, *, alpha: float = 0.05
) -> ConfidenceInterval:
    """Wilson score interval for a single binomial proportion.

    Parameters
    ----------
    successes, trials : int
        Number of conversions and number of units.
    alpha : float, default 0.05
        The interval has confidence ``1 - alpha``.

    Returns
    -------
    ConfidenceInterval
        The set of rates ``p`` not rejected by the score test
        ``|p_hat - p| / sqrt(p (1 - p) / n) <= z``::

            center = (p_hat + z^2 / (2n)) / (1 + z^2 / n)
            half   = z * sqrt(p_hat (1 - p_hat) / n + z^2 / (4 n^2)) / (1 + z^2 / n)

    Notes
    -----
    Unlike the Wald interval it never leaves [0, 1] and keeps close to nominal
    coverage for small ``n`` or rates near 0 or 1. Matches
    ``statsmodels.stats.proportion.proportion_confint(method="wilson")``.

    References
    ----------
    Wilson, E. B. (1927). Probable inference, the law of succession, and
    statistical inference. *JASA*, 22(158), 209-212.

    Examples
    --------
    >>> wilson_interval(0, 10).format(".4f")
    '[0.0000, 0.2775]'
    """
    x, n = check_binomial("arm", successes, trials)
    alpha = check_open_unit("alpha", alpha)
    return _wilson(x, n, alpha)


def compare_proportions(
    conversions_control: int,
    n_control: int,
    conversions_treatment: int,
    n_treatment: int,
    *,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
    ci_method: DifferenceCI = "newcombe",
) -> ProportionComparison:
    """Compare two conversion rates with the pooled two-proportion z-test.

    Parameters
    ----------
    conversions_control, n_control : int
        Conversions and units in the control arm.
    conversions_treatment, n_treatment : int
        Conversions and units in the treatment arm.
    alpha : float, default 0.05
        Significance level; intervals have confidence ``1 - alpha``.
    alternative : {"two-sided", "greater", "less"}, default "two-sided"
        Alternative hypothesis for ``rate_treatment - rate_control``.
    ci_method : {"newcombe", "wald"}, default "newcombe"
        Interval for the absolute difference. Newcombe's hybrid score
        interval (method 10 in Newcombe 1998) avoids the under-coverage of
        the Wald interval when rates are small or samples are modest; in
        those settings it errs on the conservative side instead.

    Returns
    -------
    ProportionComparison
        Test statistic, p-value and intervals. Intervals are always two-sided.

    Raises
    ------
    InsufficientDataError
        If the pooled rate is 0 or 1 (no variation at all), where the
        z-statistic is undefined.

    Notes
    -----
    * Test: ``z = (p_t - p_c) / sqrt(p_bar (1 - p_bar) (1/n_c + 1/n_t))`` with
      the pooled rate ``p_bar = (x_c + x_t) / (n_c + n_t)``; matches
      ``statsmodels.stats.proportion.proportions_ztest``.
    * Newcombe interval, with Wilson limits ``(l_c, u_c)`` and ``(l_t, u_t)``::

          [d - sqrt((p_t - l_t)^2 + (u_c - p_c)^2),
           d + sqrt((u_t - p_t)^2 + (p_c - l_c)^2)]

    * Relative lift: the delta method on ``log(p_t / p_c)`` gives
      ``Var ~= (1 - p_t) / x_t + (1 - p_c) / x_c``; the interval is computed
      on the log scale and mapped back with ``exp(.) - 1``, so it is
      asymmetric and never extends below -100%.

    Because the test uses the pooled variance and the interval does not, the
    two can disagree in borderline cases.

    References
    ----------
    Newcombe, R. G. (1998). Interval estimation for the difference between
    independent proportions: comparison of eleven methods. *Statistics in
    Medicine*, 17(8), 873-890.

    Examples
    --------
    >>> result = compare_proportions(480, 5000, 560, 5000)
    >>> round(result.p_value, 4)
    0.0088
    """
    x_c, n_c = check_binomial("control", conversions_control, n_control)
    x_t, n_t = check_binomial("treatment", conversions_treatment, n_treatment)
    alpha = check_open_unit("alpha", alpha)
    alternative = check_alternative(alternative)
    if ci_method not in _DIFFERENCE_CIS:
        raise InvalidInputError(
            f"ci_method must be one of {_DIFFERENCE_CIS}, got {ci_method!r}"
        )

    p_c, p_t = x_c / n_c, x_t / n_t
    pooled = (x_c + x_t) / (n_c + n_t)
    se_pooled = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n_c + 1.0 / n_t))
    if se_pooled == 0.0:
        raise InsufficientDataError(
            "both arms have a conversion rate of 0% (or both 100%); "
            "the z-test is undefined without any variation"
        )
    z = (p_t - p_c) / se_pooled
    p_value = normal_p_value(z, alternative)

    ci_control, ci_treatment = _wilson(x_c, n_c, alpha), _wilson(x_t, n_t, alpha)
    if ci_method == "newcombe":
        ci_difference = _newcombe(p_c, p_t, ci_control, ci_treatment, alpha)
    else:
        ci_difference = _wald_difference(p_c, n_c, p_t, n_t, alpha)
    relative_lift, ci_relative_lift = _log_ratio_lift(x_c, n_c, x_t, n_t, alpha)

    return ProportionComparison(
        conversions_control=x_c,
        n_control=n_c,
        conversions_treatment=x_t,
        n_treatment=n_t,
        rate_control=p_c,
        rate_treatment=p_t,
        difference=p_t - p_c,
        relative_lift=relative_lift,
        z_statistic=z,
        p_value=p_value,
        alternative=alternative,
        alpha=alpha,
        significant=p_value <= alpha,
        ci_method=ci_method,
        ci_difference=ci_difference,
        ci_relative_lift=ci_relative_lift,
        ci_control=ci_control,
        ci_treatment=ci_treatment,
    )


def _wilson(x: int, n: int, alpha: float) -> ConfidenceInterval:
    z = normal_critical_value(alpha)
    p_hat = x / n
    denominator = 1.0 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    half_width = (
        z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denominator
    )
    # The exact limits are 0 when x = 0 and 1 when x = n; set them explicitly so
    # floating-point rounding cannot leave values such as 2e-19.
    lower = 0.0 if x == 0 else center - half_width
    upper = 1.0 if x == n else center + half_width
    return ConfidenceInterval(lower, upper, 1.0 - alpha)


def _newcombe(
    p_c: float,
    p_t: float,
    ci_control: ConfidenceInterval,
    ci_treatment: ConfidenceInterval,
    alpha: float,
) -> ConfidenceInterval:
    difference = p_t - p_c
    lower = difference - math.hypot(p_t - ci_treatment.lower, ci_control.upper - p_c)
    upper = difference + math.hypot(ci_treatment.upper - p_t, p_c - ci_control.lower)
    return ConfidenceInterval(lower, upper, 1.0 - alpha)


def _wald_difference(
    p_c: float, n_c: int, p_t: float, n_t: int, alpha: float
) -> ConfidenceInterval:
    se = math.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    half_width = normal_critical_value(alpha) * se
    difference = p_t - p_c
    return ConfidenceInterval(
        difference - half_width, difference + half_width, 1.0 - alpha
    )


def _log_ratio_lift(
    x_c: int, n_c: int, x_t: int, n_t: int, alpha: float
) -> tuple[float | None, ConfidenceInterval | None]:
    if x_c == 0 or x_t == 0:
        return None, None
    ratio = (x_t / n_t) / (x_c / n_c)
    # Delta method: Var(log p_hat) ~= (1 - p) / (n p) = 1/x - 1/n for each arm.
    se_log_ratio = math.sqrt(1 / x_t - 1 / n_t + 1 / x_c - 1 / n_c)
    factor = math.exp(normal_critical_value(alpha) * se_log_ratio)
    return ratio - 1.0, ConfidenceInterval(
        ratio / factor - 1.0, ratio * factor - 1.0, 1.0 - alpha
    )

"""Sample size, power and minimum detectable effect (MDE) for two-arm experiments.

All calculations use the normal approximation for the difference between the
treatment and control estimates. Let ``n_c`` be the control sample size,
``r = n_t / n_c`` the allocation ratio, ``delta`` the true effect (treatment
minus control), ``z_a = z_{1-alpha/2}`` for a two-sided test (``z_{1-alpha}``
for a one-sided test) and ``z_b = z_{power}``.

* **Two means** with a common standard deviation ``sigma``::

      n_c = (z_a + z_b)^2 * sigma^2 * (1 + 1/r) / delta^2

* **Two proportions** ``p_c -> p_t = p_c + delta``. The analysis uses the
  *pooled* two-proportion z-test (:func:`powerlift.compare_proportions`), so the
  standard error under H0 uses the pooled rate ``p_bar = (p_c + r p_t) / (1 + r)``
  and the sampling distribution under H1 uses the unpooled variance. This is
  the classical formula without continuity correction (Fleiss, Levin & Paik
  2003, ch. 4)::

      n_c = (z_a * sqrt(p_bar (1 - p_bar) (1 + 1/r))
             + z_b * sqrt(p_c (1 - p_c) + p_t (1 - p_t) / r))^2 / delta^2

The sample-size and MDE formulas ignore the (negligible) probability of
rejecting in the wrong tail of a two-sided test, exactly like
``statsmodels.stats.proportion.samplesize_proportions_2indep_onetail``. The
``power_*`` functions count both tails, so they are exact under the normal
approximation and the power at the returned sample size is never below the
target.

References
----------
Kohavi, R., Tang, D., & Xu, Y. (2020). *Trustworthy Online Controlled
Experiments*, ch. 17. Cambridge University Press.

Fleiss, J. L., Levin, B., & Paik, M. C. (2003). *Statistical Methods for Rates
and Proportions* (3rd ed.), ch. 4. Wiley.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from scipy import optimize, stats

from powerlift._results import Result
from powerlift._validation import (
    check_count,
    check_open_unit,
    check_positive,
    check_real,
)
from powerlift.exceptions import InvalidInputError

Sides = Literal["two-sided", "one-sided"]
"""Whether the planned test is two-sided or one-sided in the direction of the MDE."""

_SIDES: tuple[Sides, ...] = ("two-sided", "one-sided")


@dataclass(frozen=True)
class SampleSizeResult(Result):
    """Required sample size for a two-arm experiment.

    Attributes
    ----------
    metric : {"proportion", "mean"}
        Type of metric the experiment is powered for.
    baseline : float or None
        Control conversion rate, or control mean (``None`` when a mean metric
        was planned without a baseline).
    sd : float or None
        Standard deviation of the metric per unit (mean metric only).
    absolute_mde : float
        Minimum detectable effect, treatment minus control.
    relative_mde : float or None
        ``absolute_mde / baseline`` when a baseline is known.
    alpha, power : float
        Significance level and target power.
    alternative : {"two-sided", "one-sided"}
        Sidedness of the planned test.
    ratio : float
        Allocation ratio ``n_treatment / n_control``.
    n_control, n_treatment, total : int
        Required sample sizes, rounded up.
    achieved_power : float
        Power at the rounded sample sizes (both rejection tails counted).
    """

    metric: Literal["proportion", "mean"]
    baseline: float | None
    sd: float | None
    absolute_mde: float
    relative_mde: float | None
    alpha: float
    power: float
    alternative: Sides
    ratio: float
    n_control: int
    n_treatment: int
    total: int
    achieved_power: float

    def duration_days(
        self, daily_traffic: float, *, traffic_fraction: float = 1.0
    ) -> int:
        """Days needed to enrol ``total`` units; see :func:`experiment_duration`."""
        return experiment_duration(
            self.total, daily_traffic, traffic_fraction=traffic_fraction
        )

    def summary(self) -> str:
        """Return a human-readable summary."""
        test = "pooled z-test" if self.metric == "proportion" else "z-test"
        lines = [
            f"Sample size: two {self.metric}s ({self.alternative} {test}, "
            f"alpha = {self.alpha:g}, power = {self.power:g})",
            *self._effect_lines(),
            f"  allocation       n_treatment / n_control = {self.ratio:g}",
            f"  required n       control {self.n_control:,} | treatment "
            f"{self.n_treatment:,} | total {self.total:,}",
            f"  achieved power   {self.achieved_power:.4f}",
        ]
        return "\n".join(lines)

    def _effect_lines(self) -> list[str]:
        if self.metric == "proportion" and self.baseline is not None:
            return [
                f"  baseline rate    {self.baseline:.2%}",
                f"  minimum effect   {self.absolute_mde * 100:+.2f} pp "
                f"({self.relative_mde or 0.0:+.2%} relative) "
                f"-> {self.baseline + self.absolute_mde:.2%}",
            ]
        lines = (
            [] if self.baseline is None else [f"  baseline mean    {self.baseline:g}"]
        )
        effect = f"  minimum effect   {self.absolute_mde:+g} (sd = {self.sd:g})"
        if self.relative_mde is not None:
            effect += f", {self.relative_mde:+.2%} relative"
        return [*lines, effect]


def sample_size_proportions(
    baseline: float,
    mde: float,
    *,
    relative: bool = False,
    alpha: float = 0.05,
    power: float = 0.80,
    alternative: Sides = "two-sided",
    ratio: float = 1.0,
) -> SampleSizeResult:
    """Sample size to compare two conversion rates with a pooled z-test.

    Parameters
    ----------
    baseline : float
        Conversion rate expected in the control arm, in (0, 1).
    mde : float
        Minimum detectable effect, treatment minus control. Interpreted as an
        absolute difference (``0.01`` = one percentage point) unless
        ``relative=True``, in which case it is a fraction of the baseline
        (``0.10`` = a 10% lift). Negative values plan for a decrease.
    relative : bool, default False
        Whether ``mde`` is relative to the baseline.
    alpha : float, default 0.05
        Significance level.
    power : float, default 0.80
        Target probability of detecting the effect; must be in [0.5, 1).
    alternative : {"two-sided", "one-sided"}, default "two-sided"
        A one-sided test looks only in the direction of ``mde``.
    ratio : float, default 1.0
        Allocation ratio ``n_treatment / n_control``.

    Returns
    -------
    SampleSizeResult
        Required sample sizes (rounded up) and the achieved power.

    Notes
    -----
    Uses the pooled-variance formula described in the module docstring;
    ``n_treatment = ceil(ratio * n_control_unrounded)``.

    Examples
    --------
    >>> result = sample_size_proportions(0.10, 0.10, relative=True)
    >>> result.n_control, result.n_treatment
    (14751, 14751)
    """
    p_control = check_open_unit("baseline", baseline)
    effect = _absolute_effect(mde, p_control, relative=relative)
    p_treatment = _treatment_rate(p_control, effect)
    alpha, power, ratio = _check_design(alpha, power, ratio)
    sides = _check_sides(alternative)

    # Standard errors for n_control = 1; for other sizes they scale by 1/sqrt(n).
    sd_null, sd_alt = _proportion_standard_errors(p_control, p_treatment, 1.0, ratio)
    n_control = _required_n_control(effect, sd_null, sd_alt, alpha, power, sides)
    n_c, n_t = math.ceil(n_control), math.ceil(ratio * n_control)
    achieved = _power_from_standard_errors(
        effect,
        *_proportion_standard_errors(p_control, p_treatment, n_c, n_t / n_c),
        alpha,
        sides,
    )
    return SampleSizeResult(
        metric="proportion",
        baseline=p_control,
        sd=None,
        absolute_mde=effect,
        relative_mde=effect / p_control,
        alpha=alpha,
        power=power,
        alternative=sides,
        ratio=ratio,
        n_control=n_c,
        n_treatment=n_t,
        total=n_c + n_t,
        achieved_power=achieved,
    )


def power_proportions(
    baseline: float,
    mde: float,
    n_control: float,
    *,
    relative: bool = False,
    alpha: float = 0.05,
    alternative: Sides = "two-sided",
    ratio: float = 1.0,
) -> float:
    """Power of the pooled two-proportion z-test for a given sample size.

    Parameters
    ----------
    baseline, mde, relative, alpha, alternative, ratio
        As in :func:`sample_size_proportions`.
    n_control : float
        Number of units in the control arm; the treatment arm has
        ``ratio * n_control`` units.

    Returns
    -------
    float
        Probability of rejecting H0 when the true effect equals ``mde``.
        For a two-sided test both rejection regions are counted.

    Examples
    --------
    >>> round(power_proportions(0.10, 0.01, 14751), 4)
    0.8
    """
    p_control = check_open_unit("baseline", baseline)
    effect = _absolute_effect(mde, p_control, relative=relative)
    p_treatment = _treatment_rate(p_control, effect)
    alpha = check_open_unit("alpha", alpha)
    n = check_positive("n_control", n_control)
    ratio = check_positive("ratio", ratio)
    sides = _check_sides(alternative)
    se_null, se_alt = _proportion_standard_errors(p_control, p_treatment, n, ratio)
    return _power_from_standard_errors(effect, se_null, se_alt, alpha, sides)


def mde_proportions(
    baseline: float,
    n_control: float,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    alternative: Sides = "two-sided",
    ratio: float = 1.0,
) -> float:
    """Smallest increase in conversion rate detectable with the given sample size.

    This inverts :func:`sample_size_proportions`: it returns the absolute
    effect ``delta > 0`` for which the required ``n_control`` equals the given
    one. Because the variance depends on ``p_c + delta``, the equation is
    solved numerically with Brent's method.

    Parameters
    ----------
    baseline : float
        Control conversion rate, in (0, 1).
    n_control : float
        Units in the control arm (treatment has ``ratio * n_control``).
    alpha, power, alternative, ratio
        As in :func:`sample_size_proportions`.

    Returns
    -------
    float
        Minimum detectable absolute increase (``0.01`` = one percentage point).
        Divide by ``baseline`` for the relative MDE.

    Raises
    ------
    InvalidInputError
        If even an increase to a 100% conversion rate would not be detected
        with the requested power.
    """
    p_control = check_open_unit("baseline", baseline)
    n = check_positive("n_control", n_control)
    alpha, power, ratio = _check_design(alpha, power, ratio)
    sides = _check_sides(alternative)
    z_alpha, z_beta = _z_values(alpha, power, sides)

    def gap(delta: float) -> float:
        # Positive while delta is still too small to reach the target power.
        se_null, se_alt = _proportion_standard_errors(
            p_control, p_control + delta, n, ratio
        )
        return z_alpha * se_null + z_beta * se_alt - delta

    largest_possible = 1.0 - p_control
    if gap(largest_possible) > 0:
        raise InvalidInputError(
            f"n_control = {n:g} is too small to detect any increase from a "
            f"baseline of {p_control:g} with power {power:g}"
        )
    return float(optimize.brentq(gap, 0.0, largest_possible, xtol=1e-12))


def sample_size_means(
    sd: float,
    mde: float,
    *,
    baseline: float | None = None,
    relative: bool = False,
    alpha: float = 0.05,
    power: float = 0.80,
    alternative: Sides = "two-sided",
    ratio: float = 1.0,
) -> SampleSizeResult:
    """Sample size to compare two means with a common standard deviation.

    Parameters
    ----------
    sd : float
        Standard deviation of the metric per unit (for example from
        historical data), assumed equal in both arms.
    mde : float
        Minimum detectable effect, treatment minus control, in the units of
        the metric, or as a fraction of ``baseline`` if ``relative=True``.
    baseline : float, optional
        Control mean. Required when ``relative=True``; otherwise only used to
        report the relative MDE.
    relative, alpha, power, alternative, ratio
        As in :func:`sample_size_proportions`.

    Returns
    -------
    SampleSizeResult
        Required sample sizes (rounded up) and the achieved power.

    Notes
    -----
    With ``alpha = 0.05``, ``power = 0.8`` and equal allocation this reduces to
    the rule of thumb ``n ~= 16 sd^2 / mde^2`` per arm (Kohavi et al. 2020).
    For small samples (fewer than ~30 units per arm) a t-based calculation
    gives slightly larger sizes.

    Examples
    --------
    >>> sample_size_means(sd=1.0, mde=0.1).n_control
    1570
    """
    sigma = check_positive("sd", sd)
    base = None if baseline is None else check_real("baseline", baseline)
    effect = _absolute_effect(mde, base, relative=relative)
    alpha, power, ratio = _check_design(alpha, power, ratio)
    sides = _check_sides(alternative)

    sd_diff = sigma * math.sqrt(1.0 + 1.0 / ratio)  # sd of the difference at n_c = 1
    n_control = _required_n_control(effect, sd_diff, sd_diff, alpha, power, sides)
    n_c, n_t = math.ceil(n_control), math.ceil(ratio * n_control)
    se = sigma * math.sqrt(1.0 / n_c + 1.0 / n_t)
    return SampleSizeResult(
        metric="mean",
        baseline=base,
        sd=sigma,
        absolute_mde=effect,
        relative_mde=None if not base else effect / base,
        alpha=alpha,
        power=power,
        alternative=sides,
        ratio=ratio,
        n_control=n_c,
        n_treatment=n_t,
        total=n_c + n_t,
        achieved_power=_power_from_standard_errors(effect, se, se, alpha, sides),
    )


def power_means(
    sd: float,
    mde: float,
    n_control: float,
    *,
    baseline: float | None = None,
    relative: bool = False,
    alpha: float = 0.05,
    alternative: Sides = "two-sided",
    ratio: float = 1.0,
) -> float:
    """Power of the two-sample z-test for a difference in means.

    Parameters
    ----------
    sd, mde, baseline, relative, alpha, alternative, ratio
        As in :func:`sample_size_means`.
    n_control : float
        Units in the control arm (treatment has ``ratio * n_control``).

    Returns
    -------
    float
        Probability of rejecting H0 when the true effect equals ``mde``
        (both rejection regions counted for a two-sided test).
    """
    sigma = check_positive("sd", sd)
    base = None if baseline is None else check_real("baseline", baseline)
    effect = _absolute_effect(mde, base, relative=relative)
    alpha = check_open_unit("alpha", alpha)
    n = check_positive("n_control", n_control)
    ratio = check_positive("ratio", ratio)
    sides = _check_sides(alternative)
    se = sigma * math.sqrt(1.0 / n + 1.0 / (ratio * n))
    return _power_from_standard_errors(effect, se, se, alpha, sides)


def mde_means(
    sd: float,
    n_control: float,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    alternative: Sides = "two-sided",
    ratio: float = 1.0,
) -> float:
    """Minimum detectable difference in means for a given sample size.

    Returns ``(z_a + z_b) * sd * sqrt(1/n_c + 1/n_t)``, the inverse of
    :func:`sample_size_means`, in the units of the metric.

    Examples
    --------
    >>> round(mde_means(sd=1.0, n_control=1570), 4)
    0.1
    """
    sigma = check_positive("sd", sd)
    n = check_positive("n_control", n_control)
    alpha, power, ratio = _check_design(alpha, power, ratio)
    z_alpha, z_beta = _z_values(alpha, power, _check_sides(alternative))
    return (z_alpha + z_beta) * sigma * math.sqrt(1.0 / n + 1.0 / (ratio * n))


def experiment_duration(
    total_sample_size: int, daily_traffic: float, *, traffic_fraction: float = 1.0
) -> int:
    """Return the number of days needed to enrol ``total_sample_size`` units.

    Parameters
    ----------
    total_sample_size : int
        Units needed across all arms (e.g. ``SampleSizeResult.total``).
    daily_traffic : float
        Eligible units arriving per day.
    traffic_fraction : float, default 1.0
        Share of eligible traffic that is put into the experiment, in (0, 1].

    Returns
    -------
    int
        ``ceil(total / (daily_traffic * traffic_fraction))``.

    Notes
    -----
    This assumes every day brings new, distinct units. When users return
    across days the number of *unique* users grows more slowly, so treat the
    result as a lower bound, and prefer running whole weeks to average over
    day-of-week effects.

    Examples
    --------
    >>> experiment_duration(29_502, daily_traffic=4_000)
    8
    """
    total = check_count("total_sample_size", total_sample_size, minimum=1)
    daily = check_positive("daily_traffic", daily_traffic)
    fraction = check_positive("traffic_fraction", traffic_fraction)
    if fraction > 1.0:
        raise InvalidInputError(f"traffic_fraction must be <= 1, got {fraction:g}")
    return math.ceil(total / (daily * fraction))


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------


def _check_sides(alternative: object) -> Sides:
    for sides in _SIDES:
        if alternative == sides:
            return sides
    raise InvalidInputError(f"alternative must be one of {_SIDES}, got {alternative!r}")


def _check_design(
    alpha: float, power: float, ratio: float
) -> tuple[float, float, float]:
    alpha = check_open_unit("alpha", alpha)
    power = check_open_unit("power", power)
    if power < 0.5:
        raise InvalidInputError(f"power must be in [0.5, 1), got {power:g}")
    return alpha, power, check_positive("ratio", ratio)


def _absolute_effect(mde: float, baseline: float | None, *, relative: bool) -> float:
    effect = check_real("mde", mde)
    if effect == 0.0:
        raise InvalidInputError("mde must be non-zero")
    if not relative:
        return effect
    if not baseline:
        raise InvalidInputError("a non-zero baseline is required when relative=True")
    return effect * baseline


def _treatment_rate(p_control: float, effect: float) -> float:
    p_treatment = p_control + effect
    if not 0.0 < p_treatment < 1.0:
        raise InvalidInputError(
            "baseline + mde must be a rate strictly between 0 and 1, "
            f"got {p_treatment:g}"
        )
    return p_treatment


def _z_values(alpha: float, power: float, sides: Sides) -> tuple[float, float]:
    tail = alpha / 2.0 if sides == "two-sided" else alpha
    return float(stats.norm.isf(tail)), float(stats.norm.ppf(power))


def _proportion_standard_errors(
    p_control: float, p_treatment: float, n_control: float, ratio: float
) -> tuple[float, float]:
    """Return the SE of p_t - p_c under H0 (pooled rate) and under H1 (unpooled)."""
    n_treatment = ratio * n_control
    pooled = (p_control + ratio * p_treatment) / (1.0 + ratio)
    se_null = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n_control + 1.0 / n_treatment))
    se_alt = math.sqrt(
        p_control * (1.0 - p_control) / n_control
        + p_treatment * (1.0 - p_treatment) / n_treatment
    )
    return se_null, se_alt


def _required_n_control(
    effect: float,
    sd_null: float,
    sd_alt: float,
    alpha: float,
    power: float,
    sides: Sides,
) -> float:
    """Unrounded n_control, given standard errors evaluated at n_control = 1."""
    z_alpha, z_beta = _z_values(alpha, power, sides)
    return ((z_alpha * sd_null + z_beta * sd_alt) / effect) ** 2


def _power_from_standard_errors(
    effect: float, se_null: float, se_alt: float, alpha: float, sides: Sides
) -> float:
    """P(|estimate| / se_null > z) when estimate ~ Normal(effect, se_alt^2).

    For a one-sided test only the tail in the direction of ``effect`` counts.
    """
    size = abs(effect)
    if sides == "one-sided":
        z = float(stats.norm.isf(alpha))
        return float(stats.norm.cdf((size - z * se_null) / se_alt))
    z = float(stats.norm.isf(alpha / 2.0))
    near_tail = stats.norm.cdf((size - z * se_null) / se_alt)
    far_tail = stats.norm.cdf((-size - z * se_null) / se_alt)
    return float(near_tail + far_tail)

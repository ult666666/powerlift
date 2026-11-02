"""Many-to-one comparisons: several treatment arms against one control (A/B/n tests).

Testing ``k`` treatments against a shared control at level ``alpha`` each lets
the chance of at least one false win grow with ``k``. Dunnett's test (Dunnett
1955) controls that family-wise error rate exactly: it compares every
treatment-minus-control t-statistic with the distribution of the *largest*
of them under the null hypothesis. The statistics are correlated through the
shared control arm, which makes Dunnett's critical value smaller (so the test
more powerful) than Bonferroni's.

All comparisons use the pooled within-arm variance of every arm, as Dunnett's
test and an analysis of variance do, so they assume equal variances across
arms. Conversion metrics are analysed as means of 0/1 outcomes; with the sample
sizes of online experiments the pooled-variance t-statistic is then within
O(1/n) of the two-proportion z-statistic.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeVar

import numpy as np
import numpy.typing as npt
from scipy import stats

from powerlift._dunnett import max_t_cdf_many, max_t_quantile
from powerlift._results import (
    ConfidenceInterval,
    Result,
    format_level,
    format_p_value,
)
from powerlift._stats import Alternative, t_p_value
from powerlift._validation import (
    as_float_array,
    check_alternative,
    check_binomial,
    check_open_unit,
)
from powerlift.exceptions import InsufficientDataError, InvalidInputError
from powerlift.multiple_testing import adjust_p_values

Adjustment = Literal["dunnett", "holm", "bonferroni", "none"]
"""Multiplicity adjustment for comparisons of several treatments with a control."""

ADJUSTMENTS: tuple[Adjustment, ...] = ("dunnett", "holm", "bonferroni", "none")

_Arm = TypeVar("_Arm")

_METHODS = {
    "dunnett": "Dunnett's many-to-one test",
    "holm": "t-tests with Holm-adjusted p-values (Bonferroni intervals)",
    "bonferroni": "t-tests with Bonferroni-adjusted p-values",
    "none": "t-tests without multiplicity adjustment",
}


@dataclass(frozen=True)
class ArmComparison(Result):
    """One treatment arm compared with the control.

    Attributes
    ----------
    name : str
        Arm label.
    n : int
        Units in the arm.
    estimate : float
        Arm mean (a conversion rate for proportions).
    difference : float
        ``estimate - control estimate``.
    se_difference : float
        Standard error of ``difference`` from the pooled variance.
    statistic : float
        ``difference / se_difference``, a t-statistic.
    p_value : float
        Unadjusted p-value of this comparison alone.
    adjusted_p_value : float
        P-value adjusted for the ``k`` comparisons.
    ci_difference : ConfidenceInterval
        Simultaneous interval for ``difference``: all ``k`` intervals cover
        their true differences with probability ``1 - alpha``. One-sided tests
        get one-sided bounds (the other end is infinite).
    relative_lift : float or None
        ``difference / control estimate``; None when the control estimate is 0.
    significant : bool
        ``adjusted_p_value <= alpha``.
    """

    name: str
    n: int
    estimate: float
    difference: float
    se_difference: float
    statistic: float
    p_value: float
    adjusted_p_value: float
    ci_difference: ConfidenceInterval
    relative_lift: float | None
    significant: bool

    def summary(self) -> str:
        """Return a one-line summary."""
        return (
            f"{self.name}: difference {self.difference:+.6g}, "
            f"CI {self.ci_difference.format('+.6g')}, "
            f"adjusted p-value = {format_p_value(self.adjusted_p_value)}"
        )


@dataclass(frozen=True)
class ManyToOneComparison(Result):
    """Several treatment arms compared with one control arm.

    Attributes
    ----------
    metric : {"mean", "proportion"}
        Kind of metric.
    adjustment : {"dunnett", "holm", "bonferroni", "none"}
        Multiplicity adjustment.
    alternative : {"two-sided", "greater", "less"}
        Alternative hypothesis for each ``treatment - control`` difference.
    alpha : float
        Family-wise significance level (per comparison for ``"none"``).
    control : str
        Control arm label.
    n_control : int
        Units in the control arm.
    estimate_control : float
        Control mean (or conversion rate).
    pooled_sd : float
        Pooled within-arm standard deviation.
    df : int
        Degrees of freedom of the pooled variance, ``N - (k + 1)``.
    critical_value : float
        Cutoff for ``|t|`` (two-sided) or ``t`` (one-sided) that the
        intervals use; for ``"dunnett"`` and ``"bonferroni"`` a comparison is
        significant exactly when its statistic passes it.
    comparisons : tuple of ArmComparison
        One entry per treatment arm, in the order given.
    """

    metric: Literal["mean", "proportion"]
    adjustment: Adjustment
    alternative: Alternative
    alpha: float
    control: str
    n_control: int
    estimate_control: float
    pooled_sd: float
    df: int
    critical_value: float
    comparisons: tuple[ArmComparison, ...]

    @property
    def method(self) -> str:
        """Name of the procedure."""
        return _METHODS[self.adjustment]

    def comparison(self, name: str) -> ArmComparison:
        """Return the comparison for the treatment arm called ``name``."""
        for arm in self.comparisons:
            if arm.name == name:
                return arm
        raise KeyError(name)

    def summary(self) -> str:
        """Return a human-readable table, one row per arm."""
        share = self.metric == "proportion"
        family = "" if self.adjustment == "none" else " (family-wise)"
        kind = "simultaneous " if self.adjustment != "none" else ""
        header = (
            f"{self.method}, {self.alternative}, alpha = {self.alpha:g}{family}\n"
            f"  pooled SD {self.pooled_sd:.6g} with df = {self.df:,}; critical value "
            f"{self.critical_value:.4f}"
        )
        lines = [
            header,
            f"  {'arm':<12}{'n':>9}  {'estimate':>10}  {'difference':>12}  "
            f"{format_level(1 - self.alpha)} {kind}CI",
            f"  {self.control:<12}{self.n_control:>9,}  "
            f"{_estimate(self.estimate_control, share):>10}",
        ]
        for arm in self.comparisons:
            interval = (
                f"[{_difference(arm.ci_difference.lower, share)}, "
                f"{_difference(arm.ci_difference.upper, share)}]"
            )
            decision = "significant" if arm.significant else "not significant"
            lines.append(
                f"  {arm.name:<12}{arm.n:>9,}  {_estimate(arm.estimate, share):>10}  "
                f"{_difference(arm.difference, share):>12}  {interval}   adjusted p = "
                f"{format_p_value(arm.adjusted_p_value)} ({decision})"
            )
        return "\n".join(lines)


def compare_means_to_control(
    control: npt.ArrayLike,
    treatments: Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike],
    *,
    control_name: str = "control",
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
    adjustment: Adjustment = "dunnett",
) -> ManyToOneComparison:
    """Compare the means of several treatment arms with a control arm.

    Parameters
    ----------
    control : array_like
        One value per randomization unit in the control arm.
    treatments : mapping of str to array_like, or sequence of array_like
        One array per treatment arm. A mapping supplies the arm names; a
        sequence gets ``"treatment 1"``, ``"treatment 2"``, ...
    control_name : str, default "control"
        Label of the control arm in summaries.
    alpha : float, default 0.05
        Family-wise significance level; simultaneous intervals have
        confidence ``1 - alpha``.
    alternative : {"two-sided", "greater", "less"}, default "two-sided"
        Alternative hypothesis for every ``treatment - control`` difference.
    adjustment : {"dunnett", "holm", "bonferroni", "none"}, default "dunnett"
        ``"dunnett"``: single-step Dunnett test with exact simultaneous
        intervals. ``"holm"`` and ``"bonferroni"``: the per-comparison
        p-values adjusted with :func:`powerlift.adjust_p_values`, with
        Bonferroni intervals (Holm has no simple matching intervals).
        ``"none"``: unadjusted tests, which do not control the family-wise
        error rate.

    Returns
    -------
    ManyToOneComparison
        Adjusted p-values, simultaneous intervals and decisions per arm.

    Notes
    -----
    With ``N`` units in ``k + 1`` arms and pooled variance ``s^2``, the
    statistic for arm ``i`` is ``t_i = (mean_i - mean_0) / (s sqrt(1/n_i + 1/n_0))``.
    Dunnett's adjusted p-value is ``P(max_j |T_j| >= |t_i|)`` under the null,
    where the ``T_j`` follow a multivariate t distribution with ``N - k - 1``
    degrees of freedom and correlations ``lambda_i lambda_j``,
    ``lambda_i = sqrt(n_i / (n_i + n_0))``. The p-values and intervals match
    :func:`scipy.stats.dunnett`, which integrates the same distribution by
    randomized quasi-Monte Carlo.

    References
    ----------
    Dunnett, C. W. (1955). A multiple comparison procedure for comparing
    several treatments with a control. *JASA*, 50(272), 1096-1121.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> control = rng.normal(10.0, 2.0, 400)
    >>> arms = {"B": rng.normal(10.0, 2.0, 400), "C": rng.normal(10.4, 2.0, 400)}
    >>> result = compare_means_to_control(control, arms)
    >>> for arm in result.comparisons:  # name, raw p-value, adjusted p-value
    ...     print(arm.name, round(arm.p_value, 4), round(arm.adjusted_p_value, 4))
    B 0.6953 0.8944
    C 0.0002 0.0004
    """
    arms = _named_arms(treatments)
    stats_by_arm = [_mean_stats(control_name, control)] + [
        _mean_stats(name, values) for name, values in arms
    ]
    return _many_to_one(
        "mean",
        control_name,
        [name for name, _ in arms],
        stats_by_arm,
        alpha,
        alternative,
        adjustment,
    )


def compare_proportions_to_control(
    control: tuple[int, int],
    treatments: Mapping[str, tuple[int, int]] | Sequence[tuple[int, int]],
    *,
    control_name: str = "control",
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
    adjustment: Adjustment = "dunnett",
) -> ManyToOneComparison:
    """Compare the conversion rates of several treatment arms with a control arm.

    Each arm is given as ``(conversions, n)``. The rates are analysed as means
    of 0/1 outcomes with :func:`compare_means_to_control`'s method, so the
    results equal Dunnett's test on the expanded 0/1 data.

    Parameters
    ----------
    control : tuple of int
        ``(conversions, n)`` of the control arm.
    treatments : mapping of str to tuple of int, or sequence of tuple of int
        ``(conversions, n)`` per treatment arm.
    control_name, alpha, alternative, adjustment
        As in :func:`compare_means_to_control`.

    Returns
    -------
    ManyToOneComparison
        Adjusted p-values, simultaneous intervals and decisions per arm.

    Examples
    --------
    >>> result = compare_proportions_to_control(
    ...     (480, 5000), {"B": (520, 5000), "C": (575, 5000)}
    ... )
    >>> [arm.significant for arm in result.comparisons]
    [False, True]
    """
    arms = _named_arms(treatments)
    stats_by_arm = [_proportion_stats(control_name, control)] + [
        _proportion_stats(name, counts) for name, counts in arms
    ]
    return _many_to_one(
        "proportion",
        control_name,
        [name for name, _ in arms],
        stats_by_arm,
        alpha,
        alternative,
        adjustment,
    )


def dunnett_critical_value(
    n_control: int,
    n_treatments: Sequence[int],
    *,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
    df: float = math.inf,
) -> float:
    """Dunnett's critical value for the given arm sizes.

    The cutoff ``c`` with ``P(max_i |T_i| <= c) = 1 - alpha`` (two-sided) or
    ``P(max_i T_i <= c) = 1 - alpha`` (one-sided, either direction) under the
    null hypothesis. ``df=math.inf`` gives the large-sample (normal) value.

    Examples
    --------
    >>> round(dunnett_critical_value(1000, [1000, 1000, 1000]), 4)
    2.349
    """
    n0 = _check_size("n_control", n_control)
    sizes = np.array(
        [_check_size(f"n_treatments[{i}]", n) for i, n in enumerate(n_treatments)]
    )
    if sizes.size == 0:
        raise InvalidInputError("n_treatments needs at least one treatment arm")
    alpha = check_open_unit("alpha", alpha)
    alternative = check_alternative(alternative)
    if not df > 0:
        raise InvalidInputError(f"df must be > 0, got {df!r}")
    lambdas = np.sqrt(sizes / (sizes + n0))
    return max_t_quantile(1.0 - alpha, lambdas, float(df), _tail(alternative))


# --------------------------------------------------------------------------
# implementation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _ArmStats:
    n: int
    mean: float
    sum_squares: float  # sum of squared deviations from the arm mean


def _named_arms(
    treatments: Mapping[str, _Arm] | Sequence[_Arm],
) -> list[tuple[str, _Arm]]:
    if isinstance(treatments, Mapping):
        arms = [(str(name), values) for name, values in treatments.items()]
    elif isinstance(treatments, Sequence) and not isinstance(treatments, str):
        arms = [(f"treatment {i}", values) for i, values in enumerate(treatments, 1)]
    else:
        raise InvalidInputError(
            "treatments must be a mapping of arm names to data or a sequence of arms"
        )
    if not arms:
        raise InvalidInputError("treatments needs at least one treatment arm")
    return arms


def _mean_stats(name: str, values: npt.ArrayLike) -> _ArmStats:
    array = as_float_array(name, values)
    mean = float(array.mean())
    return _ArmStats(array.size, mean, float(np.sum((array - mean) ** 2)))


def _proportion_stats(name: str, counts: tuple[int, int]) -> _ArmStats:
    if not isinstance(counts, tuple | list) or len(counts) != 2:
        raise InvalidInputError(f"arm {name!r} must be given as (conversions, n)")
    x, n = check_binomial(name, counts[0], counts[1])
    # For 0/1 outcomes the sum of squared deviations is x (n - x) / n.
    return _ArmStats(n, x / n, x * (n - x) / n)


def _check_size(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 1:
        raise InvalidInputError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _tail(alternative: Alternative) -> Literal["two-sided", "one-sided"]:
    return "two-sided" if alternative == "two-sided" else "one-sided"


def _many_to_one(
    metric: Literal["mean", "proportion"],
    control_name: str,
    names: list[str],
    arm_stats: list[_ArmStats],
    alpha: float,
    alternative: Alternative,
    adjustment: Adjustment,
) -> ManyToOneComparison:
    alpha = check_open_unit("alpha", alpha)
    alternative = check_alternative(alternative)
    if adjustment not in ADJUSTMENTS:
        raise InvalidInputError(
            f"adjustment must be one of {ADJUSTMENTS}, got {adjustment!r}"
        )
    if len(set(names)) != len(names) or control_name in names:
        raise InvalidInputError("arm names must be unique")
    control, arms = arm_stats[0], arm_stats[1:]
    k = len(arms)
    sizes = np.array([arm.n for arm in arms], dtype=np.float64)
    df = sum(arm.n for arm in arm_stats) - (k + 1)
    if df < 1:
        raise InsufficientDataError("the pooled variance needs more units than arms")
    pooled_variance = sum(arm.sum_squares for arm in arm_stats) / df
    if pooled_variance <= 0.0:
        raise InsufficientDataError(
            "the pooled variance is zero; the data show no variation within arms"
        )
    differences = np.array([arm.mean - control.mean for arm in arms])
    se = np.sqrt(pooled_variance * (1.0 / sizes + 1.0 / control.n))
    t = differences / se
    raw = [t_p_value(float(value), df, alternative) for value in t]

    lambdas = np.sqrt(sizes / (sizes + control.n))
    tail = _tail(alternative)
    if adjustment == "dunnett":
        # Larger statistics in the direction of the alternative are more extreme.
        oriented = (
            np.abs(t)
            if tail == "two-sided"
            else (t if alternative == "greater" else -t)
        )
        adjusted = list(1.0 - max_t_cdf_many(oriented, lambdas, df, tail))
        critical = max_t_quantile(1.0 - alpha, lambdas, df, tail)
    elif adjustment == "none":
        adjusted = raw
        critical = float(stats.t.isf(alpha / 2.0 if tail == "two-sided" else alpha, df))
    else:
        adjusted = list(
            adjust_p_values(raw, method=adjustment, alpha=alpha).adjusted_p_values
        )
        tail_alpha = alpha / k / (2.0 if tail == "two-sided" else 1.0)
        critical = float(stats.t.isf(tail_alpha, df))

    comparisons = []
    for i, (name, arm) in enumerate(zip(names, arms, strict=True)):
        margin = critical * float(se[i])
        lower = -math.inf if alternative == "less" else float(differences[i]) - margin
        upper = math.inf if alternative == "greater" else float(differences[i]) + margin
        comparisons.append(
            ArmComparison(
                name=name,
                n=arm.n,
                estimate=arm.mean,
                difference=float(differences[i]),
                se_difference=float(se[i]),
                statistic=float(t[i]),
                p_value=raw[i],
                adjusted_p_value=float(adjusted[i]),
                ci_difference=ConfidenceInterval(lower, upper, 1.0 - alpha),
                relative_lift=(
                    None if control.mean == 0 else float(differences[i]) / control.mean
                ),
                significant=bool(adjusted[i] <= alpha),
            )
        )
    return ManyToOneComparison(
        metric=metric,
        adjustment=adjustment,
        alternative=alternative,
        alpha=alpha,
        control=control_name,
        n_control=control.n,
        estimate_control=control.mean,
        pooled_sd=math.sqrt(pooled_variance),
        df=df,
        critical_value=critical,
        comparisons=tuple(comparisons),
    )


def _estimate(value: float, share: bool) -> str:
    return f"{value:.2%}" if share else f"{value:.6g}"


def _difference(value: float, share: bool) -> str:
    if math.isinf(value):
        return "-inf" if value < 0 else "+inf"
    return f"{100 * value:+.2f} pp" if share else f"{value:+.6g}"

"""Sanity checks to run before trusting an experiment's results.

* :func:`srm_test` - sample ratio mismatch (SRM): are units split between
  arms in the proportions the design asked for? An SRM usually means a bug in
  assignment, logging or filtering, and invalidates the comparison.
* :func:`simulate_aa` - A/A simulation: run a test many times on data with no
  true effect and check that its p-values are uniform and its false-positive
  rate equals alpha.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from powerlift._results import ConfidenceInterval, Result, format_p_value
from powerlift._validation import check_count, check_open_unit, check_positive
from powerlift.analysis.proportions import wilson_interval
from powerlift.exceptions import InvalidInputError


@dataclass(frozen=True)
class SRMResult(Result):
    """Result of a sample ratio mismatch check.

    Attributes
    ----------
    observed : tuple of int
        Units observed in each arm.
    expected : tuple of float
        Units expected in each arm under the planned allocation.
    allocation : tuple of float
        Planned share of traffic per arm (normalised to sum to 1).
    chi_square : float
        Pearson goodness-of-fit statistic ``sum((O - E)^2 / E)``.
    df : int
        Degrees of freedom, ``number of arms - 1``.
    p_value : float
        ``P(chi2_df >= chi_square)``.
    threshold : float
        P-value below which a mismatch is flagged.
    mismatch : bool
        ``p_value < threshold``.
    """

    observed: tuple[int, ...]
    expected: tuple[float, ...]
    allocation: tuple[float, ...]
    chi_square: float
    df: int
    p_value: float
    threshold: float
    mismatch: bool

    def summary(self) -> str:
        """Return a human-readable summary."""
        total = sum(self.observed)
        lines = ["Sample ratio mismatch check (chi-square goodness of fit)"]
        for arm, (observed, expected, share) in enumerate(
            zip(self.observed, self.expected, self.allocation, strict=True), start=1
        ):
            lines.append(
                f"  arm {arm}: observed {observed:,} ({observed / total:.3%}), "
                f"expected {expected:,.1f} ({share:.3%})"
            )
        lines.append(
            f"  chi2 = {self.chi_square:.3f}, df = {self.df}, "
            f"p-value = {format_p_value(self.p_value)}"
        )
        if self.mismatch:
            lines.append(
                f"  SRM DETECTED (p < {self.threshold:g}): do not trust the results "
                "until the cause is found"
            )
        else:
            lines.append(f"  no SRM detected (p >= {self.threshold:g})")
        return "\n".join(lines)


def srm_test(
    observed: Sequence[int],
    allocation: Sequence[float] | None = None,
    *,
    threshold: float = 0.001,
) -> SRMResult:
    """Check for a sample ratio mismatch with a chi-square goodness-of-fit test.

    Parameters
    ----------
    observed : sequence of int
        Number of units observed in each arm (two or more arms).
    allocation : sequence of float, optional
        Planned traffic split, e.g. ``[0.5, 0.5]`` or ``[1, 1, 2]``; it is
        normalised to sum to 1. Defaults to an equal split.
    threshold : float, default 0.001
        Flag a mismatch when the p-value is below this value. A strict
        threshold is customary because the check runs on every experiment and
        a false alarm stops the analysis (Kohavi, Tang & Xu 2020, ch. 21).

    Returns
    -------
    SRMResult
        Statistic, p-value and the ``mismatch`` flag.

    Notes
    -----
    ``chi2 = sum_i (O_i - E_i)^2 / E_i`` with ``E_i = N * share_i`` is compared
    with a chi-square distribution with ``k - 1`` degrees of freedom; this is
    the same test as ``scipy.stats.chisquare(observed, expected)``.

    References
    ----------
    Fabijan, A., et al. (2019). Diagnosing sample ratio mismatch in online
    controlled experiments: a taxonomy and rules of thumb for practitioners.
    *KDD 2019*.

    Examples
    --------
    >>> srm_test([50_000, 48_900]).mismatch
    True
    """
    counts = tuple(
        check_count(f"observed[{i}]", value) for i, value in enumerate(observed)
    )
    if len(counts) < 2:
        raise InvalidInputError("observed needs counts for at least two arms")
    total = sum(counts)
    if total == 0:
        raise InvalidInputError("observed must contain at least one unit")
    weights = (
        [1.0] * len(counts)
        if allocation is None
        else [check_positive(f"allocation[{i}]", w) for i, w in enumerate(allocation)]
    )
    if len(weights) != len(counts):
        raise InvalidInputError(
            f"allocation has {len(weights)} entries but observed has {len(counts)}"
        )
    shares = tuple(w / math.fsum(weights) for w in weights)
    expected = tuple(total * share for share in shares)
    chi_square = math.fsum(
        (o - e) ** 2 / e for o, e in zip(counts, expected, strict=True)
    )
    df = len(counts) - 1
    p_value = float(stats.chi2.sf(chi_square, df))
    threshold = check_open_unit("threshold", threshold)
    return SRMResult(
        observed=counts,
        expected=expected,
        allocation=shares,
        chi_square=chi_square,
        df=df,
        p_value=p_value,
        threshold=threshold,
        mismatch=p_value < threshold,
    )


@dataclass(frozen=True)
class AASimulationResult(Result):
    """Calibration of a test under the null hypothesis (A/A simulation).

    Attributes
    ----------
    n_simulations : int
        Number of simulated A/A experiments.
    alpha : float
        Significance level used to count false positives.
    seed : int or None
        Seed of the random generator.
    false_positive_rate : float
        Share of experiments with ``p_value <= alpha``; should be close to
        ``alpha`` for a calibrated test.
    false_positive_ci : ConfidenceInterval
        95% Wilson interval for the false-positive rate (Monte Carlo error).
    ks_statistic, ks_p_value : float
        Kolmogorov-Smirnov test of the p-values against Uniform(0, 1). A
        small ``ks_p_value`` means the p-values are not uniform.
    p_values : tuple of float
        All simulated p-values.
    """

    n_simulations: int
    alpha: float
    seed: int | None
    false_positive_rate: float
    false_positive_ci: ConfidenceInterval
    ks_statistic: float
    ks_p_value: float
    p_values: tuple[float, ...] = field(repr=False)

    def summary(self) -> str:
        """Return a human-readable summary."""
        return "\n".join(
            [
                f"A/A simulation: {self.n_simulations:,} experiments "
                f"(seed {self.seed})",
                f"  false-positive rate  {self.false_positive_rate:.2%} "
                f"(95% CI {self.false_positive_ci.format('.2%')}), "
                f"nominal {self.alpha:.2%}",
                f"  KS test vs Uniform(0, 1): D = {self.ks_statistic:.4f}, "
                f"p-value = {format_p_value(self.ks_p_value)}",
            ]
        )


def simulate_aa(
    p_value_fn: Callable[[np.random.Generator], float],
    *,
    n_simulations: int = 1_000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> AASimulationResult:
    """Run many A/A experiments and check that a test is calibrated.

    Under the null hypothesis a valid test produces p-values that are
    uniformly distributed on [0, 1], so a share ``alpha`` of A/A experiments
    is (falsely) significant. Systematic deviations reveal a broken test, for
    example one that ignores clustering or uses the wrong variance.

    Parameters
    ----------
    p_value_fn : callable
        Simulates one A/A experiment with the given ``numpy.random.Generator``
        and returns the p-value of the test being checked.
    n_simulations : int, default 1000
        Number of simulated experiments.
    alpha : float, default 0.05
        Significance level for counting false positives.
    seed : int, optional
        Seed for ``numpy.random.default_rng``; set it for reproducible runs.

    Returns
    -------
    AASimulationResult
        False-positive rate with its Monte Carlo interval and a
        Kolmogorov-Smirnov uniformity test of the p-values.

    Examples
    --------
    >>> from powerlift import compare_means
    >>> def one_experiment(rng):
    ...     return compare_means(rng.normal(size=200), rng.normal(size=200)).p_value
    >>> result = simulate_aa(one_experiment, n_simulations=500, seed=1)
    >>> result.false_positive_ci.contains(0.05)
    True
    """
    n = check_count("n_simulations", n_simulations, minimum=1)
    alpha = check_open_unit("alpha", alpha)
    rng = np.random.default_rng(seed)
    p_values = np.array([float(p_value_fn(rng)) for _ in range(n)])
    if not np.all((p_values >= 0.0) & (p_values <= 1.0)):
        raise InvalidInputError("p_value_fn must return p-values between 0 and 1")
    false_positives = int(np.count_nonzero(p_values <= alpha))
    ks = stats.kstest(p_values, "uniform")
    return AASimulationResult(
        n_simulations=n,
        alpha=alpha,
        seed=seed,
        false_positive_rate=false_positives / n,
        false_positive_ci=wilson_interval(false_positives, n, alpha=0.05),
        ks_statistic=float(ks.statistic),
        ks_p_value=float(ks.pvalue),
        p_values=tuple(p_values.tolist()),
    )

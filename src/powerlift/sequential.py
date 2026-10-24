"""Sequential monitoring: group sequential boundaries and always-valid inference.

Checking an experiment repeatedly and stopping the first time ``p < 0.05``
inflates the false-positive rate: with five equally spaced looks it is about
14%, with twenty about 25% (Armitage, McPherson & Rowe 1969). Two remedies are
implemented:

* **Group sequential designs** (Jennison & Turnbull 2000): plan ``K`` looks
  and use stricter critical values so that the overall type-I error is
  ``alpha``. The critical values are computed exactly (up to numerical
  integration error) from the joint normal distribution of the sequential
  z-statistics: with information fractions ``t_1 < ... < t_K = 1`` (share of
  the final sample size available at each look) the statistics are
  multivariate normal with ``corr(Z_i, Z_j) = sqrt(t_i / t_j)`` for
  ``t_i <= t_j``.
* **Always-valid inference** with the mixture sequential probability ratio
  test (mSPRT, Johari, Pekelis & Walsh 2017): p-values and confidence
  sequences that stay valid however often, and whenever, the data are
  checked.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy import optimize, stats

from powerlift._results import Result, format_p_value
from powerlift._validation import (
    FloatArray,
    as_float_array,
    check_count,
    check_open_unit,
    check_positive,
    check_real,
)
from powerlift.exceptions import InvalidInputError

BoundaryMethod = Literal[
    "pocock", "obrien-fleming", "lan-demets-obrien-fleming", "lan-demets-pocock"
]
_METHODS: tuple[BoundaryMethod, ...] = (
    "pocock",
    "obrien-fleming",
    "lan-demets-obrien-fleming",
    "lan-demets-pocock",
)
_METHOD_NAMES = {
    "pocock": "Pocock",
    "obrien-fleming": "O'Brien-Fleming",
    "lan-demets-obrien-fleming": "Lan-DeMets (O'Brien-Fleming-type spending)",
    "lan-demets-pocock": "Lan-DeMets (Pocock-type spending)",
}

# SciPy evaluates multivariate normal rectangle probabilities in three or more
# dimensions by randomized quasi-Monte Carlo (Genz's algorithm). A fixed seed
# makes the boundaries reproducible. Only small "exit" probabilities are ever
# evaluated, and with this tolerance the boundaries agree with a 100x stricter
# tolerance to within 5e-5 for up to 10 looks, at a fraction of the cost.
_MVN_ABSOLUTE_ERROR = 1e-6
_MVN_SEED = 0
# Root-finding tolerance on the z scale, well below the integration error.
_ROOT_TOLERANCE = 1e-7


@dataclass(frozen=True)
class GroupSequentialBoundaries(Result):
    """Two-sided critical values for a group sequential test.

    At look ``k`` reject H0 if ``|Z_k| >= z_boundaries[k - 1]``, where ``Z_k``
    is the usual z-statistic computed from all data collected so far.

    Attributes
    ----------
    method : str
        Boundary family.
    alpha : float
        Overall two-sided type-I error.
    information_fractions : tuple of float
        Share of the maximum sample size available at each look.
    z_boundaries : tuple of float
        Critical values for ``|Z|`` at each look.
    nominal_alpha : tuple of float
        Equivalent two-sided p-value thresholds, ``2 * (1 - Phi(c_k))``.
    cumulative_alpha : tuple of float
        Probability under H0 of having rejected by look ``k``; the last value
        equals ``alpha`` up to numerical error.
    """

    method: BoundaryMethod
    alpha: float
    information_fractions: tuple[float, ...]
    z_boundaries: tuple[float, ...]
    nominal_alpha: tuple[float, ...]
    cumulative_alpha: tuple[float, ...]

    def first_crossing(self, z_statistics: Sequence[float]) -> int | None:
        """Return the first look (1-based) at which ``|z|`` reaches the boundary.

        Parameters
        ----------
        z_statistics : sequence of float
            Z-statistics observed at the looks so far (at most one per look).

        Returns
        -------
        int or None
            Look number at which H0 is rejected, or ``None`` if no boundary
            has been crossed.
        """
        z = as_float_array("z_statistics", z_statistics, min_size=1)
        if z.size > len(self.z_boundaries):
            raise InvalidInputError(
                f"got {z.size} z-statistics for a design with "
                f"{len(self.z_boundaries)} looks"
            )
        for look, (value, boundary) in enumerate(
            zip(z, self.z_boundaries, strict=False), start=1
        ):
            if abs(value) >= boundary:
                return look
        return None

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"{_METHOD_NAMES[self.method]} boundaries: two-sided alpha = "
            f"{self.alpha:g}, {len(self.z_boundaries)} looks",
            "  look  information  |z| boundary  nominal alpha  cumulative alpha",
        ]
        for look, (t, z, nominal, cumulative) in enumerate(
            zip(
                self.information_fractions,
                self.z_boundaries,
                self.nominal_alpha,
                self.cumulative_alpha,
                strict=True,
            ),
            start=1,
        ):
            lines.append(
                f"  {look:<5} {t:<12.3f} {z:<13.4f} {nominal:<14.6f} {cumulative:.6f}"
            )
        return "\n".join(lines)


def group_sequential_boundaries(
    looks: int | Sequence[float],
    *,
    method: BoundaryMethod = "obrien-fleming",
    alpha: float = 0.05,
) -> GroupSequentialBoundaries:
    """Compute two-sided group sequential boundaries for ``K`` interim looks.

    Parameters
    ----------
    looks : int or sequence of float
        Either the number of equally spaced looks ``K`` or the information
        fractions of the looks (strictly increasing, in (0, 1], last = 1).
    method : str, default "obrien-fleming"
        * ``"pocock"``: the same critical value ``C_P`` at every look.
        * ``"obrien-fleming"``: ``c_k = C_B / sqrt(t_k)``, very strict early
          and close to the fixed-sample value at the end.
        * ``"lan-demets-obrien-fleming"`` / ``"lan-demets-pocock"``: alpha
          spending (Lan & DeMets 1983). Each boundary is chosen so that the
          probability of first crossing at look ``k`` equals
          ``alpha*(t_k) - alpha*(t_{k-1})``, with the spending functions
          ``alpha*(t) = 4 (1 - Phi(z_{1 - alpha/4} / sqrt(t)))`` (each side
          spends ``alpha/2`` O'Brien-Fleming style, the convention of
          ``gsDesign`` and ``ldbounds``) and
          ``alpha*(t) = alpha * log(1 + (e - 1) t)``. Unlike the classical
          designs, spending boundaries only need the information fractions
          of the looks that have actually happened.
    alpha : float, default 0.05
        Overall two-sided type-I error.

    Returns
    -------
    GroupSequentialBoundaries
        Critical values, nominal levels and cumulative alpha per look.

    Notes
    -----
    The probability of stopping at look ``k`` under H0 is the multivariate
    normal probability ``P(|Z_j| < c_j for j < k, |Z_k| >= c_k)``, a rectangle
    probability evaluated with ``scipy.stats.multivariate_normal.cdf`` (exact
    in one and two dimensions, randomized quasi-Monte Carlo with a fixed seed
    beyond). Classical constants are found with Brent's method so that these
    probabilities sum to ``alpha``. For ``alpha = 0.05`` and ``K = 5`` this
    reproduces Jennison & Turnbull's ``C_P = 2.413`` and ``C_B = 2.040``.
    Boundaries are accurate to about 1e-4 or better. The cost grows with
    ``K``: under a second for 10 looks, up to about 10 seconds for 20.

    References
    ----------
    Jennison, C., & Turnbull, B. W. (2000). *Group Sequential Methods with
    Applications to Clinical Trials*, ch. 2 and 7. Chapman & Hall/CRC.

    Lan, K. K. G., & DeMets, D. L. (1983). Discrete sequential boundaries for
    clinical trials. *Biometrika*, 70(3), 659-663.

    Examples
    --------
    >>> design = group_sequential_boundaries(5, method="pocock")
    >>> [round(z, 3) for z in design.z_boundaries]
    [2.413, 2.413, 2.413, 2.413, 2.413]
    """
    fractions = _information_fractions(looks)
    alpha = check_open_unit("alpha", alpha)
    if alpha >= 0.5:
        raise InvalidInputError(
            f"alpha must be < 0.5 for a two-sided test, got {alpha:g}"
        )
    if method == "pocock":
        bounds = _solve_constant(np.ones(fractions.size), fractions, alpha)
    elif method == "obrien-fleming":
        bounds = _solve_constant(1.0 / np.sqrt(fractions), fractions, alpha)
    elif method == "lan-demets-obrien-fleming":
        bounds = _spending_bounds(fractions, alpha, _obrien_fleming_spending)
    elif method == "lan-demets-pocock":
        bounds = _spending_bounds(fractions, alpha, _pocock_spending)
    else:
        raise InvalidInputError(f"method must be one of {_METHODS}, got {method!r}")

    cumulative = np.cumsum(_exit_probabilities(bounds, fractions))
    return GroupSequentialBoundaries(
        method=method,
        alpha=alpha,
        information_fractions=tuple(fractions.tolist()),
        z_boundaries=tuple(bounds.tolist()),
        nominal_alpha=tuple((2.0 * stats.norm.sf(bounds)).tolist()),
        cumulative_alpha=tuple(cumulative.tolist()),
    )


def crossing_probability(
    z_boundaries: Sequence[float], information_fractions: Sequence[float] | None = None
) -> float:
    """Probability under H0 that ``|Z_k| >= c_k`` at some look.

    This is the overall type-I error of the rule "reject the first time the
    z-statistic crosses the boundary". With a constant boundary of 1.96 it
    shows how peeking inflates the error rate.

    Parameters
    ----------
    z_boundaries : sequence of float
        Positive critical values ``c_1, ..., c_K``.
    information_fractions : sequence of float, optional
        Information fractions of the looks; equally spaced by default.

    Returns
    -------
    float
        The overall two-sided crossing probability.

    Examples
    --------
    >>> round(crossing_probability([1.959964] * 5), 3)  # five looks at p < 0.05
    0.142
    """
    bounds = as_float_array("z_boundaries", z_boundaries, min_size=1)
    if np.any(bounds <= 0):
        raise InvalidInputError("z_boundaries must be positive")
    fractions = _information_fractions(
        bounds.size if information_fractions is None else information_fractions
    )
    if fractions.size != bounds.size:
        raise InvalidInputError(
            f"got {bounds.size} boundaries but {fractions.size} information fractions"
        )
    return float(np.sum(_exit_probabilities(bounds, fractions)))


@dataclass(frozen=True)
class AlwaysValidResult(Result):
    """Always-valid p-values and confidence sequence from the mSPRT.

    Attributes
    ----------
    tau : float
        Standard deviation of the normal mixing distribution of effects.
    alpha : float
        Level of the confidence sequence and of the rejection rule.
    null_value : float
        Effect under H0.
    estimates, standard_errors : tuple of float
        The inputs, one entry per look.
    p_values : tuple of float
        Always-valid p-values; non-increasing over looks.
    ci_lower, ci_upper : tuple of float
        Confidence sequence: the true effect lies inside every interval with
        probability at least ``1 - alpha``, simultaneously over all looks.
    rejection_look : int or None
        First look (1-based) with ``p_value <= alpha``, if any.
    """

    tau: float
    alpha: float
    null_value: float
    estimates: tuple[float, ...]
    standard_errors: tuple[float, ...]
    p_values: tuple[float, ...]
    ci_lower: tuple[float, ...]
    ci_upper: tuple[float, ...]
    rejection_look: int | None

    def summary(self) -> str:
        """Return a human-readable summary."""
        decision = (
            f"H0 (effect = {self.null_value:g}) rejected at look {self.rejection_look}"
            if self.rejection_look is not None
            else f"H0 (effect = {self.null_value:g}) not rejected"
        )
        return "\n".join(
            [
                f"mSPRT always-valid inference (tau = {self.tau:g}, "
                f"alpha = {self.alpha:g}), {len(self.estimates)} looks",
                f"  latest estimate      {self.estimates[-1]:+.6g} "
                f"(se {self.standard_errors[-1]:.4g})",
                f"  always-valid p-value {format_p_value(self.p_values[-1])}",
                f"  confidence sequence  [{self.ci_lower[-1]:+.6g}, "
                f"{self.ci_upper[-1]:+.6g}]",
                f"  {decision}",
            ]
        )


def always_valid_inference(
    estimates: npt.ArrayLike,
    standard_errors: npt.ArrayLike,
    *,
    tau: float,
    alpha: float = 0.05,
    null_value: float = 0.0,
) -> AlwaysValidResult:
    """Always-valid inference for a difference with the normal-mixture SPRT.

    Parameters
    ----------
    estimates : array_like
        Estimated effect (e.g. difference in means or proportions) at each
        look, computed from all data collected so far.
    standard_errors : array_like
        Standard error of each estimate (e.g. ``sqrt(s_c^2/n_c + s_t^2/n_t)``).
    tau : float
        Standard deviation of the ``Normal(0, tau^2)`` mixing distribution over
        the effect, in the units of the estimate. Validity holds for any
        ``tau``; power is best when ``tau`` is close to the size of effects you
        expect (for example the MDE).
    alpha : float, default 0.05
        Error level: ``P(any p_n <= alpha | H0) <= alpha``.
    null_value : float, default 0.0
        Effect under the null hypothesis.

    Returns
    -------
    AlwaysValidResult
        Always-valid p-values, confidence sequence and the first rejection.

    Notes
    -----
    With ``theta_n`` the estimate and ``V_n`` its variance, the mixture
    likelihood ratio of ``N(theta, V_n)`` versus ``N(theta_0, V_n)`` with
    ``theta ~ N(theta_0, tau^2)`` is::

        Lambda_n = sqrt(V_n / (V_n + tau^2))
                   * exp(tau^2 (theta_n - theta_0)^2 / (2 V_n (V_n + tau^2)))

    Under H0, ``Lambda_n`` is a nonnegative martingale with mean 1, so by
    Ville's inequality ``P(sup_n Lambda_n >= 1/alpha) <= alpha``. Hence
    ``p_n = min(1, 1 / max_{m <= n} Lambda_m)`` is an always-valid p-value.
    Inverting ``Lambda_m(theta_0) < 1/alpha`` gives intervals
    ``theta_m +/- sqrt(2 V_m (V_m + tau^2) / tau^2
    * log(sqrt((V_m + tau^2) / V_m) / alpha))``, intersected over looks.
    Variances are plug-in estimates (normal approximation), as in Johari et
    al.; the guarantee is asymptotic in the sample size.

    References
    ----------
    Johari, R., Koomen, P., Pekelis, L., & Walsh, D. (2017). Peeking at A/B
    tests: why it matters, and what to do about it. *KDD 2017*.

    Johari, R., Pekelis, L., & Walsh, D. (2022). Always valid inference:
    continuous monitoring of A/B tests. *Operations Research*, 70(3).
    """
    theta = as_float_array("estimates", estimates, min_size=1)
    se = as_float_array("standard_errors", standard_errors, min_size=1)
    if theta.size != se.size:
        raise InvalidInputError(
            "estimates and standard_errors differ in length "
            f"({theta.size} != {se.size})"
        )
    if np.any(se <= 0):
        raise InvalidInputError("standard_errors must be positive")
    tau = check_positive("tau", tau)
    alpha = check_open_unit("alpha", alpha)
    null_value = check_real("null_value", null_value)

    variance = se**2
    log_lr = _log_mixture_likelihood_ratio(theta - null_value, variance, tau)
    p_values = np.minimum(1.0, np.exp(-np.maximum.accumulate(log_lr)))
    half_width = np.sqrt(
        2.0
        * variance
        * (variance + tau**2)
        / tau**2
        * (math.log(1.0 / alpha) + 0.5 * np.log((variance + tau**2) / variance))
    )
    lower = np.maximum.accumulate(theta - half_width)
    upper = np.minimum.accumulate(theta + half_width)
    rejected = np.flatnonzero(p_values <= alpha)
    return AlwaysValidResult(
        tau=tau,
        alpha=alpha,
        null_value=null_value,
        estimates=tuple(theta.tolist()),
        standard_errors=tuple(se.tolist()),
        p_values=tuple(p_values.tolist()),
        ci_lower=tuple(lower.tolist()),
        ci_upper=tuple(upper.tolist()),
        rejection_look=int(rejected[0]) + 1 if rejected.size else None,
    )


def _log_mixture_likelihood_ratio(
    deviation: FloatArray, variance: FloatArray, tau: float
) -> FloatArray:
    """Return the log of the normal-mixture likelihood ratio, elementwise.

    ``deviation`` is ``theta_n - theta_0`` and ``variance`` is ``V_n``; see
    :func:`always_valid_inference`. Working on the log scale avoids overflow
    when the evidence is overwhelming.
    """
    total = variance + tau**2
    return 0.5 * np.log(variance / total) + tau**2 * deviation**2 / (
        2.0 * variance * total
    )


# --------------------------------------------------------------------------
# Internal helpers for group sequential boundaries
# --------------------------------------------------------------------------


def _information_fractions(looks: int | Sequence[float]) -> FloatArray:
    if isinstance(looks, int) and not isinstance(looks, bool):
        k = check_count("looks", looks, minimum=1)
        return np.arange(1, k + 1, dtype=np.float64) / k
    fractions = as_float_array("information_fractions", looks, min_size=1)
    if np.any(fractions <= 0) or np.any(fractions > 1):
        raise InvalidInputError("information fractions must lie in (0, 1]")
    if np.any(np.diff(fractions) <= 0):
        raise InvalidInputError("information fractions must be strictly increasing")
    if fractions[-1] != 1.0:
        raise InvalidInputError("the last information fraction must be 1")
    return fractions


def _correlation(fractions: FloatArray) -> FloatArray:
    """Correlation of the sequential z-statistics: sqrt(t_i / t_j) for t_i <= t_j."""
    return np.sqrt(
        np.minimum.outer(fractions, fractions) / np.maximum.outer(fractions, fractions)
    )


def _upper_exit_probability(bounds: FloatArray, fractions: FloatArray) -> float:
    """P(|Z_j| < c_j for j < k and Z_k >= c_k) under H0, with k = len(bounds)."""
    k = bounds.size
    if k == 1:
        return float(stats.norm.sf(bounds[0]))
    lower = np.append(-bounds[:-1], bounds[-1])
    upper = np.append(bounds[:-1], np.inf)
    return float(
        stats.multivariate_normal.cdf(
            upper,
            mean=np.zeros(k),
            cov=_correlation(fractions[:k]),
            lower_limit=lower,
            abseps=_MVN_ABSOLUTE_ERROR,
            rng=np.random.default_rng(_MVN_SEED),
        )
    )


def _exit_probabilities(bounds: FloatArray, fractions: FloatArray) -> FloatArray:
    """Two-sided probability of stopping at each look under H0.

    The boundaries are symmetric, so exiting through the lower boundary is as
    likely as exiting through the upper one.
    """
    return np.array(
        [
            2.0 * _upper_exit_probability(bounds[:k], fractions)
            for k in range(1, bounds.size + 1)
        ]
    )


def _solve_constant(
    shape: FloatArray, fractions: FloatArray, alpha: float
) -> FloatArray:
    """Scale ``shape`` (``shape[-1] == 1``) so the crossing probability is alpha."""
    lowest = float(stats.norm.isf(alpha / 2))  # the final look alone spends alpha
    if shape.size == 1:
        return np.array([lowest])
    highest = float(stats.norm.isf(alpha / (2 * shape.size)))  # Bonferroni bound

    def excess(constant: float) -> float:
        return float(np.sum(_exit_probabilities(constant * shape, fractions))) - alpha

    constant = float(optimize.brentq(excess, lowest, highest, xtol=_ROOT_TOLERANCE))
    return constant * shape


def _obrien_fleming_spending(t: float, alpha: float) -> float:
    return float(4.0 * stats.norm.sf(stats.norm.isf(alpha / 4) / math.sqrt(t)))


def _pocock_spending(t: float, alpha: float) -> float:
    return alpha * math.log(1.0 + (math.e - 1.0) * t)


def _spending_bounds(
    fractions: FloatArray, alpha: float, spending: Callable[[float, float], float]
) -> FloatArray:
    bounds: list[float] = []
    spent_before = 0.0
    for look, t in enumerate(fractions, start=1):
        spent = spending(float(t), alpha)
        bounds.append(
            _next_bound(
                np.array(bounds), fractions[:look], spent - spent_before, spent_before
            )
        )
        spent_before = spent
    return np.array(bounds)


def _next_bound(
    previous: FloatArray, fractions: FloatArray, target: float, spent_before: float
) -> float:
    """Return the next critical value, whose two-sided exit probability is target."""

    def excess(c: float) -> float:
        upper_exit = _upper_exit_probability(np.append(previous, c), fractions)
        return 2.0 * upper_exit - target

    # P(Z_k >= c) - spent_before <= P(exit up at look k) <= P(Z_k >= c), so the
    # excess is positive where P(Z_k >= c) = target + 2 * spent_before and
    # negative where P(Z_k >= c) = target / 4. Both bracket ends are small
    # probabilities (fast to integrate) and the margins keep the sign change
    # clear of integration error.
    lowest = float(stats.norm.isf(target + 2.0 * spent_before))
    highest = float(stats.norm.isf(target / 4.0))
    return float(optimize.brentq(excess, lowest, highest, xtol=_ROOT_TOLERANCE))

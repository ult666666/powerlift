"""Bayesian analysis of conversion rates with the Beta-Binomial model.

Each arm's conversion rate ``p`` gets an independent ``Beta(a, b)`` prior.
After ``x`` conversions among ``n`` units the posterior is
``Beta(a + x, b + n - x)`` (conjugacy), so every quantity below is a
functional of two Beta distributions.

Priors
------
The default ``Beta(1, 1)`` is uniform on [0, 1]: every conversion rate is
equally plausible a priori, and it adds the equivalent of one conversion and
one non-conversion to each arm. ``Beta(0.5, 0.5)`` (Jeffreys) is the other
common non-informative choice. At the sample sizes of online experiments the
data dominate either one. An informative prior such as
``Beta(100 * p0, 100 * (1 - p0))`` encodes a historical rate ``p0`` with the
weight of 100 prior users; it should be justified before use, because it
shrinks both arms towards ``p0``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import integrate, special, stats

from powerlift._results import ConfidenceInterval, Result, format_level
from powerlift._validation import (
    check_binomial,
    check_count,
    check_open_unit,
    check_positive,
)


@dataclass(frozen=True)
class BetaPosterior(Result):
    """Beta posterior of one arm's conversion rate.

    Attributes
    ----------
    alpha, beta : float
        Posterior parameters ``prior_alpha + conversions`` and
        ``prior_beta + n - conversions``.
    mean : float
        Posterior mean ``alpha / (alpha + beta)``.
    credible_interval : ConfidenceInterval
        Equal-tailed credible interval.
    """

    alpha: float
    beta: float
    mean: float
    credible_interval: ConfidenceInterval

    def summary(self) -> str:
        """Return a one-line summary."""
        return (
            f"Beta({self.alpha:g}, {self.beta:g}): mean {self.mean:.2%}, "
            f"{format_level(self.credible_interval.level)} credible interval "
            f"{self.credible_interval.format('.2%')}"
        )


@dataclass(frozen=True)
class BayesianComparison(Result):
    """Posterior comparison of two conversion rates.

    Attributes
    ----------
    prior_alpha, prior_beta : float
        Parameters of the Beta prior used for both arms.
    control, treatment : BetaPosterior
        Posterior of each arm.
    prob_treatment_better : float
        ``P(p_treatment > p_control | data)``, by numerical integration.
    expected_loss_treatment : float
        ``E[max(p_control - p_treatment, 0)]``: the conversion rate expected
        to be lost by shipping the treatment if it is in fact worse.
    expected_loss_control : float
        ``E[max(p_treatment - p_control, 0)]``: the expected loss of keeping
        the control.
    difference_interval : ConfidenceInterval
        Credible interval for ``p_treatment - p_control`` (Monte Carlo).
    relative_lift_interval : ConfidenceInterval
        Credible interval for ``p_treatment / p_control - 1`` (Monte Carlo).
    n_draws, seed : int
        Monte Carlo settings for the two intervals above.
    """

    prior_alpha: float
    prior_beta: float
    control: BetaPosterior
    treatment: BetaPosterior
    prob_treatment_better: float
    expected_loss_treatment: float
    expected_loss_control: float
    difference_interval: ConfidenceInterval
    relative_lift_interval: ConfidenceInterval
    n_draws: int
    seed: int

    def summary(self) -> str:
        """Return a human-readable summary."""
        level = format_level(self.difference_interval.level)
        difference = (
            f"[{self.difference_interval.lower * 100:+.2f} pp, "
            f"{self.difference_interval.upper * 100:+.2f} pp]"
        )
        return "\n".join(
            [
                f"Bayesian Beta-Binomial comparison, prior "
                f"Beta({self.prior_alpha:g}, {self.prior_beta:g}) for each arm",
                f"  control      {self.control.summary()}",
                f"  treatment    {self.treatment.summary()}",
                f"  P(treatment > control)          {self.prob_treatment_better:.4f}",
                "  expected loss, ship treatment   "
                f"{self.expected_loss_treatment * 100:.4f} pp",
                "  expected loss, keep control     "
                f"{self.expected_loss_control * 100:.4f} pp",
                f"  difference     {level} credible interval {difference}",
                f"  relative lift  {level} credible interval "
                f"{self.relative_lift_interval.format('+.2%')}",
                f"  (intervals for the difference and lift: {self.n_draws:,} Monte "
                f"Carlo draws, seed {self.seed})",
            ]
        )


def compare_proportions_bayes(
    conversions_control: int,
    n_control: int,
    conversions_treatment: int,
    n_treatment: int,
    *,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    credible_level: float = 0.95,
    n_draws: int = 200_000,
    seed: int = 0,
) -> BayesianComparison:
    """Compare two conversion rates with independent Beta-Binomial models.

    Parameters
    ----------
    conversions_control, n_control : int
        Conversions and units in the control arm.
    conversions_treatment, n_treatment : int
        Conversions and units in the treatment arm.
    prior_alpha, prior_beta : float, default 1.0
        Beta prior shared by both arms (uniform by default; see the module
        docstring for alternatives).
    credible_level : float, default 0.95
        Probability content of the equal-tailed credible intervals.
    n_draws : int, default 200000
        Posterior draws used for the difference and relative-lift intervals.
    seed : int, default 0
        Seed of the generator for those draws, so results are reproducible.

    Returns
    -------
    BayesianComparison
        Posteriors, ``P(treatment > control)``, expected losses and credible
        intervals.

    Notes
    -----
    ``P(T > C) = integral of f_T(x) F_C(x) dx`` is computed by adaptive
    quadrature (see :func:`prob_beta_greater`). The expected losses use the
    identity ``x f(x; a, b) = a / (a + b) * f(x; a + 1, b)``, which gives::

        E[max(C - T, 0)] = E[C] P(C' > T) - E[T] P(C > T')

    with ``C' ~ Beta(a_C + 1, b_C)`` and ``T' ~ Beta(a_T + 1, b_T)``, so they
    also reduce to one-dimensional integrals. Only the two intervals use Monte
    Carlo.

    References
    ----------
    Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.), ch. 2.
    CRC Press.

    Examples
    --------
    >>> result = compare_proportions_bayes(480, 5000, 560, 5000)
    >>> round(result.prob_treatment_better, 4)
    0.9956
    """
    x_c, n_c = check_binomial("control", conversions_control, n_control)
    x_t, n_t = check_binomial("treatment", conversions_treatment, n_treatment)
    a0 = check_positive("prior_alpha", prior_alpha)
    b0 = check_positive("prior_beta", prior_beta)
    level = check_open_unit("credible_level", credible_level)
    draws = check_count("n_draws", n_draws, minimum=100)
    seed = check_count("seed", seed)

    control = _posterior(a0 + x_c, b0 + n_c - x_c, level)
    treatment = _posterior(a0 + x_t, b0 + n_t - x_t, level)
    a_c, b_c, a_t, b_t = control.alpha, control.beta, treatment.alpha, treatment.beta

    loss_treatment = control.mean * prob_beta_greater(
        a_c + 1, b_c, a_t, b_t
    ) - treatment.mean * prob_beta_greater(a_c, b_c, a_t + 1, b_t)
    loss_control = treatment.mean * prob_beta_greater(
        a_t + 1, b_t, a_c, b_c
    ) - control.mean * prob_beta_greater(a_t, b_t, a_c + 1, b_c)

    rng = np.random.default_rng(seed)
    p_c = rng.beta(a_c, b_c, size=draws)
    p_t = rng.beta(a_t, b_t, size=draws)
    tails = [(1 - level) / 2, (1 + level) / 2]
    difference = np.quantile(p_t - p_c, tails)
    lift = np.quantile(p_t / p_c - 1, tails)
    return BayesianComparison(
        prior_alpha=a0,
        prior_beta=b0,
        control=control,
        treatment=treatment,
        prob_treatment_better=prob_beta_greater(a_t, b_t, a_c, b_c),
        # Clip rounding noise of order 1e-12 when the loss is essentially zero.
        expected_loss_treatment=max(0.0, loss_treatment),
        expected_loss_control=max(0.0, loss_control),
        difference_interval=ConfidenceInterval(
            float(difference[0]), float(difference[1]), level
        ),
        relative_lift_interval=ConfidenceInterval(
            float(lift[0]), float(lift[1]), level
        ),
        n_draws=draws,
        seed=seed,
    )


def prob_beta_greater(
    alpha_x: float, beta_x: float, alpha_y: float, beta_y: float
) -> float:
    """Return ``P(X > Y)`` for independent Beta random variables X and Y.

    ``X ~ Beta(alpha_x, beta_x)`` and ``Y ~ Beta(alpha_y, beta_y)``.

    The probability ``integral_0^1 f_X(u) F_Y(u) du`` is evaluated with
    adaptive Gauss-Kronrod quadrature (``scipy.integrate.quad``) over the
    central ``1 - 2e-15`` of X's distribution, so the truncation error is
    below ``2e-15`` and the result is accurate to about ``1e-10``.

    Examples
    --------
    >>> round(prob_beta_greater(2, 1, 1, 1), 6)  # P(max(U1, U2) > U3)
    0.666667
    """
    a_x, b_x = check_positive("alpha_x", alpha_x), check_positive("beta_x", beta_x)
    a_y, b_y = check_positive("alpha_y", alpha_y), check_positive("beta_y", beta_y)
    log_norm = special.betaln(a_x, b_x)

    def integrand(u: float) -> float:
        log_density = (a_x - 1) * math.log(u) + (b_x - 1) * math.log1p(-u) - log_norm
        return math.exp(log_density) * float(special.betainc(a_y, b_y, u))

    lower = float(stats.beta.ppf(1e-15, a_x, b_x))
    upper = float(stats.beta.isf(1e-15, a_x, b_x))
    # Telling the integrator where Y's CDF rises helps when the posteriors are
    # far apart relative to their widths.
    median_y = float(stats.beta.median(a_y, b_y))
    points = [median_y] if lower < median_y < upper else None
    value, _ = integrate.quad(
        integrand, lower, upper, points=points, limit=200, epsabs=1e-13, epsrel=1e-10
    )
    return min(1.0, max(0.0, float(value)))


def _posterior(alpha: float, beta: float, level: float) -> BetaPosterior:
    low, high = stats.beta.ppf([(1 - level) / 2, (1 + level) / 2], alpha, beta)
    return BetaPosterior(
        alpha=alpha,
        beta=beta,
        mean=alpha / (alpha + beta),
        credible_interval=ConfidenceInterval(float(low), float(high), level),
    )

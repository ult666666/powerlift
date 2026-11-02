"""The distribution behind Dunnett's many-to-one comparisons.

Comparing ``k`` treatment arms with one control gives ``k`` t-statistics

    T_i = (mean_i - mean_0) / (s * sqrt(1/n_i + 1/n_0)),   i = 1..k,

which share the control mean and the pooled standard deviation ``s``. Under
the null hypothesis they follow a multivariate t distribution with ``df``
degrees of freedom whose correlations have a *product form*,
``corr(T_i, T_j) = lambda_i * lambda_j`` with ``lambda_i = sqrt(n_i / (n_i + n_0))``.

The product form lets one write ``T_i = (lambda_i Z_0 + sqrt(1 - lambda_i^2) E_i) / S``
with independent standard normals ``Z_0, E_1, ..., E_k`` and an independent
``S = sqrt(chi2_df / df)``. Given ``Z_0`` and ``S`` the statistics are
independent, so a k-dimensional integral becomes a two-dimensional one
(Dunnett 1955; Dunnett 1989, algorithm AS 251)::

    P(max_i |T_i| <= c) = E_S E_Z0 [ prod_i (Phi(a_i+) - Phi(a_i-)) ],
    a_i+- = (+-c S - lambda_i Z_0) / sqrt(1 - lambda_i^2)

and ``P(max_i T_i <= c)`` likewise with ``prod_i Phi(a_i+)``. Both expectations
are computed with the trapezoidal rule, which converges exponentially fast for
smooth integrands that decay like a normal density (Trefethen & Weideman 2014):
``Z_0`` on a grid over [-9, 9], and ``S`` through its normal score ``y`` (with
``S = sqrt(chi2.ppf(Phi(y), df) / df)``) on a grid over [-8.5, 8.5]. The result is
deterministic: the tests require agreement with closed forms to 1e-12 and with
SciPy's general multivariate normal and t integrators to 1e-6 and 1e-5, whereas
:func:`scipy.stats.dunnett` uses randomized quasi-Monte Carlo integration whose
p-values move by up to about 1e-4 from one random seed to the next.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Literal

import numpy as np
from scipy import optimize, special, stats

from powerlift._validation import FloatArray
from powerlift.exceptions import InvalidInputError

Tail = Literal["two-sided", "one-sided"]

# Smallest tail probability 1 - level that max_t_quantile accepts. With one
# comparison, where a closed form exists, the critical value at 1e-10 is within a
# relative 1e-6 of it for df >= 5 (less accurate for fewer df), but 1e-3 off at
# 1e-14 even for df = inf. At 1e-16 a finite df made the search return the end of
# its bracket, and below that it never ended.
MIN_TAIL_PROBABILITY = 1e-10
# The bracket search gives up beyond this cutoff instead of doubling forever.
_MAX_CUTOFF = 2.0**64

# Trapezoid steps. Each factor of the integrand changes over a distance of about
# sqrt(1 - lambda^2) / lambda in Z_0, so the Z_0 step shrinks for arms much
# larger than the control. The tests compare with closed forms and SciPy at 1e-8.
_Z_STEP, _Z_LIMIT = 0.125, 9.0
_Y_STEP, _Y_LIMIT = 0.25, 8.5
_MAX_CORRELATION_FACTOR = 0.9999


@lru_cache(maxsize=16)
def _z_grid(step: float) -> tuple[FloatArray, FloatArray]:
    """Nodes for Z_0 and their trapezoid weights times the normal density."""
    z = np.arange(-_Z_LIMIT, _Z_LIMIT + step / 2, step)
    return z, step * stats.norm.pdf(z)


@lru_cache(maxsize=64)
def _scale_grid(df: float) -> tuple[FloatArray, FloatArray]:
    """Nodes for S = sqrt(chi2_df / df) and weights; one node S = 1 if df is inf."""
    if math.isinf(df):
        return np.ones(1), np.ones(1)
    y = np.arange(-_Y_LIMIT, _Y_LIMIT + _Y_STEP / 2, _Y_STEP)
    # chi2.isf(norm.sf(y)) keeps full precision in the upper tail.
    chi2 = np.where(
        y < 0,
        stats.chi2.ppf(stats.norm.cdf(y), df),
        stats.chi2.isf(stats.norm.sf(y), df),
    )
    return np.sqrt(chi2 / df), _Y_STEP * stats.norm.pdf(y)


def max_t_cdf(c: float, lambdas: FloatArray, df: float, tail: Tail) -> float:
    """``P(max_i |T_i| <= c)`` (two-sided) or ``P(max_i T_i <= c)`` (one-sided).

    Parameters
    ----------
    c : float
        Cutoff (non-negative for the two-sided probability).
    lambdas : array of float
        Correlation factors ``lambda_i`` in [0, 1), one per comparison.
    df : float
        Degrees of freedom of the pooled variance; ``math.inf`` for normal
        statistics.
    tail : {"two-sided", "one-sided"}
        Which probability to compute.
    """
    return float(max_t_cdf_many(np.array([c], dtype=np.float64), lambdas, df, tail)[0])


def max_t_cdf_many(
    cutoffs: FloatArray, lambdas: FloatArray, df: float, tail: Tail
) -> FloatArray:
    """:func:`max_t_cdf` for several cutoffs at once (one array operation)."""
    lam = np.minimum(np.asarray(lambdas, dtype=np.float64), _MAX_CORRELATION_FACTOR)
    spread = np.sqrt(1.0 - lam**2)
    narrowest = float(np.min(spread / np.maximum(lam, 1e-12)))
    z, z_weights = _z_grid(min(_Z_STEP, 2.0 ** math.floor(math.log2(narrowest / 4))))
    s, s_weights = _scale_grid(df)
    # Axes: cutoff, S node, Z_0 node, comparison.
    centre = lam * z[:, np.newaxis]
    bound = np.asarray(cutoffs, dtype=np.float64)[:, np.newaxis] * s
    bound = bound[:, :, np.newaxis, np.newaxis]
    upper = special.ndtr((bound - centre) / spread)
    if tail == "two-sided":
        upper -= special.ndtr((-bound - centre) / spread)
    probability: FloatArray = (upper.prod(axis=3) @ z_weights) @ s_weights
    return np.clip(probability, 0.0, 1.0)


def max_t_quantile(level: float, lambdas: FloatArray, df: float, tail: Tail) -> float:
    """Return the cutoff ``c`` with ``max_t_cdf(c, ...) = level``: Dunnett's value.

    Results are cached, because simulations and repeated readouts ask for the
    same critical value many times.

    Raises
    ------
    InvalidInputError
        If ``1 - level`` (the ``alpha`` of the test) is below
        :data:`MIN_TAIL_PROBABILITY`, where the integration is no longer accurate.
    """
    if not 1.0 - level >= MIN_TAIL_PROBABILITY:
        raise InvalidInputError(
            f"alpha must be at least {MIN_TAIL_PROBABILITY:g} for Dunnett's critical "
            f"value, got {1.0 - level:.3g}: smaller tail probabilities are beyond the "
            "accuracy of the numerical integration"
        )
    key = tuple(float(x) for x in np.asarray(lambdas, dtype=np.float64))
    return _quantile(float(level), key, float(df), tail)


@lru_cache(maxsize=256)
def _quantile(level: float, lambdas: tuple[float, ...], df: float, tail: Tail) -> float:
    lam = np.array(lambdas)

    def gap(c: float) -> float:
        return max_t_cdf(c, lam, df, tail) - level

    upper = 2.0
    while gap(upper) < 0.0:
        upper *= 2.0
        if upper > _MAX_CUTOFF:
            raise InvalidInputError(
                f"no critical value below {_MAX_CUTOFF:g} reaches probability {level!r}"
            )
    lower = 0.0 if tail == "two-sided" else -upper
    return float(optimize.brentq(gap, lower, upper, xtol=1e-12, rtol=1e-12))

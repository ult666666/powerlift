"""Small statistical helpers shared across modules."""

from __future__ import annotations

from typing import Literal

from scipy import stats

Alternative = Literal["two-sided", "greater", "less"]
"""Alternative hypothesis for the effect ``treatment - control``.

``"greater"`` means H1: treatment > control, ``"less"`` means H1: treatment <
control. The names follow :func:`scipy.stats.ttest_ind`.
"""

ALTERNATIVES: tuple[Alternative, ...] = ("two-sided", "greater", "less")


def normal_p_value(z: float, alternative: Alternative) -> float:
    """P-value of a z-statistic under a standard normal null distribution."""
    if alternative == "two-sided":
        return float(2.0 * stats.norm.sf(abs(z)))
    if alternative == "greater":
        return float(stats.norm.sf(z))
    return float(stats.norm.cdf(z))


def t_p_value(t: float, df: float, alternative: Alternative) -> float:
    """P-value of a t-statistic with ``df`` degrees of freedom."""
    if alternative == "two-sided":
        return float(2.0 * stats.t.sf(abs(t), df))
    if alternative == "greater":
        return float(stats.t.sf(t, df))
    return float(stats.t.cdf(t, df))


def normal_critical_value(alpha: float) -> float:
    """Two-sided standard normal critical value ``z_{1 - alpha/2}``."""
    return float(stats.norm.isf(alpha / 2.0))

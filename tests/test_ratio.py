"""Tests for the delta-method ratio metric analysis."""

from typing import Any

import numpy as np
import pytest

from powerlift import (
    InsufficientDataError,
    InvalidInputError,
    compare_means,
    compare_ratio_metrics,
)


def test_constant_denominator_reduces_to_a_difference_in_means() -> None:
    rng = np.random.default_rng(0)
    y_c, y_t = rng.normal(10, 3, 500), rng.normal(10.5, 4, 400)
    ratio = compare_ratio_metrics(y_c, np.ones(500), y_t, np.ones(400))
    welch = compare_means(y_c, y_t)
    assert ratio.difference == pytest.approx(welch.difference)
    assert ratio.se_difference == pytest.approx(welch.se_difference)
    assert ratio.statistic == pytest.approx(welch.statistic)
    assert ratio.df is None  # normal reference distribution


def test_matches_the_covariance_form_of_the_delta_method() -> None:
    rng = np.random.default_rng(1)
    sessions_c = rng.poisson(3, 800) + 1
    sessions_t = rng.poisson(3, 700) + 1
    revenue_c = sessions_c * rng.gamma(2.0, 1.5, 800)
    revenue_t = sessions_t * rng.gamma(2.0, 1.6, 700)
    result = compare_ratio_metrics(revenue_c, sessions_c, revenue_t, sessions_t)

    def delta_variance(y: np.ndarray, x: np.ndarray) -> float:
        # Deng, Knoblich & Lu (2018): Var(Y/X) with sample (co)variances.
        cov = np.cov(y, x, ddof=1)
        r = y.mean() / x.mean()
        return float(
            (cov[0, 0] - 2 * r * cov[0, 1] + r**2 * cov[1, 1])
            / (len(y) * x.mean() ** 2)
        )

    expected_se = np.sqrt(
        delta_variance(revenue_c, sessions_c) + delta_variance(revenue_t, sessions_t)
    )
    assert result.estimate_control == pytest.approx(revenue_c.sum() / sessions_c.sum())
    assert result.estimate_treatment == pytest.approx(
        revenue_t.sum() / sessions_t.sum()
    )
    assert result.se_difference == pytest.approx(expected_se, rel=1e-12)


def _simulate_arm(
    rng: np.random.Generator, n_users: int, lift: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Users with correlated sessions: a user-level spend factor drives all sessions."""
    sessions = rng.poisson(2.0, n_users) + 1
    user_factor = rng.gamma(1.0, 1.0, n_users) * (1 + lift)
    per_session = np.repeat(user_factor, sessions) * rng.exponential(
        1.0, sessions.sum()
    )
    owners = np.repeat(np.arange(n_users), sessions)
    revenue = np.bincount(owners, weights=per_session, minlength=n_users)
    return revenue, sessions.astype(float), per_session


def test_delta_method_standard_error_matches_simulation() -> None:
    rng = np.random.default_rng(2018)
    differences, delta_ses, naive_ses = [], [], []
    hits = 0
    for _ in range(2_000):
        revenue_c, sessions_c, per_session_c = _simulate_arm(rng, 1_000, 0.0)
        revenue_t, sessions_t, per_session_t = _simulate_arm(rng, 1_000, 0.0)
        result = compare_ratio_metrics(revenue_c, sessions_c, revenue_t, sessions_t)
        differences.append(result.difference)
        delta_ses.append(result.se_difference)
        hits += result.ci_difference.contains(0.0)
        # The naive analysis treats every session as an independent observation.
        naive_ses.append(
            np.sqrt(
                per_session_c.var(ddof=1) / per_session_c.size
                + per_session_t.var(ddof=1) / per_session_t.size
            )
        )
    empirical_sd = np.std(differences, ddof=1)
    # The delta-method SE agrees with the true sampling variability ...
    assert np.mean(delta_ses) / empirical_sd == pytest.approx(1.0, abs=0.06)
    assert hits / 2_000 == pytest.approx(0.95, abs=0.02)
    # ... while the session-level SE is far too small, inflating false positives.
    assert np.mean(naive_ses) / empirical_sd < 0.8


def test_errors() -> None:
    with pytest.raises(InvalidInputError, match="same length"):
        compare_ratio_metrics([1.0, 2.0], [1.0, 1.0, 1.0], [1.0, 2.0], [1.0, 1.0])
    with pytest.raises(InvalidInputError, match="positive mean"):
        compare_ratio_metrics([1.0, 2.0], [0.0, 0.0], [1.0, 2.0], [1.0, 1.0])
    # Revenue exactly proportional to sessions: every unit has the same ratio,
    # so the delta-method variance is zero in both arms.
    with pytest.raises(InsufficientDataError, match="standard error"):
        compare_ratio_metrics([2.0, 4.0], [1.0, 2.0], [3.0, 6.0], [1.0, 2.0])
    unknown: Any = "up"
    with pytest.raises(InvalidInputError, match="alternative"):
        compare_ratio_metrics([1, 2], [1, 1], [1, 2], [1, 1], alternative=unknown)


def test_summary_uses_z() -> None:
    result = compare_ratio_metrics(
        [1.0, 2.0, 3.0], [1, 2, 2], [2.0, 2.5, 4.0], [1, 2, 2]
    )
    assert "Delta-method z-test (ratio metric)" in result.summary()
    assert "z = " in result.summary()

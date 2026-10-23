"""Tests for CUPED variance reduction (Deng, Xu, Kohavi & Walker 2013)."""

import numpy as np
import pytest
from scipy import stats

from powerlift import (
    InsufficientDataError,
    InvalidInputError,
    compare_means,
    compare_means_cuped,
)


def _experiment(
    rng: np.random.Generator, n: int, rho: float, effect: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pre-period covariate X and outcome Y with corr(X, Y) = rho in each arm."""
    noise_sd = np.sqrt(1 - rho**2)
    x_c, x_t = rng.normal(5, 1, n), rng.normal(5, 1, n)
    y_c = 10 + rho * (x_c - 5) + noise_sd * rng.normal(size=n)
    y_t = 10 + effect + rho * (x_t - 5) + noise_sd * rng.normal(size=n)
    return y_c, x_c, y_t, x_t


def test_theta_is_the_pooled_regression_slope() -> None:
    rng = np.random.default_rng(0)
    y_c, x_c, y_t, x_t = _experiment(rng, 300, 0.6, 0.2)
    result = compare_means_cuped(y_c, x_c, y_t, x_t)
    pooled = stats.linregress(np.concatenate([x_c, x_t]), np.concatenate([y_c, y_t]))
    assert result.theta == pytest.approx(pooled.slope)
    assert result.correlation == pytest.approx(pooled.rvalue)
    # The unadjusted comparison is the plain Welch test.
    assert result.unadjusted == compare_means(y_c, y_t)


def test_variance_reduction_matches_rho_squared_and_estimate_is_unbiased() -> None:
    rng = np.random.default_rng(2013)
    rho, effect, n, n_sims = 0.7, 0.1, 500, 2_000
    raw, adjusted, reported, hits = [], [], [], 0
    for _ in range(n_sims):
        result = compare_means_cuped(*_experiment(rng, n, rho, effect))
        raw.append(result.unadjusted.difference)
        adjusted.append(result.adjusted.difference)
        reported.append(result.variance_reduction)
        hits += result.adjusted.ci_difference.contains(effect)

    # Unbiased: the mean estimate is within 4 Monte Carlo standard errors.
    mc_se = np.std(adjusted, ddof=1) / np.sqrt(n_sims)
    assert abs(np.mean(adjusted) - effect) < 4 * mc_se
    # The sampling variance of the estimate shrinks by about rho^2 = 0.49 ...
    observed_reduction = 1 - np.var(adjusted, ddof=1) / np.var(raw, ddof=1)
    assert observed_reduction == pytest.approx(rho**2, abs=0.03)
    # ... which is also what each analysis reports on average,
    assert np.mean(reported) == pytest.approx(rho**2, abs=0.01)
    # and the adjusted intervals keep their nominal coverage.
    assert hits / n_sims == pytest.approx(0.95, abs=0.015)


def test_variance_reduction_in_one_large_experiment() -> None:
    rng = np.random.default_rng(1)
    result = compare_means_cuped(*_experiment(rng, 50_000, 0.5, 0.0))
    assert result.variance_reduction == pytest.approx(0.25, abs=0.01)
    assert result.adjusted.ci_difference.width < result.unadjusted.ci_difference.width


def test_uncorrelated_covariate_changes_little() -> None:
    rng = np.random.default_rng(2)
    y_c, y_t = rng.normal(size=2_000), rng.normal(size=2_000)
    x_c, x_t = rng.normal(size=2_000), rng.normal(size=2_000)
    result = compare_means_cuped(y_c, x_c, y_t, x_t)
    assert abs(result.variance_reduction) < 0.01
    assert result.adjusted.difference == pytest.approx(
        result.unadjusted.difference, abs=0.01
    )


def test_errors() -> None:
    with pytest.raises(InsufficientDataError, match="constant"):
        compare_means_cuped([1.0, 2.0], [3.0, 3.0], [1.5, 2.5], [3.0, 3.0])
    with pytest.raises(InvalidInputError, match="same length"):
        compare_means_cuped([1.0, 2.0], [3.0, 3.0, 1.0], [1.5, 2.5], [3.0, 4.0])


def test_summary_and_to_dict() -> None:
    rng = np.random.default_rng(3)
    result = compare_means_cuped(*_experiment(rng, 1_000, 0.8, 0.1))
    text = result.summary()
    assert text.startswith("CUPED (Deng et al. 2013)")
    assert "variance of the estimate reduced by" in text
    assert "unadjusted" in text
    data = result.to_dict()
    assert data["adjusted"]["method"] == "Welch's t-test on CUPED-adjusted metric"
    assert data["unadjusted"]["method"] == "Welch's t-test"

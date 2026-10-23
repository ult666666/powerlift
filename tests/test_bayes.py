"""Tests for the Beta-Binomial model.

P(X > Y) is checked against Evan Miller's closed form for integer parameters,
    P(X > Y) = sum_{i=0}^{a_x - 1} B(a_y + i, b_y + b_x)
               / ((b_x + i) B(1 + i, b_x) B(a_y, b_y)),
and everything else against Monte Carlo with a fixed seed.
"""

from typing import Any

import numpy as np
import pytest
from scipy import special, stats

from powerlift import (
    InvalidInputError,
    compare_proportions_bayes,
    prob_beta_greater,
)


def _closed_form(a_x: int, b_x: int, a_y: int, b_y: int) -> float:
    i = np.arange(a_x)
    log_terms = (
        special.betaln(a_y + i, b_y + b_x)
        - np.log(b_x + i)
        - special.betaln(1 + i, b_x)
        - special.betaln(a_y, b_y)
    )
    return float(np.exp(log_terms).sum())


@pytest.mark.parametrize(
    "parameters",
    [
        (2, 1, 1, 1),
        (5, 3, 4, 6),
        (10, 10, 10, 10),
        (561, 4441, 481, 4521),
        (481, 4521, 561, 4441),
        (5_101, 94_901, 5_001, 95_001),
        (1, 1_000, 30, 971),
        (10, 199_993, 3, 200_000),
    ],
)
def test_prob_greater_matches_closed_form(
    parameters: tuple[int, int, int, int],
) -> None:
    assert prob_beta_greater(*parameters) == pytest.approx(
        _closed_form(*parameters), rel=1e-8, abs=1e-12
    )


def test_prob_greater_matches_monte_carlo_for_non_integer_parameters() -> None:
    rng = np.random.default_rng(0)
    x = rng.beta(12.5, 300.5, 2_000_000)
    y = rng.beta(9.5, 310.5, 2_000_000)
    estimate = np.mean(x > y)
    standard_error = np.sqrt(estimate * (1 - estimate) / x.size)
    assert prob_beta_greater(12.5, 300.5, 9.5, 310.5) == pytest.approx(
        estimate, abs=4 * standard_error
    )


def test_prob_greater_symmetry() -> None:
    forward = prob_beta_greater(30.5, 70.5, 25.0, 80.0)
    backward = prob_beta_greater(25.0, 80.0, 30.5, 70.5)
    assert forward + backward == pytest.approx(1.0, abs=1e-10)
    assert prob_beta_greater(3.0, 4.0, 3.0, 4.0) == pytest.approx(0.5, abs=1e-12)


def test_posteriors_are_conjugate_updates() -> None:
    result = compare_proportions_bayes(
        480, 5000, 560, 5000, prior_alpha=0.5, prior_beta=0.5, credible_level=0.9
    )
    assert (result.control.alpha, result.control.beta) == (480.5, 4520.5)
    assert (result.treatment.alpha, result.treatment.beta) == (560.5, 4440.5)
    assert result.control.mean == pytest.approx(480.5 / 5001)
    low, high = stats.beta.ppf([0.05, 0.95], 560.5, 4440.5)
    assert result.treatment.credible_interval.lower == pytest.approx(low)
    assert result.treatment.credible_interval.upper == pytest.approx(high)
    assert result.prob_treatment_better == pytest.approx(
        prob_beta_greater(560.5, 4440.5, 480.5, 4520.5)
    )


@pytest.mark.parametrize("counts", [(480, 5000, 560, 5000), (48, 500, 50, 500)])
def test_expected_loss_and_intervals_match_monte_carlo(
    counts: tuple[int, int, int, int],
) -> None:
    result = compare_proportions_bayes(*counts)
    rng = np.random.default_rng(42)
    p_c = rng.beta(result.control.alpha, result.control.beta, 2_000_000)
    p_t = rng.beta(result.treatment.alpha, result.treatment.beta, 2_000_000)
    for exact, losses in (
        (result.expected_loss_treatment, np.maximum(p_c - p_t, 0)),
        (result.expected_loss_control, np.maximum(p_t - p_c, 0)),
    ):
        standard_error = losses.std() / np.sqrt(losses.size)
        assert exact == pytest.approx(losses.mean(), abs=4 * standard_error + 1e-12)
    # E[max(T - C, 0)] - E[max(C - T, 0)] = E[T - C] exactly.
    assert (
        result.expected_loss_control - result.expected_loss_treatment
        == pytest.approx(result.treatment.mean - result.control.mean, abs=1e-10)
    )
    quantiles = np.quantile(p_t - p_c, [0.025, 0.975])
    assert result.difference_interval.lower == pytest.approx(quantiles[0], abs=5e-4)
    assert result.difference_interval.upper == pytest.approx(quantiles[1], abs=5e-4)
    assert result.relative_lift_interval.contains(
        result.treatment.mean / result.control.mean - 1
    )


def test_results_are_reproducible_and_summarised() -> None:
    first = compare_proportions_bayes(10, 100, 20, 100, seed=7)
    assert first == compare_proportions_bayes(10, 100, 20, 100, seed=7)
    text = first.summary()
    assert "prior Beta(1, 1)" in text
    assert "P(treatment > control)" in text
    assert "seed 7" in text
    assert first.to_dict()["control"]["alpha"] == 11.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"prior_alpha": 0.0}, "prior_alpha"),
        ({"credible_level": 1.0}, "credible_level"),
        ({"n_draws": 10}, "n_draws"),
    ],
)
def test_invalid_inputs(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(InvalidInputError, match=message):
        compare_proportions_bayes(1, 10, 2, 10, **kwargs)
    with pytest.raises(InvalidInputError, match="beta_y"):
        prob_beta_greater(1.0, 1.0, 1.0, -1.0)

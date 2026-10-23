"""Tests for the SRM check and the A/A simulation utility.

The A/A tests double as the calibration check for every frequentist test in
the library: under the null hypothesis their p-values must be uniform and the
false-positive rate must equal alpha.
"""

from collections.abc import Callable, Sequence

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy import stats

from powerlift import (
    InvalidInputError,
    compare_means,
    compare_means_cuped,
    compare_proportions,
    compare_ratio_metrics,
    simulate_aa,
    srm_test,
)


@given(
    observed=st.lists(st.integers(0, 1_000_000), min_size=2, max_size=6).filter(
        lambda counts: sum(counts) > 0
    ),
    data=st.data(),
)
def test_srm_matches_scipy_chisquare(observed: list[int], data: st.DataObject) -> None:
    weights = data.draw(
        st.lists(st.floats(0.05, 10.0), min_size=len(observed), max_size=len(observed))
    )
    result = srm_test(observed, weights)
    expected = np.asarray(weights) / np.sum(weights) * sum(observed)
    reference = stats.chisquare(observed, expected)
    assert result.chi_square == pytest.approx(reference.statistic, rel=1e-9, abs=1e-9)
    assert result.p_value == pytest.approx(reference.pvalue, rel=1e-9, abs=1e-12)
    assert result.df == len(observed) - 1
    assert sum(result.allocation) == pytest.approx(1.0)
    assert result.mismatch == (result.p_value < 0.001)


def test_srm_examples() -> None:
    flagged = srm_test([50_000, 48_900])
    assert flagged.chi_square == pytest.approx(2 * 550**2 / 49_450)
    assert flagged.mismatch
    assert "SRM DETECTED" in flagged.summary()

    fine = srm_test([50_000, 49_800])
    assert not fine.mismatch
    assert "no SRM detected" in fine.summary()

    # A 25/25/50 design given as weights.
    three_arms = srm_test([2_500, 2_480, 5_020], [1, 1, 2])
    assert three_arms.allocation == pytest.approx((0.25, 0.25, 0.5))
    assert three_arms.expected == pytest.approx((2_500.0, 2_500.0, 5_000.0))
    assert not three_arms.mismatch
    assert three_arms.to_dict()["observed"] == [2_500, 2_480, 5_020]


def test_srm_false_alarm_rate_is_the_threshold() -> None:
    rng = np.random.default_rng(11)
    counts = rng.multinomial(100_000, [0.5, 0.5], size=4_000)
    alarms = [srm_test(c.tolist(), threshold=0.05).mismatch for c in counts]
    assert np.mean(alarms) == pytest.approx(0.05, abs=0.012)


@pytest.mark.parametrize(
    ("observed", "allocation", "message"),
    [
        ([10], None, "at least two"),
        ([0, 0], None, "at least one unit"),
        ([10, 10], [0.5, 0.3, 0.2], "entries"),
        ([10, -1], None, ">= 0"),
        ([10, 10], [0.5, 0.0], "allocation"),
    ],
)
def test_srm_invalid_inputs(
    observed: list[int], allocation: Sequence[float] | None, message: str
) -> None:
    with pytest.raises(InvalidInputError, match=message):
        srm_test(observed, allocation)
    with pytest.raises(InvalidInputError, match="threshold"):
        srm_test([10, 10], threshold=1.0)


def _proportions_aa(rng: np.random.Generator) -> float:
    conversions = rng.binomial(5_000, 0.10, size=2)
    return compare_proportions(
        int(conversions[0]), 5_000, int(conversions[1]), 5_000
    ).p_value


def _welch_aa(rng: np.random.Generator) -> float:
    # Skewed, revenue-like data.
    return compare_means(rng.lognormal(0, 1, 500), rng.lognormal(0, 1, 500)).p_value


def _ratio_aa(rng: np.random.Generator) -> float:
    sessions = rng.poisson(2.0, size=(2, 400)) + 1
    revenue = sessions * rng.gamma(1.0, 1.0, size=(2, 400))
    return compare_ratio_metrics(
        revenue[0], sessions[0], revenue[1], sessions[1]
    ).p_value


def _cuped_aa(rng: np.random.Generator) -> float:
    pre = rng.normal(0, 1, size=(2, 400))
    post = 0.8 * pre + rng.normal(0, 0.6, size=(2, 400))
    return compare_means_cuped(post[0], pre[0], post[1], pre[1]).adjusted.p_value


@pytest.mark.parametrize(
    "p_value_fn", [_proportions_aa, _welch_aa, _ratio_aa, _cuped_aa]
)
def test_implemented_tests_are_calibrated_under_aa(
    p_value_fn: Callable[[np.random.Generator], float],
) -> None:
    result = simulate_aa(p_value_fn, n_simulations=2_000, seed=2026)
    # Monte Carlo standard error of the false-positive rate is about 0.005.
    assert result.false_positive_rate == pytest.approx(0.05, abs=0.015)
    assert result.false_positive_ci.contains(0.05)
    assert result.ks_p_value > 0.001


def test_aa_simulation_detects_a_miscalibrated_test() -> None:
    def session_level_test(rng: np.random.Generator) -> float:
        # Treats each session as independent although sessions of the same user
        # share a spend factor: the classic mistake the delta method fixes.
        arms = []
        for _ in range(2):
            sessions = rng.poisson(3.0, 300) + 1
            spend = np.repeat(rng.gamma(0.5, 2.0, 300), sessions)
            arms.append(spend * rng.exponential(1.0, sessions.sum()))
        return compare_means(arms[0], arms[1]).p_value

    result = simulate_aa(session_level_test, n_simulations=1_000, seed=3)
    assert result.false_positive_rate > 0.15
    assert result.ks_p_value < 1e-6
    assert not result.false_positive_ci.contains(0.05)


def test_simulate_aa_is_reproducible_and_summarises() -> None:
    first = simulate_aa(_welch_aa, n_simulations=50, seed=5)
    second = simulate_aa(_welch_aa, n_simulations=50, seed=5)
    assert first.p_values == second.p_values
    assert len(first.p_values) == 50
    assert "A/A simulation: 50 experiments (seed 5)" in first.summary()
    assert "p_values" not in repr(first)


def test_simulate_aa_validates_inputs() -> None:
    with pytest.raises(InvalidInputError, match="between 0 and 1"):
        simulate_aa(lambda rng: 1.5, n_simulations=3)
    with pytest.raises(InvalidInputError, match="n_simulations"):
        simulate_aa(_welch_aa, n_simulations=0)

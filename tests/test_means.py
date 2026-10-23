"""Tests for Welch's t-test against scipy.stats."""

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy import stats

from powerlift import (
    InsufficientDataError,
    InvalidInputError,
    compare_means,
    compare_means_from_stats,
)
from powerlift._stats import Alternative

samples = arrays(
    np.float64,
    st.integers(2, 60),
    elements=st.floats(-1e3, 1e3, allow_nan=False, allow_subnormal=False),
)


# SciPy warns about precision loss when an arm is constant; the reference
# values are still exact in that case, so the warning is not an error here.
@pytest.mark.filterwarnings("ignore:Precision loss occurred:RuntimeWarning")
@given(
    control=samples,
    treatment=samples,
    alternative=st.sampled_from(["two-sided", "greater", "less"]),
)
def test_matches_scipy_welch_ttest(
    control: np.ndarray, treatment: np.ndarray, alternative: Alternative
) -> None:
    assume(np.ptp(control) > 1e-3 or np.ptp(treatment) > 1e-3)
    result = compare_means(control, treatment, alternative=alternative)
    reference = stats.ttest_ind(
        treatment, control, equal_var=False, alternative=alternative
    )
    assert result.statistic == pytest.approx(reference.statistic, rel=1e-9, abs=1e-9)
    assert result.p_value == pytest.approx(reference.pvalue, rel=1e-8, abs=1e-12)
    assert result.df == pytest.approx(reference.df, rel=1e-9)
    assert result.difference == pytest.approx(
        treatment.mean() - control.mean(), abs=1e-9
    )
    if alternative == "two-sided":
        low, high = reference.confidence_interval(confidence_level=0.95)
        assert result.ci_difference.lower == pytest.approx(low, rel=1e-9, abs=1e-9)
        assert result.ci_difference.upper == pytest.approx(high, rel=1e-9, abs=1e-9)


def test_from_stats_matches_scipy() -> None:
    result = compare_means_from_stats(
        mean_control=10.0,
        sd_control=4.0,
        n_control=250,
        mean_treatment=10.9,
        sd_treatment=6.0,
        n_treatment=180,
        alpha=0.01,
    )
    reference = stats.ttest_ind_from_stats(
        10.9, 6.0, 180, 10.0, 4.0, 250, equal_var=False
    )
    assert result.statistic == pytest.approx(reference.statistic, rel=1e-12)
    assert result.p_value == pytest.approx(reference.pvalue, rel=1e-12)
    # 99% interval with the Welch-Satterthwaite degrees of freedom.
    se = np.sqrt(16 / 250 + 36 / 180)
    df = se**4 / ((16 / 250) ** 2 / 249 + (36 / 180) ** 2 / 179)
    half_width = stats.t.isf(0.005, df) * se
    assert result.df == pytest.approx(df)
    assert result.ci_difference.lower == pytest.approx(0.9 - half_width)
    assert result.ci_difference.upper == pytest.approx(0.9 + half_width)
    assert result.ci_difference.level == pytest.approx(0.99)


def test_relative_lift_delta_method_formula() -> None:
    result = compare_means_from_stats(
        mean_control=50.0,
        sd_control=20.0,
        n_control=1000,
        mean_treatment=52.0,
        sd_treatment=21.0,
        n_treatment=1000,
    )
    var_c, var_t = 20.0**2 / 1000, 21.0**2 / 1000
    se = np.sqrt(var_t / 50.0**2 + 52.0**2 * var_c / 50.0**4)
    assert result.relative_lift == pytest.approx(0.04)
    assert result.ci_relative_lift is not None
    assert result.ci_relative_lift.lower == pytest.approx(0.04 - 1.959963984540054 * se)
    assert result.ci_relative_lift.upper == pytest.approx(0.04 + 1.959963984540054 * se)


def test_intervals_have_nominal_coverage_on_skewed_data() -> None:
    # Revenue-like lognormal data with a true 5% lift; 2,000 seeded experiments.
    rng = np.random.default_rng(42)
    mu, sigma, n = 2.0, 1.0, 2_000
    true_mean = np.exp(mu + sigma**2 / 2)
    true_difference, true_lift = 0.05 * true_mean, 0.05
    hits_difference = hits_lift = 0
    for _ in range(2_000):
        control = rng.lognormal(mu, sigma, n)
        treatment = 1.05 * rng.lognormal(mu, sigma, n)
        result = compare_means(control, treatment)
        assert result.ci_relative_lift is not None
        hits_difference += result.ci_difference.contains(true_difference)
        hits_lift += result.ci_relative_lift.contains(true_lift)
    # Monte Carlo standard error of a coverage estimate is about 0.005.
    assert hits_difference / 2_000 == pytest.approx(0.95, abs=0.015)
    assert hits_lift / 2_000 == pytest.approx(0.95, abs=0.015)


def test_zero_control_mean_has_no_relative_lift() -> None:
    result = compare_means([-1.0, 1.0, 0.0], [1.0, 2.0, 3.0])
    assert result.relative_lift is None
    assert result.ci_relative_lift is None
    assert "n/a" in result.summary()


def test_extreme_magnitudes_do_not_underflow() -> None:
    # Regression tests for cases found by Hypothesis: squaring tiny numbers
    # used to underflow to zero and divide by zero.
    tiny_means = compare_means([1.25e-153, 1.25e-153], [0.0, 1.0])
    assert np.isfinite(tiny_means.statistic)
    # Variances of order 1e-180 are representable but their squares are not;
    # the statistic and df are scale-invariant, so compare with rescaled data.
    tiny_variances = compare_means([1e-90, 2e-90], [0.0, 3e-90])
    rescaled = compare_means([1.0, 2.0], [0.0, 3.0])
    assert tiny_variances.df == pytest.approx(rescaled.df)
    assert tiny_variances.statistic == pytest.approx(rescaled.statistic)


def test_one_constant_arm_is_allowed() -> None:
    result = compare_means([5.0, 5.0, 5.0], [4.0, 6.0, 7.0])
    # Only the treatment variance contributes: df = n_t - 1.
    assert result.df == pytest.approx(2.0)


def test_errors() -> None:
    with pytest.raises(InsufficientDataError, match="zero variance"):
        compare_means([1.0, 1.0], [2.0, 2.0])
    with pytest.raises(InvalidInputError, match="NaN"):
        compare_means([1.0, np.nan], [1.0, 2.0])
    with pytest.raises(InvalidInputError, match="at least 2"):
        compare_means([1.0], [1.0, 2.0])
    with pytest.raises(InvalidInputError, match="non-negative"):
        compare_means_from_stats(
            mean_control=1.0,
            sd_control=-1.0,
            n_control=10,
            mean_treatment=1.0,
            sd_treatment=1.0,
            n_treatment=10,
        )
    with pytest.raises(InvalidInputError, match="n_treatment"):
        compare_means_from_stats(
            mean_control=1.0,
            sd_control=1.0,
            n_control=10,
            mean_treatment=1.0,
            sd_treatment=1.0,
            n_treatment=1,
        )


def test_summary() -> None:
    result = compare_means_from_stats(
        mean_control=50.0,
        sd_control=20.0,
        n_control=1000,
        mean_treatment=52.0,
        sd_treatment=21.0,
        n_treatment=1000,
    )
    text = result.summary()
    assert text.startswith("Welch's t-test, two-sided, alpha = 0.05")
    assert "t = 2.181 (df = 1993.3)" in text
    assert "+4.00%" in text
    assert result.to_dict()["df"] == pytest.approx(result.df)

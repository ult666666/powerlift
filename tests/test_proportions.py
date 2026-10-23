"""Tests for conversion-rate analysis against statsmodels.

statsmodels' sample 1 is our treatment arm, so differences and ratios are
``treatment - control`` and ``treatment / control`` in both libraries.
"""

import math
from typing import Any

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from scipy import stats
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportion_confint,
    proportions_ztest,
)

from powerlift import (
    InsufficientDataError,
    InvalidInputError,
    compare_proportions,
    wilson_interval,
)
from powerlift._stats import Alternative
from powerlift.analysis.proportions import DifferenceCI

SM_ALTERNATIVE = {"two-sided": "two-sided", "greater": "larger", "less": "smaller"}


@st.composite
def arm(draw: st.DrawFn, max_n: int = 20_000) -> tuple[int, int]:
    n = draw(st.integers(1, max_n))
    return draw(st.integers(0, n)), n


@given(
    control=arm(),
    treatment=arm(),
    alternative=st.sampled_from(["two-sided", "greater", "less"]),
)
def test_z_test_matches_statsmodels(
    control: tuple[int, int], treatment: tuple[int, int], alternative: Alternative
) -> None:
    (x_c, n_c), (x_t, n_t) = control, treatment
    pooled = (x_c + x_t) / (n_c + n_t)
    assume(0.0 < pooled < 1.0)
    result = compare_proportions(x_c, n_c, x_t, n_t, alternative=alternative)
    z, p = proportions_ztest(
        count=[x_t, x_c], nobs=[n_t, n_c], alternative=SM_ALTERNATIVE[alternative]
    )
    assert result.z_statistic == pytest.approx(z, rel=1e-9, abs=1e-12)
    assert result.p_value == pytest.approx(p, rel=1e-9, abs=1e-15)
    assert result.significant == (result.p_value <= 0.05)


@given(data=arm(), alpha=st.sampled_from([0.01, 0.05, 0.10, 0.20]))
def test_wilson_interval_matches_statsmodels(
    data: tuple[int, int], alpha: float
) -> None:
    x, n = data
    ours = wilson_interval(x, n, alpha=alpha)
    low, high = proportion_confint(x, n, alpha=alpha, method="wilson")
    assert ours.lower == pytest.approx(low, abs=1e-12)
    assert ours.upper == pytest.approx(high, abs=1e-12)
    assert ours.level == pytest.approx(1 - alpha)
    assert 0.0 <= ours.lower <= x / n <= ours.upper <= 1.0


@given(control=arm(), treatment=arm(), alpha=st.sampled_from([0.01, 0.05, 0.10]))
def test_difference_intervals_match_statsmodels(
    control: tuple[int, int], treatment: tuple[int, int], alpha: float
) -> None:
    (x_c, n_c), (x_t, n_t) = control, treatment
    assume(0 < x_c + x_t < n_c + n_t)
    methods: tuple[tuple[DifferenceCI, str], ...] = (
        ("newcombe", "newcomb"),
        ("wald", "wald"),
    )
    for ours_method, sm_method in methods:
        result = compare_proportions(
            x_c, n_c, x_t, n_t, alpha=alpha, ci_method=ours_method
        )
        low, high = confint_proportions_2indep(
            x_t, n_t, x_c, n_c, method=sm_method, compare="diff", alpha=alpha
        )
        assert result.ci_difference.lower == pytest.approx(low, abs=1e-12)
        assert result.ci_difference.upper == pytest.approx(high, abs=1e-12)
        assert result.ci_difference.contains(result.difference)


@given(control=arm(), treatment=arm(), alpha=st.sampled_from([0.01, 0.05, 0.10]))
def test_relative_lift_interval_matches_statsmodels_log_ratio(
    control: tuple[int, int], treatment: tuple[int, int], alpha: float
) -> None:
    (x_c, n_c), (x_t, n_t) = control, treatment
    assume(x_c > 0 and x_t > 0 and (x_c < n_c or x_t < n_t))
    result = compare_proportions(x_c, n_c, x_t, n_t, alpha=alpha)
    low, high = confint_proportions_2indep(
        x_t, n_t, x_c, n_c, method="log", compare="ratio", alpha=alpha
    )
    assert result.relative_lift == pytest.approx((x_t / n_t) / (x_c / n_c) - 1)
    assert result.ci_relative_lift is not None
    assert result.ci_relative_lift.lower == pytest.approx(low - 1, rel=1e-9, abs=1e-12)
    assert result.ci_relative_lift.upper == pytest.approx(high - 1, rel=1e-9, abs=1e-12)
    assert result.ci_relative_lift.lower > -1.0


def test_worked_example() -> None:
    result = compare_proportions(480, 5000, 560, 5000)
    assert result.rate_control == pytest.approx(0.096)
    assert result.rate_treatment == pytest.approx(0.112)
    assert result.difference == pytest.approx(0.016)
    # Pooled rate 0.104: z = 0.016 / sqrt(0.104 * 0.896 * 2 / 5000)
    assert result.z_statistic == pytest.approx(
        0.016 / math.sqrt(0.104 * 0.896 * 0.0004)
    )
    assert result.p_value == pytest.approx(2 * stats.norm.sf(result.z_statistic))
    assert result.p_value == pytest.approx(0.008775, abs=1e-6)
    assert result.significant
    assert result.relative_lift == pytest.approx(1 / 6)


def test_one_sided_p_values_split_the_two_sided_one() -> None:
    two_sided = compare_proportions(480, 5000, 560, 5000)
    greater = compare_proportions(480, 5000, 560, 5000, alternative="greater")
    less = compare_proportions(480, 5000, 560, 5000, alternative="less")
    assert greater.p_value == pytest.approx(two_sided.p_value / 2)
    assert less.p_value == pytest.approx(1 - two_sided.p_value / 2)


def test_zero_conversions_in_one_arm_drops_the_relative_lift() -> None:
    result = compare_proportions(0, 1000, 12, 1000)
    assert result.relative_lift is None
    assert result.ci_relative_lift is None
    assert result.ci_control.lower == 0.0
    assert "n/a" in result.summary()


def test_degenerate_data_raises() -> None:
    with pytest.raises(InsufficientDataError, match="0%"):
        compare_proportions(0, 100, 0, 100)
    with pytest.raises(InsufficientDataError):
        compare_proportions(100, 100, 50, 50)


@pytest.mark.parametrize(
    ("args", "kwargs", "message"),
    [
        ((11, 10, 1, 10), {}, "cannot exceed"),
        ((1, 0, 1, 10), {}, "n_control"),
        ((1.5, 10, 1, 10), {}, "integer"),
        ((1, 10, 1, 10), {"alpha": 1.5}, "alpha"),
        ((1, 10, 1, 10), {"alternative": "larger"}, "alternative"),
        ((1, 10, 1, 10), {"ci_method": "exact"}, "ci_method"),
    ],
)
def test_invalid_inputs(
    args: tuple[Any, ...], kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(InvalidInputError, match=message):
        compare_proportions(*args, **kwargs)


def test_wald_under_covers_where_newcombe_does_not() -> None:
    # Small samples and rates are where the Wald interval under-covers
    # (Newcombe 1998). Seeded simulation with 4,000 replications; the Monte
    # Carlo standard error of each coverage estimate is about 0.004.
    rng = np.random.default_rng(1998)
    p_c, p_t, n = 0.10, 0.20, 25
    x_c = rng.binomial(n, p_c, size=4_000)
    x_t = rng.binomial(n, p_t, size=4_000)
    covered: dict[DifferenceCI, int] = {"newcombe": 0, "wald": 0}
    analysed = 0
    for c, t in zip(x_c, x_t, strict=True):
        if c + t == 0:
            continue  # no conversions at all: the z-test is undefined
        analysed += 1
        for method in covered:
            result = compare_proportions(int(c), n, int(t), n, ci_method=method)
            covered[method] += result.ci_difference.contains(p_t - p_c)
    assert covered["wald"] / analysed < 0.94
    assert covered["newcombe"] / analysed >= 0.95


def test_summary_and_to_dict() -> None:
    result = compare_proportions(480, 5000, 560, 5000)
    text = result.summary()
    assert "480 / 5,000 =  9.60%" in text
    assert "+1.60 pp" in text
    assert "(Newcombe)" in text
    assert "significant at alpha = 0.05" in text
    data = result.to_dict()
    assert data["ci_difference"]["level"] == pytest.approx(0.95)
    assert data["ci_method"] == "newcombe"
    wald = compare_proportions(480, 5000, 560, 5000, ci_method="wald")
    assert "(Wald)" in wald.summary()

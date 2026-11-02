"""Many-to-one comparisons against scipy.stats.dunnett, closed forms and simulation."""

import json
import math
import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from powerlift import (
    InsufficientDataError,
    InvalidInputError,
    adjust_p_values,
    compare_means_to_control,
    compare_proportions_to_control,
    dunnett_critical_value,
)
from powerlift._dunnett import max_t_cdf, max_t_quantile
from powerlift._stats import Alternative
from powerlift.analysis.many_to_one import ADJUSTMENTS, Adjustment

ALTERNATIVES: tuple[Alternative, ...] = ("two-sided", "greater", "less")

# ------------------------------------------------- the max-|t| distribution


@pytest.mark.parametrize("df", [math.inf, 3.0, 25.0, 1e6])
def test_one_comparison_is_the_t_distribution(df: float) -> None:
    reference = stats.norm if math.isinf(df) else stats.t(df)
    for c in (0.3, 1.7, 3.2):
        two = max_t_cdf(c, np.array([0.6]), df, "two-sided")
        assert two == pytest.approx(1 - 2 * reference.sf(c), abs=1e-12)
        one = max_t_cdf(c, np.array([0.6]), df, "one-sided")
        assert one == pytest.approx(reference.cdf(c), abs=1e-12)


def test_uncorrelated_normal_statistics_multiply() -> None:
    for k in (2, 5):
        for c in (1.0, 2.5):
            assert max_t_cdf(c, np.zeros(k), math.inf, "two-sided") == pytest.approx(
                (2 * stats.norm.cdf(c) - 1) ** k, abs=1e-12
            )
            assert max_t_cdf(c, np.zeros(k), math.inf, "one-sided") == pytest.approx(
                stats.norm.cdf(c) ** k, abs=1e-12
            )


@pytest.mark.parametrize("seed", range(3))
def test_matches_scipy_general_multivariate_integrators(seed: int) -> None:
    """SciPy integrates any correlation matrix by randomized quasi-Monte Carlo."""
    rng = np.random.default_rng(seed)
    k = int(rng.integers(2, 6))
    sizes = rng.integers(20, 500, k + 1)
    lambdas = np.sqrt(sizes[1:] / (sizes[1:] + sizes[0]))
    corr = np.outer(lambdas, lambdas)
    np.fill_diagonal(corr, 1.0)
    c = float(rng.uniform(1.5, 3.0))
    normal = stats.multivariate_normal(np.zeros(k), corr, abseps=1e-9, releps=1e-9)
    expected = normal.cdf(np.full(k, c), lower_limit=np.full(k, -c), rng=0)
    assert max_t_cdf(c, lambdas, math.inf, "two-sided") == pytest.approx(
        expected, abs=1e-6
    )
    df = float(rng.integers(4, 60))
    expected_t = stats.multivariate_t(np.zeros(k), corr, df=df).cdf(
        np.full(k, c), maxpts=2_000_000, random_state=0
    )
    assert max_t_cdf(c, lambdas, df, "one-sided") == pytest.approx(expected_t, abs=1e-5)


def test_quantile_inverts_the_cdf_and_grows_with_k() -> None:
    previous = 0.0
    for k in (1, 2, 4, 8):
        lambdas = np.full(k, math.sqrt(0.5))
        c = max_t_quantile(0.95, lambdas, 40.0, "two-sided")
        assert max_t_cdf(c, lambdas, 40.0, "two-sided") == pytest.approx(
            0.95, abs=1e-10
        )
        assert c > previous  # more comparisons, larger cutoff
        previous = c
    # One comparison: the usual t quantile.
    assert dunnett_critical_value(30, [30], df=58) == pytest.approx(
        stats.t.isf(0.025, 58)
    )


@pytest.mark.parametrize("df", [math.inf, 597.0, 30.0])
def test_critical_values_stay_exact_down_to_the_smallest_supported_alpha(
    df: float,
) -> None:
    # One comparison has a closed form, so it shows how far the integration holds.
    reference = stats.norm if math.isinf(df) else stats.t(df)
    for alpha in (1e-8, 1e-10):
        c = dunnett_critical_value(100, [100], alpha=alpha, df=df)
        assert c == pytest.approx(reference.isf(alpha / 2), abs=1e-6)


@pytest.mark.parametrize("df", [math.inf, 597.0, 30.0])
def test_alpha_beyond_the_integration_accuracy_is_rejected(df: float) -> None:
    # Below 1e-10 the trapezoidal sums cannot resolve the tail: at 1e-16 the
    # search returned 16 or 32 instead of about 8.3 to 16.7, and smaller values
    # never returned with a finite df.
    with pytest.raises(InvalidInputError, match="alpha must be at least 1e-10"):
        dunnett_critical_value(100, [100], alpha=1e-16, df=df)


def test_dunnett_analysis_rejects_alpha_beyond_the_integration_accuracy() -> None:
    rng = np.random.default_rng(0)
    control, arms = rng.normal(size=200), [rng.normal(size=200), rng.normal(size=200)]
    with pytest.raises(InvalidInputError, match="alpha must be at least 1e-10"):
        compare_means_to_control(control, arms, alpha=1e-16)
    # The adjustments that do not integrate the max-|t| distribution still work.
    result = compare_means_to_control(
        control, arms, alpha=1e-16, adjustment="bonferroni"
    )
    assert result.critical_value == pytest.approx(stats.t.isf(1e-16 / 4, 597))


def test_the_bracket_search_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    # With a finite df the integration never reaches probability 1. Without the
    # alpha floor, the search must give up instead of doubling its bracket forever.
    monkeypatch.setattr("powerlift._dunnett.MIN_TAIL_PROBABILITY", 0.0)
    with pytest.raises(InvalidInputError, match="no critical value below"):
        max_t_quantile(1.0, np.array([0.5]), 597.0, "two-sided")


# ------------------------------------------------- against scipy.stats.dunnett

arm_means = st.floats(-1.0, 1.0)


@settings(max_examples=40)
@given(
    k=st.integers(1, 5),
    sizes=st.lists(st.integers(3, 60), min_size=6, max_size=6),
    means=st.lists(arm_means, min_size=6, max_size=6),
    alternative=st.sampled_from(ALTERNATIVES),
    seed=st.integers(0, 2**32 - 1),
)
def test_matches_scipy_dunnett(
    k: int, sizes: list[int], means: list[float], alternative: Alternative, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    control = rng.normal(means[0], 1.0, sizes[0])
    treatments = [rng.normal(means[i], 1.0, sizes[i]) for i in range(1, k + 1)]
    result = compare_means_to_control(control, treatments, alternative=alternative)
    reference = stats.dunnett(
        *treatments, control=control, alternative=alternative, random_state=0
    )
    arms = result.comparisons
    np.testing.assert_allclose(
        [a.statistic for a in arms], reference.statistic, rtol=1e-10
    )
    # SciPy's randomized integration moves its p-values by up to ~1e-4 between seeds.
    np.testing.assert_allclose(
        [a.adjusted_p_value for a in arms], reference.pvalue, atol=5e-4
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        interval = reference.confidence_interval(confidence_level=0.95)
    if caught:  # SciPy's own interval search did not converge; nothing to compare
        return
    for bound, theirs in (("lower", interval.low), ("upper", interval.high)):
        ours = np.array([getattr(a.ci_difference, bound) for a in arms])
        theirs = np.asarray(theirs, dtype=np.float64)
        np.testing.assert_array_equal(np.isinf(ours), np.isinf(theirs))
        finite = np.isfinite(ours)
        se = np.array([a.se_difference for a in arms])
        # Bounds differ only through the critical value: compare it in t units.
        np.testing.assert_allclose(
            ours[finite] / se[finite], theirs[finite] / se[finite], atol=5e-3
        )


def zero_one(conversions: int, n: int) -> np.ndarray:
    outcomes: np.ndarray = np.r_[np.ones(conversions), np.zeros(n - conversions)]
    return outcomes


def test_proportions_are_dunnett_on_zero_one_outcomes() -> None:
    counts = [(480, 5000), (520, 5000), (575, 5000), (505, 4800)]
    result = compare_proportions_to_control(counts[0], counts[1:])
    data = [zero_one(*arm) for arm in counts]
    same = compare_means_to_control(data[0], data[1:])
    for ours, theirs in zip(result.comparisons, same.comparisons, strict=True):
        assert ours.adjusted_p_value == pytest.approx(
            theirs.adjusted_p_value, abs=1e-12
        )
        assert ours.ci_difference.lower == pytest.approx(
            theirs.ci_difference.lower, abs=1e-12
        )
    reference = stats.dunnett(*data[1:], control=data[0], random_state=0)
    np.testing.assert_allclose(
        [a.adjusted_p_value for a in result.comparisons], reference.pvalue, atol=5e-4
    )
    assert result.metric == "proportion"
    assert "pp" in result.summary()


def test_one_treatment_is_students_t_test() -> None:
    rng = np.random.default_rng(4)
    control, treatment = rng.normal(0, 1, 40), rng.normal(0.5, 1, 35)
    reference = stats.ttest_ind(treatment, control, equal_var=True)
    for adjustment in ADJUSTMENTS:
        arm = compare_means_to_control(control, [treatment], adjustment=adjustment)
        assert arm.comparisons[0].p_value == pytest.approx(reference.pvalue, rel=1e-9)
        assert arm.comparisons[0].adjusted_p_value == pytest.approx(
            reference.pvalue, rel=1e-6
        )


# ------------------------------------------------- adjustments and intervals


@pytest.fixture(scope="module")
def four_arms() -> tuple[np.ndarray, list[np.ndarray]]:
    rng = np.random.default_rng(11)
    control = rng.normal(10.0, 2.0, 300)
    return control, [rng.normal(mu, 2.0, 300) for mu in (10.0, 10.3, 10.5)]


def test_holm_and_bonferroni_reuse_adjust_p_values(
    four_arms: tuple[np.ndarray, list[np.ndarray]],
) -> None:
    control, treatments = four_arms
    raw = [a.p_value for a in compare_means_to_control(control, treatments).comparisons]
    for method in ("holm", "bonferroni"):
        result = compare_means_to_control(control, treatments, adjustment=method)
        expected = adjust_p_values(raw, method=method).adjusted_p_values
        assert tuple(a.adjusted_p_value for a in result.comparisons) == expected
        # Holm has no simple simultaneous intervals, so it reports Bonferroni's.
        assert result.critical_value == pytest.approx(stats.t.isf(0.05 / 6, result.df))
    unadjusted = compare_means_to_control(control, treatments, adjustment="none")
    assert [a.adjusted_p_value for a in unadjusted.comparisons] == raw


@pytest.mark.parametrize("adjustment", ["dunnett", "bonferroni", "none"])
@pytest.mark.parametrize("alternative", ALTERNATIVES)
def test_decisions_match_intervals(
    four_arms: tuple[np.ndarray, list[np.ndarray]],
    adjustment: Adjustment,
    alternative: Alternative,
) -> None:
    control, treatments = four_arms
    result = compare_means_to_control(
        control, treatments, adjustment=adjustment, alternative=alternative
    )
    for arm in result.comparisons:
        excludes_zero = not arm.ci_difference.contains(0.0)
        assert arm.significant == excludes_zero
    if alternative == "greater":
        assert all(math.isinf(a.ci_difference.upper) for a in result.comparisons)
    if alternative == "less":
        assert all(math.isinf(a.ci_difference.lower) for a in result.comparisons)


def test_dunnett_is_sharper_than_bonferroni(
    four_arms: tuple[np.ndarray, list[np.ndarray]],
) -> None:
    control, treatments = four_arms
    dunnett = compare_means_to_control(control, treatments)
    bonferroni = compare_means_to_control(control, treatments, adjustment="bonferroni")
    assert dunnett.critical_value < bonferroni.critical_value
    for d, b in zip(dunnett.comparisons, bonferroni.comparisons, strict=True):
        assert d.p_value <= d.adjusted_p_value <= b.adjusted_p_value


def test_results_serialize_to_strict_json(
    four_arms: tuple[np.ndarray, list[np.ndarray]],
) -> None:
    control, treatments = four_arms
    result = compare_means_to_control(
        control, {"B": treatments[0], "C": treatments[1]}, alternative="greater"
    )
    payload = result.to_dict()
    assert payload["comparisons"][0]["ci_difference"]["upper"] is None  # +infinity
    assert json.loads(json.dumps(payload, allow_nan=False))["control"] == "control"
    assert result.comparison("C").name == "C"
    with pytest.raises(KeyError):
        result.comparison("Z")
    assert "Dunnett's many-to-one test, greater" in str(result)
    assert "+inf]" in str(result)
    assert result.comparisons[0].summary().startswith("B: difference")


# ------------------------------------------------- family-wise error rate


def test_family_wise_error_rate_under_the_global_null() -> None:
    """Simulate experiments where no arm differs; count those with any false win."""
    k, n, experiments = 4, 200, 1_000
    rng = np.random.default_rng(2026)
    any_win = dict.fromkeys(ADJUSTMENTS, 0)
    for _ in range(experiments):
        control = rng.normal(0.0, 1.0, n)
        treatments = [rng.normal(0.0, 1.0, n) for _ in range(k)]
        for adjustment in ADJUSTMENTS:
            result = compare_means_to_control(
                control, treatments, adjustment=adjustment
            )
            any_win[adjustment] += any(a.significant for a in result.comparisons)
    df = (k + 1) * n - (k + 1)
    lambdas = np.full(k, math.sqrt(0.5))
    bonferroni_cut = float(stats.t.isf(0.05 / (2 * k), df))
    exact = {
        "dunnett": 0.05,
        "bonferroni": 1 - max_t_cdf(bonferroni_cut, lambdas, df, "two-sided"),
        "holm": 1 - max_t_cdf(bonferroni_cut, lambdas, df, "two-sided"),  # same event
        "none": 1 - max_t_cdf(float(stats.t.isf(0.025, df)), lambdas, df, "two-sided"),
    }
    for adjustment, wins in any_win.items():
        rate, target = wins / experiments, exact[adjustment]
        assert abs(rate - target) < 4 * math.sqrt(target * (1 - target) / experiments)
    assert exact["none"] > 0.15  # unadjusted tests: about one experiment in six
    assert exact["bonferroni"] < 0.05


# ------------------------------------------------- validation


def test_rejects_invalid_inputs() -> None:
    data = [1.0, 2.0, 3.0]
    with pytest.raises(InvalidInputError, match="adjustment"):
        compare_means_to_control(data, [data], adjustment="sidak")  # type: ignore[arg-type]
    with pytest.raises(InvalidInputError, match="at least one treatment"):
        compare_means_to_control(data, [])
    with pytest.raises(InvalidInputError, match="unique"):
        compare_means_to_control(data, {"control": data})
    with pytest.raises(InvalidInputError, match="mapping"):
        compare_means_to_control(data, "abc")
    with pytest.raises(InvalidInputError, match="conversions, n"):
        compare_proportions_to_control((1, 10), [(1, 2, 3)])  # type: ignore[list-item]
    with pytest.raises(InvalidInputError, match="cannot exceed"):
        compare_proportions_to_control((1, 10), [(11, 10)])
    with pytest.raises(InsufficientDataError, match="zero"):
        compare_means_to_control([1.0, 1.0], [[2.0, 2.0]])
    with pytest.raises(InsufficientDataError, match="more units than arms"):
        compare_proportions_to_control((0, 1), [(1, 1)])
    with pytest.raises(InvalidInputError, match="alpha"):
        compare_means_to_control(data, [data], alpha=1.5)
    with pytest.raises(InvalidInputError, match="positive integer"):
        dunnett_critical_value(0, [10])
    with pytest.raises(InvalidInputError, match="at least one"):
        dunnett_critical_value(10, [])
    with pytest.raises(InvalidInputError, match="df"):
        dunnett_critical_value(10, [10], df=0)

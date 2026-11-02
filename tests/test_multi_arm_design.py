"""Multi-arm sample sizes: reductions to known cases and power checked by simulation."""

import json
import math

import numpy as np
import pytest
from scipy import optimize, stats

from powerlift import (
    InvalidInputError,
    compare_means_to_control,
    compare_proportions_to_control,
    dunnett_critical_value,
    sample_size_means,
    sample_size_multi_arm_means,
    sample_size_multi_arm_proportions,
)


def test_one_treatment_reduces_to_the_two_arm_formula() -> None:
    multi = sample_size_multi_arm_means(sd=2.0, mde=0.25, n_treatments=1)
    two_arm = sample_size_means(sd=2.0, mde=0.25)
    assert multi.critical_value == pytest.approx(stats.norm.isf(0.025), abs=1e-9)
    assert (multi.n_control, multi.n_per_treatment) == (
        two_arm.n_control,
        two_arm.n_treatment,
    )
    assert multi.achieved_power == pytest.approx(two_arm.achieved_power, abs=1e-12)


def test_means_formula_by_hand() -> None:
    # Three treatments, equal allocation: c = Dunnett's two-sided 5% value for
    # correlation 1/2, and n = ((c + z_0.8) * sd / mde)^2 * 2 per arm.
    plan = sample_size_multi_arm_means(sd=1.0, mde=0.1, n_treatments=3)
    c = dunnett_critical_value(10**9, [10**9] * 3)
    assert plan.critical_value == pytest.approx(c, abs=1e-6)
    expected = ((c + stats.norm.ppf(0.8)) / 0.1) ** 2 * 2
    assert plan.n_per_treatment == math.ceil(expected)
    assert plan.total == plan.n_control + 3 * plan.n_per_treatment
    assert plan.achieved_power >= 0.8


@pytest.mark.parametrize(("k", "ratio"), [(1, 1.0), (3, 1.0), (4, 2.0)])
def test_proportions_plan_solves_its_power_equation(k: int, ratio: float) -> None:
    """Re-derive the plan numerically from the power of the planned analysis."""
    p_c, p_t = 0.10, 0.11
    plan = sample_size_multi_arm_proportions(
        p_c, 0.01, n_treatments=k, control_ratio=ratio
    )
    c = plan.critical_value

    def power(n: float) -> float:
        # The analysis pools p (1 - p) over the control (r n units at p_c) and
        # k treatment arms (n units each at p_t); its difference has the unpooled SD.
        n_c = ratio * n
        pooled = (n_c * p_c * (1 - p_c) + k * n * p_t * (1 - p_t)) / (n_c + k * n)
        se_null = math.sqrt(pooled * (1 / n + 1 / n_c))
        se_alt = math.sqrt(p_t * (1 - p_t) / n + p_c * (1 - p_c) / n_c)
        return float(stats.norm.cdf((0.01 - c * se_null) / se_alt))

    root = optimize.brentq(lambda n: power(n) - 0.8, 10.0, 1e8, xtol=1e-9)
    assert plan.n_per_treatment == math.ceil(root)


def test_bonferroni_is_a_conservative_fallback() -> None:
    for k in (2, 4, 8):
        dunnett = sample_size_multi_arm_proportions(0.05, 0.005, n_treatments=k)
        bonferroni = sample_size_multi_arm_proportions(
            0.05, 0.005, n_treatments=k, adjustment="bonferroni"
        )
        assert bonferroni.critical_value > dunnett.critical_value
        assert bonferroni.n_per_treatment > dunnett.n_per_treatment


def test_square_root_allocation_needs_fewer_units() -> None:
    equal = sample_size_multi_arm_means(sd=1.0, mde=0.1, n_treatments=4)
    root = sample_size_multi_arm_means(
        sd=1.0, mde=0.1, n_treatments=4, control_ratio=2.0
    )
    assert root.total < equal.total
    assert root.n_control == pytest.approx(2 * root.n_per_treatment, abs=1)


def test_one_sided_plans_need_fewer_units() -> None:
    two = sample_size_multi_arm_means(sd=1.0, mde=0.1, n_treatments=3)
    one = sample_size_multi_arm_means(
        sd=1.0, mde=0.1, n_treatments=3, alternative="one-sided"
    )
    assert one.critical_value < two.critical_value
    assert one.n_per_treatment < two.n_per_treatment


def test_planned_power_matches_simulated_means_experiments() -> None:
    plan = sample_size_multi_arm_means(sd=1.0, mde=0.1, n_treatments=3)
    rng = np.random.default_rng(5)
    experiments, detected = 600, 0
    for _ in range(experiments):
        control = rng.normal(0.0, 1.0, plan.n_control)
        arms = [rng.normal(0.1, 1.0, plan.n_per_treatment) for _ in range(3)]
        detected += compare_means_to_control(control, arms).comparisons[0].significant
    error = math.sqrt(plan.achieved_power * (1 - plan.achieved_power) / experiments)
    assert abs(detected / experiments - plan.achieved_power) < 4 * error


@pytest.mark.parametrize("adjustment", ["dunnett", "bonferroni"])
def test_planned_power_matches_simulated_conversion_experiments(
    adjustment: str,
) -> None:
    plan = sample_size_multi_arm_proportions(
        0.10,
        0.10,
        relative=True,
        n_treatments=3,
        adjustment=adjustment,  # type: ignore[arg-type]
    )
    rng = np.random.default_rng(6)
    experiments, detected = 1_500, 0
    n, n0 = plan.n_per_treatment, plan.n_control
    for _ in range(experiments):
        control = (int(rng.binomial(n0, 0.10)), n0)
        arms = [(int(rng.binomial(n, 0.11)), n) for _ in range(3)]
        result = compare_proportions_to_control(control, arms, adjustment=adjustment)  # type: ignore[arg-type]
        detected += result.comparisons[0].significant
    error = math.sqrt(plan.achieved_power * (1 - plan.achieved_power) / experiments)
    assert abs(detected / experiments - plan.achieved_power) < 4 * error


def test_summary_duration_and_json() -> None:
    plan = sample_size_multi_arm_proportions(0.10, 0.10, relative=True, n_treatments=3)
    text = str(plan)
    assert "3 treatment arms vs one control" in text
    assert "each treatment 19,424" in text
    assert plan.duration_days(daily_traffic=10_000) == 8
    assert json.loads(json.dumps(plan.to_dict(), allow_nan=False))["n_treatments"] == 3
    means = sample_size_multi_arm_means(sd=2.0, mde=0.5, n_treatments=2)
    assert "minimum effect   +0.5 (sd = 2)" in str(means)


def test_rejects_invalid_plans() -> None:
    with pytest.raises(InvalidInputError, match="n_treatments"):
        sample_size_multi_arm_means(sd=1.0, mde=0.1, n_treatments=0)
    with pytest.raises(InvalidInputError, match="adjustment"):
        sample_size_multi_arm_means(sd=1.0, mde=0.1, n_treatments=2, adjustment="holm")  # type: ignore[arg-type]
    with pytest.raises(InvalidInputError, match="ratio"):
        sample_size_multi_arm_means(sd=1.0, mde=0.1, n_treatments=2, control_ratio=0.0)
    with pytest.raises(InvalidInputError, match="baseline"):
        sample_size_multi_arm_proportions(1.5, 0.1, n_treatments=2)

"""Tests for powerlift.design: sample size, power, MDE and duration.

References used here:
* statsmodels.stats.proportion.samplesize_proportions_2indep_onetail and
  power_proportions_2indep implement the same pooled-null / unpooled-
  alternative normal approximation (statsmodels' sample 1 is our treatment
  arm and its ``ratio`` is n_control / n_treatment).
* statsmodels.stats.power.NormalIndPower for two means with known SD.
"""

import math
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import (
    power_proportions_2indep,
    samplesize_proportions_2indep_onetail,
)

from powerlift import (
    InvalidInputError,
    experiment_duration,
    mde_means,
    mde_proportions,
    power_means,
    power_proportions,
    sample_size_means,
    sample_size_proportions,
)
from powerlift.design import Sides

Z_975 = 1.959963984540054
Z_80 = 0.8416212335729143


def _sm_alternative(sides: Sides, effect: float) -> str:
    if sides == "two-sided":
        return "two-sided"
    return "larger" if effect > 0 else "smaller"


PROPORTION_CASES = [
    # baseline, absolute effect, alpha, power, sides, ratio
    (0.10, 0.01, 0.05, 0.80, "two-sided", 1.0),
    (0.10, -0.02, 0.05, 0.90, "two-sided", 1.0),
    (0.03, 0.006, 0.01, 0.80, "two-sided", 1.0),
    (0.50, 0.05, 0.10, 0.95, "one-sided", 1.0),
    (0.20, -0.03, 0.05, 0.80, "one-sided", 1.0),
    (0.10, 0.02, 0.05, 0.80, "two-sided", 2.0),
    (0.25, 0.04, 0.05, 0.85, "one-sided", 0.5),
]


@pytest.mark.parametrize(
    ("baseline", "effect", "alpha", "power", "sides", "ratio"), PROPORTION_CASES
)
def test_proportion_sample_size_matches_statsmodels(
    baseline: float,
    effect: float,
    alpha: float,
    power: float,
    sides: Sides,
    ratio: float,
) -> None:
    result = sample_size_proportions(
        baseline, effect, alpha=alpha, power=power, alternative=sides, ratio=ratio
    )
    n_treatment_sm = samplesize_proportions_2indep_onetail(
        diff=effect,
        prop2=baseline,
        power=power,
        ratio=1.0 / ratio,
        alpha=alpha,
        alternative=_sm_alternative(sides, effect),
    )
    # statsmodels returns the unrounded treatment size; we round up.
    assert result.n_treatment == math.ceil(n_treatment_sm)
    assert result.n_control == math.ceil(n_treatment_sm / ratio - 1e-9)
    assert result.total == result.n_control + result.n_treatment
    assert result.achieved_power >= power


@pytest.mark.parametrize(
    ("baseline", "effect", "alpha", "power", "sides", "ratio"), PROPORTION_CASES
)
def test_proportion_power_matches_statsmodels(
    baseline: float,
    effect: float,
    alpha: float,
    power: float,
    sides: Sides,
    ratio: float,
) -> None:
    for n_control in (500, 2_000, 10_000):
        ours = power_proportions(
            baseline, effect, n_control, alpha=alpha, alternative=sides, ratio=ratio
        )
        reference = power_proportions_2indep(
            diff=effect,
            prop2=baseline,
            nobs1=ratio * n_control,
            ratio=1.0 / ratio,
            alpha=alpha,
            alternative=_sm_alternative(sides, effect),
        ).power
        assert ours == pytest.approx(reference, rel=1e-10, abs=1e-12)


def test_proportion_sample_size_closed_form_spot_check() -> None:
    # 10% -> 12%, two-sided alpha = 0.05, power = 0.8, equal allocation.
    p_bar = 0.11
    numerator = Z_975 * math.sqrt(2 * p_bar * (1 - p_bar)) + Z_80 * math.sqrt(
        0.10 * 0.90 + 0.12 * 0.88
    )
    expected = math.ceil((numerator / 0.02) ** 2)
    assert expected == 3841
    assert sample_size_proportions(0.10, 0.02).n_control == expected
    # The relative form gives the same answer.
    assert sample_size_proportions(0.10, 0.20, relative=True).n_control == expected


@pytest.mark.parametrize(
    ("effect_size", "alpha", "power", "sides", "ratio"),
    [
        (0.1, 0.05, 0.80, "two-sided", 1.0),
        (0.25, 0.01, 0.90, "two-sided", 1.0),
        (0.05, 0.05, 0.80, "one-sided", 1.0),
        (0.2, 0.05, 0.80, "two-sided", 3.0),
        (0.3, 0.10, 0.95, "one-sided", 0.5),
    ],
)
def test_mean_design_matches_statsmodels_normal_ind_power(
    effect_size: float, alpha: float, power: float, sides: Sides, ratio: float
) -> None:
    sd = 4.0
    effect = effect_size * sd
    sm_alternative = "two-sided" if sides == "two-sided" else "larger"
    reference_n = NormalIndPower().solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        ratio=ratio,
        alternative=sm_alternative,
    )
    result = sample_size_means(
        sd, effect, alpha=alpha, power=power, alternative=sides, ratio=ratio
    )
    # statsmodels solves the exact two-tailed power equation; the closed form
    # ignores the far tail, which changes n by far less than one unit.
    assert reference_n <= result.n_control < reference_n + 1.05

    for n_control in (50, 400, 3_000):
        reference_power = NormalIndPower().power(
            effect_size=effect_size,
            nobs1=n_control,
            alpha=alpha,
            ratio=ratio,
            alternative=sm_alternative,
        )
        ours = power_means(
            sd, effect, n_control, alpha=alpha, alternative=sides, ratio=ratio
        )
        assert ours == pytest.approx(reference_power, rel=1e-10)


def test_mean_rule_of_thumb_and_closed_form() -> None:
    # n = 2 (z_{0.975} + z_{0.8})^2 sd^2 / delta^2 ~= 15.7 sd^2 / delta^2
    expected = math.ceil(2 * (Z_975 + Z_80) ** 2 / 0.1**2)
    result = sample_size_means(1.0, 0.1)
    assert result.n_control == expected == 1570
    assert result.n_control == pytest.approx(16 / 0.1**2, rel=0.02)
    assert mde_means(1.0, 1570) == pytest.approx(0.1, rel=1e-4)


def test_mean_relative_mde_uses_baseline() -> None:
    absolute = sample_size_means(20.0, 2.0, baseline=50.0)
    relative = sample_size_means(20.0, 0.04, baseline=50.0, relative=True)
    assert absolute.n_control == relative.n_control
    assert relative.relative_mde == pytest.approx(0.04)
    assert sample_size_means(20.0, 2.0).relative_mde is None
    assert power_means(20.0, 0.04, 1570, baseline=50.0, relative=True) == pytest.approx(
        power_means(20.0, 2.0, 1570)
    )


def test_power_is_symmetric_for_two_sided_mean_tests() -> None:
    assert power_means(1.0, 0.1, 1000) == pytest.approx(power_means(1.0, -0.1, 1000))


@given(
    baseline=st.floats(0.01, 0.6),
    relative_mde=st.floats(0.05, 0.5),
    power=st.floats(0.5, 0.99),
    alpha=st.floats(0.001, 0.2),
    ratio=st.floats(0.25, 4.0),
    sides=st.sampled_from(["two-sided", "one-sided"]),
)
def test_proportion_round_trips(
    baseline: float,
    relative_mde: float,
    power: float,
    alpha: float,
    ratio: float,
    sides: Sides,
) -> None:
    result = sample_size_proportions(
        baseline,
        relative_mde,
        relative=True,
        alpha=alpha,
        power=power,
        alternative=sides,
        ratio=ratio,
    )
    # Rounding up can only add power ...
    assert result.achieved_power >= power - 1e-9
    achieved = power_proportions(
        baseline,
        relative_mde,
        result.n_control,
        relative=True,
        alpha=alpha,
        alternative=sides,
        ratio=result.n_treatment / result.n_control,
    )
    assert achieved == pytest.approx(result.achieved_power)
    # ... and the MDE at the rounded size is at most the requested one.
    mde = mde_proportions(
        baseline,
        result.n_control,
        alpha=alpha,
        power=power,
        alternative=sides,
        ratio=ratio,
    )
    assert mde <= result.absolute_mde * (1 + 1e-9)
    assert mde == pytest.approx(result.absolute_mde, rel=0.05)


@given(
    n_control=st.floats(200, 1e6),
    baseline=st.floats(0.01, 0.9),
    sides=st.sampled_from(["two-sided", "one-sided"]),
)
def test_mde_proportions_inverts_sample_size(
    n_control: float, baseline: float, sides: Sides
) -> None:
    mde = mde_proportions(baseline, n_control, alternative=sides)
    if baseline + mde >= 1.0:
        return  # an increase that large cannot be planned for
    result = sample_size_proportions(baseline, mde, alternative=sides)
    # Planning for exactly the MDE recovers n_control (rounded up).
    tolerance = 1e-6 * n_control
    assert n_control - tolerance <= result.n_control <= n_control + 1 + tolerance


@given(
    mde=st.floats(0.01, 1.0),
    factor=st.floats(1.1, 3.0),
    power=st.floats(0.5, 0.95),
)
def test_sample_size_monotonicity(mde: float, factor: float, power: float) -> None:
    smaller_effect = sample_size_means(1.0, mde / factor, power=power)
    larger_effect = sample_size_means(1.0, mde, power=power)
    assert smaller_effect.n_control >= larger_effect.n_control
    more_power = sample_size_means(1.0, mde, power=min(0.99, power * 1.04))
    assert more_power.n_control >= larger_effect.n_control
    stricter = sample_size_means(1.0, mde, power=power, alpha=0.01)
    assert stricter.n_control >= larger_effect.n_control


def test_experiment_duration() -> None:
    assert experiment_duration(29_502, 4_000) == 8
    assert experiment_duration(10_000, 1_000) == 10
    assert experiment_duration(10_000, 1_000, traffic_fraction=0.5) == 20
    plan = sample_size_proportions(0.10, 0.10, relative=True)
    assert plan.duration_days(4_000) == 8


def test_summary_and_to_dict() -> None:
    plan = sample_size_proportions(0.10, 0.10, relative=True)
    text = plan.summary()
    assert "control 14,751 | treatment 14,751 | total 29,502" in text
    assert "+1.00 pp (+10.00% relative) -> 11.00%" in text
    data = plan.to_dict()
    assert data["n_control"] == 14_751
    assert data["metric"] == "proportion"

    mean_plan = sample_size_means(20.0, 0.04, baseline=50.0, relative=True)
    assert "baseline mean    50" in mean_plan.summary()
    assert "+4.00% relative" in mean_plan.summary()
    assert "baseline" not in sample_size_means(20.0, 2.0).summary()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"baseline": 1.2, "mde": 0.01}, "baseline"),
        ({"baseline": 0.1, "mde": 0.0}, "non-zero"),
        ({"baseline": 0.1, "mde": 0.95}, "between 0 and 1"),
        ({"baseline": 0.1, "mde": 0.01, "power": 0.3}, "power"),
        ({"baseline": 0.1, "mde": 0.01, "alpha": 0.0}, "alpha"),
        ({"baseline": 0.1, "mde": 0.01, "ratio": -1.0}, "ratio"),
        ({"baseline": 0.1, "mde": 0.01, "alternative": "greater"}, "alternative"),
    ],
)
def test_invalid_proportion_inputs(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(InvalidInputError, match=message):
        sample_size_proportions(**kwargs)


def test_invalid_mean_and_duration_inputs() -> None:
    with pytest.raises(InvalidInputError, match="non-zero baseline"):
        sample_size_means(1.0, 0.1, relative=True)
    with pytest.raises(InvalidInputError, match="sd"):
        sample_size_means(0.0, 0.1)
    with pytest.raises(InvalidInputError, match="n_control"):
        power_means(1.0, 0.1, 0)
    with pytest.raises(InvalidInputError, match="traffic_fraction"):
        experiment_duration(100, 10, traffic_fraction=1.5)
    with pytest.raises(InvalidInputError, match="daily_traffic"):
        experiment_duration(100, 0)
    with pytest.raises(InvalidInputError, match="too small"):
        mde_proportions(0.5, 2)

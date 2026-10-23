"""Tests for multiple-comparison corrections against statsmodels."""

from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy import stats
from statsmodels.stats.multitest import multipletests

from powerlift import InvalidInputError, adjust_p_values
from powerlift.multiple_testing import CorrectionMethod

p_value_lists = st.lists(
    st.one_of(st.floats(0.0, 1.0), st.sampled_from([0.0, 0.01, 0.05, 0.5, 1.0])),
    min_size=1,
    max_size=40,
)
SM_METHOD = {"bonferroni": "bonferroni", "holm": "holm", "bh": "fdr_bh"}
METHODS: tuple[CorrectionMethod, ...] = ("bonferroni", "holm", "bh")


@given(p_values=p_value_lists, method=st.sampled_from(METHODS))
def test_matches_statsmodels_multipletests(
    p_values: list[float], method: CorrectionMethod
) -> None:
    result = adjust_p_values(p_values, method=method, alpha=0.05)
    reject, corrected, _, _ = multipletests(
        p_values, alpha=0.05, method=SM_METHOD[method]
    )
    np.testing.assert_allclose(
        result.adjusted_p_values, corrected, rtol=1e-12, atol=1e-15
    )
    # Decisions agree except possibly for values that tie with alpha up to rounding.
    away_from_alpha = np.abs(np.asarray(result.adjusted_p_values) - 0.05) > 1e-12
    np.testing.assert_array_equal(
        np.asarray(result.reject)[away_from_alpha], reject[away_from_alpha]
    )


@given(p_values=p_value_lists)
def test_ordering_properties(p_values: list[float]) -> None:
    raw = np.asarray(p_values)
    bonferroni = np.asarray(adjust_p_values(raw, method="bonferroni").adjusted_p_values)
    holm = np.asarray(adjust_p_values(raw, method="holm").adjusted_p_values)
    bh = np.asarray(adjust_p_values(raw, method="bh").adjusted_p_values)
    tolerance = 1e-12
    assert np.all(raw <= bh + tolerance)
    assert np.all(bh <= holm + tolerance)
    assert np.all(holm <= bonferroni + tolerance)
    assert np.all(bonferroni <= 1.0)
    # Adjusted p-values preserve the order of the raw ones.
    order = np.argsort(raw, kind="stable")
    for adjusted in (bonferroni, holm, bh):
        assert np.all(np.diff(adjusted[order]) >= -tolerance)


@given(p_values=p_value_lists, seed=st.integers(0, 2**32 - 1))
def test_permutation_equivariance(p_values: list[float], seed: int) -> None:
    permutation = np.random.default_rng(seed).permutation(len(p_values))
    shuffled = [p_values[i] for i in permutation]
    for method in METHODS:
        original = np.asarray(
            adjust_p_values(p_values, method=method).adjusted_p_values
        )
        permuted = adjust_p_values(shuffled, method=method).adjusted_p_values
        np.testing.assert_allclose(permuted, original[permutation])


def test_hand_computed_example() -> None:
    p_values = [0.01, 0.04, 0.03, 0.005]
    assert adjust_p_values(
        p_values, method="bonferroni"
    ).adjusted_p_values == pytest.approx((0.04, 0.16, 0.12, 0.02))
    # Holm: sorted 0.005*4, 0.01*3, 0.03*2, 0.04*1 -> running max.
    assert adjust_p_values(p_values, method="holm").adjusted_p_values == pytest.approx(
        (0.03, 0.06, 0.06, 0.02)
    )
    # BH: sorted 0.005*4/1, 0.01*4/2, 0.03*4/3, 0.04*4/4 -> running min from the top.
    bh = adjust_p_values(p_values, method="bh")
    assert bh.adjusted_p_values == pytest.approx((0.02, 0.04, 0.04, 0.02))
    assert bh.reject == (True, True, True, True)


def test_error_rates_by_simulation() -> None:
    # 10 metrics, 8 with no effect and 2 with a real effect, 4,000 experiments.
    rng = np.random.default_rng(1995)
    z = rng.normal(size=(4_000, 10))
    z[:, :2] += 3.0
    p = 2 * stats.norm.sf(np.abs(z))
    is_null = np.arange(10) >= 2
    any_false = {"none": 0, "holm": 0, "bh": 0}
    false_discovery_proportions = []
    for row in p:
        any_false["none"] += bool(np.any((row <= 0.05) & is_null))
        holm = np.asarray(adjust_p_values(row, method="holm").reject)
        any_false["holm"] += bool(np.any(holm & is_null))
        bh = np.asarray(adjust_p_values(row, method="bh").reject)
        any_false["bh"] += bool(np.any(bh & is_null))
        n_rejected = int(np.sum(bh))
        false_discoveries = int(np.sum(bh & is_null))
        false_discovery_proportions.append(false_discoveries / max(1, n_rejected))
    fwer = {name: count / 4_000 for name, count in any_false.items()}
    assert fwer["none"] == pytest.approx(1 - 0.95**8, abs=0.03)  # about 34%
    assert fwer["holm"] <= 0.05 + 0.01
    assert np.mean(false_discovery_proportions) <= 0.05 + 0.01


def test_errors_and_summary() -> None:
    with pytest.raises(InvalidInputError, match="between 0 and 1"):
        adjust_p_values([0.5, 1.2])
    unknown: Any = "sidak"
    with pytest.raises(InvalidInputError, match="method"):
        adjust_p_values([0.5], method=unknown)
    with pytest.raises(InvalidInputError, match="at least 1"):
        adjust_p_values([])
    text = adjust_p_values([0.01, 0.2], method="bh").summary()
    assert "Benjamini-Hochberg correction, FDR controlled at 0.05" in text
    assert "yes" in text
    assert "FWER" in adjust_p_values([0.01, 0.2]).summary()

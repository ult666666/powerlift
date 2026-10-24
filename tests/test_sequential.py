"""Tests for group sequential boundaries and mSPRT always-valid inference.

Reference values:
* Pocock constants C_P(K, 0.05) and O'Brien-Fleming constants C_B(K, 0.05)
  from Jennison & Turnbull (2000, ch. 2), given there to three decimals.
* Lan-DeMets O'Brien-Fleming-type spending, K = 5, two-sided 0.05:
  4.877, 3.357, 2.680, 2.290, 2.031 (as produced by gsDesign / ldbounds).
* Type-I error of repeatedly testing at |z| >= 1.96 with K equally spaced
  looks (Armitage, McPherson & Rowe 1969; Jennison & Turnbull 2000, ch. 1).
"""

import math
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy import stats

from powerlift import (
    InvalidInputError,
    always_valid_inference,
    crossing_probability,
    group_sequential_boundaries,
)
from powerlift.sequential import BoundaryMethod

METHODS: tuple[BoundaryMethod, ...] = (
    "pocock",
    "obrien-fleming",
    "lan-demets-obrien-fleming",
    "lan-demets-pocock",
)
JT_POCOCK = {2: 2.178, 3: 2.289, 4: 2.361, 5: 2.413, 10: 2.555}
JT_OBRIEN_FLEMING = {2: 1.977, 3: 2.004, 4: 2.024, 5: 2.040, 10: 2.087}
# Table values are rounded to 3 decimals; integration error is below 1e-4.
TABLE_TOLERANCE = 1e-3


@pytest.mark.parametrize(("looks", "constant"), JT_POCOCK.items())
def test_pocock_constants_match_jennison_turnbull(looks: int, constant: float) -> None:
    design = group_sequential_boundaries(looks, method="pocock")
    assert design.z_boundaries == pytest.approx([constant] * looks, abs=TABLE_TOLERANCE)


@pytest.mark.parametrize(("looks", "constant"), JT_OBRIEN_FLEMING.items())
def test_obrien_fleming_constants_match_jennison_turnbull(
    looks: int, constant: float
) -> None:
    design = group_sequential_boundaries(looks, method="obrien-fleming")
    expected = [constant * math.sqrt(looks / k) for k in range(1, looks + 1)]
    assert design.z_boundaries == pytest.approx(
        expected, abs=TABLE_TOLERANCE * math.sqrt(looks)
    )
    assert design.z_boundaries[-1] == pytest.approx(constant, abs=TABLE_TOLERANCE)


def test_obrien_fleming_five_looks_boundaries() -> None:
    design = group_sequential_boundaries(5, method="obrien-fleming")
    assert design.z_boundaries == pytest.approx(
        [4.562, 3.226, 2.634, 2.281, 2.040], abs=TABLE_TOLERANCE
    )


def test_lan_demets_obrien_fleming_matches_published_bounds() -> None:
    design = group_sequential_boundaries(5, method="lan-demets-obrien-fleming")
    assert design.z_boundaries == pytest.approx(
        [4.877, 3.357, 2.680, 2.290, 2.031], abs=TABLE_TOLERANCE
    )


@pytest.mark.parametrize("method", ["lan-demets-obrien-fleming", "lan-demets-pocock"])
def test_spending_designs_spend_exactly_the_spending_function(
    method: BoundaryMethod,
) -> None:
    fractions = [0.2, 0.45, 0.7, 0.85, 1.0]
    design = group_sequential_boundaries(fractions, method=method)
    if method == "lan-demets-pocock":
        spent = [0.05 * math.log(1 + (math.e - 1) * t) for t in fractions]
    else:
        z = stats.norm.isf(0.05 / 4)
        spent = [4 * stats.norm.sf(z / math.sqrt(t)) for t in fractions]
    assert design.cumulative_alpha == pytest.approx(spent, abs=1e-5)


@pytest.mark.parametrize(
    ("looks", "expected"),
    [(2, 0.083), (3, 0.107), (4, 0.126), (5, 0.142), (10, 0.193), (20, 0.248)],
)
def test_naive_peeking_error_rates_match_the_classic_values(
    looks: int, expected: float
) -> None:
    critical = stats.norm.isf(0.025)
    assert crossing_probability([critical] * looks) == pytest.approx(expected, abs=1e-3)


def _simulate_z_paths(
    rng: np.random.Generator, fractions: np.ndarray, n_paths: int
) -> np.ndarray:
    """Sequential z-statistics Z_k = W(t_k) / sqrt(t_k) for Brownian motion W."""
    increments = rng.normal(size=(n_paths, fractions.size)) * np.sqrt(
        np.diff(fractions, prepend=0.0)
    )
    # np.asarray pins the return type, which some NumPy stub versions infer as Any.
    return np.asarray(
        np.cumsum(increments, axis=1) / np.sqrt(fractions), dtype=np.float64
    )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "fractions", [[0.2, 0.4, 0.6, 0.8, 1.0], [0.1, 0.35, 0.5, 1.0]]
)
def test_type_one_error_by_simulation(
    method: BoundaryMethod, fractions: list[float]
) -> None:
    design = group_sequential_boundaries(fractions, method=method)
    rng = np.random.default_rng([METHODS.index(method), len(fractions)])
    n_paths = 400_000
    z = _simulate_z_paths(rng, np.asarray(fractions), n_paths)
    crossed = np.abs(z) >= np.asarray(design.z_boundaries)
    stopped = crossed.any(axis=1)
    # Monte Carlo standard error of the overall rate is about 0.00034.
    assert stopped.mean() == pytest.approx(0.05, abs=0.0015)
    first_look = np.argmax(crossed, axis=1)[stopped]
    per_look = np.bincount(first_look, minlength=len(fractions)) / n_paths
    expected = np.diff(design.cumulative_alpha, prepend=0.0)
    standard_errors = np.sqrt(expected * (1 - expected) / n_paths)
    assert np.all(np.abs(per_look - expected) <= 5 * standard_errors + 1e-5)


@pytest.mark.parametrize("method", METHODS)
def test_single_look_is_the_fixed_sample_test(method: BoundaryMethod) -> None:
    design = group_sequential_boundaries(1, method=method)
    assert design.z_boundaries == pytest.approx((stats.norm.isf(0.025),))
    assert design.cumulative_alpha == pytest.approx((0.05,))
    assert design.nominal_alpha == pytest.approx((0.05,))


def test_boundaries_are_deterministic() -> None:
    first = group_sequential_boundaries(6, method="lan-demets-pocock")
    second = group_sequential_boundaries(6, method="lan-demets-pocock")
    assert first == second


def test_first_crossing() -> None:
    design = group_sequential_boundaries(5, method="obrien-fleming")
    assert design.first_crossing([1.0, 2.0, 3.0]) == 3
    assert design.first_crossing([-4.7]) == 1
    assert design.first_crossing([1.0, 3.0, 2.5, 2.2, 2.0]) is None
    with pytest.raises(InvalidInputError, match="6 z-statistics"):
        design.first_crossing([0.0] * 6)


def test_summary_and_to_dict() -> None:
    design = group_sequential_boundaries(3, method="pocock", alpha=0.01)
    text = design.summary()
    assert text.startswith("Pocock boundaries: two-sided alpha = 0.01, 3 looks")
    assert len(text.splitlines()) == 5
    data = design.to_dict()
    assert data["method"] == "pocock"
    assert len(data["z_boundaries"]) == 3


@pytest.mark.parametrize(
    ("looks", "kwargs", "message"),
    [
        (0, {}, "looks"),
        ([0.5, 0.4, 1.0], {}, "strictly increasing"),
        ([0.2, 0.5], {}, "last information fraction"),
        ([0.0, 1.0], {}, r"\(0, 1\]"),
        (3, {"method": "haybittle"}, "method"),
        (3, {"alpha": 0.6}, "alpha"),
    ],
)
def test_invalid_designs(looks: Any, kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(InvalidInputError, match=message):
        group_sequential_boundaries(looks, **kwargs)


def test_crossing_probability_validation() -> None:
    with pytest.raises(InvalidInputError, match="positive"):
        crossing_probability([2.0, -1.0])
    with pytest.raises(InvalidInputError, match="information fractions"):
        crossing_probability([2.0, 2.0], [0.5, 0.75, 1.0])
    assert crossing_probability([2.0, 2.0], [0.5, 1.0]) == pytest.approx(
        crossing_probability([2.0, 2.0])
    )


# --------------------------------------------------------------------------
# mSPRT
# --------------------------------------------------------------------------


@given(
    estimate=st.floats(-5, 5),
    se=st.floats(0.01, 2.0),
    tau=st.floats(0.01, 3.0),
)
def test_single_look_p_value_is_the_inverse_likelihood_ratio(
    estimate: float, se: float, tau: float
) -> None:
    # Mixture marginal N(0, se^2 + tau^2) over the null density N(0, se^2),
    # on the log scale because the densities underflow for large |estimate|.
    log_likelihood_ratio = stats.norm.logpdf(
        estimate, scale=math.hypot(se, tau)
    ) - stats.norm.logpdf(estimate, scale=se)
    result = always_valid_inference([estimate], [se], tau=tau)
    expected = min(1.0, math.exp(-log_likelihood_ratio))
    assert result.p_values[0] == pytest.approx(expected, rel=1e-9, abs=1e-300)


@given(
    estimates=st.lists(st.floats(-3, 3), min_size=1, max_size=30),
    se_scale=st.floats(0.05, 1.0),
    tau=st.floats(0.05, 2.0),
    alpha=st.sampled_from([0.01, 0.05, 0.1]),
)
def test_p_values_and_confidence_sequence_are_dual(
    estimates: list[float], se_scale: float, tau: float, alpha: float
) -> None:
    # Standard errors shrink like 1/sqrt(n) as data accumulate.
    se = [se_scale / math.sqrt(n) for n in range(1, len(estimates) + 1)]
    result = always_valid_inference(estimates, se, tau=tau, alpha=alpha)
    p = np.asarray(result.p_values)
    lower, upper = np.asarray(result.ci_lower), np.asarray(result.ci_upper)
    assert np.all(np.diff(p) <= 0)  # p-values never increase
    assert np.all(np.diff(lower) >= 0)  # intervals are nested
    assert np.all(np.diff(upper) <= 0)
    excludes_zero = (lower > 0) | (upper < 0)
    # Away from the exact boundary, rejecting H0 <=> 0 is outside the interval.
    with np.errstate(divide="ignore"):  # p underflows to 0 for huge evidence
        clear = np.abs(np.log(p) - math.log(alpha)) > 1e-9
    np.testing.assert_array_equal(excludes_zero[clear], (p <= alpha)[clear])


def _stream_estimates(
    rng: np.random.Generator, effect: float, looks: int, batch: int
) -> tuple[np.ndarray, np.ndarray]:
    """Running difference in means (unit variance) after each batch per arm."""
    control = rng.normal(0.0, 1.0, size=(looks, batch)).sum(axis=1)
    treatment = rng.normal(effect, 1.0, size=(looks, batch)).sum(axis=1)
    n = batch * np.arange(1, looks + 1)
    estimates = (np.cumsum(treatment) - np.cumsum(control)) / n
    return estimates, np.sqrt(2.0 / n)


def test_msprt_controls_type_one_error_under_continuous_monitoring() -> None:
    rng = np.random.default_rng(2017)
    n_sims, looks = 2_000, 400
    msprt_rejections = naive_rejections = 0
    for _ in range(n_sims):
        estimates, se = _stream_estimates(rng, 0.0, looks, batch=5)
        result = always_valid_inference(estimates, se, tau=0.1)
        msprt_rejections += result.rejection_look is not None
        naive_rejections += bool(np.any(np.abs(estimates / se) >= 1.959964))
    # Ville's inequality bounds the mSPRT error by alpha; naive peeking at 400
    # looks is wrong far more often (the classic value is about 50%).
    assert msprt_rejections / n_sims <= 0.05
    assert naive_rejections / n_sims > 0.3


def test_msprt_detects_a_real_effect_and_covers_it() -> None:
    rng = np.random.default_rng(1)
    estimates, se = _stream_estimates(rng, 0.2, looks=200, batch=20)
    result = always_valid_inference(estimates, se, tau=0.2)
    assert result.rejection_look is not None
    assert result.ci_lower[-1] <= 0.2 <= result.ci_upper[-1]
    assert result.p_values[-1] <= 0.05
    assert f"rejected at look {result.rejection_look}" in result.summary()


def test_msprt_summary_validation_and_null_value() -> None:
    result = always_valid_inference([0.01, 0.02], [0.5, 0.4], tau=0.1)
    assert result.rejection_look is None
    assert "not rejected" in result.summary()
    shifted = always_valid_inference([1.01, 1.02], [0.5, 0.4], tau=0.1, null_value=1.0)
    assert shifted.p_values == pytest.approx(result.p_values)
    with pytest.raises(InvalidInputError, match="differ in length"):
        always_valid_inference([0.1, 0.2], [0.1], tau=0.1)
    with pytest.raises(InvalidInputError, match="positive"):
        always_valid_inference([0.1], [0.0], tau=0.1)
    with pytest.raises(InvalidInputError, match="tau"):
        always_valid_inference([0.1], [0.1], tau=0.0)

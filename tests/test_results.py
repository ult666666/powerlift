"""Tests for the shared result machinery and validation helpers."""

import json
from dataclasses import FrozenInstanceError, dataclass

import numpy as np
import pytest

import powerlift
from powerlift import ConfidenceInterval, InvalidInputError
from powerlift._results import Result, format_level, format_p_value, verdict
from powerlift._validation import (
    as_float_array,
    check_alternative,
    check_binomial,
    check_count,
    check_open_unit,
    check_positive,
    check_real,
)


@dataclass(frozen=True)
class _Nested(Result):
    interval: ConfidenceInterval
    values: tuple[float, ...]
    count: np.int64

    def summary(self) -> str:
        return "nested"


def test_version_is_exposed() -> None:
    assert powerlift.__version__ == "0.1.0"


def test_confidence_interval_helpers() -> None:
    ci = ConfidenceInterval(-0.5, 1.5, 0.95)
    assert ci.width == pytest.approx(2.0)
    assert ci.contains(0.0)
    assert ci.contains(-0.5)
    assert not ci.contains(2.0)
    assert ci.format(".1f") == "[-0.5, 1.5]"
    assert str(ci) == "95% interval [-0.5, 1.5]"


def test_results_are_frozen() -> None:
    ci = ConfidenceInterval(0.0, 1.0, 0.9)
    attribute = "lower"
    with pytest.raises(FrozenInstanceError):
        setattr(ci, attribute, 2.0)


def test_to_dict_is_json_serialisable_and_recursive() -> None:
    nested = _Nested(ConfidenceInterval(0.1, 0.2, 0.95), (1.0, 2.0), np.int64(3))
    data = nested.to_dict()
    assert data == {
        "interval": {"lower": 0.1, "upper": 0.2, "level": 0.95},
        "values": [1.0, 2.0],
        "count": 3,
    }
    assert type(data["count"]) is int
    assert json.loads(json.dumps(data)) == data
    assert str(nested) == "nested"


def test_formatting_helpers() -> None:
    assert format_p_value(0.04321) == "0.0432"
    assert format_p_value(3e-7) == "< 0.0001"
    assert format_level(0.95) == "95%"
    assert format_level(0.975) == "97.5%"
    assert verdict(0.01, 0.05) == "significant at alpha = 0.05"
    assert verdict(0.2, 0.05) == "not significant at alpha = 0.05"


@pytest.mark.parametrize("bad", [True, "1", None, float("nan"), float("inf")])
def test_check_real_rejects_non_finite_or_non_numeric(bad: object) -> None:
    with pytest.raises(InvalidInputError):
        check_real("x", bad)


def test_numeric_validators_accept_numpy_scalars() -> None:
    assert check_real("x", np.float32(1.5)) == 1.5
    assert check_count("n", np.int64(7)) == 7
    assert check_positive("x", 2) == 2.0
    assert check_open_unit("p", 0.25) == 0.25


@pytest.mark.parametrize(
    ("validator", "value"),
    [
        (check_positive, 0.0),
        (check_open_unit, 1.0),
        (check_open_unit, 0.0),
    ],
)
def test_range_validators(validator: object, value: float) -> None:
    assert callable(validator)
    with pytest.raises(InvalidInputError):
        validator("x", value)


@pytest.mark.parametrize("bad", [1.5, True, -1, "3"])
def test_check_count_rejects_bad_values(bad: object) -> None:
    with pytest.raises(InvalidInputError):
        check_count("n", bad)


def test_check_binomial() -> None:
    assert check_binomial("control", 3, 10) == (3, 10)
    with pytest.raises(InvalidInputError, match="cannot exceed"):
        check_binomial("control", 11, 10)
    with pytest.raises(InvalidInputError, match="n_control"):
        check_binomial("control", 0, 0)


def test_check_alternative() -> None:
    assert check_alternative("less") == "less"
    with pytest.raises(InvalidInputError, match="alternative"):
        check_alternative("bigger")


def test_as_float_array() -> None:
    np.testing.assert_array_equal(as_float_array("x", [1, 2, 3]), [1.0, 2.0, 3.0])
    with pytest.raises(InvalidInputError, match="one-dimensional"):
        as_float_array("x", [[1.0, 2.0]])
    with pytest.raises(InvalidInputError, match="at least 2"):
        as_float_array("x", [1.0])
    with pytest.raises(InvalidInputError, match="NaN"):
        as_float_array("x", [1.0, float("nan")])
    with pytest.raises(InvalidInputError, match="sequence of numbers"):
        as_float_array("x", ["a", "b"])


def test_p_value_helpers_match_scipy() -> None:
    from scipy import stats

    from powerlift._stats import normal_critical_value, normal_p_value, t_p_value

    assert normal_p_value(1.96, "two-sided") == pytest.approx(0.0499958, rel=1e-5)
    assert normal_p_value(1.5, "greater") == pytest.approx(stats.norm.sf(1.5))
    assert normal_p_value(1.5, "less") == pytest.approx(stats.norm.cdf(1.5))
    assert t_p_value(-2.0, 10.0, "two-sided") == pytest.approx(2 * stats.t.sf(2.0, 10))
    assert t_p_value(2.0, 10.0, "greater") == pytest.approx(stats.t.sf(2.0, 10))
    assert t_p_value(2.0, 10.0, "less") == pytest.approx(stats.t.cdf(2.0, 10))
    assert normal_critical_value(0.05) == pytest.approx(1.959964, rel=1e-6)

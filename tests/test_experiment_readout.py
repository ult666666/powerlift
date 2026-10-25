"""The example must gate inference before it computes an effect."""

import csv
import importlib.util
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "data" / "checkout.csv"
FINAL = date(2026, 9, 15)


def load_example() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "experiment_readout", ROOT / "examples" / "experiment_readout.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_seed_reproduces_the_committed_data(tmp_path: Path) -> None:
    example = load_example()
    output = tmp_path / "checkout.csv"
    example.generate_data(output)
    assert output.read_bytes() == DATA.read_bytes()
    rows = example.read_experiment(output)
    assert len(rows) == 400
    assert len({row.user_id for row in rows}) == 400


def test_complete_experiment_reports_uncertainty_and_variance_reduction() -> None:
    example = load_example()
    result = example.analyze(example.read_experiment(DATA), FINAL)
    assert result.srm.observed == (200, 200)
    assert not result.srm.mismatch
    assert result.blocked_reason is None
    assert result.comparison is not None
    assert result.comparison.variance_reduction > 0.5
    interval = result.comparison.adjusted.ci_difference
    assert interval.lower < 0 < interval.upper
    report = example.render_readout(result)
    assert "SYNTHETIC" in report
    assert "not a launch recommendation" in report


def test_srm_blocks_all_effect_inference() -> None:
    example = load_example()
    rows = example.read_experiment(DATA)
    # All control users but only 20 treatment users: a broken logging export.
    broken = [row for row in rows if row.group == "control"] + [
        row for row in rows if row.group == "treatment"
    ][:20]
    # CUPED would reject a constant covariate if called before the SRM gate.
    broken = [replace(row, pre_revenue=1.0) for row in broken]
    result = example.analyze(broken, FINAL)
    assert result.srm.mismatch
    assert result.comparison is None
    report = example.render_readout(result)
    assert "BLOCKED" in report
    assert "CUPED estimate" not in report


@pytest.mark.parametrize("as_of", [date(2026, 9, 1), date(2026, 9, 14)])
def test_unfinished_outcome_window_blocks_inference(as_of: date) -> None:
    example = load_example()
    result = example.analyze(example.read_experiment(DATA), as_of)
    assert not result.srm.mismatch
    assert result.comparison is None
    assert "outcome window" in result.blocked_reason


def test_balanced_but_insufficient_enrollment_blocks_inference() -> None:
    example = load_example()
    rows = example.read_experiment(DATA)
    short = [row for row in rows if row.group == "control"][:100] + [
        row for row in rows if row.group == "treatment"
    ][:100]
    result = example.analyze(short, FINAL)
    assert not result.srm.mismatch
    assert result.comparison is None
    assert "enrollment" in result.blocked_reason


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "", "oops"])
def test_invalid_revenue_is_rejected(tmp_path: Path, value: str) -> None:
    example = load_example()
    path = tmp_path / "bad.csv"
    write_rows(
        path,
        [
            {
                "user_id": "u1",
                "group": "control",
                "pre_revenue_usd": "10",
                "revenue_usd": value,
            }
        ],
    )
    with pytest.raises(ValueError, match="revenue_usd"):
        example.read_experiment(path)


def test_duplicate_users_across_arms_are_rejected(tmp_path: Path) -> None:
    example = load_example()
    path = tmp_path / "duplicate.csv"
    write_rows(
        path,
        [
            {
                "user_id": "same",
                "group": group,
                "pre_revenue_usd": "10",
                "revenue_usd": "12",
            }
            for group in ("control", "treatment")
        ],
    )
    with pytest.raises(ValueError, match="duplicate user_id"):
        example.read_experiment(path)


@pytest.mark.parametrize("group", ["variant-c", "", "Control"])
def test_unknown_groups_are_rejected(tmp_path: Path, group: str) -> None:
    example = load_example()
    path = tmp_path / "group.csv"
    write_rows(
        path,
        [
            {
                "user_id": "u1",
                "group": group,
                "pre_revenue_usd": "10",
                "revenue_usd": "12",
            }
        ],
    )
    with pytest.raises(ValueError, match="group"):
        example.read_experiment(path)


def test_missing_columns_are_named(tmp_path: Path) -> None:
    example = load_example()
    path = tmp_path / "missing.csv"
    write_rows(path, [{"user_id": "u1", "group": "control"}])
    with pytest.raises(ValueError, match="pre_revenue_usd"):
        example.read_experiment(path)


def test_duplicate_header_is_rejected_before_values_can_be_overwritten(
    tmp_path: Path,
) -> None:
    example = load_example()
    path = tmp_path / "duplicate-header.csv"
    path.write_text(
        "user_id,group,pre_revenue_usd,revenue_usd,revenue_usd\nu1,control,10,999,12\n"
    )
    with pytest.raises(ValueError, match="columns"):
        example.read_experiment(path)


def test_cli_writes_report_and_pending_run_exits_nonzero(tmp_path: Path) -> None:
    example = load_example()
    output = tmp_path / "readout.md"
    assert example.main(["--as-of", "2026-09-15", "--output", str(output)]) == 0
    assert "CUPED estimate" in output.read_text()
    assert example.main(["--as-of", "2026-09-14", "--output", str(output)]) == 1
    pending = output.read_text()
    assert "BLOCKED" in pending
    assert "CUPED estimate" not in pending


def test_cli_regenerates_data_and_reports_bad_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    example = load_example()
    output = tmp_path / "checkout.csv"
    assert example.main(["--generate-data", str(output)]) == 0
    assert output.read_bytes() == DATA.read_bytes()
    assert (
        example.main(["--csv", str(tmp_path / "missing.csv"), "--as-of", "2026-09-15"])
        == 2
    )
    assert "error:" in capsys.readouterr().err

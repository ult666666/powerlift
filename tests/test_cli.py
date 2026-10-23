"""Tests for the command-line interface."""

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from powerlift.cli import main


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    status = main(list(argv))
    captured = capsys.readouterr()
    return status, captured.out, captured.err


def test_sample_size_text_and_duration(capsys: pytest.CaptureFixture[str]) -> None:
    status, out, _ = run(
        capsys,
        "sample-size",
        "--baseline", "0.10", "--mde", "0.10", "--relative", "--daily-traffic", "4000",
    )  # fmt: skip
    assert status == 0
    assert "control 14,751 | treatment 14,751 | total 29,502" in out
    assert "duration         8 days at 4,000 eligible units/day" in out


def test_sample_size_json_and_means(capsys: pytest.CaptureFixture[str]) -> None:
    status, out, _ = run(
        capsys, "sample-size", "--metric", "mean", "--sd", "1", "--mde", "0.1", "--json"
    )
    assert status == 0
    assert json.loads(out)["n_control"] == 1570
    status, out, _ = run(
        capsys,
        "sample-size",
        "--baseline", "0.2", "--mde", "0.02", "--one-sided", "--daily-traffic", "1000",
        "--traffic-fraction", "0.5", "--json",
    )  # fmt: skip
    data = json.loads(out)
    assert data["plan"]["alternative"] == "one-sided"
    assert data["duration_days"] == -(-data["plan"]["total"] // 500)


def test_sample_size_errors(capsys: pytest.CaptureFixture[str]) -> None:
    status, _, err = run(capsys, "sample-size", "--mde", "0.01")
    assert status == 1
    assert "--baseline" in err
    status, _, err = run(capsys, "sample-size", "--metric", "mean", "--mde", "1")
    assert status == 1
    assert "--sd is required" in err


def test_analyze_counts_with_bayes(capsys: pytest.CaptureFixture[str]) -> None:
    status, out, _ = run(
        capsys,
        "analyze",
        "--control",
        "480/5000",
        "--treatment",
        "560/5,000",
        "--bayes",
    )
    assert status == 0
    assert "z = 2.621, p-value = 0.0088" in out
    assert "P(treatment > control)          0.9956" in out
    status, out, _ = run(
        capsys, "analyze", "--control", "480/5000", "--treatment", "560/5000", "--json"
    )
    assert json.loads(out)["p_value"] == pytest.approx(0.0087746, abs=1e-6)


def test_analyze_argument_errors(capsys: pytest.CaptureFixture[str]) -> None:
    status, _, err = run(capsys, "analyze", "--control", "480/5000")
    assert status == 1
    assert "--treatment" in err
    status, _, err = run(capsys, "analyze", "--control", "2/1", "--treatment", "1/1")
    assert status == 1
    assert "cannot exceed" in err
    with pytest.raises(SystemExit) as exit_info:
        main(["analyze", "--control", "lots", "--treatment", "1/2"])
    assert exit_info.value.code == 2
    assert "CONVERSIONS/N" in capsys.readouterr().err


@pytest.fixture
def experiment_csv(tmp_path: Path) -> Path:
    rng = np.random.default_rng(3)
    lines = ["group,revenue,converted,sessions,pre_revenue"]
    for group, lift in (("control", 0.0), ("treatment", 0.5), ("holdout", 0.0)):
        for _ in range(300):
            pre = rng.gamma(2.0, 5.0)
            sessions = int(rng.poisson(2) + 1)
            revenue = 0.8 * pre + rng.normal(lift, 3.0)
            lines.append(
                f"{group},{revenue:.4f},{int(rng.random() < 0.1)},{sessions},{pre:.4f}"
            )
    path = tmp_path / "experiment.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        (["--value-col", "revenue"], "Welch's t-test, two-sided"),
        (
            ["--value-col", "converted", "--metric", "proportion"],
            "Two-proportion z-test",
        ),
        (
            [
                "--value-col",
                "revenue",
                "--metric",
                "ratio",
                "--denominator-col",
                "sessions",
            ],
            "Delta-method z-test (ratio metric)",
        ),
        (["--value-col", "revenue", "--covariate-col", "pre_revenue"], "CUPED (Deng"),
    ],
)
def test_analyze_csv(
    capsys: pytest.CaptureFixture[str],
    experiment_csv: Path,
    extra: list[str],
    expected: str,
) -> None:
    status, out, err = run(capsys, "analyze", "--csv", str(experiment_csv), *extra)
    assert status == 0, err
    assert expected in out
    _, out, _ = run(capsys, "analyze", "--csv", str(experiment_csv), *extra, "--json")
    data = json.loads(out)
    # 300 units per arm: the third ("holdout") arm in the file is ignored.
    assert data.get("n_control", data.get("adjusted", {}).get("n_control")) == 300


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--value-col", "missing"], "no column(s) missing"),
        (["--value-col", "revenue", "--metric", "proportion"], "0/1 values"),
        (["--value-col", "revenue", "--metric", "ratio"], "--denominator-col"),
        (
            ["--value-col", "revenue", "--metric", "ratio",
             "--denominator-col", "sessions", "--covariate-col", "pre_revenue"],
            "only supported for --metric mean",
        ),
        (
            ["--value-col", "revenue", "--treatment-label", "variant"],
            "no rows with group",
        ),
        (["--value-col", "revenue", "--control", "1/2"], "not both"),
    ],
)  # fmt: skip
def test_analyze_csv_errors(
    capsys: pytest.CaptureFixture[str],
    experiment_csv: Path,
    extra: list[str],
    message: str,
) -> None:
    status, _, err = run(capsys, "analyze", "--csv", str(experiment_csv), *extra)
    assert status == 1
    assert message in err


def test_analyze_csv_bad_values(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("group,value\ncontrol,1.0\ntreatment,oops\n", encoding="utf-8")
    status, _, err = run(capsys, "analyze", "--csv", str(path))
    assert status == 1
    assert "line 3: value = 'oops' is not a number" in err
    status, _, err = run(capsys, "analyze", "--csv", str(tmp_path / "absent.csv"))
    assert status == 1
    assert "No such file" in err


def test_srm(capsys: pytest.CaptureFixture[str]) -> None:
    status, out, _ = run(capsys, "srm", "--observed", "50421", "49579")
    assert status == 0
    assert "no SRM detected" in out
    status, out, _ = run(
        capsys, "srm", "--observed", "5000", "5600", "--allocation", "1", "1", "--json"
    )
    assert json.loads(out)["mismatch"] is True


def test_boundaries(capsys: pytest.CaptureFixture[str]) -> None:
    status, out, _ = run(capsys, "boundaries", "--looks", "5")
    assert status == 0
    assert "O'Brien-Fleming boundaries" in out
    assert "4.5617" in out
    status, out, _ = run(
        capsys,
        "boundaries",
        "--information-fractions", "0.5", "1",
        "--method", "lan-demets-pocock", "--json",
    )  # fmt: skip
    assert json.loads(out)["information_fractions"] == [0.5, 1.0]
    status, _, err = run(capsys, "boundaries")
    assert status == 1
    assert "--looks" in err


def test_version_and_module_entry_point(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == "powerlift 0.1.0"
    monkeypatch.setattr(sys, "argv", ["powerlift", "srm", "--observed", "10", "10"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("powerlift", run_name="__main__")
    assert exit_info.value.code == 0
    assert "Sample ratio mismatch check" in capsys.readouterr().out

"""Smoke test for examples/multi_arm_simulation.py so the example cannot rot."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "multi_arm_simulation.py"


def _load_example() -> ModuleType:
    spec = importlib.util.spec_from_file_location("multi_arm_simulation", EXAMPLE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_example_prints_both_tables_and_writes_the_csv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    example = _load_example()
    path = tmp_path / "arms.csv"
    assert example.main(["--experiments", "10", "--csv", str(path)]) == 0
    out = capsys.readouterr().out
    assert "Family-wise error rate at alpha = 0.05" in out
    assert "Power to detect the first treatment arm" in out
    rows = path.read_text(encoding="utf-8").splitlines()
    assert rows[0] == "user_id,arm,revenue_usd"
    assert len(rows) == 1 + 4 * 2_000
    assert {row.split(",")[1] for row in rows[1:]} == {"A", "B", "C", "D"}
    assert example.exact_fwer(3, 500, "dunnett") == 0.05
    assert example.exact_fwer(3, 500, "none") > example.exact_fwer(3, 500, "holm")

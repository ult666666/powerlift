"""Smoke test for examples/peeking_simulation.py so the example cannot rot."""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "peeking_simulation.py"


def _load_example() -> ModuleType:
    spec = importlib.util.spec_from_file_location("peeking_simulation", EXAMPLE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_peeking_simulation_runs_and_plots(tmp_path: Path) -> None:
    example = _load_example()
    rng = np.random.default_rng(0)
    aa = example.simulate(rng, 0.10, 1_476, 200)
    ab = example.simulate(rng, 0.11, 1_476, 200)
    for rates in (aa, ab):
        assert set(rates) == {*example.RULES, "fixed-horizon"}
        for values in rates.values():
            assert values.shape == (example.LOOKS,)
            assert np.all(np.diff(values) >= 0)  # cumulative rejection rates
    # Naive peeking can only reject more often than the single final look.
    assert aa["naive"][-1] >= aa["fixed-horizon"][-1]
    output = tmp_path / "chart.png"
    example.plot(aa, ab, 1_476, 200, 0, output)
    assert output.stat().st_size > 10_000

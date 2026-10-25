# Experiment Readout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the complete fictional CSV → SRM gate → CUPED readout runnable in one command.

**Architecture:** Add an example that composes the existing statistical APIs. Keep all data and analysis assumptions local to the example; preserve the library and CLI API.

**Tech Stack:** Python 3.11+, stdlib CSV/dataclasses/argparse, existing NumPy/powerlift.

**Spec:** `docs/superpowers/specs/2026-10-25-experiment-readout-design.md`

## Global Constraints

- 400 fictional users, 200 per arm, seed 42; one row per randomized user.
- One primary metric: 14-day net revenue; August covariate predates September 1 randomization.
- Fixed outcome window ends 2026-09-15; block effect inference before it closes, below 200 per arm, or on SRM at threshold 0.001.
- No new dependencies, statistical primitives, launch recommendations or adoption claims.

## Review Focus

- Duplicate IDs across arms must fail rather than double-count users.
- Unknown groups, missing columns and nonfinite revenue must fail clearly.
- SRM must prevent the comparison object from being computed.
- Missing enrollment or unfinished outcomes must prevent inference.
- Seed regeneration and reading the committed CSV must give identical report output.

### Task 1: Runnable example, data and walkthrough

**Files:** Create `examples/experiment_readout.py`, `examples/data/checkout.csv`, `docs/EXPERIMENT_READOUT.md`, `tests/test_experiment_readout.py`; modify `README.md`.

**Interfaces:** `generate_data(path: Path) -> None`, `read_experiment(path: Path) -> list[Observation]`, `analyze(rows: list[Observation], as_of: date) -> Readout`, `render_readout(result: Readout) -> str`, `main(argv: Sequence[str] | None = None) -> int`. `Readout.comparison` is `CupedResult | None`; blocked reports contain no effect inference.

- [ ] Write tests exercising seed reproducibility, valid readout, SRM blocking, horizon/enrollment blocking, malformed/duplicate CSV, and CLI output/exit behavior.
- [ ] Run `uv run pytest tests/test_experiment_readout.py -q` and observe failure because the example is absent.
- [ ] Implement the typed example and generate the fictional CSV; run focused tests and resolve failures.
- [ ] Add the linked walkthrough and measured expected output, including covariate provenance and fixed-horizon assumptions.
- [ ] Run `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`, `uv run pytest --cov`, README doctests and the documented CLI commands. Expected: exit 0, no failures; SRM/pending cases exit 1 with inference suppressed.
- [ ] Commit the complete, verified change on `feat/experiment-readout`.

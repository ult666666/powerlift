"""A fictional, fixed-horizon CSV → SRM → CUPED experiment readout.

Run ``uv run python examples/experiment_readout.py --as-of 2026-09-15``.
The fixed example plan and covariate provenance are explained in
``docs/EXPERIMENT_READOUT.md``. This is a workflow example, not a general
experiment platform: changing the experiment requires changing its plan.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from powerlift import compare_means_cuped, srm_test
from powerlift.cuped import CupedResult
from powerlift.exceptions import PowerliftError
from powerlift.sanity import SRMResult

DATA = Path(__file__).resolve().parent / "data" / "checkout.csv"
FIELDS = ("user_id", "group", "pre_revenue_usd", "revenue_usd")
USERS_PER_ARM = 200
OUTCOME_WINDOW_END = date(2026, 9, 15)


@dataclass(frozen=True)
class Observation:
    """One randomized user's pre-period and outcome-window net revenue."""

    user_id: str
    group: str
    pre_revenue: float
    revenue: float


@dataclass(frozen=True)
class Readout:
    """Sanity check, then either a comparison or a reason inference is blocked."""

    as_of: date
    srm: SRMResult
    comparison: CupedResult | None
    blocked_reason: str | None


def generate_data(path: Path) -> None:
    """Reproduce the fictional seed-42 CSV; signed revenue can include refunds."""
    rng = np.random.default_rng(42)
    groups = np.array(["control"] * USERS_PER_ARM + ["treatment"] * USERS_PER_ARM)
    rng.shuffle(groups)
    pre = rng.gamma(2.0, 10.0, size=len(groups))
    revenue = 0.8 * pre + rng.normal(0.0, 8.0, size=len(groups))
    revenue += 0.5 * (groups == "treatment")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(FIELDS)
        for index, (group, before, after) in enumerate(
            zip(groups, pre, revenue, strict=True), start=1
        ):
            writer.writerow(
                [f"user-{index:04d}", group, f"{before:.2f}", f"{after:.2f}"]
            )


def read_experiment(path: Path) -> list[Observation]:
    """Read complete per-user rows, rejecting duplicates and malformed values."""
    rows: list[Observation] = []
    seen: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if (
            reader.fieldnames is None
            or len(reader.fieldnames) != len(FIELDS)
            or set(reader.fieldnames) != set(FIELDS)
        ):
            raise ValueError(f"CSV columns must be exactly {', '.join(FIELDS)}")
        for line, row in enumerate(reader, start=2):
            if None in row or any(row[field] is None for field in FIELDS):
                raise ValueError(f"line {line}: expected {len(FIELDS)} columns")
            user_id = row["user_id"].strip()
            if not user_id:
                raise ValueError(f"line {line}: user_id cannot be blank")
            if user_id in seen:
                raise ValueError(f"line {line}: duplicate user_id {user_id!r}")
            seen.add(user_id)
            group = row["group"].strip()
            if group not in ("control", "treatment"):
                raise ValueError(f"line {line}: group must be control or treatment")
            values = []
            for field in ("pre_revenue_usd", "revenue_usd"):
                try:
                    value = float(row[field])
                except ValueError as exc:
                    raise ValueError(
                        f"line {line}: {field} must be a finite number"
                    ) from exc
                if not math.isfinite(value):
                    raise ValueError(f"line {line}: {field} must be a finite number")
                values.append(value)
            rows.append(Observation(user_id, group, values[0], values[1]))
    if not rows:
        raise ValueError("CSV contains no randomized users")
    return rows


def analyze(rows: list[Observation], as_of: date) -> Readout:
    """Check the planned split and completed horizon before computing any effect."""
    control = [row for row in rows if row.group == "control"]
    treatment = [row for row in rows if row.group == "treatment"]
    sanity = srm_test([len(control), len(treatment)], allocation=[0.5, 0.5])
    reason = None
    if sanity.mismatch:
        reason = "sample ratio mismatch: investigate assignment, logging and filtering"
    elif as_of < OUTCOME_WINDOW_END:
        reason = f"outcome window is incomplete until {OUTCOME_WINDOW_END}"
    elif min(len(control), len(treatment)) < USERS_PER_ARM:
        reason = f"enrollment is below the planned {USERS_PER_ARM} users per arm"
    if reason is not None:
        return Readout(as_of, sanity, None, reason)
    comparison = compare_means_cuped(
        [row.revenue for row in control],
        [row.pre_revenue for row in control],
        [row.revenue for row in treatment],
        [row.pre_revenue for row in treatment],
    )
    return Readout(as_of, sanity, comparison, None)


def render_readout(result: Readout) -> str:
    """Render uncertainty and assumptions without making a launch recommendation."""
    lines = [
        "# Fictional checkout experiment — SYNTHETIC DATA",
        "",
        f"Analysis as of {result.as_of}; "
        f"fixed outcome window ends {OUTCOME_WINDOW_END}.",
        "Primary metric: 14-day net revenue per randomized user "
        "(USD, including refunds).",
        "Covariate: August net revenue, measured before September 1 randomization.",
        f"Example plan: 50/50 allocation, {USERS_PER_ARM} users per arm, "
        "one final analysis.",
        "",
        f"Observed users: control {result.srm.observed[0]}, "
        f"treatment {result.srm.observed[1]}.",
        f"SRM p-value: {result.srm.p_value:.4f} "
        f"(block below {result.srm.threshold:g}).",
        "",
    ]
    if result.comparison is None:
        lines.extend(
            [f"**BLOCKED:** {result.blocked_reason}.", "Effect inference suppressed."]
        )
    else:
        comparison = result.comparison
        lines.extend(
            [
                "| Analysis | Treatment - control (USD/user) | 95% CI | p-value |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for name, estimate in (
            ("Unadjusted context", comparison.unadjusted),
            ("CUPED estimate (primary)", comparison.adjusted),
        ):
            interval = estimate.ci_difference
            lines.append(
                f"| {name} | {estimate.difference:+.3f} | "
                f"[{interval.lower:+.3f}, {interval.upper:+.3f}] | "
                f"{estimate.p_value:.4f} |"
            )
        lines.extend(
            [
                "",
                f"Estimated variance reduction: {comparison.variance_reduction:.1%}.",
            ]
        )
    lines.extend(
        [
            "",
            "This is not a launch recommendation, and the small synthetic sample",
            "is not a power plan. docs/EXPERIMENT_READOUT.md lists the assumptions.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Write a readout: exit 0 ready, 1 blocked or 2 invalid; or generate data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DATA)
    parser.add_argument(
        "--as-of", type=date.fromisoformat, help="date outcomes are complete through"
    )
    parser.add_argument(
        "--output", type=Path, help="write Markdown here instead of stdout"
    )
    parser.add_argument(
        "--generate-data",
        type=Path,
        help="reproduce the seed-42 fictional CSV and exit",
    )
    args = parser.parse_args(argv)
    try:
        if args.generate_data is not None:
            generate_data(args.generate_data)
            print(f"wrote synthetic data to {args.generate_data}")
            return 0
        if args.as_of is None:
            parser.error("--as-of is required for analysis")
        result = analyze(read_experiment(args.csv), args.as_of)
        report = render_readout(result)
        if args.output is None:
            print(report, end="")
        else:
            args.output.write_text(report, encoding="utf-8")
        return 1 if result.blocked_reason else 0
    except (OSError, ValueError, PowerliftError) as exc:
        print(f"experiment-readout: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

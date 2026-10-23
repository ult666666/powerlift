"""Command-line interface: ``powerlift <command> [options]``.

Commands
--------
sample-size  Sample size (and optional duration) for a two-arm test.
analyze      Analyse results from counts or from a CSV file of per-unit data.
srm          Sample ratio mismatch check.
boundaries   Group sequential boundaries for interim analyses.

Every command prints a readable summary, or JSON with ``--json``. Invalid
input exits with status 1 and a message on stderr; usage errors exit with 2.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from powerlift import __version__
from powerlift._results import Result
from powerlift._validation import FloatArray
from powerlift.analysis import (
    compare_means,
    compare_proportions,
    compare_ratio_metrics,
)
from powerlift.bayes import compare_proportions_bayes
from powerlift.cuped import compare_means_cuped
from powerlift.design import (
    SampleSizeResult,
    Sides,
    sample_size_means,
    sample_size_proportions,
)
from powerlift.exceptions import InvalidInputError, PowerliftError
from powerlift.sanity import srm_test
from powerlift.sequential import group_sequential_boundaries

Handler = Callable[[argparse.Namespace], list[Result]]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return the process exit status."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler: Handler = args.handler
    try:
        results = handler(args)
    except (PowerliftError, OSError) as exc:
        print(f"powerlift {args.command}: error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        payload: Any = [r.to_dict() for r in results]
        print(json.dumps(payload[0] if len(payload) == 1 else payload, indent=2))
    else:
        print("\n\n".join(r.summary() for r in results))
    return 0


# --------------------------------------------------------------------------
# sample-size
# --------------------------------------------------------------------------


def _sample_size(args: argparse.Namespace) -> list[Result]:
    alternative: Sides = "one-sided" if args.one_sided else "two-sided"
    plan: SampleSizeResult
    if args.metric == "proportion":
        if args.baseline is None:
            raise InvalidInputError(
                "--baseline (the control conversion rate) is required"
            )
        plan = sample_size_proportions(
            args.baseline,
            args.mde,
            relative=args.relative,
            alpha=args.alpha,
            power=args.power,
            alternative=alternative,
            ratio=args.ratio,
        )
    else:
        if args.sd is None:
            raise InvalidInputError("--sd is required for --metric mean")
        plan = sample_size_means(
            args.sd,
            args.mde,
            baseline=args.baseline,
            relative=args.relative,
            alpha=args.alpha,
            power=args.power,
            alternative=alternative,
            ratio=args.ratio,
        )
    if args.daily_traffic is None:
        return [plan]
    days = plan.duration_days(
        args.daily_traffic, traffic_fraction=args.traffic_fraction
    )
    return [_PlanWithDuration(plan, days, args.daily_traffic, args.traffic_fraction)]


@dataclass(frozen=True)
class _PlanWithDuration(Result):
    """A sample-size plan together with the time needed to enrol it."""

    plan: SampleSizeResult
    duration_days: int
    daily_traffic: float
    traffic_fraction: float

    def summary(self) -> str:
        enrolled = (
            ""
            if self.traffic_fraction == 1.0
            else f", {self.traffic_fraction:.0%} enrolled"
        )
        return (
            f"{self.plan.summary()}\n  duration         {self.duration_days} days at "
            f"{self.daily_traffic:,.0f} eligible units/day{enrolled}"
        )


# --------------------------------------------------------------------------
# analyze
# --------------------------------------------------------------------------


def _analyze(args: argparse.Namespace) -> list[Result]:
    if args.csv is not None:
        if args.control is not None or args.treatment is not None:
            raise InvalidInputError(
                "use either --csv or --control/--treatment, not both"
            )
        return _analyze_csv(args)
    if args.control is None or args.treatment is None:
        raise InvalidInputError(
            "give --control and --treatment as CONVERSIONS/N, or --csv"
        )
    (x_c, n_c), (x_t, n_t) = args.control, args.treatment
    results: list[Result] = [
        compare_proportions(
            x_c, n_c, x_t, n_t, alpha=args.alpha, alternative=args.alternative
        )
    ]
    if args.bayes:
        results.append(compare_proportions_bayes(x_c, n_c, x_t, n_t))
    return results


def _analyze_csv(args: argparse.Namespace) -> list[Result]:
    columns = {"value": args.value_col}
    if args.metric == "ratio":
        if args.denominator_col is None:
            raise InvalidInputError("--metric ratio needs --denominator-col")
        columns["denominator"] = args.denominator_col
    if args.covariate_col is not None:
        if args.metric != "mean":
            raise InvalidInputError(
                "--covariate-col (CUPED) is only supported for --metric mean"
            )
        columns["covariate"] = args.covariate_col
    data = _read_arms(
        args.csv, args.group_col, (args.control_label, args.treatment_label), columns
    )
    control, treatment = data[args.control_label], data[args.treatment_label]
    common = {"alpha": args.alpha, "alternative": args.alternative}

    if args.metric == "proportion":
        counts = []
        for arm in (control, treatment):
            values = arm["value"]
            if not np.all((values == 0) | (values == 1)):
                raise InvalidInputError("--metric proportion needs 0/1 values")
            counts.append((int(values.sum()), values.size))
        (x_c, n_c), (x_t, n_t) = counts
        return [compare_proportions(x_c, n_c, x_t, n_t, **common)]
    if args.metric == "ratio":
        return [
            compare_ratio_metrics(
                control["value"],
                control["denominator"],
                treatment["value"],
                treatment["denominator"],
                **common,
            )
        ]
    if args.covariate_col is not None:
        return [
            compare_means_cuped(
                control["value"],
                control["covariate"],
                treatment["value"],
                treatment["covariate"],
                **common,
            )
        ]
    return [compare_means(control["value"], treatment["value"], **common)]


def _read_arms(
    path: Path, group_col: str, labels: tuple[str, str], columns: dict[str, str]
) -> dict[str, dict[str, FloatArray]]:
    """Read the requested numeric columns of a CSV file, split by arm label."""
    rows: dict[str, dict[str, list[float]]] = {
        label: {key: [] for key in columns} for label in labels
    }
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [
            c
            for c in (group_col, *columns.values())
            if c not in (reader.fieldnames or [])
        ]
        if missing:
            raise InvalidInputError(
                f"{path.name} has no column(s) {', '.join(missing)}"
            )
        for line, row in enumerate(reader, start=2):
            group = row[group_col]
            if group not in rows:
                continue  # other arms (e.g. a second treatment) are ignored
            for key, column in columns.items():
                try:
                    rows[group][key].append(float(row[column]))
                except ValueError as exc:
                    raise InvalidInputError(
                        f"{path.name}, line {line}: {column} = {row[column]!r} "
                        "is not a number"
                    ) from exc
    for label in labels:
        if not rows[label]["value"]:
            raise InvalidInputError(
                f"no rows with {group_col} == {label!r} in {path.name}"
            )
    return {
        label: {key: np.asarray(values) for key, values in arm.items()}
        for label, arm in rows.items()
    }


# --------------------------------------------------------------------------
# srm and boundaries
# --------------------------------------------------------------------------


def _srm(args: argparse.Namespace) -> list[Result]:
    return [srm_test(args.observed, args.allocation, threshold=args.threshold)]


def _boundaries(args: argparse.Namespace) -> list[Result]:
    looks = args.information_fractions if args.information_fractions else args.looks
    if looks is None:
        raise InvalidInputError("give --looks K or --information-fractions")
    return [group_sequential_boundaries(looks, method=args.method, alpha=args.alpha)]


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


def _counts(text: str) -> tuple[int, int]:
    """Parse ``CONVERSIONS/N`` such as ``480/5000``."""
    try:
        conversions, n = (
            int(part.replace(",", "").replace("_", "")) for part in text.split("/")
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected CONVERSIONS/N such as 480/5000, got {text!r}"
        ) from exc
    return conversions, n


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="powerlift",
        description="Statistics for A/B tests: power analysis, analysis, SRM checks "
        "and sequential boundaries.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--json", action="store_true", help="print JSON instead of text"
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    size = commands.add_parser(
        "sample-size",
        parents=[common],
        help="sample size and duration for a two-arm test",
    )
    size.add_argument("--metric", choices=["proportion", "mean"], default="proportion")
    size.add_argument("--baseline", type=float, help="control conversion rate or mean")
    size.add_argument(
        "--mde", type=float, required=True, help="minimum detectable effect"
    )
    size.add_argument(
        "--relative", action="store_true", help="MDE is relative to baseline"
    )
    size.add_argument("--sd", type=float, help="standard deviation (--metric mean)")
    size.add_argument("--alpha", type=float, default=0.05)
    size.add_argument("--power", type=float, default=0.80)
    size.add_argument("--one-sided", action="store_true", help="plan a one-sided test")
    size.add_argument(
        "--ratio", type=float, default=1.0, help="n_treatment / n_control"
    )
    size.add_argument("--daily-traffic", type=float, help="eligible units per day")
    size.add_argument("--traffic-fraction", type=float, default=1.0)
    size.set_defaults(handler=_sample_size)

    analyze = commands.add_parser(
        "analyze", parents=[common], help="analyse counts or a CSV of per-unit data"
    )
    analyze.add_argument("--control", type=_counts, metavar="X/N")
    analyze.add_argument("--treatment", type=_counts, metavar="X/N")
    analyze.add_argument(
        "--bayes", action="store_true", help="add a Beta-Binomial analysis"
    )
    analyze.add_argument("--csv", type=Path, help="CSV file with one row per unit")
    analyze.add_argument("--group-col", default="group")
    analyze.add_argument("--value-col", default="value")
    analyze.add_argument("--control-label", default="control")
    analyze.add_argument("--treatment-label", default="treatment")
    analyze.add_argument(
        "--metric", choices=["mean", "proportion", "ratio"], default="mean"
    )
    analyze.add_argument(
        "--denominator-col", help="denominator column (--metric ratio)"
    )
    analyze.add_argument("--covariate-col", help="pre-experiment covariate for CUPED")
    analyze.add_argument("--alpha", type=float, default=0.05)
    analyze.add_argument(
        "--alternative", choices=["two-sided", "greater", "less"], default="two-sided"
    )
    analyze.set_defaults(handler=_analyze)

    srm = commands.add_parser(
        "srm", parents=[common], help="sample ratio mismatch check"
    )
    srm.add_argument("--observed", type=int, nargs="+", required=True, metavar="N")
    srm.add_argument("--allocation", type=float, nargs="+", metavar="SHARE")
    srm.add_argument("--threshold", type=float, default=0.001)
    srm.set_defaults(handler=_srm)

    bounds = commands.add_parser(
        "boundaries", parents=[common], help="group sequential boundaries"
    )
    bounds.add_argument("--looks", type=int, help="number of equally spaced looks")
    bounds.add_argument(
        "--information-fractions",
        type=float,
        nargs="+",
        metavar="T",
        help="e.g. 0.25 0.5 1",
    )
    bounds.add_argument(
        "--method",
        choices=[
            "pocock",
            "obrien-fleming",
            "lan-demets-obrien-fleming",
            "lan-demets-pocock",
        ],
        default="obrien-fleming",
    )
    bounds.add_argument("--alpha", type=float, default=0.05)
    bounds.set_defaults(handler=_boundaries)
    return parser


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

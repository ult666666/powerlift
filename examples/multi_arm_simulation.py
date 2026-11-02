"""Family-wise error and power of multi-arm (A/B/n) tests, checked by simulation.

Run from the repository root:

    uv run python examples/multi_arm_simulation.py

All data are SYNTHETIC. The script runs the public analysis functions on many
simulated experiments and prints two tables:

1. Family-wise error rate under the global null (no arm differs from the
   control): the share of experiments with at least one false win, with no
   adjustment, Bonferroni, Holm and Dunnett. The exact rate of each rule comes
   from the distribution of the largest of the correlated t-statistics.
2. Power: the sample sizes from ``sample_size_multi_arm_means`` and
   ``sample_size_multi_arm_proportions``, and the share of simulated
   experiments at those sizes in which the planned test (Dunnett or
   Bonferroni) detects the first treatment arm.

``--csv PATH`` also writes a synthetic per-user CSV with four arms (A is the
control) for trying ``powerlift analyze --csv`` with several
``--treatment-label`` values.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy import stats

from powerlift import (
    compare_means_to_control,
    compare_proportions_to_control,
    sample_size_multi_arm_means,
    sample_size_multi_arm_proportions,
)
from powerlift._dunnett import max_t_cdf
from powerlift.analysis.many_to_one import Adjustment

ALPHA = 0.05
RULES: tuple[Adjustment, ...] = ("none", "bonferroni", "holm", "dunnett")


def exact_fwer(k: int, n: int, rule: Adjustment) -> float:
    """Return P(any rejection) under the global null, from the max-|t| distribution."""
    df = (k + 1) * n - (k + 1)
    lambdas = np.full(k, math.sqrt(0.5))  # equal arm sizes
    if rule == "dunnett":
        return ALPHA
    # Holm rejects something exactly when Bonferroni does: when min p <= alpha / k.
    tail = ALPHA / 2 if rule == "none" else ALPHA / (2 * k)
    return 1.0 - max_t_cdf(float(stats.t.isf(tail, df)), lambdas, df, "two-sided")


def fwer_rows(
    rng: np.random.Generator, experiments: int, n: int, ks: Sequence[int]
) -> list[str]:
    """Return one table row per rule: measured and exact error rates for each k.

    Every simulated experiment has k + 1 identical arms and is analysed with
    all four rules, so the rules are compared on the same data.
    """
    measured: dict[Adjustment, list[float]] = {rule: [] for rule in RULES}
    for k in ks:
        wins = dict.fromkeys(RULES, 0)
        for _ in range(experiments):
            control = rng.normal(0.0, 1.0, n)
            treatments = [rng.normal(0.0, 1.0, n) for _ in range(k)]
            for rule in RULES:
                result = compare_means_to_control(
                    control, treatments, adjustment=rule, alpha=ALPHA
                )
                wins[rule] += any(arm.significant for arm in result.comparisons)
        for rule in RULES:
            measured[rule].append(wins[rule] / experiments)
    return [
        f"  {rule:<12}"
        + "".join(
            f"{rate:>10.1%}{exact_fwer(k, n, rule):>8.1%}"
            for k, rate in zip(ks, measured[rule], strict=True)
        )
        for rule in RULES
    ]


def detected_means(
    rng: np.random.Generator, experiments: int, k: int
) -> tuple[str, float, float]:
    """Return the planned and simulated power of Dunnett's test for a mean metric."""
    plan = sample_size_multi_arm_means(sd=1.0, mde=0.1, n_treatments=k)
    hits = 0
    for _ in range(experiments):
        control = rng.normal(0.0, 1.0, plan.n_control)
        arms = [rng.normal(0.1, 1.0, plan.n_per_treatment) for _ in range(k)]
        hits += compare_means_to_control(control, arms).comparisons[0].significant
    label = f"mean +0.1 sd, {k} arms, Dunnett, n = {plan.n_per_treatment:,}/arm"
    return label, plan.achieved_power, hits / experiments


def detected_proportions(
    rng: np.random.Generator, experiments: int, k: int, adjustment: Adjustment
) -> tuple[str, float, float]:
    """Return the planned and simulated power for a conversion metric."""
    if adjustment not in ("dunnett", "bonferroni"):
        raise ValueError(f"no sample-size formula for {adjustment!r}")
    plan = sample_size_multi_arm_proportions(
        0.10, 0.10, relative=True, n_treatments=k, adjustment=adjustment
    )
    n, n0 = plan.n_per_treatment, plan.n_control
    hits = 0
    for _ in range(experiments):
        control = (int(rng.binomial(n0, 0.10)), n0)
        arms = [(int(rng.binomial(n, 0.11)), n) for _ in range(k)]
        result = compare_proportions_to_control(control, arms, adjustment=adjustment)
        hits += result.comparisons[0].significant
    label = f"10% -> 11%, {k} arms, {adjustment.title()}, n = {n:,}/arm"
    return label, plan.achieved_power, hits / experiments


def write_csv(path: Path, seed: int, n_per_arm: int = 2_000) -> None:
    """Synthetic revenue per user: arms B and C match A; D adds $1.00 on average."""
    rng = np.random.default_rng(seed)
    lifts = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 1.0}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["user_id", "arm", "revenue_usd"])
        user = 0
        for arm, lift in lifts.items():
            for value in rng.gamma(2.0, 5.0, n_per_arm) + lift:
                user += 1
                writer.writerow([f"user-{user:05d}", arm, f"{value:.2f}"])


def main(argv: Sequence[str] | None = None) -> int:
    """Print the error-rate and power tables; optionally write the demo CSV."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiments", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--csv", type=Path, help="also write a synthetic 4-arm CSV")
    args = parser.parse_args(argv)
    if args.csv is not None:  # its own generator, so the tables do not depend on it
        write_csv(args.csv, seed=args.seed + 1)
        print(f"wrote {args.csv}")
    rng = np.random.default_rng(args.seed)

    n, ks = 500, (3, 6)
    print(
        f"Family-wise error rate at alpha = {ALPHA:g}, global null, {n} units per arm, "
        f"{args.experiments:,} SYNTHETIC experiments per column"
    )
    print(f"  {'':<12}" + "".join(f"{f'{k} treatments':>18}" for k in ks))
    print(
        f"  {'adjustment':<12}" + "".join(f"{'measured':>10}{'exact':>8}" for _ in ks)
    )
    print("\n".join(fwer_rows(rng, args.experiments, n, ks)))

    print(
        f"\nPower to detect the first treatment arm, {args.experiments:,} experiments "
        "per row"
    )
    print(f"  {'plan':<50}{'planned':>9}{'simulated':>11}")
    for label, planned, simulated in (
        detected_means(rng, args.experiments, 3),
        detected_proportions(rng, args.experiments, 3, "dunnett"),
        detected_proportions(rng, args.experiments, 3, "bonferroni"),
    ):
        print(f"  {label:<50}{planned:>9.1%}{simulated:>11.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

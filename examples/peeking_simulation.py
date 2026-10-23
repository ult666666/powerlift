"""Peeking simulation: what repeated significance testing does to error rates.

Experiments are planned with powerlift for a 10% baseline conversion rate and a
+10% relative lift (80% power, alpha = 0.05), then analysed at 10 equally
spaced interim looks. Four stopping rules are compared:

* naive peeking: stop the first time the pooled z-test gives p < 0.05;
* Pocock group sequential boundaries;
* O'Brien-Fleming group sequential boundaries;
* the mSPRT always-valid p-value (Johari et al. 2017).

A/A experiments (no true effect) measure the false-positive rate; A/B
experiments whose true lift equals the planned one measure power.

Run from the repository root::

    uv run python examples/peeking_simulation.py

The script prints the results as a Markdown table and writes
``docs/images/peeking_simulation.png``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from powerlift import (
    always_valid_inference,
    compare_proportions,
    group_sequential_boundaries,
    sample_size_proportions,
)

BASELINE = 0.10
RELATIVE_LIFT = 0.10
LOOKS = 10
ALPHA = 0.05
Z_CRITICAL = 1.959963984540054  # two-sided 5% critical value of a single test
TAU = BASELINE * RELATIVE_LIFT  # mSPRT mixing sd: the planned absolute effect
DEFAULT_SIMULATIONS = 10_000
DEFAULT_SEED = 2026
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1] / "docs" / "images" / "peeking_simulation.png"
)

RULES = ("naive", "pocock", "obrien-fleming", "msprt")
LABELS = {
    "naive": "Naive peeking (p < 0.05 at any look)",
    "pocock": "Pocock boundaries",
    "obrien-fleming": "O'Brien-Fleming boundaries",
    "msprt": "mSPRT always-valid p-value",
}
# Categorical slots 1-4 of the reference palette, validated for colour-vision
# deficiency; identity is also carried by the legend and the Markdown table.
COLORS = {
    "naive": "#2a78d6",
    "pocock": "#eb6834",
    "obrien-fleming": "#1baf7a",
    "msprt": "#eda100",
}
SURFACE, INK, INK_SECONDARY, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

Rates = dict[str, npt.NDArray[np.float64]]


def simulate(
    rng: np.random.Generator, p_treatment: float, n_per_look: int, n_simulations: int
) -> Rates:
    """Share of experiments that have rejected H0 by each look, per stopping rule."""
    shape = (n_simulations, LOOKS)
    conversions_c = rng.binomial(n_per_look, BASELINE, size=shape).cumsum(axis=1)
    conversions_t = rng.binomial(n_per_look, p_treatment, size=shape).cumsum(axis=1)
    n = n_per_look * np.arange(1, LOOKS + 1)
    rate_c, rate_t = conversions_c / n, conversions_t / n

    # Pooled two-proportion z-statistic at every look (vectorised form of
    # powerlift.compare_proportions, checked against it below).
    pooled = (conversions_c + conversions_t) / (2 * n)
    z = (rate_t - rate_c) / np.sqrt(pooled * (1 - pooled) * 2 / n)
    for look in range(LOOKS):
        reference = compare_proportions(
            int(conversions_c[0, look]),
            int(n[look]),
            int(conversions_t[0, look]),
            int(n[look]),
        )
        assert np.isclose(reference.z_statistic, z[0, look])

    rejected: dict[str, npt.NDArray[np.bool_]] = {"naive": np.abs(z) >= Z_CRITICAL}
    for method in ("pocock", "obrien-fleming"):
        design = group_sequential_boundaries(LOOKS, method=method, alpha=ALPHA)
        rejected[method] = np.abs(z) >= np.asarray(design.z_boundaries)

    # The mSPRT uses the difference in rates and its unpooled standard error.
    difference = rate_t - rate_c
    se = np.sqrt(rate_c * (1 - rate_c) / n + rate_t * (1 - rate_t) / n)
    rejected["msprt"] = np.array(
        [
            np.asarray(always_valid_inference(d, s, tau=TAU, alpha=ALPHA).p_values)
            <= ALPHA
            for d, s in zip(difference, se, strict=True)
        ]
    )

    # "Rejected by look k" = rejected at any look up to k.
    rates = {
        rule: np.logical_or.accumulate(rejected[rule], axis=1).mean(axis=0)
        for rule in RULES
    }
    rates["fixed-horizon"] = np.full(LOOKS, np.mean(np.abs(z[:, -1]) >= Z_CRITICAL))
    return rates


def _style(ax: Axes, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold", pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelcolor=INK_SECONDARY, labelsize=9, length=0, pad=6)
    ax.set_xticks(range(1, LOOKS + 1))
    ax.set_xlim(0.6, LOOKS + 0.9)
    ax.set_xlabel(
        "Interim look (10 = planned sample size)", color=INK_SECONDARY, fontsize=9
    )
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))


def _line(ax: Axes, rule: str, values: npt.NDArray[np.float64]) -> None:
    ax.plot(
        np.arange(1, LOOKS + 1),
        values,
        color=COLORS[rule],
        linewidth=1.8,
        marker="o",
        markersize=5,
        markeredgecolor=SURFACE,
        markeredgewidth=1.2,
        solid_capstyle="round",
        label=LABELS[rule],
    )


def plot(
    aa: Rates, ab: Rates, n_per_look: int, n_simulations: int, seed: int, path: Path
) -> None:
    """Draw the two-panel chart: false-positive rate (A/A) and power (A/B)."""
    figure = Figure(figsize=(11, 4.8), dpi=150, facecolor=SURFACE, layout="constrained")
    ax_aa, ax_ab = figure.subplots(1, 2)

    _style(ax_aa, "No true effect: share of A/A tests declared significant")
    for rule in RULES:
        _line(ax_aa, rule, aa[rule])
    ax_aa.axhline(ALPHA, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax_aa.annotate(
        "target alpha = 5%",
        (LOOKS + 0.85, ALPHA),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=INK_SECONDARY,
    )
    ax_aa.annotate(
        f"{aa['naive'][-1]:.1%}",
        (LOOKS, aa["naive"][-1]),
        xytext=(7, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color=INK,
    )
    ax_aa.set_ylim(0, 0.25)

    _style(ax_ab, "True +10% lift: share of A/B tests declared significant")
    # Naive peeking is not a valid test, so it gets no power curve.
    for rule in RULES[1:]:
        _line(ax_ab, rule, ab[rule])
    # The fixed-horizon test decides once, at look 10; its marker is nudged
    # right so that it does not hide the O'Brien-Fleming end point.
    fixed = ab["fixed-horizon"][-1]
    ax_ab.plot(
        [LOOKS + 0.35],
        [fixed],
        marker="D",
        markersize=6,
        color=INK_SECONDARY,
        markeredgecolor=SURFACE,
        markeredgewidth=1.2,
        linestyle="none",
    )
    ax_ab.annotate(
        f"fixed-horizon test\n(single look): {fixed:.1%}",
        (LOOKS + 0.35, fixed),
        xytext=(-10, 10),
        textcoords="offset points",
        ha="right",
        fontsize=8.5,
        color=INK_SECONDARY,
    )
    ax_ab.set_ylim(0, 1)

    handles, labels = ax_aa.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside upper center",
        ncols=4,
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
    )
    figure.supxlabel(
        f"{n_simulations:,} simulated experiments per panel (seed {seed}); baseline "
        f"conversion 10%, {LOOKS} looks of {n_per_look:,} users per arm.",
        x=0.01,
        ha="left",
        fontsize=8,
        color=MUTED,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor=SURFACE)


def main() -> None:
    """Run the simulation, print a Markdown table and save the chart."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    plan = sample_size_proportions(BASELINE, RELATIVE_LIFT, relative=True, alpha=ALPHA)
    n_per_look = -(-plan.n_control // LOOKS)  # ceiling division
    rng = np.random.default_rng(args.seed)
    aa = simulate(rng, BASELINE, n_per_look, args.simulations)
    ab = simulate(rng, BASELINE * (1 + RELATIVE_LIFT), n_per_look, args.simulations)

    print(
        f"{args.simulations:,} A/A and {args.simulations:,} A/B experiments (seed "
        f"{args.seed}); {LOOKS} looks of {n_per_look:,} users per arm "
        f"(plan: {plan.n_control:,} per arm)\n"
    )
    print("| Stopping rule | False-positive rate (A/A) | Power (true +10% lift) |")
    print("|---|---:|---:|")
    names = {"fixed-horizon": "Fixed horizon (single look at the end)", **LABELS}
    for rule in ("fixed-horizon", *RULES):
        print(f"| {names[rule]} | {aa[rule][-1]:.1%} | {ab[rule][-1]:.1%} |")

    plot(aa, ab, n_per_look, args.simulations, args.seed, args.output)
    print(f"\nChart written to {args.output.name}")


if __name__ == "__main__":
    main()

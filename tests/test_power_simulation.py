"""End-to-end check: the planned sample size delivers the planned power.

Experiments are simulated at the size returned by powerlift.design and
analysed with powerlift.analysis; the share of significant results must match
the target power up to Monte Carlo error.
"""

import numpy as np
import pytest

from powerlift import (
    compare_means,
    compare_proportions,
    sample_size_means,
    sample_size_proportions,
)


def test_simulated_power_of_the_z_test_matches_the_plan() -> None:
    plan = sample_size_proportions(0.20, 0.20, relative=True, power=0.80)
    rng = np.random.default_rng(20261023)
    n_sims = 2_000
    conversions_c = rng.binomial(plan.n_control, 0.20, size=n_sims)
    conversions_t = rng.binomial(plan.n_treatment, 0.24, size=n_sims)
    rejections = [
        compare_proportions(
            int(c), plan.n_control, int(t), plan.n_treatment
        ).significant
        for c, t in zip(conversions_c, conversions_t, strict=True)
    ]
    # Binomial standard error of the estimate is about 0.009.
    assert np.mean(rejections) == pytest.approx(0.80, abs=0.03)


def test_simulated_power_of_welch_matches_the_mean_plan() -> None:
    plan = sample_size_means(10.0, 2.0, power=0.90)
    rng = np.random.default_rng(7)
    rejections = [
        compare_means(
            rng.normal(50.0, 10.0, plan.n_control),
            rng.normal(52.0, 10.0, plan.n_treatment),
        ).significant
        for _ in range(1_000)
    ]
    assert np.mean(rejections) == pytest.approx(0.90, abs=0.03)

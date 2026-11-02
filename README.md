# powerlift

A Python library for planning and reading out A/B tests: sample sizes and power,
frequentist and Bayesian analysis, Dunnett's test for several variants against one
control, CUPED variance reduction, sequential monitoring and sample-ratio checks.
Results are checked against SciPy, statsmodels and published tables.

[![CI](https://github.com/ult666666/powerlift/actions/workflows/ci.yml/badge.svg)](https://github.com/ult666666/powerlift/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11 | 3.12 | 3.13](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](pyproject.toml)

Most A/B-testing mistakes are statistical. Stopping a test at the first significant
daily look gives nearly four times the planned false-positive rate over ten looks.
Comparing three variants with a control, without adjustment, finds a false winner in
12.5% of experiments where nothing works. powerlift implements the standard remedies
as small, typed functions.

## Install

```bash
pip install "git+https://github.com/ult666666/powerlift.git"
```

It needs Python 3.11 or newer; the [v0.1.0 release](https://github.com/ult666666/powerlift/releases/tag/v0.1.0)
wheel predates the A/B/n features. For development, clone and run `uv sync`.

## Plan a test, then read it out

```python
>>> from powerlift import compare_proportions, sample_size_proportions
>>> plan = sample_size_proportions(baseline=0.10, mde=0.10, relative=True)
>>> print(plan)
Sample size: two proportions (two-sided pooled z-test, alpha = 0.05, power = 0.8)
  baseline rate    10.00%
  minimum effect   +1.00 pp (+10.00% relative) -> 11.00%
  allocation       n_treatment / n_control = 1
  required n       control 14,751 | treatment 14,751 | total 29,502
  achieved power   0.8000
>>> plan.duration_days(daily_traffic=4_000)
8
>>> print(compare_proportions(480, 5_000, 560, 5_000))
Two-proportion z-test (pooled), two-sided, alpha = 0.05
  control      480 / 5,000 =  9.60%   95% Wilson CI [8.81%, 10.45%]
  treatment    560 / 5,000 = 11.20%   95% Wilson CI [10.36%, 12.10%]
  difference   +1.60 pp   95% CI [+0.40 pp, +2.80 pp] (Newcombe)
  lift         +16.67%   95% CI [+3.95%, +30.94%] (log-ratio delta method)
  z = 2.621, p-value = 0.0088 (significant at alpha = 0.05)

```

## Modules

| Module | What it does |
|---|---|
| `powerlift.design` | Sample size, power and MDE for two proportions (pooled z-test) or two means; sample size and per-arm power for several treatment arms under Dunnett's test (or Bonferroni); experiment duration from daily traffic |
| `powerlift.analysis` | Two-proportion z-test with Wilson, Newcombe and Wald intervals and a log-ratio interval for the relative lift; Welch's t-test; delta-method analysis of ratio metrics; several treatment arms against one control with Dunnett's test (adjusted p-values, simultaneous intervals) or Holm and Bonferroni |
| `powerlift.sanity` | SRM chi-square check for any number of arms; A/A simulation of a test's false-positive rate and p-value uniformity |
| `powerlift.cuped` | CUPED variance reduction with a pre-experiment covariate |
| `powerlift.multiple_testing` | Bonferroni, Holm and Benjamini-Hochberg adjustments |
| `powerlift.sequential` | Pocock, O'Brien-Fleming and Lan-DeMets group sequential boundaries; mSPRT always-valid p-values and confidence sequences |
| `powerlift.bayes` | Beta-Binomial posteriors, P(B > A), expected loss and credible intervals |
| `powerlift` CLI | `sample-size`, `analyze` (counts, or a CSV with two or more arms), `srm` and `boundaries`, with text or JSON output ([docs/CLI.md](docs/CLI.md)) |

Every result is a frozen dataclass with `.to_dict()` (JSON-ready) and `.summary()`
(also used by `print`). Invalid input raises `powerlift.InvalidInputError` with a
message that names the argument.

## Several treatment arms

Dunnett's test compares every variant with the control while holding the family-wise
error at `alpha`, and it gives simultaneous intervals. Conversion rates are given as
`(conversions, n)` per arm:

```python
>>> from powerlift import compare_proportions_to_control, sample_size_multi_arm_proportions
>>> result = compare_proportions_to_control(
...     (1_000, 10_000), {"B": (1_050, 10_000), "C": (1_080, 10_000), "D": (1_120, 10_000)}
... )
>>> print(result)
Dunnett's many-to-one test, two-sided, alpha = 0.05 (family-wise)
  pooled SD 0.308142 with df = 39,996; critical value 2.3491
  arm                 n    estimate    difference  95% simultaneous CI
  control        10,000      10.00%
  B              10,000      10.50%      +0.50 pp  [-0.52 pp, +1.52 pp]   adjusted p = 0.5214 (not significant)
  C              10,000      10.80%      +0.80 pp  [-0.22 pp, +1.82 pp]   adjusted p = 0.1630 (not significant)
  D              10,000      11.20%      +1.20 pp  [+0.18 pp, +2.22 pp]   adjusted p = 0.0164 (significant)
>>> round(result.comparison("D").p_value, 4)  # D's p-value before the adjustment
0.0059
>>> plan = sample_size_multi_arm_proportions(0.10, 0.10, relative=True, n_treatments=3)
>>> print(plan)
Sample size: 3 treatment arms vs one control (two-sided Dunnett test, family-wise alpha = 0.05, power = 0.8 per arm)
  baseline rate    10.00%
  minimum effect   +1.00 pp (+10.00% relative) in each treatment arm
  critical value   2.3490
  allocation       n_control / n_treatment = 1
  required n       control 19,424 | each treatment 19,424 | total 77,696
  achieved power   0.8000 per treatment arm
>>> sample_size_multi_arm_proportions(
...     0.10, 0.10, relative=True, n_treatments=3, adjustment="bonferroni"
... ).total
79912
>>> sample_size_multi_arm_proportions(  # control sqrt(3) times larger than each variant
...     0.10, 0.10, relative=True, n_treatments=3, control_ratio=3 ** 0.5
... ).total
73077

```

`compare_means_to_control` does the same for per-unit values such as revenue, and
`adjustment=` switches to Holm, Bonferroni or no adjustment. Planning for Bonferroni
instead of Dunnett costs 2,216 extra users here. A control arm `sqrt(3)` times larger
than each variant saves 4,619.

## Variance reduction, sequential monitoring and Bayesian analysis

```python
>>> import numpy as np
>>> from powerlift import compare_means_cuped
>>> rng = np.random.default_rng(0)
>>> pre = rng.gamma(2.0, 10.0, size=(2, 5_000))  # pre-period spend per user
>>> post = 0.8 * pre + rng.normal(0.0, 8.0, size=(2, 5_000))
>>> post[1] += 0.5  # true treatment effect
>>> print(compare_means_cuped(post[0], pre[0], post[1], pre[1]))
CUPED (Deng et al. 2013): theta = 0.8082, corr(metric, covariate) = 0.820
  variance of the estimate reduced by 67.2% (corr^2 = 67.2%)
  unadjusted  difference +0.262933   95% CI [-0.285125, +0.81099]   p-value = 0.3470
  adjusted    difference +0.402497   95% CI [+0.0888451, +0.716148]   p-value = 0.0119

>>> from powerlift import adjust_p_values
>>> adjust_p_values([0.01, 0.04, 0.03, 0.005], method="holm").adjusted_p_values
(0.03, 0.06, 0.06, 0.02)

>>> from powerlift import crossing_probability, group_sequential_boundaries
>>> round(crossing_probability([1.96] * 10), 3)  # 10 looks at |z| >= 1.96
0.193
>>> design = group_sequential_boundaries(5, method="obrien-fleming")
>>> [round(z, 3) for z in design.z_boundaries]
[4.562, 3.226, 2.634, 2.281, 2.04]
>>> design.first_crossing([1.2, 2.9, 2.7])  # z-statistics at looks 1-3
3

>>> from powerlift import always_valid_inference
>>> monitor = always_valid_inference(
...     estimates=[0.004, 0.006, 0.009], standard_errors=[0.006, 0.004, 0.003], tau=0.01
... )
>>> print(monitor)
mSPRT always-valid inference (tau = 0.01, alpha = 0.05), 3 looks
  latest estimate      +0.009 (se 0.003)
  always-valid p-value 0.0561
  confidence sequence  [-0.000123794, +0.0181238]
  H0 (effect = 0) not rejected

>>> from powerlift import compare_proportions_bayes
>>> round(compare_proportions_bayes(480, 5_000, 560, 5_000).prob_treatment_better, 4)
0.9956

```

The mSPRT example shows the price of peeking. On the last look alone, a
fixed-horizon z-test gives `z = 3` (p = 0.0027). After three looks, the always-valid
p-value is 0.056.

`simulate_aa` checks that a test is calibrated: on experiments with no true effect it
should be falsely significant about `alpha` of the time, with uniform p-values.

```python
>>> from powerlift import compare_means, simulate_aa
>>> def one_aa_test(rng):
...     return compare_means(rng.lognormal(size=500), rng.lognormal(size=500)).p_value
>>> print(simulate_aa(one_aa_test, n_simulations=2_000, seed=1))
A/A simulation: 2,000 experiments (seed 1)
  false-positive rate  4.45% (95% CI [3.63%, 5.44%]), nominal 5.00%
  KS test vs Uniform(0, 1): D = 0.0163, p-value = 0.6557

```

## How it works

Each test is computed from first principles (z, Welch t, the delta method), with
intervals that behave well for small rates (Wilson, Newcombe). Sequential z-statistics
are multivariate normal with `corr(Z_i, Z_j) = sqrt(t_i / t_j)`, which gives the group
sequential boundaries. Comparisons that share a control arm become independent once
you condition on it, so Dunnett's distribution is a two-dimensional integral for any
number of arms. The mSPRT's likelihood ratio is a martingale under H0, which keeps its
p-value valid however often you look.

- [docs/METHODOLOGY.md](docs/METHODOLOGY.md): formulas, assumptions and references.
- [docs/DESIGN.md](docs/DESIGN.md): architecture, trade-offs and how each result is checked.
- [docs/SIMULATIONS.md](docs/SIMULATIONS.md): error rates and power, by simulation.
- [docs/EXPERIMENT_READOUT.md](docs/EXPERIMENT_READOUT.md): a full readout from per-user CSV data.

## Development

```bash
uv sync                                              # environment from uv.lock
uv run pytest --cov                                  # tests, docstring examples, coverage
uv run pytest --doctest-glob="README.md" README.md   # this README's examples
```

CI also runs ruff, strict mypy, a CLI smoke test, and this README's examples against an
installed wheel, on Python 3.11, 3.12 and 3.13.

## Limits

- Frequentist methods use large-sample normal approximations, with no exact tests.
- Multi-arm comparisons are many-to-one and single-step, with equal variances assumed.
- CUPED, ratio metrics, sequential and Bayesian methods handle two arms only.
- Group sequential designs are two-sided and efficacy-only, with no futility bounds.
- The Bayesian module covers conversion metrics with independent Beta priors.

The full list, with the roadmap, is in
[docs/DESIGN.md](docs/DESIGN.md#limitations-and-roadmap).

Built with AI coding assistants. The tests and CI checks described above verify its
behaviour.

## License

[MIT](LICENSE). Copyright (c) 2026 Charles Dai.

# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Multi-arm (A/B/n) experiments: `compare_means_to_control` and
  `compare_proportions_to_control` compare several treatment arms with one
  control using Dunnett's test, with adjusted p-values and simultaneous
  intervals (one-sided or two-sided), or Holm, Bonferroni or no adjustment.
  The Dunnett distribution is computed deterministically from a
  two-dimensional integral; `dunnett_critical_value` exposes its quantile.
- `sample_size_multi_arm_means` and `sample_size_multi_arm_proportions`:
  sample sizes and per-arm power for several treatment arms under Dunnett's
  test, with Bonferroni as a conservative option and an adjustable control
  allocation.
- `powerlift analyze --csv` accepts `--treatment-label` more than once for a
  multi-arm readout: an SRM check across all arms (`--allocation`), then the
  many-to-one comparison (`--adjustment`).
- `examples/multi_arm_simulation.py`: family-wise error under the global null
  and planned versus simulated power.

- `examples/experiment_readout.py`: a fixed-horizon readout of a fictional
  checkout experiment that runs a per-user CSV through SRM, outcome-window and
  enrollment gates before a CUPED analysis. The walkthrough is in
  `docs/EXPERIMENT_READOUT.md`.
- A release workflow that builds the wheel and sdist from a version tag and
  attaches them, with SHA-256 checksums, to the GitHub release.

### Changed

- `Result.to_dict()` returns `None` for infinite or NaN floats, so JSON output
  stays valid. No earlier result contained such values.

### Fixed

- The README quick start installed the v0.1.0 release wheel, which predates the
  A/B/n features: the multi-arm README examples failed with `ImportError`, and
  the multi-arm `analyze` command silently ran a two-arm Welch test on the last
  `--treatment-label`. The quick start now installs from GitHub and says what
  the release lacks. CI runs the README examples against an installed wheel.
- Dunnett critical values for `alpha` of about 1e-16 or less were wrong (the
  search returned the end of its bracket, such as 16.0), and with a finite `df`
  smaller values made `compare_means_to_control` run forever. `alpha` below
  1e-10 now raises `InvalidInputError`, and the bracket search is bounded.

## [0.1.0] - 2026-10-23

### Added

- Experiment design: sample size, power and minimum detectable effect for two
  proportions (pooled z-test) and two means, and experiment duration.
- Analysis: pooled two-proportion z-test with Wilson, Newcombe and Wald
  intervals and a log-ratio interval for the relative lift; Welch's t-test;
  delta-method analysis of ratio metrics.
- Sanity checks: sample ratio mismatch test and A/A simulation utility.
- CUPED variance reduction.
- Bonferroni, Holm and Benjamini-Hochberg p-value adjustments.
- Group sequential boundaries (Pocock, O'Brien-Fleming, Lan-DeMets spending)
  and mSPRT always-valid p-values and confidence sequences.
- Beta-Binomial Bayesian comparison of conversion rates.
- `powerlift` command-line interface with `sample-size`, `analyze`, `srm` and
  `boundaries` commands.
- Peeking simulation example and documentation of the methodology and design.

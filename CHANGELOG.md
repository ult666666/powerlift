# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

The `powerlift` package itself is unchanged since 0.1.0.

### Added

- `examples/experiment_readout.py`: a fixed-horizon readout of a fictional
  checkout experiment that runs a per-user CSV through SRM, outcome-window and
  enrollment gates before a CUPED analysis. The walkthrough is in
  `docs/EXPERIMENT_READOUT.md`.
- A release workflow that builds the wheel and sdist from a version tag and
  attaches them, with SHA-256 checksums, to the GitHub release.

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

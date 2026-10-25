# Experiment readout example

Build an additive, runnable CSV-to-readout example for a fictional checkout experiment. Existing APIs already provide CSV statistics, SRM and CUPED separately; the missing use case is a complete workflow with explicit analysis gates.

Use `examples/experiment_readout.py`, a checked-in 400-user seed-42 CSV and a linked walkthrough. The example has one predeclared primary metric (14-day net revenue per randomized user), a 50/50 allocation, 200 users per arm and a fixed outcome-window end of 2026-09-15. The covariate is net revenue during August, measured before September 1 randomization. Signed revenue permits refunds. These are illustrative design constants, not an adequately powered production design.

CSV validation rejects duplicate users, unknown groups, nonfinite values and missing columns. Run SRM before inference. An SRM, incomplete outcome window or insufficient enrollment produces a blocked report with no effect estimates or p-values. Otherwise render the existing raw/CUPED comparison, confidence interval and variance reduction. Passing SRM is one check, not proof of valid assignment; the readout makes no shipping recommendation. The analyst remains responsible for covariate provenance and complete outcome collection.

Approaches considered: expanding the public CLI (larger API commitment), notebook (extra tooling), or runnable example (selected for a bounded, dependency-free addition using current APIs). No statistical primitives change. Add real positive/negative tests and document regeneration and expected output without fabricated adoption or performance claims.

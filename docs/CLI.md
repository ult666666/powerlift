# Command line

The `powerlift` command has four subcommands: `sample-size`, `analyze`, `srm` and
`boundaries`. The transcripts below come from a source checkout, where `uv run` puts
the command on the path; after `pip install`, drop the `uv run` prefix.

## Analyse counts, check the split, plan interim looks

```console
$ uv run powerlift analyze --control 480/5000 --treatment 560/5000 --bayes
Two-proportion z-test (pooled), two-sided, alpha = 0.05
  control      480 / 5,000 =  9.60%   95% Wilson CI [8.81%, 10.45%]
  treatment    560 / 5,000 = 11.20%   95% Wilson CI [10.36%, 12.10%]
  difference   +1.60 pp   95% CI [+0.40 pp, +2.80 pp] (Newcombe)
  lift         +16.67%   95% CI [+3.95%, +30.94%] (log-ratio delta method)
  z = 2.621, p-value = 0.0088 (significant at alpha = 0.05)

Bayesian Beta-Binomial comparison, prior Beta(1, 1) for each arm
  control      Beta(481, 4521): mean 9.62%, 95% credible interval [8.81%, 10.45%]
  treatment    Beta(561, 4441): mean 11.22%, 95% credible interval [10.36%, 12.10%]
  P(treatment > control)          0.9956
  expected loss, ship treatment   0.0008 pp
  expected loss, keep control     1.6002 pp
  difference     95% credible interval [+0.40 pp, +2.79 pp]
  relative lift  95% credible interval [+3.92%, +30.90%]
  (intervals for the difference and lift: 200,000 Monte Carlo draws, seed 0)

$ uv run powerlift srm --observed 50421 49579
Sample ratio mismatch check (chi-square goodness of fit)
  arm 1: observed 50,421 (50.421%), expected 50,000.0 (50.000%)
  arm 2: observed 49,579 (49.579%), expected 50,000.0 (50.000%)
  chi2 = 7.090, df = 1, p-value = 0.0078
  no SRM detected (p >= 0.001)

$ uv run powerlift boundaries --looks 4 --method lan-demets-obrien-fleming
Lan-DeMets (O'Brien-Fleming-type spending) boundaries: two-sided alpha = 0.05, 4 looks
  look  information  |z| boundary  nominal alpha  cumulative alpha
  1     0.250        4.3326        0.000015       0.000015
  2     0.500        2.9631        0.003045       0.003051
  3     0.750        2.3591        0.018321       0.019299
  4     1.000        2.0141        0.044000       0.050000
```

`powerlift sample-size --baseline 0.10 --mde 0.10 --relative --daily-traffic 4000`
prints the plan from the first example in the [README](../README.md), plus the line
`duration         8 days at 4,000 eligible units/day`.
`powerlift analyze --csv data.csv --group-col group --value-col revenue` runs
Welch's t-test on per-unit data; add `--metric proportion` (0/1 values),
`--metric ratio --denominator-col sessions`, or `--covariate-col pre_revenue`
(CUPED). Every command accepts `--json`; see `uv run powerlift <command> --help`.

## Several treatment arms from a CSV

From the command line, repeat `--treatment-label` to read out a CSV with
several arms. The readout starts with an SRM check across all arms (pass the
planned split with `--allocation`, control first). The synthetic file below
comes from `uv run python examples/multi_arm_simulation.py --csv arms.csv`;
its arms B and C match control A, and D adds $1.00 per user:

```console
$ uv run powerlift analyze --csv arms.csv --group-col arm --value-col revenue_usd --control-label A --treatment-label B --treatment-label C --treatment-label D
Sample ratio mismatch check (chi-square goodness of fit)
  arm 1: observed 2,000 (25.000%), expected 2,000.0 (25.000%)
  arm 2: observed 2,000 (25.000%), expected 2,000.0 (25.000%)
  arm 3: observed 2,000 (25.000%), expected 2,000.0 (25.000%)
  arm 4: observed 2,000 (25.000%), expected 2,000.0 (25.000%)
  chi2 = 0.000, df = 3, p-value = 1.0000
  no SRM detected (p >= 0.001)

Dunnett's many-to-one test, two-sided, alpha = 0.05 (family-wise)
  pooled SD 7.23568 with df = 7,996; critical value 2.3494
  arm                 n    estimate    difference  95% simultaneous CI
  A               2,000     10.0568
  B               2,000     10.0752      +0.01837  [-0.519206, +0.555946]   adjusted p = 0.9996 (not significant)
  C               2,000     10.2287     +0.171955  [-0.365621, +0.709531]   adjusted p = 0.7934 (not significant)
  D               2,000     11.0045      +0.94768  [+0.410104, +1.48526]   adjusted p = 0.0001 (significant)
```

The v0.1.0 release wheel predates this: its `analyze` keeps only the last
`--treatment-label`, so the command above runs a two-arm test there. Install v0.2.0
or later to get the multi-arm readout.

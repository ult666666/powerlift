# Design notes

How powerlift is put together, the decisions behind it, and how its
correctness is checked. Statistical formulas are in
[METHODOLOGY.md](METHODOLOGY.md).

## Architecture

```
src/powerlift/
  exceptions.py        PowerliftError > InvalidInputError, InsufficientDataError, ConvergenceError
  _validation.py       argument checks shared by every public function
  _stats.py            p-values and critical values (z and t)
  _results.py          Result base class (to_dict, summary), ConfidenceInterval
  design.py            sample size, power, MDE, duration
  analysis/            proportions.py, means.py, ratio.py (+ _common.py result builder)
  cuped.py             variance reduction on top of analysis.compare_means
  sanity.py            SRM test, A/A simulation
  multiple_testing.py  Bonferroni, Holm, Benjamini-Hochberg
  sequential.py        group sequential boundaries, mSPRT
  bayes.py             Beta-Binomial model
  cli.py, __main__.py  argparse front end (`powerlift ...`, `python -m powerlift`)
```

Dependencies point one way: the private helpers know nothing about the
statistics, the statistical modules depend only on the helpers (CUPED and the
A/A utility also reuse `analysis`), and only `cli.py` knows about all of them.
Runtime dependencies are NumPy and SciPy; statsmodels, matplotlib, Hypothesis
and the tooling are development-only.

**Functions in, frozen dataclasses out.** The public API is plain functions
(like SciPy and statsmodels) that return immutable result objects. Each result
derives from `Result`, which supplies `to_dict()` (recursively converts nested
results, tuples and NumPy scalars into JSON-ready Python types) and `__str__`
via the subclass's `summary()`. Immutability makes results safe to cache and
compare (`==` works, which the tests use for determinism checks).

**Validation at the boundary.** Every public function validates its inputs
first and raises `InvalidInputError` (a `ValueError` subclass) naming the
argument, so mistakes do not surface later as NaNs or NumPy errors. Data that
cannot support a computation (for example zero variance in both arms) raises
`InsufficientDataError`. Booleans are rejected where numbers are expected,
since `True` passed as a count is almost always a bug.

**Typing.** The package is `mypy --strict` clean and ships `py.typed`. SciPy
and statsmodels have no type information, so a single, commented mypy
override treats them as untyped, and every value coming back from them is
converted with `float(...)` at the call site. There are no `type: ignore`
comments in the code base.

## Key decisions and trade-offs

**Design formulas match the test that will be run.** The proportion sample
size uses the pooled variance under H0 and the unpooled variance under H1,
because the analysis uses the pooled z-test. This is exactly statsmodels'
`samplesize_proportions_2indep_onetail`, so the test suite can check it
directly. The closed-form sample size ignores the far rejection tail (standard
practice); the `power_*` functions include it, and the result reports the
achieved power at the rounded sizes, which is therefore never below the target.

**Interval choices.** Wald intervals under-cover with small samples or rates
near 0, so the difference in proportions defaults to Newcombe's hybrid score
interval (Wald stays available). For the relative lift of a conversion rate the
interval is built on the log ratio, which respects the -100% floor and matches
statsmodels' `method="log"`. Means can be zero or negative, so their relative
lift uses the symmetric delta-method interval instead. All intervals are
two-sided at `1 - alpha`, even for one-sided tests; this keeps one interval
per result and is listed as a limitation.

**Welch, not Student.** Welch's t-test does not assume equal variances and
loses almost nothing when they are equal, so it is the only mean test.

**Ratio metrics by linearization.** The delta-method variance is computed as
`Var(Y - R X) / (n mean(X)^2)`, which equals the textbook three-term formula
but is a single, numerically stable variance computation.

**CUPED with a pooled theta.** `theta` is estimated on both arms together (the
covariate is independent of assignment, so pooling cannot bias the estimate)
and the adjusted values go through the same Welch test, so CUPED reuses the
existing, validated code path instead of duplicating it.

**Group sequential boundaries from multivariate normal probabilities.** The
probability of first crossing at each look is a rectangle probability of the
joint normal distribution of the sequential z-statistics. Two choices matter
for accuracy and speed:

1. *Sum small exit probabilities instead of computing `1 - P(no crossing)`.*
   SciPy's quasi-Monte Carlo integrator struggles to resolve a probability
   near 1 to high absolute precision (measured: 0.6 s for 5 looks and 17 s for
   20 looks per evaluation, regardless of tolerance), while each exit
   probability is small and converges quickly.
2. *Bracket the spending-function roots tightly.* The bracket ends are chosen
   from single-look tail probabilities so that the integrator never evaluates
   a large probability.

The QMC tolerance (`1e-6`) and a fixed seed were chosen after measuring that
the boundaries agree with a 100x stricter tolerance to within `5e-5` for up to
10 looks, while a 10x stricter tolerance already made them about 10x slower.
On the development machine, 10 looks take under a second for every method;
20 looks take 1.5-2 s for Pocock and O'Brien-Fleming and 6-11 s for the
Lan-DeMets designs.
A recursive numerical integration (Jennison & Turnbull ch. 19) would be faster
and is on the roadmap; the multivariate normal formulation was kept because it
is short and easy to verify.

**Lan-DeMets convention.** Two conventions exist for "O'Brien-Fleming-type"
spending in a two-sided test: applying `alpha*(t)` to the total alpha (first
boundary 4.38 for `K = 5`) or letting each side spend `alpha/2` (first boundary
4.88). powerlift uses the second, which is what gsDesign and ldbounds report,
and documents it.

**mSPRT on the log scale.** The mixture likelihood ratio is computed as a log
to avoid overflow when evidence is overwhelming; p-values use the running
maximum of the log ratio and the confidence sequence is the running
intersection, so p-values never increase and intervals are nested. The
p-value and the interval are dual by construction, which the tests check.

**Bayesian quantities by quadrature.** `P(B > A)` and the expected losses are
one-dimensional integrals, so they are computed by adaptive quadrature (about
1e-10 accuracy, under a millisecond) rather than Monte Carlo; the expected
loss uses the identity `u f(u; a, b) = a/(a+b) f(u; a+1, b)` to reuse the same
integral. Only the credible intervals for the difference and the lift use
seeded Monte Carlo draws.

**Reproducibility.** Every random component takes a seed (A/A simulation,
Bayesian intervals) or uses a fixed internal seed (quasi-Monte Carlo
integration). Dependency versions are locked in `uv.lock`; the lower bounds in
`pyproject.toml` (NumPy 2.4, SciPy 1.17) are the oldest versions CI actually
exercises (the lock resolves them for Python 3.11), rather than guesses.

## How correctness is verified

Four kinds of evidence, all in `tests/`:

1. **Reference parity.** Results are compared with independent
   implementations: statsmodels (`power_proportions_2indep`,
   `samplesize_proportions_2indep_onetail`, `NormalIndPower`,
   `proportions_ztest`, `proportion_confint`, `confint_proportions_2indep`,
   `multipletests`) and SciPy (`ttest_ind`, `ttest_ind_from_stats`,
   `chisquare`, `linregress`), often on Hypothesis-generated inputs.
2. **Published values.** Pocock and O'Brien-Fleming constants from Jennison &
   Turnbull (K = 2-5 and 10), Lan-DeMets O'Brien-Fleming-type bounds for
   K = 5, the classic error rates of naive repeated testing, Evan Miller's
   closed form for P(B > A).
3. **Simulation.** Seeded Monte Carlo checks of the properties that matter:
   planned sample sizes deliver planned power, intervals have nominal
   coverage, CUPED is unbiased and removes about `rho^2` of the variance, every
   test is calibrated in A/A simulations (and a deliberately wrong
   session-level analysis is caught), group sequential designs have type-I
   error alpha over 400,000 simulated paths, the mSPRT keeps its error below
   alpha under continuous monitoring, Holm controls FWER and BH controls FDR.
   Tolerances are set from the Monte Carlo standard error (typically 3-5 SE).
4. **Properties and invariants.** Hypothesis checks round trips (the MDE at
   the planned size gives back the planned size), monotonicity, ordering of
   corrections (BH <= Holm <= Bonferroni), permutation equivariance and the
   p-value / confidence-sequence duality. It also found real bugs: squaring
   values near 1e-150 underflowed to zero and divided by zero, now fixed and
   covered by regression tests.

Examples in docstrings and in the README are executed as doctests, the peeking
example has a smoke test, and coverage is measured with branch coverage
(CI fails below 90%).

## Questions worth being able to answer

- *Why does peeking inflate the false-positive rate?* Each look is another
  chance for a random walk to cross 1.96; the crossing probability grows with
  the number of looks (19% after 10, about 25% after 20 equally spaced looks)
  and tends to 1 with continuous monitoring.
- *Pocock or O'Brien-Fleming?* Pocock uses the same boundary at every look, so
  it stops early more often but needs a stricter final boundary (lower power
  at the planned size). O'Brien-Fleming is very strict early and nearly
  unchanged at the end, so it costs little power; in the peeking simulation
  its power is 78.3% against 80.0% for a single look.
- *Why is CUPED unbiased?* The covariate is measured before randomization,
  so its mean is the same in both arms in expectation; subtracting
  `theta (X - mean(X))` changes each arm's mean by a quantity with expectation
  zero while removing the part of the variance explained by X.
- *When is the mSPRT the right tool?* When the number or timing of looks is
  not fixed in advance (dashboards, continuous monitoring). The price is
  power at any fixed sample size.
- *Why Holm over Bonferroni?* Same guarantee (FWER under any dependence), never
  fewer rejections.

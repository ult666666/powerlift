# Methodology

This document lists the formula, assumptions and references behind every
method in powerlift. Notation: control arm `c`, treatment arm `t`, sample
sizes `n_c`, `n_t`, allocation ratio `r = n_t / n_c`, effect
`delta = treatment - control`, significance level `alpha`,
`z_q = Phi^{-1}(q)` the standard normal quantile. Unless stated otherwise all
intervals are two-sided with confidence `1 - alpha`, and a test is significant
when `p <= alpha`.

## 1. Experiment design (`powerlift.design`)

All design calculations use the normal approximation to the sampling
distribution of the difference in estimates, as is standard for online
experiments (Kohavi, Tang & Xu 2020, ch. 17). Let `z_a = z_{1-alpha/2}`
(two-sided) or `z_{1-alpha}` (one-sided, in the direction of the MDE) and
`z_b = z_{power}`.

**Two means**, common standard deviation `sigma`:

```
n_c = (z_a + z_b)^2 * sigma^2 * (1 + 1/r) / delta^2
```

With `alpha = 0.05`, `power = 0.8` and `r = 1` this is `15.7 sigma^2 / delta^2`,
the familiar `16 sigma^2 / delta^2` rule of thumb.

**Two proportions** `p_c -> p_t = p_c + delta`. The analysis uses the pooled
z-test, so the null standard error uses the pooled rate
`p_bar = (p_c + r p_t) / (1 + r)` and the alternative uses the unpooled
variance (Fleiss, Levin & Paik 2003, ch. 4, without continuity correction):

```
n_c = (z_a * sqrt(p_bar (1 - p_bar) (1 + 1/r))
       + z_b * sqrt(p_c (1 - p_c) + p_t (1 - p_t) / r))^2 / delta^2
```

This equals statsmodels' `samplesize_proportions_2indep_onetail` (whose sample
1 is our treatment arm and whose `ratio` is `n_c / n_t`).

**Rounding.** `n_c = ceil(n_c*)` and `n_t = ceil(r * n_c*)`, where `n_c*` is
the unrounded solution.

**Power.** With `se_0` and `se_1` the standard errors of the difference under
H0 and H1 at the given sample sizes, the test rejects when
`|estimate| / se_0 > z_a`, so for a two-sided test

```
power = Phi((|delta| - z_a se_0) / se_1) + Phi((-|delta| - z_a se_0) / se_1)
```

(only the first term for a one-sided test). The sample-size formula drops the
second, negligible term; the power functions keep it, so the power at the
returned sample size is never below the target.

**MDE.** For means, `delta = (z_a + z_b) sigma sqrt(1/n_c + 1/n_t)`. For
proportions the variance depends on `p_c + delta`, so the sample-size equation
`z_a se_0(delta) + z_b se_1(delta) = delta` is solved with Brent's method for
an increase `delta > 0`.

**Duration.** `days = ceil(total / (daily_traffic * traffic_fraction))`,
assuming each day brings new units. With returning users, unique users
accumulate more slowly, so this is a lower bound; running whole weeks also
averages over day-of-week effects (Kohavi et al. 2020).

**Assumptions:** independent units, large samples (normal approximation), the
SD or baseline rate is known in advance (in practice estimated from historical
data).

## 2. Analysis of conversion metrics (`compare_proportions`)

With `x` conversions out of `n` units, `p = x / n`.

**Pooled two-proportion z-test.**
`p_bar = (x_c + x_t) / (n_c + n_t)`,
`z = (p_t - p_c) / sqrt(p_bar (1 - p_bar) (1/n_c + 1/n_t))`.
Two-sided `p = 2 (1 - Phi(|z|))`; `"greater"` uses `1 - Phi(z)` and `"less"`
uses `Phi(z)`. Undefined (an error is raised) when `p_bar` is 0 or 1.

**Wilson score interval** for one arm (Wilson 1927): the set of `p` not
rejected by the score test,

```
center = (p_hat + z^2/(2n)) / (1 + z^2/n)
half   = z sqrt(p_hat (1 - p_hat)/n + z^2/(4n^2)) / (1 + z^2/n)
```

with `z = z_{1-alpha/2}`. It stays inside [0, 1] and has near-nominal coverage
for small `n` and extreme rates.

**Newcombe hybrid score interval** for `d = p_t - p_c` (method 10 of Newcombe
1998), combining the Wilson limits `(l, u)` of each arm:

```
[d - sqrt((p_t - l_t)^2 + (u_c - p_c)^2),  d + sqrt((u_t - p_t)^2 + (p_c - l_c)^2)]
```

**Wald interval** (optional): `d +/- z sqrt(p_c(1-p_c)/n_c + p_t(1-p_t)/n_t)`.
It under-covers for small samples or rates near 0; the test suite shows this by
simulation.

**Relative lift** `p_t / p_c - 1`, delta method on the log ratio (the Katz
log interval): `Var(log p_hat) ~= (1 - p) / (n p) = 1/x - 1/n`, so

```
se_log = sqrt(1/x_t - 1/n_t + 1/x_c - 1/n_c)
CI     = (p_t / p_c) * exp(+/- z se_log) - 1
```

The interval is asymmetric and never extends below -100%. It is undefined when
an arm has zero conversions (reported as missing).

**Caveat:** the test uses the pooled variance and the intervals do not, so in
borderline cases they can disagree.

## 3. Analysis of continuous metrics (`compare_means`)

**Welch's t-test** (Welch 1947). With sample variances `s^2` (`ddof = 1`):

```
se = sqrt(s_c^2/n_c + s_t^2/n_t)
t  = (mean_t - mean_c) / se
df = (s_c^2/n_c + s_t^2/n_t)^2 / ((s_c^2/n_c)^2/(n_c - 1) + (s_t^2/n_t)^2/(n_t - 1))
```

The difference interval uses `t_{df, 1-alpha/2}`. Welch's test does not assume
equal variances; for skewed metrics it relies on the central limit theorem,
which the A/A tests check for lognormal data at 500 units per arm.

**Relative lift** `mean_t / mean_c - 1`, first-order delta method for a ratio
of independent estimates (Deng, Knoblich & Lu 2018; Kohavi et al. 2020, ch. 18):

```
Var(T / C) ~= Var(T) / C^2 + T^2 Var(C) / C^4
```

with a normal critical value. Means can be zero or negative, so this
symmetric interval is used instead of the log-ratio one; it is unreliable when
the control mean is close to zero relative to its standard error.

## 4. Ratio metrics (`compare_ratio_metrics`)

When users are randomized but the metric is a ratio of sums, such as revenue
per session `R = sum(Y) / sum(X)`, observations within a user are correlated,
and treating sessions as independent understates the variance: in the test
suite's seeded A/A simulation a session-level t-test is falsely significant in
18.8% of experiments instead of 5%. Each randomized unit contributes
`(Y_i, X_i)` and, by the delta method (Deng, Knoblich & Lu 2018):

```
Var(R) ~= (s_Y^2 - 2 R s_XY + R^2 s_X^2) / (n * mean(X)^2) = Var(Y - R X) / (n * mean(X)^2)
```

Arms are independent; the difference uses a z-test and normal interval and the
relative lift uses the delta method of section 3. Assumes many units per arm
and a denominator mean well away from zero.

## 5. Sanity checks (`powerlift.sanity`)

**Sample ratio mismatch.** Pearson goodness-of-fit test of the observed counts
`O_i` against `E_i = N * share_i`:
`chi2 = sum (O_i - E_i)^2 / E_i` with `k - 1` degrees of freedom. A mismatch is
flagged when `p < 0.001` by default: the check runs on every experiment, so a
strict threshold keeps false alarms rare (Kohavi et al. 2020, ch. 21; Fabijan
et al. 2019).

**A/A simulation.** `simulate_aa` runs a user-supplied test on simulated
experiments without a true effect and reports the false-positive rate (with a
95% Wilson interval for its Monte Carlo error) and a Kolmogorov-Smirnov test of
the p-values against Uniform(0, 1). Under H0 a valid test has uniform p-values
(exactly for continuous statistics, approximately for discrete ones such as
binomial counts).

## 6. CUPED (`compare_means_cuped`)

Deng, Xu, Kohavi & Walker (2013). With a pre-experiment covariate `X` (for
example the same metric in the weeks before the test):

```
theta  = cov(Y, X) / var(X)               (pooled over both arms)
Y_cv   = Y - theta (X - mean(X))
```

Because `X` is measured before assignment, `E[X]` is the same in both arms and
the adjusted difference in means remains unbiased for the treatment effect,
while its variance is multiplied by `1 - rho^2`, where `rho = corr(Y, X)`. The
adjusted values are analysed with Welch's t-test. `theta` is treated as fixed;
estimating it adds variance of order `1/n`. The reported variance reduction is
`1 - se_adjusted^2 / se_unadjusted^2`.

## 7. Multiple comparisons (`adjust_p_values`)

For `m` p-values sorted as `p_(1) <= ... <= p_(m)`:

- **Bonferroni:** `min(1, m p_i)`; controls the family-wise error rate (FWER)
  under any dependence.
- **Holm** (1979): `adj_(i) = max_{j <= i} min(1, (m - j + 1) p_(j))`; controls
  FWER under any dependence and is uniformly more powerful than Bonferroni.
- **Benjamini-Hochberg** (1995): `adj_(i) = min_{j >= i} min(1, m p_(j) / j)`;
  controls the false discovery rate under independence or positive
  regression dependence.

A hypothesis is rejected when its adjusted p-value is `<= alpha`.

## 8. Sequential testing (`powerlift.sequential`)

### 8.1 Group sequential boundaries

With information fractions `t_1 < ... < t_K = 1` (the share of the final
sample available at each look), the z-statistics computed on cumulative data
are jointly normal with

```
corr(Z_i, Z_j) = sqrt(t_i / t_j)    for t_i <= t_j
```

(Jennison & Turnbull 2000, ch. 3). A two-sided design rejects at look `k` if
`|Z_k| >= c_k`. Under H0 the probability of stopping first at look `k` is

```
P(|Z_1| < c_1, ..., |Z_{k-1}| < c_{k-1}, |Z_k| >= c_k)
    = 2 P(|Z_j| < c_j for j < k, Z_k >= c_k)
```

(symmetry), a rectangle probability of a `k`-dimensional normal distribution.
It is computed with `scipy.stats.multivariate_normal.cdf` and `lower_limit`
(exact in one and two dimensions, randomized quasi-Monte Carlo with a fixed
seed and absolute tolerance `1e-6` beyond). Summing these small probabilities
is cheaper and more accurate than computing `1 - P(no crossing)`.

- **Pocock:** `c_k = C_P` for all `k`.
- **O'Brien-Fleming:** `c_k = C_B / sqrt(t_k)`.

The constant solves `sum_k P(stop at k) = alpha` (Brent's method on the
bracket `[z_{1-alpha/2}, z_{1-alpha/(2K)}]`: the final look alone spends at
least alpha at the lower end; Bonferroni bounds the error at the upper end).
For `alpha = 0.05`, `K = 5` this gives `C_P = 2.4132` and `C_B = 2.0401`,
matching Jennison & Turnbull's 2.413 and 2.040.

- **Lan-DeMets alpha spending** (Lan & DeMets 1983): look by look, `c_k` is
  chosen so that `P(stop at k) = alpha*(t_k) - alpha*(t_{k-1})`. Two spending
  functions are provided:
  - O'Brien-Fleming type, each side spending `alpha/2`
    (the convention of the R packages gsDesign and ldbounds):
    `alpha*(t) = 4 (1 - Phi(z_{1-alpha/4} / sqrt(t)))`;
  - Pocock type: `alpha*(t) = alpha log(1 + (e - 1) t)`.

  For `K = 5`, two-sided 0.05 and O'Brien-Fleming-type spending this gives
  4.877, 3.357, 2.680, 2.290, 2.031, the published values. The Pocock-type
  boundaries are validated by simulation (overall type-I error and the alpha
  spent at each look).

Accuracy: comparing with a 100 times stricter integration tolerance, the
boundaries agree to within `5e-5` for up to 10 looks.

### 8.2 Always-valid inference: mSPRT

Johari, Koomen, Pekelis & Walsh (2017); Johari, Pekelis & Walsh (2022). Let
`theta_n` be the estimated effect at look `n` with variance `V_n`. For
`H0: theta = theta_0`, mix the alternative over `theta ~ N(theta_0, tau^2)`:

```
Lambda_n = sqrt(V_n / (V_n + tau^2)) * exp(tau^2 (theta_n - theta_0)^2 / (2 V_n (V_n + tau^2)))
```

`Lambda_n` is the ratio of the marginal density of `theta_n` under the mixture,
`N(theta_0, V_n + tau^2)`, to its density under H0, `N(theta_0, V_n)`. Under H0
it is a nonnegative martingale with mean 1, so Ville's inequality gives
`P(sup_n Lambda_n >= 1/alpha) <= alpha`. Therefore

- always-valid p-value: `p_n = min(1, 1 / max_{m <= n} Lambda_m)`;
- confidence sequence: the intersection over `m <= n` of
  `theta_m +/- sqrt(2 V_m (V_m + tau^2) / tau^2 * log(sqrt((V_m + tau^2)/V_m) / alpha))`,

valid simultaneously over all looks, however many there are and whenever
monitoring stops. Rejecting H0 when `p_n <= alpha` is equivalent to
`theta_0` leaving the confidence sequence. powerlift uses plug-in variances
(normal approximation), so the guarantee is asymptotic in the sample size.
`tau` should be on the scale of plausible effects (for example the MDE);
validity holds for any `tau`, power does not.

## 9. Bayesian analysis of conversion rates (`compare_proportions_bayes`)

Independent `Beta(a, b)` priors and binomial likelihoods give the posteriors
`Beta(a + x, b + n - x)`. Default prior `Beta(1, 1)` (uniform); `Beta(0.5, 0.5)`
(Jeffreys) is the other common choice; with experiment-sized samples either is
dominated by the data. Informative priors shrink both arms towards the prior
mean and should be justified.

- **Credible intervals:** equal-tailed posterior quantiles.
- **P(T > C)** `= integral_0^1 f_T(u) F_C(u) du`, by adaptive Gauss-Kronrod
  quadrature over the central `1 - 2e-15` of T's posterior. The tests compare it
  with Evan Miller's closed form for integer parameters,
  `sum_{i=0}^{a_T - 1} B(a_C + i, b_C + b_T) / ((b_T + i) B(1 + i, b_T) B(a_C, b_C))`.
- **Expected loss** of shipping the treatment, `E[max(p_c - p_t, 0)]`: using
  `u f(u; a, b) = a/(a + b) f(u; a + 1, b)`,

  ```
  E[max(C - T, 0)] = E[C] P(C' > T) - E[T] P(C > T')
  ```

  with `C' ~ Beta(a_C + 1, b_C)` and `T' ~ Beta(a_T + 1, b_T)`, so it reduces
  to the same one-dimensional integrals. The loss of keeping the control is
  symmetric, and the two differ by exactly `E[T] - E[C]`.
- **Difference and relative-lift intervals:** quantiles of seeded Monte Carlo
  draws (200,000 by default).

## References

- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *JRSS B*, 57(1), 289-300.
- Deng, A., Knoblich, U., & Lu, J. (2018). Applying the delta method in metric analytics: a practical guide with novel ideas. *KDD 2018*.
- Deng, A., Xu, Y., Kohavi, R., & Walker, T. (2013). Improving the sensitivity of online controlled experiments by utilizing pre-experiment data. *WSDM 2013*.
- Fabijan, A., et al. (2019). Diagnosing sample ratio mismatch in online controlled experiments: a taxonomy and rules of thumb for practitioners. *KDD 2019*.
- Fleiss, J. L., Levin, B., & Paik, M. C. (2003). *Statistical Methods for Rates and Proportions* (3rd ed.). Wiley.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65-70.
- Jennison, C., & Turnbull, B. W. (2000). *Group Sequential Methods with Applications to Clinical Trials*. Chapman & Hall/CRC.
- Johari, R., Koomen, P., Pekelis, L., & Walsh, D. (2017). Peeking at A/B tests: why it matters, and what to do about it. *KDD 2017*.
- Johari, R., Pekelis, L., & Walsh, D. (2022). Always valid inference: continuous monitoring of A/B tests. *Operations Research*, 70(3).
- Kohavi, R., Tang, D., & Xu, Y. (2020). *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing*. Cambridge University Press.
- Lan, K. K. G., & DeMets, D. L. (1983). Discrete sequential boundaries for clinical trials. *Biometrika*, 70(3), 659-663.
- Newcombe, R. G. (1998). Interval estimation for the difference between independent proportions: comparison of eleven methods. *Statistics in Medicine*, 17(8), 873-890.
- O'Brien, P. C., & Fleming, T. R. (1979). A multiple testing procedure for clinical trials. *Biometrics*, 35(3), 549-556.
- Pocock, S. J. (1977). Group sequential methods in the design and analysis of clinical trials. *Biometrika*, 64(2), 191-199.
- Welch, B. L. (1947). The generalization of "Student's" problem when several different population variances are involved. *Biometrika*, 34(1-2), 28-35.
- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *JASA*, 22(158), 209-212.

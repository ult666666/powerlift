# A complete experiment readout from a CSV

This fictional checkout experiment shows how to connect a per-user export to
sanity checks and an uncertainty-aware analysis. It uses the existing
`powerlift.srm_test` and `powerlift.compare_means_cuped` APIs. All 400 users and
all revenue amounts are synthetic; there was no live experiment.

From the repository root after `uv sync`, run:

```bash
uv run python examples/experiment_readout.py --as-of 2026-09-15
```

To save the same Markdown report:

```bash
uv run python examples/experiment_readout.py --as-of 2026-09-15 --output /tmp/checkout-readout.md
```

## The example's plan

| Item | Fixed assumption |
| --- | --- |
| Randomization unit | A user, exactly once; planned allocation 50/50 |
| Primary metric | Net revenue in USD per randomized user over September 1–14, including refunds and zeros |
| Covariate | The same user's net revenue during August, before September 1 randomization |
| Final analysis | Once outcomes are complete through September 14; `--as-of 2026-09-15` or later |
| Enrollment | At least 200 users in each arm |
| Analysis | Two-sided CUPED-adjusted Welch test, alpha 0.05; unadjusted comparison shown for context |
| Sanity gate | Block if the SRM p-value is below 0.001 |

The 200-user count keeps the example small. It is **not a sample-size or power
recommendation**. A real experiment needs a design based on its baseline
variance, minimum meaningful effect, traffic and outcome-maturation window;
see `powerlift sample-size --metric mean --help`.

The script's fixed plan is deliberately local to this example. A different
experiment requires changing the plan as well as the data. `--as-of` is the
analyst's assertion that outcome collection is complete through that date; the
CSV alone cannot prove it. Late enrollments require waiting for their full
outcome windows. A calendar date and a sufficient count do not establish
complete follow-up by themselves.

The CSV has one row per randomized user:

```text
user_id,group,pre_revenue_usd,revenue_usd
```

Both revenue columns contain signed net amounts, so refunds may make them
negative. Keep randomized users with zero revenue. Filtering to converters
would condition on a treatment-affected outcome and bias the comparison.
Duplicate users, duplicate or missing column headers, unknown arm labels,
malformed amounts and NaN/infinity are rejected. The example does not silently
drop bad rows or impute missing covariates.

## What the result means

The checked-in seed-42 data gives this output (rounded by the script):

| Analysis | Treatment - control (USD/user) | 95% CI | p-value |
| --- | ---: | ---: | ---: |
| Unadjusted context | -0.875 | [-3.652, +1.902] | 0.5361 |
| CUPED estimate (primary) | -1.538 | [-3.175, +0.099] | 0.0655 |

The split is 200/200, giving an SRM p-value of 1.0000. The CUPED adjustment
reduces the estimated variance of the comparison by **65.2%**. Its interval is
narrower but still includes zero, so this sample does not establish a direction
of effect.

The simulator adds $0.50 to treatment revenue, plus independent noise, and
rounds amounts to cents. This small sample still estimates a negative effect:
one sample's confidence interval can miss the generating effect. The library's
simulation tests cover calibration.

### Caveats

- Not rejecting zero does not show the arms are equivalent or rule out a
  business-relevant loss.
- The unadjusted row is context. Do not switch to it when its result looks
  better.
- The covariate must be measured **before treatment can affect it**. A column
  named `pre_revenue_usd` does not prove that; in a real export, check time
  boundaries and joins. September revenue, post-assignment sessions or a later
  snapshot are unsuitable substitutes.
- CUPED here estimates one pooled adjustment coefficient and treats it as
  fixed, matching the library's [methodology](METHODOLOGY.md).
- Passing SRM only means this check found no allocation problem. It does not
  validate assignment, logging, complete outcomes or covariate provenance.
- Do not rerun this fixed-horizon test to pick a stopping time. If interim
  monitoring is part of the plan, use the
  [sequential methods](../README.md#peeking-simulation) instead.

## See the gates work

Before the outcome window closes, the command prints `BLOCKED`, suppresses all
effect estimates, intervals and effect p-values, and exits 1:

```bash
uv run python examples/experiment_readout.py --as-of 2026-09-14
```

To simulate a broken assignment/logging export, keep all controls but only 20
treatment users:

```bash
uv run python - <<'PY'
import csv
from pathlib import Path

source = Path("examples/data/checkout.csv")
with source.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    fields = reader.fieldnames
    rows = list(reader)
control = [row for row in rows if row["group"] == "control"]
treatment = [row for row in rows if row["group"] == "treatment"][:20]
with Path("/tmp/checkout-broken.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(control + treatment)
PY
uv run python examples/experiment_readout.py --csv /tmp/checkout-broken.csv --as-of 2026-09-15
```

The SRM gate blocks inference before CUPED runs. Investigate the source of the
missing users; changing the planned split after observing the data or filtering
the other arm to match does not repair randomization. Balanced enrollment below
200 users per arm also blocks the example's final analysis.

Exit codes are 0 for a completed, ungated readout; 1 for a blocked readout; 2
for invalid input or an I/O error.

## Reproduce the data

```bash
uv run python examples/experiment_readout.py --generate-data /tmp/checkout-seed42.csv
cmp examples/data/checkout.csv /tmp/checkout-seed42.csv
uv run pytest tests/test_experiment_readout.py -q
```

Regeneration uses NumPy's seeded generator and the versions in `uv.lock`.
No network, database, plotting dependency or external data is needed after
installation. The test suite checks that generation reproduces the committed
CSV byte for byte, the readout handles the full CSV, and failed sanity/horizon
gates never return effect inference.

# Pre-registered prediction for the past-threshold sweep

Committed **before** the measurements it predicts existed. `git log` on this
file is the timestamp; run E's data lands in a later commit.

## Why bother

ERRATA A7 argues that the slowdown is ordinary speculative-decoding economics —
the per-round cost of drafting, times how much is drafted, divided by how much
is accepted — and that no MoE-specific mechanism is needed. That is an
explanatory claim, and an explanatory claim that cannot fail is worthless. So:
fit the model on the draft lengths already measured, extrapolate it past the
threshold MoESD's coverage argument is about, and write the numbers down first.

## The model

Fitted on `n_max` ∈ {1, 2, 4, 8, 16, 32} from
`data/C_master_matrix_think_on/`, five repeats each:

```
ms_per_token = 27.56 * (rounds per generated token)
             +  5.54 * (draft tokens per generated token)
             + 12.83
```

with `rounds per generated token = 1 / (acceptance × n_max + 1)`.

**R² = 0.9956** on six points and three parameters, so three degrees of
freedom — impressive but not overwhelming. Honest limits:

- Draft volume alone already reaches R² = 0.9471. The per-round term takes it
  to 0.9956, removing 91 % of the remaining error, and it is what makes the
  small-`n_max` end fit at all — `n_max` 1 has the *lowest* draft volume and
  the *second-highest* cost, which volume alone cannot express.
- Leave-one-out mean absolute error is 1.63 ms/token against values spanning
  28.1–57.9, worst at the edge (`n_max` 32, +4.99).
- The fitted intercept, 12.83 ms/token, is **58 % above** the measured
  no-speculation cost of 8.11. The coefficients are not physical constants and
  must not be read as one.
- The two regressors correlate at r = −0.665 (VIF 1.79), so they are separable,
  but not cleanly.

Acceptance and draft volume are extrapolated by a power law in `n_max`, which
both follow closely over the fitted range.

## The prediction

| `n_max` | predicted acceptance | predicted draft/gen | predicted ms/token | **predicted tok/s** |
|---:|---:|---:|---:|---:|
| 64 | 6.4 % | 10.20 | 74.76 | **13.4** |
| 96 | 4.9 % | 13.86 | 94.42 | **10.6** |
| 128 | 4.1 % | 17.24 | 112.67 | **8.9** |

Baseline on this host and binary is ~123 tok/s.

## What would falsify what

The model has **no term for expert coverage**. It says cost keeps rising
monotonically with draft length, for ever. MoESD's expected-coverage argument
says something different: with 256 routed experts and top-8 routing, a draft of
95 tokens is where the expected fraction of routed experts touched passes 95 %,
and beyond that the union of expert slices should stop growing, so the cost per
drafted token should amortise.

The sweep reaches 63.8 % coverage at `n_max` 32, 86.9 % at 64, **95.3 % at 96**
and 98.3 % at 128 — so it crosses the threshold.

- **If the measurements land near the table above**, the coverage argument gains
  no support on this hardware: a model that knows nothing about experts
  predicted the result.
- **If they substantially beat the table at 96 and 128**, the model is missing
  physics that the coverage argument supplies, and ERRATA A7 needs weakening.

Either way this is answered by measurement rather than by assertion, which is
the thing the audited version of this repository failed to do.

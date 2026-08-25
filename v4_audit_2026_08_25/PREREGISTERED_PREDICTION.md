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


---

# Outcome

Measured after the fact, `data/E_past_threshold/`, three repeats per arm,
run-to-run SD 0.03–0.17 tok/s.

| `n_max` | predicted tok/s | measured tok/s | error |
|---:|---:|---:|---:|
| 64 | 13.4 | 12.4 | −7.5 % |
| 96 | 10.6 | 10.0 | −5.7 % |
| 128 | 8.9 | 8.9 | **0.0 %** |

Every measurement lands at or slightly below the registered number — reality is
never *better* than the coverage-blind model expected.

**But the agreement is partly luck, and saying so matters more than the win.**
Feeding the model its *measured* inputs instead of the power-law extrapolation
makes it over-predict cost by +11.9 %, +15.9 % and +24.8 %. The extrapolation
under-predicted draft volume and over-predicted acceptance, and those two errors
happened to cancel the model's own bias. A prediction that is right for
partially wrong reasons is worth recording as exactly that.

## What the data settled instead

Chasing that discrepancy produced a cleaner result than the original model. Over
the whole sweep, pooling every repeat:

```
ms per generated token = 27.00 + 4.040 × (draft tokens per generated token)
R² = 0.99303,  n_max = 1 … 128
```

One regressor, no expert term, 99.3 % of the variance across **3.1 % → 98.3 %**
expected routed-expert coverage. The slope reads sensibly: 4.04 ms per
speculated position against a measured 7.87 ms no-speculation decode step, so
each drafted token costs about half a target step — an autoregressive 0.8 B
drafter plus its share of the verify pass.

**Step in the residuals at the 95.3 % coverage point: −0.39 percentage points.**
Mean residual is −0.27 % below it and −0.67 % at or above it. There is no knee,
no break, and nothing for a coverage threshold to explain.

End-to-end throughput, which is what a user actually gets: 31.1, 34.2, **35.6**,
32.1, 23.7, 17.3, 12.4, 10.0, 8.9 tok/s against a 123.4 baseline. It peaks at
`n_max` 4 and declines monotonically after, straight through the threshold.

## Two hypotheses this killed along the way

**Cost per drafted token amortises, so MoESD gains support.** It does fall, 48.4
ms at `n_max` 1 to 4.8 at 128, and fitting `a + b/n` on the sub-threshold points
alone puts the supra-threshold ones 24–34 % *below* the curve, which looks like
a mechanism. It is not: that subset is dominated by `n_max` 1, 2 and 4, where
marginal cost is 6–10× the asymptote, and it drags the intercept up. Fit all
nine points and the effect goes away.

**Speculative state checkpointing dominates the cost.** The server logs make
this tempting — 1639 checkpoints at 101.3 MiB each for a single 300-token
request at `n_max` 1, and `the context does not support partial sequence
removal` still printed on post-merge master. But checkpoint traffic per
generated token *falls* from 55.4 MiB at `n_max` 1 to 24.9 at 128 while cost
*rises* from 32 to 113 ms. The correlation is −0.52 and the implied bandwidth is
negative. Refuted by its own test.

## Limits

Nine points and two parameters. Residuals form an arc of about ±11 %, so the law
is an approximation, not a physical identity — and the intercept, 27.00 ms, is
3.4× the measured no-speculation step, so it absorbs whatever per-round cost the
single regressor cannot express. One host, one target, one drafter, thinking on.
The claim is bounded to that.

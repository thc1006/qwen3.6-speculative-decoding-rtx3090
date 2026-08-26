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
  to 0.9956, removing 92 % of the remaining error, and it is what makes the
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
run-to-run SD 0.05, 0.16 and 0.11 tok/s at `n_max` 64, 96 and 128.

Every figure below is recomputed by `analysis/past_threshold_fit.py` from the
committed data and asserted in `analysis/verify_claims.py`. It was not, until
2026-08-27: this was the one analysis in the repository with no code path, and
recomputing it found **eight wrong figures** — marked where each appears — and
two more that looked wrong and were not, because the conventions behind them
had never been written down. The prediction above, committed before the data
existed, reproduces to the last decimal.

| `n_max` | registered tok/s | measured tok/s | error |
|---:|---:|---:|---:|
| 64 | 13.4 | 12.38 | −7.6 % |
| 96 | 10.6 | 9.98 | −5.8 % |
| 128 | 8.9 | 8.85 | **−0.6 %** |

Every measurement lands below the registered number — reality is never *better*
than the coverage-blind model expected. The error is the measured rate against
the number this file committed in advance; **until 2026-08-27 this column read
−7.5 %, −5.7 % and 0.0 %**, computed from the rounded display values rather than
the measurements, which made the third row look exact when it is 0.6 % low.

**But the agreement is partly luck, and saying so matters more than the win.**
Feeding the model its *measured* inputs instead of the power-law extrapolation
makes it over-predict cost by +12.3 %, +18.1 % and +25.2 %. The extrapolation
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
speculated position against a measured 8.11 ms no-speculation decode step, so
each drafted token costs almost exactly half a target step — an autoregressive
0.8 B drafter plus its share of the verify pass. That step read **7.87 ms**
until 2026-08-27, which is repeat 0 of the five the model is fitted on.

**Step in the residuals at the 95.3 % coverage point: −0.39 percentage points.**
Mean residual is −0.27 % below it and −0.67 % at or above it. Residual is
`(measured − predicted) / predicted`, the column
[A10](../ERRATA.md#a10-the-single-regressor-law-is-falsified-out-of-sample-and-p_min-is-the-lever-that-matters)
publishes, so a negative residual is the law over-predicting cost; that
convention was not stated here until 2026-08-27, and reading the step in the
opposite one gives −0.13. Either way it is two orders of magnitude below the
±11 % scatter it would have to rise out of. There is no knee, no break, and
nothing for a coverage threshold to explain.

Pooled decode rate: 31.1, 34.2, **35.6**, 32.1, 23.7, 17.3, 12.4, 10.0, 8.9
tok/s against a 123.4 baseline. It peaks at `n_max` 4 and declines
monotonically after, straight through the threshold. End to end against the
wall clock, which is what a user actually gets, the same nine are 30.2, 33.2,
**34.5**, 31.1, 23.2, 17.0, 12.2, 9.9, 8.8 against a **110.8** baseline — the
shape is identical, and the baseline loses the most because a 300-token
request at 123 tok/s spends the largest share of its wall clock outside decode.
Until 2026-08-27 the decode figures were published under the end-to-end
label.

## Two hypotheses this killed along the way

**Cost per drafted token amortises, so MoESD gains support.** It does fall, 48.4
ms at `n_max` 1 to 4.8 at 128 — cost over the no-speculation step, divided by
draft volume — and fitting `a + b/n` on the six fitted points alone, `n_max` 1
to 32, puts the three that came after 24–34 % *below* the curve (−24.5 %,
−29.4 %, −34.1 %), which looks like a mechanism. It is not: that subset is
dominated by `n_max` 1, 2 and 4, where marginal cost is 6–10× the asymptote,
and it drags the intercept up. Fit all nine points and the same three sit
−12.5 %, −17.7 % and −22.9 % off it, and the effect goes away entirely once the
curve is allowed to see them.

**Speculative state checkpointing dominates the cost.** The server logs make
this tempting — 1639 checkpoints at a server-reported 82.079 MiB each in one
`n_max` 1 arm-run, and `the context does not support partial sequence removal`
still printed on post-merge master. But checkpoint traffic per generated token
*falls* from 44.8 MiB at `n_max` 1 to 20.2 at 128 while cost *rises* from 32 to
113 ms. The correlation is −0.52 and the implied bandwidth is negative. Refuted
by its own test.

Two numbers in that paragraph were wrong until 2026-08-27. The 1639 checkpoints
are one arm-run of **ten** 300-token requests, 163.9 per request, not one
request. And the traffic figures read 55.4 and 24.9 because they were computed
at 101.3 MiB per checkpoint — the sum
[A12](../ERRATA.md#a12-what-the-checkpoint-path-costs-measured-with-timers-in-the-source)
withdrew, 82.079 plus the 19.266 draft component it already contains. The
correlation is scale-free and does not move, so the refutation stands on the
same evidence it always did.

## Limits

Nine points and two parameters. Residuals form an arc of about ±11 %, so the law
is an approximation, not a physical identity — and the intercept, 27.00 ms, is
3.3× the measured no-speculation step, so it absorbs whatever per-round cost the
single regressor cannot express. One host, one target, one drafter, thinking on.
The claim is bounded to that.

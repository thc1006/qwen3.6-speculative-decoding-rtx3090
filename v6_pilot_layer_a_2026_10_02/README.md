# Plan Z layer A, a pilot on a second card, 2026-10-02

A **pilot**, and the directory name says so because the card is not the one every
published round was measured on. Taken while the `3090` bench host was offline, on
the dev box's own RTX 3090 — same GA102, same GDDR6X, a different cooler and a
420 W limit against the bench host's 350.

Pre-registered in
[`../v4_audit_2026_08_25/PROSPECTIVE_PLAN_Z_MEMORY_STATE.md`](../v4_audit_2026_08_25/PROSPECTIVE_PLAN_Z_MEMORY_STATE.md)
before it ran. Every figure below is derived by
[`../analysis/layer_a_slope.py`](../analysis/layer_a_slope.py), which also reads
this file back and fails if a cell here is not what the data gives.

## The question

Does achieved GDDR6X bandwidth move with thermal state at **locked** clocks? ERRATA
A16 is a 3.93 % step in decode rate with nothing recorded distinguishing its two
levels, and memory-subsystem thermal state is one of the three candidates it names.

## What was measured

Core clock and memory clock locked at 1800 and 9501 MHz, held for **all 2358**
telemetry samples at **sd 0.000** — `-lgc` is a request and the power cap outranks
it, so holding was checked against the trace rather than asserted. 2100 and 1950 MHz
were tried first and did not hold; the hold test stepped down.

One streaming kernel over two four gibibyte buffers, nothing else on the card.
An ascending phase from a cooled card to its plateau, then a descending phase of
short windows with idle gaps, which crosses the same temperatures in the other
direction at different elapsed times.

| | |
|---|---|
| the thermal range | 51.6 to 69.0 °C |
| R, the bandwidth change across it | 0.014 % |
| the slope B | -0.00099 % per °C |
| the 95 % upper bound on its magnitude | 0.00140 % per °C |
| the noise floor, 64 windows at the plateau | 0.0123 % |
| the shared bins the control used | 9 |
| the mean phase offset in them | 0.022 % |

## The reading

Plan Z's pre-registered rule excludes the hypothesis if R is below A16's 3.93 %
step. It is **0.014 %**.

Put against A16's own numbers rather than an invented divergence: A16's two levels
differ by 1.67 °C of core temperature, so for memory thermal state to produce a
3.93 % step through bandwidth, with elasticity at the fully memory-bound ceiling of
one, the sensitivity would have to be **2.35 % per °C**. That is **1,678** times the
upper bound measured here. Equivalently, at that upper bound the memory would have
to diverge by thousands of degrees between the two levels.

And the step is 319 noise floors from zero, so the instrument is not the limit.

**The faster of A16's two levels is also the hotter one**, by 1.67 °C, which is the
opposite of what a thermal explanation predicts. This measurement says the
bandwidth channel cannot carry the step; A16's own thermal record says the sign is
wrong as well.

## What this does not establish

- **It is not the bench host's card.** A null here is evidence about GA102 with
  GDDR6X as a class. It is not `GPU-f9db9841`'s number: board designs differ in how
  the back-side memory modules are cooled, so the onset temperature can differ.
- **The thermal index is CORE temperature.** The memory register is unreadable on
  this host: `CONFIG_IO_STRICT_DEVMEM=y` makes the kernel refuse to map an IO region
  a driver has claimed, and the `nvidia` driver claims BAR0.
- **The control ran but cannot attribute.** The two phases agree to 0.022 % at
  matched temperatures, and they are two separate invocations of the load: a
  constant offset is what run-to-run variation looks like, and seven invocations at
  a flat plateau spread 0.013 %, about the same size. A balanced test the same day
  also refuted the obvious instrument explanation — five and ten second windows
  differ by 0.006 % at a flat plateau, inside the scatter. Attribution needs
  replicated pairs, or both phases inside one process.
- **It bounds the bandwidth channel only.** A thermal effect acting through memory
  LATENCY rather than streaming throughput is untouched by this: the load is a pure
  stream, and a decode step's KV-cache access is scattered, where row-buffer
  behaviour and latency matter. That is a different hypothesis and this does not
  speak to it.
- The range is 17.4 °C, not the whole span the card can reach. Below about 51 °C the
  load heats the card faster than a window can resolve, so starting colder adds
  nothing.

## What is here

`data/pilot_layer_a_pilotB_20261002_171035/` holds the manifest, the calibration at
five core clocks, the three hold tests with their verdicts, both phases, and the
one second telemetry trace the clocks were checked against.

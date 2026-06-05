# Oscillator Hunt Sheet

Worksheet for identifying oscillators in a new domain.

## Instructions

For each candidate cyclic process, fill in one row. Not all candidates will become oscillators -- filtering happens after.

## Template

| # | Process | Frequency (Hz) | Channel | Observable? | Data Source | Layer | Keep? |
|---|---------|----------------|---------|-------------|-------------|-------|-------|
| 1 | | | P / I / S | direct / inferred | | micro / meso / macro | Y / N |
| 2 | | | | | | | |
| 3 | | | | | | | |
| 4 | | | | | | | |
| 5 | | | | | | | |
| 6 | | | | | | | |
| 7 | | | | | | | |
| 8 | | | | | | | |

## Questions per Candidate

1. **What cycles?** Describe the repeating process. What goes up and comes back down?
2. **What frequency?** Rough estimate in Hz. Order of magnitude is enough.
3. **Continuous or discrete?** Continuous waveform (P), event stream (I), or state sequence (S)?
4. **Observable or latent?** Can you measure it directly from a sensor/API/log? Or must it be inferred from proxy signals?
5. **Data source?** Sensor name, API endpoint, log field, metric name.
6. **Timescale tier?** Does it belong to the fastest (micro), middle (meso), or slowest (macro) layer?

## Filtering Criteria

Drop candidates that are:

- Not actually cyclic (monotonic trends, one-shot events)
- Unobservable with no viable proxy
- Redundant with another candidate (pick the more direct measurement)
- Outside the frequency range of interest (> 100x faster or slower than the control loop)

## Example: Cloud Queue System

| # | Process | Frequency | Channel | Observable? | Data Source | Layer | Keep? |
|---|---------|-----------|---------|-------------|-------------|-------|-------|
| 1 | Queue depth oscillation | 0.5 Hz | P | direct | metrics API | micro | Y |
| 2 | Request arrival rate | 10 Hz | I | direct | access log timestamps | micro | Y |
| 3 | Retry burst cycle | 0.1 Hz | I | direct | retry log timestamps | micro | Y |
| 4 | P99 latency oscillation | 0.05 Hz | P | direct | latency histogram | meso | Y |
| 5 | Error rate cycle | 0.01 Hz | P | direct | error counter | macro | Y |
| 6 | Deploy cadence | 0.001 Hz | S | direct | deploy events | macro | N (too slow) |

## Why this worksheet drives better bindings

The value of this sheet is not in the table itself; it is in forcing a disciplined
pre-binding filter. Most unstable onboarding issues come from treating non-oscillatory
signals as oscillators. This worksheet makes those assumptions explicit before YAML
or domainpack structure is authored.

Use the sheet in this order:

1. collect a complete candidate list,
2. reject non-cyclic signals,
3. assign channel tags conservatively,
4. keep candidates that have direct observables and plausible frequencies.

That sequence keeps the first production draft smaller and easier to validate.

## Practical review criteria

Teams should only pass a candidate to binding authoring when:

- it has a defendable frequency estimate,
- channel assignment is consistent across related signals,
- and there is at least one viable raw data source path for every kept row.

Everything else stays in discovery and should be revisited only after a concrete
measurement or transformation step is available.

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

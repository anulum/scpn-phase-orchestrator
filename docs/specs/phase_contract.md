# Phase Contract

Every oscillator produces a `PhaseState` with the following fields:

| Field | Type | Contract |
|-------|------|----------|
| `theta` | float | Phase in `[0, 2*pi)`. Wrapping enforced by extractor. |
| `omega` | float | Instantaneous angular frequency, rad/s. May be negative. |
| `amplitude` | float | Signal amplitude. Channel-specific meaning. |
| `quality` | float | Confidence in `[0, 1]`. 0 = unreliable. 1 = perfect. |
| `channel` | str | `"P"`, `"I"`, or `"S"`. Set by extractor. |
| `node_id` | str | Unique identifier for this oscillator instance. |

## Extraction Requirements

1. `theta` MUST be in `[0, 2*pi)` after extraction. Extractors apply `% TWO_PI`.
2. `omega` is computed from the signal, not assumed constant. Physical: phase gradient. Informational: median inter-event frequency. Symbolic: phase difference / dt.
3. `quality` reflects measurement reliability, not system health. Low quality means the phase estimate is uncertain.
4. `node_id` must match an entry in the binding spec `oscillator_ids`.

## Quality Gating

- `PhaseQualityScorer.downweight_mask(min_quality=0.3)` returns weight 0 for states below threshold.
- `detect_collapse(threshold=0.1)` returns True when majority of states are below 0.1.
- Down-weighted oscillators still participate in UPDE integration but with reduced coupling effect (multiply Knm row by quality weight).

## Phase Wrapping

The UPDE engine wraps output phases via `% TWO_PI` after every integration step. Phase differences use the standard `sin(theta_j - theta_i)` form, which handles wrapping implicitly.

## References

- **[gabor1946]** D. Gabor (1946). Theory of communication. *J. IEE* 93, 429–457. — Analytic signal underlying P-channel phase extraction.
- **[pikovsky2001]** A. Pikovsky, M. Rosenblum & J. Kurths (2001). *Synchronization: A Universal Concept in Nonlinear Sciences*. Cambridge UP. — Instantaneous phase and frequency conventions.

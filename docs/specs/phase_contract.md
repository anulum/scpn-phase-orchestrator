# Phase Contract

The Phase Contract defines the universal interface between signal sources
(oscillators) and the UPDE integration engine. Every oscillator — physical,
informational, or symbolic — must produce a `PhaseState` that satisfies
this contract. Violations break the coupling pipeline downstream.

## PhaseState Fields

| Field | Type | Contract |
|-------|------|----------|
| `theta` | `float` | Phase in `[0, 2π)`. Wrapping enforced by extractor. |
| `omega` | `float` | Instantaneous angular frequency, rad/s. May be negative. |
| `amplitude` | `float` | Signal amplitude. Channel-specific meaning (see below). |
| `quality` | `float` | Confidence in `[0, 1]`. 0 = unreliable. 1 = perfect. |
| `channel` | `str` | `"P"`, `"I"`, or `"S"`. Set by extractor class. |
| `node_id` | `str` | Unique identifier for this oscillator instance. |

### Field Semantics by Channel

**Physical (`"P"`)** — theta is the Hilbert-transform analytic phase of a
continuous signal (EEG, MEG, accelerometer, voltage). Omega is the
instantaneous frequency from the phase gradient. Amplitude is the
analytic signal envelope.

**Informational (`"I"`)** — theta maps discrete events (spikes, network
packets, heartbeats) to a continuous phase on `[0, 2π)` via ring
projection: `theta = 2π * (t - t_prev) / (t_next - t_prev)`. Omega is
the median inter-event frequency. Amplitude is the event rate.

**Symbolic (`"S"`)** — theta encodes a discrete state transition as
`2π * state_index / n_states`. Omega is computed as phase difference
over dt between consecutive observations. Amplitude is the confidence
or probability of the current state assignment.

## Extraction Requirements

1. **Wrapping** — `theta` MUST be in `[0, 2π)` after extraction.
   Extractors apply `theta % TWO_PI`. The UPDE engine re-wraps after
   every integration step, so a transiently out-of-range theta from
   the extractor will not propagate, but correct wrapping at source
   prevents coupling distortions during the first step.

2. **Omega computation** — `omega` is computed from the signal, not
   assumed constant. Each channel type has a specific method:
   - P-channel: phase gradient via `d(unwrapped_theta)/dt`.
   - I-channel: median of `2π / IEI` where IEI is inter-event interval.
   - S-channel: `(theta_current - theta_previous) / dt`.
   Negative omega is valid and indicates clockwise rotation (e.g.,
   retrograde propagation in cortical waves).

3. **Quality** — reflects measurement reliability, not system health.
   Low quality means the phase estimate is uncertain. Causes include:
   - Low SNR in the P-channel signal.
   - Irregular event timing in the I-channel (high coefficient of
     variation of inter-event intervals).
   - Ambiguous state assignment in the S-channel.
   Quality is always in `[0, 1]`. Extractors must clamp to this range.

4. **Node identity** — `node_id` must match an entry in the binding
   spec `oscillator_ids`. Mismatched IDs cause `BindingLoadError` at
   pipeline setup. Format is free-form string; convention is
   `"{channel}_{source}_{index}"` (e.g., `"P_eeg_Cz"`, `"I_spike_0"`).

## Quality Gating

The `PhaseQualityScorer` provides three operations that downstream
components (coupling builder, supervisor) use to handle unreliable phases:

### `score(phase_states) -> float`

Weighted average quality across all phase states. Weights are the
amplitudes (floored at 1e-12 to prevent division by zero). Returns a
scalar in `[0, 1]` representing global extraction confidence.

### `downweight_mask(phase_states, min_quality=0.3) -> NDArray`

Returns an `(N,)` weight array where entries below `min_quality` are
set to 0.0 and entries at or above retain their original quality value.
This mask multiplies into the coupling matrix row-wise:

```python
weights = scorer.downweight_mask(states, min_quality=0.3)
knm_effective = knm * weights[:, None]  # row-wise scaling
```

The effect: oscillators with uncertain phase estimates decouple from the
network rather than injecting noise. The threshold 0.3 is empirical;
see `docs/ASSUMPTIONS.md` § Quality Gating for calibration details.

### `detect_collapse(phase_states, threshold=0.1) -> bool`

Returns `True` when more than half of the phase states have quality
below `threshold`. This triggers the supervisor to enter `Degraded`
regime and reduce coupling strength to prevent divergence.

Collapse detection is a hard safety boundary. The threshold 0.1 means
fewer than 10% of samples produce a usable phase estimate — the
oscillator is effectively offline.

## Phase Wrapping in the UPDE Engine

The UPDE engine wraps output phases via `theta % TWO_PI` after every
integration step. Phase differences in the coupling term use the
standard Kuramoto form `sin(theta_j - theta_i)`, which handles
wrapping implicitly because sine is `2π`-periodic.

For adaptive-step methods (RK45), the error estimate operates on
unwrapped phases to avoid discontinuities at the `0/2π` boundary.
The 5th-order solution is wrapped only after step acceptance.

### Wrapping and Winding Numbers

Wrapping destroys information about cumulative rotation. The
`monitor.winding` module reconstructs winding numbers by summing
wrapped phase increments:

```python
dtheta = diff(phases_history, axis=0)
dtheta_wrapped = (dtheta + pi) % TWO_PI - pi  # wrap to [-pi, pi]
cumulative = sum(dtheta_wrapped, axis=0)
winding = floor(cumulative / TWO_PI)
```

This is exact for timesteps where the true phase increment is less
than `pi` (satisfied when `dt * max(|omega|) < pi`).

## Phase Lag (Alpha)

Phase lags `alpha_ij` shift the coupling interaction:

```
coupling_ij = K_ij * sin(theta_j - theta_i - alpha_ij)
```

The lag rotates the preferred phase relationship away from perfect
synchrony. At `alpha=0`, oscillators prefer to lock in-phase. At
`alpha=pi/2`, they prefer a quarter-cycle lead/lag. The `LagModel`
class manages per-pair or per-layer lag matrices.

Lags are NOT part of the PhaseState — they are coupling parameters.
But they interact with the phase contract because the coupling term
assumes theta values in `[0, 2π)`.

## External Drive (Zeta, Psi)

The external drive term `zeta * sin(Psi - theta_i)` pulls all
oscillators towards a global target phase `Psi` with strength `zeta`.
This is how the Active Inference controller acts on the network:

- `zeta > 0` with `Psi` aligned to current mean phase → reinforce sync.
- `zeta > 0` with `Psi` anti-aligned → suppress sync (push apart).
- `zeta = 0` → no external drive, free-running Kuramoto dynamics.

The PhaseState does not carry `zeta` or `Psi` — these are control
inputs set by the supervisor.

## Invariants

The following invariants hold at all times in a correctly wired pipeline:

1. **Range**: `0 <= theta < 2*pi` for every PhaseState emitted by an
   extractor and for every phase in the UPDE state vector.

2. **Finite**: `theta`, `omega`, `amplitude` are finite floats. NaN or
   Inf in any field causes `IntegrationDiverged` error.

3. **Quality range**: `0 <= quality <= 1`.

4. **Channel validity**: `channel in {"P", "I", "S"}`.

5. **Node binding**: `node_id` exists in the active binding spec.

6. **Determinism**: given the same signal and sample rate, an extractor
   produces the same PhaseState (no hidden random state).

## Rust FFI Considerations

When the `spo_kernel` Rust FFI is available, the UPDE integration and
order parameter computations run in Rust. The Rust code enforces the
same wrapping (`rem_euclid(TAU)`) and NaN/Inf checks. Phase arrays
cross the FFI boundary as contiguous `f64` numpy arrays — the Rust
side receives a borrowed slice, computes in-place, and returns. No
copying occurs for the integration hot path.

The `PlasticityModel` in Rust also respects the phase contract: it
reads `theta_j - theta_i` from the phase array and computes
`cos(theta_j - theta_i)` for eligibility traces. The coupling matrix
`K_nm` is updated in-place after each integration step.

## Validation and Error Handling

### Input Validation in the UPDE Engine

Before each integration step, the engine validates all input arrays:

```python
# Pseudocode — actual implementation in upde.py and sparse_upde.rs
for theta in phases:
    assert isfinite(theta)  # NaN/Inf → IntegrationDiverged
for omega in omegas:
    assert isfinite(omega)
for k in knm_values:
    assert isfinite(k)
```

If any value is non-finite, the engine raises `IntegrationDiverged`
immediately rather than propagating corrupted state. This is a hard
stop — the supervisor must intervene (reduce coupling, switch to
recovery regime) before integration can resume.

### Dimension Checks

The engine verifies array dimensions at each step:

- `len(phases) == N` (number of oscillators)
- `len(omegas) == N`
- `len(knm) == N * N` (dense) or `len(row_ptr) == N + 1` (sparse)
- `len(alpha) == len(knm)` (same sparsity pattern)

Dimension mismatches raise `InvalidDimension`. This catches binding
spec changes that were not propagated to the coupling builder.

### PhaseState Validation at Extraction

Extractors should validate their output before returning:

```python
def _validate(state: PhaseState) -> None:
    assert 0.0 <= state.theta < TWO_PI
    assert math.isfinite(state.omega)
    assert math.isfinite(state.amplitude)
    assert 0.0 <= state.quality <= 1.0
    assert state.channel in ("P", "I", "S")
```

The pipeline does not currently enforce this at the boundary (for
performance), so extractors bear responsibility. The
`test_phase_contract.py` test suite verifies contract compliance for
all three built-in extractors.

## Pipeline Integration Example

A minimal pipeline exercising the full phase contract:

```python
from scpn_phase_orchestrator import (
    UPDEEngine, CouplingBuilder, PhaseQualityScorer,
    SupervisorPolicy, BoundaryObserver,
)
from scpn_phase_orchestrator.oscillators import (
    PhysicalExtractor, InformationalExtractor,
)
from scpn_phase_orchestrator.binding import load_binding_spec

# 1. Load binding spec (maps oscillator IDs to coupling topology)
spec = load_binding_spec("path/to/binding.yaml")

# 2. Build coupling matrix from binding spec
builder = CouplingBuilder(n=spec.n_oscillators)
knm = builder.build(spec)

# 3. Extract phases from raw signals
p_extractor = PhysicalExtractor()
i_extractor = InformationalExtractor()
p_states = p_extractor.extract(eeg_signal, sample_rate=256.0)
i_states = i_extractor.extract(spike_train, sample_rate=1000.0)
all_states = p_states + i_states

# 4. Quality gating
scorer = PhaseQualityScorer()
if scorer.detect_collapse(all_states):
    # Enter degraded regime — too many unreliable phases
    ...
weights = scorer.downweight_mask(all_states, min_quality=0.3)

# 5. Integrate one step
engine = UPDEEngine(n=len(all_states))
phases = [s.theta for s in all_states]
omegas = [s.omega for s in all_states]
engine.step(phases, omegas, knm * weights[:, None], zeta=0.0, psi=0.0)
```

Each step in this pipeline depends on the phase contract being
satisfied. If an extractor emits `theta=NaN`, the engine catches it
at step 5. If `node_id` is wrong, the binding loader catches it at
step 1. If quality is out of range, the scorer produces incorrect
masks at step 4.

## Sheaf Extension

The `SheafUPDEEngine` extends the phase contract to vector-valued
phases. Instead of a scalar `theta` per oscillator, each oscillator
carries a `D`-dimensional phase vector `theta_{i,d}` for
`d = 0, ..., D-1`. The contract generalises:

- **Range**: each component `theta_{i,d}` in `[0, 2pi)`.
- **Coupling**: restriction maps `B_ij` are `D x D` matrices replacing
  scalar `K_ij`. The coupling term becomes
  `sum_j sum_k B_ij^{dk} sin(theta_{j,k} - theta_{i,d})`.
- **External drive**: `psi` becomes a `D`-vector.
- **Order parameter**: computed per component or as the mean over
  components.

All other invariants (finite, quality range, node binding) apply
unchanged. The sheaf extension is experimental and documented further
in `docs/concepts/sheaf_topology.md`.

## Stuart-Landau Extension

The `StuartLandauEngine` adds amplitude dynamics:

```
dr_i/dt = (mu_i - r_i^2) * r_i + epsilon * sum_j K^r_ij * r_j * cos(theta_j - theta_i)
```

The `amplitude` field in `PhaseState` maps directly to `r_i`. The
phase contract for theta, omega, quality, and channel applies unchanged.
The additional invariant is `r_i >= 0` (enforced by clamping after
each step).

## Testing the Phase Contract

The test suite `tests/test_phase_contract.py` verifies:

1. All three extractors produce `theta` in `[0, 2pi)` for synthetic
   signals with known phase.
2. Quality is in `[0, 1]` under normal and adversarial inputs (zero
   signal, constant signal, signal with NaN samples).
3. Omega sign matches expected rotation direction.
4. Node IDs match the binding spec after a full extraction round.
5. The UPDE engine rejects NaN/Inf in any input field.
6. Quality gating correctly zeros coupling for unreliable oscillators.
7. Winding numbers computed from a trajectory match analytical
   predictions for constant-frequency oscillators.

Property-based tests (Hypothesis) additionally verify:

- For any array of floats passed through wrapping, the result is in
  `[0, 2pi)`.
- For any sequence of wrapped phase increments, the reconstructed
  winding number equals `floor(total_unwrapped_phase / 2pi)`.

## References

- **[gabor1946]** D. Gabor (1946). Theory of communication. *J. IEE* 93, 429-457. — Analytic signal underlying P-channel phase extraction.
- **[pikovsky2001]** A. Pikovsky, M. Rosenblum & J. Kurths (2001). *Synchronization: A Universal Concept in Nonlinear Sciences*. Cambridge UP. — Instantaneous phase and frequency conventions.
- **[kuramoto1984]** Y. Kuramoto (1984). *Chemical Oscillations, Waves, and Turbulence*. Springer. — Phase coupling with `sin(theta_j - theta_i - alpha)` form.
- **[friston2010]** K. J. Friston (2010). The free-energy principle: a unified brain theory? *Nature Rev. Neuroscience* 11, 127-138. — Active Inference framework for zeta/Psi control.
- **[strogatz2000]** S. H. Strogatz (2000). From Kuramoto to Crawford: exploring the onset of synchronization in populations of coupled oscillators. *Physica D* 143, 1-20. — Winding numbers and phase coherence.

# Memory Imprint Model

> **Provenance:** The imprint model is original to SCPN Phase Orchestrator.
> The exponential forgetting form is standard in memory/learning models
> (cf. Ebbinghaus 1885), but the specific application to coupling-matrix
> modulation is novel. Parameters (`decay_rate`, `saturation`) require
> domain-specific calibration — see [ASSUMPTIONS.md](../ASSUMPTIONS.md).

## Purpose

The imprint model tracks cumulative exposure history per oscillator.
The imprint vector `M_k` modulates coupling strength and phase lag,
encoding long-term adaptation without requiring explicit retraining or
configuration changes. It is the mechanism by which the system learns
from its own dynamics.

Use cases:

- **Habituation**: repeated stimulation reduces coupling sensitivity
  (imprint decays back, but leaves a residual offset).
- **Sensitisation**: prolonged high-coherence states strengthen coupling
  pathways (positive feedback through K modulation).
- **Drift adaptation**: slow environmental changes are absorbed by the
  imprint, keeping the system tuned without supervisor intervention.
- **Learning**: in BCI (Brain-Computer Interface) applications, the
  imprint encodes user-specific neural patterns over sessions.

## Dynamics

### Update Rule

```
M_k(t + dt) = M_k(t) * exp(-decay_rate * dt) + exposure_k * dt
M_k = clip(M_k, 0, saturation)
```

This is a leaky integrator with exponential decay and hard saturation.

### Parameters

| Parameter | Type | Meaning | Typical range |
|-----------|------|---------|---------------|
| `decay_rate` | `float` | Exponential forgetting rate (1/s). | 0.001 - 0.1 |
| `saturation` | `float` | Maximum imprint magnitude. | 1.0 - 5.0 |
| `exposure_k` | `float` | Instantaneous exposure signal for oscillator k. | Signal-dependent |

### Decay Behaviour

At `decay_rate = 0.01`, the half-life is `ln(2) / 0.01 ≈ 69` seconds.
After 5 half-lives (~345 s), the imprint has decayed to ~3% of its
peak. This timescale is appropriate for session-level adaptation in
real-time control systems.

For slower adaptation (hours/days), use `decay_rate = 0.0001`.
For rapid plasticity (sub-second), use `decay_rate = 0.1`.

### Saturation

The clip to `[0, saturation]` prevents runaway accumulation. Without
saturation, sustained high exposure would drive `M_k` to infinity,
causing coupling matrix overflow. The saturation value determines the
maximum multiplicative effect on K:

```
K_ij_effective_max = K_ij * (1 + saturation)
```

At `saturation = 2.0`, coupling can at most triple. At `saturation = 5.0`,
it can sextuple. These bounds should be coordinated with the
`ActionProjector` value limits on K.

## Modulation Targets

The imprint vector modulates the coupling pipeline through two
independent channels: coupling strength (K) and phase lag (alpha).
Which channels are active is configured in the binding spec.

### Coupling Modulation

```
K_ij_effective = K_ij * (1 + M_i)
```

Row-wise scaling: oscillator i's incoming coupling is scaled by its
imprint value. High imprint = stronger coupling. This models the
intuition that well-established connections are more influential.

The scaling is applied before each UPDE integration step:

```python
model = ImprintModel(decay_rate=0.01, saturation=2.0)
state = ImprintState(n=8)

# Each step:
model.update(state, exposure=current_exposure, dt=0.01)
knm_effective = model.modulate_coupling(state, knm_base)
engine.step(phases, omegas, knm_effective, zeta, psi, alpha)
```

### Lag Modulation

```
alpha_ij_effective = alpha_ij + M_i
```

Additive shift: oscillator i's incoming phase lag is increased by its
imprint value. High imprint introduces a directional bias — the
oscillator "expects" incoming signals to arrive with a delay
proportional to past exposure.

This models adaptation to consistent phase relationships: if oscillator
i consistently receives signals from j at a specific phase offset, the
imprint shifts alpha to encode that relationship.

### Configuring Modulation Targets

The binding spec `imprint_model.modulates` list controls which knobs
are affected:

```yaml
imprint_model:
  decay_rate: 0.01
  saturation: 2.0
  modulates: ["K", "alpha"]
```

Valid targets: `"K"`, `"alpha"`, or both. Omitting a target disables
that modulation channel. Using both simultaneously creates coupled
adaptation: imprint strengthens connections (K) while also shifting
their preferred phase relationship (alpha).

## Exposure Signal

The `exposure_k` input to the update rule can be any scalar signal
per oscillator. Common choices:

| Source | Formula | Use case |
|--------|---------|----------|
| Coherence | `R_local_k` (local order parameter) | Strengthen pathways that produce coherence. |
| Amplitude | `amplitude_k` from PhaseState | Track oscillator activity level. |
| Quality | `quality_k` from PhaseState | Penalise unreliable connections. |
| External | Supervisor-provided signal | Reward/punishment for reinforcement learning. |
| Constant | `1.0` | Pure time-based accumulation (clock). |

The exposure signal is NOT clamped — negative exposure is valid and
causes the imprint to decrease faster than natural decay. This allows
active forgetting driven by, e.g., error signals.

## Attribution

`ImprintState.attribution` maps source labels to contribution
magnitudes. It tracks where exposure came from for audit and
explainability:

```python
state.attribution
# {"eeg_alpha": 0.45, "spike_rate": 0.22, "external_reward": 0.15}
```

Attribution is updated alongside the imprint vector. When multiple
exposure sources contribute in the same step, their contributions are
tracked proportionally. This enables post-hoc analysis of what drove
the adaptation.

## Lifecycle

1. **Initialise**: `M_k = 0` for all oscillators. No prior exposure.
2. **Update**: each integration step, call
   `ImprintModel.update(state, exposure, dt)`.
3. **Modulate**: before UPDE integration, apply modulation via
   `modulate_coupling()` and/or `modulate_lag()`.
4. **Attribute**: attribution dict updated alongside exposure sources.
5. **Persist** (optional): save `ImprintState` to disk for
   cross-session continuity.

### State Persistence

The `ImprintState` can be serialised to JSON for persistence across
sessions:

```python
import json
serialised = state.to_dict()
json.dump(serialised, open("imprint_state.json", "w"))

# Next session:
state = ImprintState.from_dict(json.load(open("imprint_state.json")))
```

This is critical for BCI applications where the imprint encodes
user-specific calibration that accumulates over days or weeks.

## Rust Implementation

The `ImprintModel` has a Rust counterpart in `spo-engine::imprint`.
The Rust path handles the update and modulation in a single pass over
the arrays, avoiding Python loop overhead for large N. The FFI wrapper
`PyImprintModel` is registered in `spo_kernel`.

```rust
let mut model = ImprintModel::new(decay_rate, saturation, n);
model.update(&exposure, dt);
model.modulate_coupling(&mut knm, n);
```

The Rust implementation is bit-for-bit identical to the Python version
for all inputs (verified by `tests/test_ffi_parity.py`).

## Interaction with Plasticity

The imprint model operates at a slower timescale than the plasticity
model (`PlasticityModel`). Plasticity modifies K_nm within each
integration step based on instantaneous phase correlations (Hebbian
learning). Imprint modifies the _base_ K_nm over many steps based on
cumulative exposure.

The two are composable:

```
K_step = K_base * (1 + M_i)          # imprint modulation
K_step += plasticity_delta            # Hebbian update within step
```

This separation of timescales mirrors biological systems where
synaptic plasticity (milliseconds) and structural adaptation (hours)
operate in parallel.

## When to Use

Enable the imprint model when the system has history-dependent
coupling:

- **Enable**: habituation, sensitisation, learning, drift adaptation,
  user calibration, long-running sessions.
- **Disable**: time-invariant coupling, short simulations, benchmark
  runs where adaptation would confound results.

The computational cost of the imprint model is O(N) per step — negligible
compared to the O(N^2) coupling computation in the UPDE engine.

## Testing

The test suite covers:

- Exponential decay convergence (verify half-life matches analytical
  prediction).
- Saturation clamping (verify M_k never exceeds saturation).
- Coupling modulation correctness (verify row-wise scaling).
- Lag modulation correctness (verify additive shift).
- Attribution tracking (verify contributions are proportional).
- Rust-Python parity (verify identical output for random inputs).
- Zero exposure leaves imprint unchanged (minus natural decay).
- Negative exposure accelerates decay.

## Design Decisions

**Why a scalar imprint per oscillator, not per pair?** A per-pair
imprint `M_ij` would allow edge-specific adaptation but scales as
O(N^2) in memory and update cost. The per-oscillator vector `M_k`
scales as O(N) and covers the dominant use cases (node-level
habituation and sensitisation). Edge-specific adaptation is handled
by the plasticity model instead.

**Why exponential decay, not power-law?** Exponential decay has a
single parameter (decay_rate) and is analytically tractable. Power-law
forgetting (as observed in some human memory studies) would require
tracking the full history for accurate computation. The exponential
approximation is sufficient for real-time control and enables the
simple update rule `M *= exp(-rate * dt)`.

**Why not merge with plasticity?** Plasticity operates at the coupling
level (K_ij changes per step based on phase correlation). Imprint
operates at the node level (M_k changes based on exposure). Merging
them would conflate two distinct timescales and adaptation mechanisms.
Keeping them separate allows independent tuning and clear attribution
of which mechanism drove a particular change.

**Why saturation, not soft-clamp?** Hard saturation (clip) is simpler
to reason about and test. A soft-clamp (sigmoid) would smooth the
boundary but introduce a nonlinearity that interacts with the
modulation formula in hard-to-predict ways. The hard saturation is
transparent: M_k is always in `[0, saturation]`.

## Example: BCI Session Adaptation

A Brain-Computer Interface session runs for 30 minutes:

1. **Minutes 0-5** — user is unfamiliar with the interface. Phase
   quality is low, coherence R fluctuates. Imprint grows slowly
   (low exposure due to low quality gating).
2. **Minutes 5-15** — user adapts. Quality rises, exposure signal
   increases. Imprint accumulates, strengthening the coupling
   pathways that produce stable phase relationships.
3. **Minutes 15-25** — steady state. Imprint saturates on the
   dominant pathways. The system is "tuned" to the user's neural
   signature.
4. **Minutes 25-30** — fatigue. Quality drops, exposure decreases.
   Imprint begins to decay on less-used pathways, but the dominant
   ones remain near saturation due to accumulated history.

At session end, `ImprintState` is saved. Next session starts from the
saved state, so the system does not need to re-learn from scratch.
Over multiple sessions, the imprint converges to a user-specific
profile that should reduce re-calibration time (not yet measured —
this is a design expectation, not a validated result).

## References

- **[ebbinghaus1885]** H. Ebbinghaus (1885). *Uber das Gedachtnis*.
  Duncker & Humblot. — Exponential forgetting curve.
- All constants in this module are original to SPO. See
  [ASSUMPTIONS.md](../ASSUMPTIONS.md) § Imprint Model for provenance
  of `decay_rate` and `saturation` defaults.
- Plasticity model: `scpn_phase_orchestrator.coupling.plasticity`.
- Imprint source: `scpn_phase_orchestrator.imprint`.

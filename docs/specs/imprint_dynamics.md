# Imprint Dynamics

## Model

Imprint tracks cumulative exposure per oscillator. The state vector **m_k(t)** evolves as:

```
m_k(t+dt) = m_k(t) * exp(-decay_rate * dt) + exposure_k * dt
m_k = clip(m_k, 0, saturation)
```

Where `exposure_k` is the coherence (R) of the layer containing oscillator k.

## Parameters

| Parameter | Type | Constraint | Source |
|-----------|------|------------|--------|
| `decay_rate` | float | >= 0 | `binding_spec.imprint_model.decay_rate` |
| `saturation` | float | > 0 | `binding_spec.imprint_model.saturation` |

## Modulation

Imprint modulates the coupling matrix and phase lags before each UPDE step:

**Coupling modulation** — rows of K_nm scaled by accumulated exposure:

```
K'_nm = K_nm * (1 + m_n)
```

Oscillators with higher imprint receive stronger coupling from all neighbours.

**Lag modulation** — antisymmetric offset added to alpha:

```
alpha'_nm = alpha_nm + (m_n - m_m)
```

Oscillators with asymmetric exposure histories develop phase lead/lag relative to each other.

## State

`ImprintState` is a frozen dataclass:

| Field | Type | Description |
|-------|------|-------------|
| `m_k` | NDArray (N,) | Per-oscillator imprint accumulation |
| `last_update` | float | Simulation time of last update |
| `attribution` | dict[str, float] | Optional provenance metadata |

## Invariants

- `m_k >= 0` always (clipped after update).
- `m_k <= saturation` always (clipped after update).
- Decay is exponential: without exposure, `m_k -> 0` as `t -> inf`.
- With constant exposure `e`, steady state is `m_k = min(e / decay_rate, saturation)`.

## Integration Point

When `binding_spec.imprint_model` is present, the CLI `run` loop:

1. Initialises `ImprintModel` from spec parameters and `ImprintState(m_k=zeros)`.
2. Before each UPDE step, applies `modulate_coupling` and `modulate_lag` to the effective K_nm and alpha.
3. After computing layer states, updates imprint with per-oscillator exposure derived from layer R values.

## References

- `src/scpn_phase_orchestrator/imprint/update.py` — `ImprintModel` class.
- `src/scpn_phase_orchestrator/imprint/state.py` — `ImprintState` dataclass.
- `docs/concepts/memory_imprint.md` — conceptual motivation.

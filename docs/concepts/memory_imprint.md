# Memory Imprint Model

> **Provenance:** The imprint model is original to SCPN Phase Orchestrator.
> The exponential forgetting form is standard in memory/learning models
> (cf. Ebbinghaus 1885), but the specific application to coupling-matrix
> modulation is novel. Parameters (`decay_rate`, `saturation`) require
> domain-specific calibration — see [ASSUMPTIONS.md](../ASSUMPTIONS.md).

## Purpose

Track cumulative exposure history per oscillator. The imprint vector `M_k` modulates coupling and phase lag, encoding long-term adaptation.

## Dynamics

```
M_k(t + dt) = M_k(t) * exp(-decay_rate * dt) + exposure_k * dt
M_k = clip(M_k, 0, saturation)
```

- **decay_rate:** Exponential forgetting rate. Higher = faster fade.
- **saturation:** Upper bound on imprint magnitude. Prevents runaway accumulation.
- **exposure_k:** Instantaneous exposure signal for oscillator k.

## Modulation

### Coupling modulation

```
K_ij_effective = K_ij * (1 + M_i)
```

Oscillators with high imprint couple more strongly. Row-wise scaling of Knm.

### Lag modulation

```
alpha_ij_effective = alpha_ij + M_i
```

High imprint shifts the phase lag, introducing a directional bias.

### Configuring modulation targets

The binding spec `imprint_model.modulates` list controls which knobs are affected:

```yaml
imprint_model:
  decay_rate: 0.01
  saturation: 2.0
  modulates: ["K", "alpha"]
```

## Attribution

`ImprintState.attribution` maps source labels to contribution magnitudes. Tracks where exposure came from for audit and explainability.

## Lifecycle

1. Initialise `M_k = 0` for all oscillators.
2. Each step: call `ImprintModel.update(state, exposure, dt)`.
3. Before UPDE integration: apply modulation via `modulate_coupling()` and/or `modulate_lag()`.
4. Attribution updated alongside exposure sources.

## When to Use

Enable the imprint model when the system has history-dependent coupling -- habituation, sensitisation, learning, or drift adaptation. Omit when coupling is time-invariant.

## References

All constants in this module are original to SPO. See [ASSUMPTIONS.md](../ASSUMPTIONS.md) § Imprint Model for provenance of `decay_rate` and `saturation` defaults.

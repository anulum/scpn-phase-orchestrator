# Imprint

The imprint subsystem implements history-dependent coupling modulation —
a computational model of learning and memory in coupled oscillator networks.
Oscillators that have been consistently active accumulate an *imprint*
$m_k$ that modulates their coupling strength, phase lag, and bifurcation
parameter. This creates a memory effect: frequently synchronised
oscillator pairs develop stronger coupling over time, analogous to
Hebbian synaptic strengthening in neural systems.

## Theory

The imprint model draws from three theoretical traditions:

1. **Hebbian plasticity** (Hebb 1949): "neurons that fire together wire together."
   In phase oscillator terms, oscillators with correlated phases strengthen
   their mutual coupling.

2. **Exponential trace models** (Friston 2005, Philos. Trans. R. Soc. B 360:815-836):
   The free energy framework uses exponential accumulation with decay
   to model synaptic efficacy changes.

3. **Imprint dynamics** in the SCPN model: the 15+1 layer architecture
   uses imprints as the mechanism by which L16 (the cybernetic closure)
   shapes the lower layers' coupling topology over time.

## Dynamics

The imprint variable $m_k$ for oscillator $k$ evolves as:

$$m_k(t + \Delta t) = m_k(t) \cdot e^{-\lambda \Delta t} + E_k \cdot \Delta t$$

where:

- $\lambda$ — decay rate (how fast the memory fades without reinforcement)
- $E_k$ — exposure (input from the current oscillator activity)
- $\Delta t$ — timestep

The result is clipped to $[0, m_{\text{sat}}]$ where $m_{\text{sat}}$
is the saturation threshold, preventing runaway accumulation.

**Decay rate interpretation:**

| $\lambda$ | Timescale | Analogy |
|-----------|-----------|---------|
| 0.0 | Permanent | Long-term potentiation |
| 0.01 | ~100 steps | Working memory |
| 0.1 | ~10 steps | Sensory adaptation |
| 1.0 | ~1 step | No memory (reactive) |

## Modulation Channels

The imprint modulates three UPDE parameters:

### Coupling ($K_{nm}$)

$$K_{nm}^{\prime} = K_{nm} \cdot (1 + m_k)$$

Rows of the coupling matrix are scaled by $(1 + m_k)$, so oscillators
with high imprint couple more strongly to all their neighbours.

### Phase Lag ($\alpha_{ij}$)

$$\alpha_{ij}^{\prime} = \alpha_{ij} + (m_i - m_j)$$

The antisymmetric offset breaks phase-lag symmetry toward observed
phase relationships. If oscillator $i$ has higher imprint than $j$,
the lag shifts to favour the pattern that produced the imprint.

### Bifurcation ($\mu_k$)

$$\mu_k^{\prime} = \mu_k \cdot (1 + m_k)$$

For Stuart-Landau dynamics, the bifurcation parameter controls
the distance from the Hopf bifurcation. Higher imprint pushes
the oscillator further into the limit-cycle regime.

!!! note "Amplitude coupling excluded"
    The amplitude coupling topology $K_{nm}^r$ is *not* modulated by
    imprint. It is fixed by the binding specification. This is intentional:
    amplitude coupling represents physical connectivity (e.g. axon tracts),
    which does not change on the timescale of imprint dynamics.

## Usage

```python
from scpn_phase_orchestrator.imprint.state import ImprintState
from scpn_phase_orchestrator.imprint.update import ImprintModel

# Create model: slow decay, saturation at 5.0
model = ImprintModel(decay_rate=0.01, saturation=5.0)

# Initial state: no imprint
state = ImprintState(m_k=np.zeros(n), last_update=0.0)

# Update with exposure from oscillator activity
exposure = compute_exposure(phases, knm)  # domain-specific
state = model.update(state, exposure, dt=0.01)

# Modulate coupling for next UPDE step
knm_modulated = model.modulate_coupling(knm, state)
alpha_modulated = model.modulate_lag(alpha, state)
```

## Pipeline integration

```
UPDEEngine.step() ──→ phases ──→ compute_exposure()
                                        │
                                        ↓
                                 ImprintModel.update()
                                        │
                                        ↓
                                 ImprintState (m_k updated)
                                        │
                      ┌─────────────────┼──────────────────┐
                      ↓                 ↓                  ↓
              modulate_coupling  modulate_lag     modulate_mu
              K_nm' = K·(1+m)   α' = α+(Δm)     μ' = μ·(1+m)
                      │                 │                  │
                      ↓                 ↓                  ↓
              UPDEEngine.step(K_nm', ..., α', ...)  ← next cycle
```

## ImprintState (frozen dataclass)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `m_k` | `NDArray` | required | Imprint values per oscillator |
| `last_update` | `float` | required | Timestamp of last update |
| `attribution` | `dict[str, float]` | `{}` | Source attribution weights |

The `attribution` dict tracks which input sources contributed to the
imprint (e.g., `{"eeg_alpha": 0.6, "emg_burst": 0.4}`). This supports
explainability: when the imprint modulates coupling, the attribution
records which signals caused the modulation.

### Validation

`ImprintModel.__init__` validates:
- `decay_rate >= 0` (non-negative; 0 = permanent memory)
- `saturation > 0` (positive; prevents runaway accumulation)

## ImprintModel

```python
ImprintModel(decay_rate: float, saturation: float)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `update` | `(state, exposure, dt) → ImprintState` | Exponential update + decay |
| `modulate_coupling` | `(knm, state) → NDArray` | K' = K · (1 + m_k) |
| `modulate_lag` | `(alpha, state) → NDArray` | α' = α + (m_i - m_j) |
| `modulate_mu` | `(mu, state) → NDArray` | μ' = μ · (1 + m_k) |

**Performance:** `update(n=64)` < 5 ms.

## API Reference

### State

::: scpn_phase_orchestrator.imprint.state

### Model

::: scpn_phase_orchestrator.imprint.update

# Drivers

External forcing functions that inject the drive signal $\Psi(t)$ into
the Kuramoto equation. Each driver corresponds to one of the three
oscillator channels (Physical, Informational, Symbolic) and produces
a time-varying phase target that pulls oscillators toward a desired
synchronisation pattern.

## Theory

The drive term in the Kuramoto ODE is:

$$\frac{d\theta_i}{dt} = \omega_i + \sum_j K_{ij} \sin(\theta_j - \theta_i - \alpha_{ij}) + \zeta \sin(\Psi - \theta_i)$$

The third term $\zeta \sin(\Psi - \theta_i)$ represents external forcing.
The supervisor controls $\zeta$ (drive strength); the driver controls
$\Psi$ (drive phase). When $\zeta > 0$, oscillators are pulled toward
the target phase $\Psi(t)$.

## Physical Driver

Sinusoidal external drive:

$$\Psi_P(t) = A \sin(2\pi f t)$$

Models periodic physical forcing — cardiac pacemaker signals,
power grid reference frequency (50/60 Hz), mechanical vibration
sources, plasma heating pulses.

**Parameters:**

| Parameter | Type | Constraint | Description |
|-----------|------|------------|-------------|
| `frequency` | `float` | > 0 | Drive frequency in Hz |
| `amplitude` | `float` | any | Peak amplitude (default 1.0) |

**Batch mode:** `compute_batch(t_array)` vectorises over a NumPy array
of time values for efficient integration.

::: scpn_phase_orchestrator.drivers.psi_physical

## Informational Driver

Linear ramp drive with modular wrapping:

$$\Psi_I(t) = 2\pi f_c t \pmod{2\pi}$$

Models information cadence — packet arrival rates, event stream
clocks, data pipeline heartbeats. The phase advances at a constant
rate $f_c$ Hz, producing a sawtooth waveform that resets at $2\pi$.

This driver is appropriate when the domain has a natural clock rate
(e.g. 100 Hz monitoring cadence) and the goal is to synchronise
oscillators to that cadence.

**Parameters:**

| Parameter | Type | Constraint | Description |
|-----------|------|------------|-------------|
| `cadence_hz` | `float` | > 0 | Information cadence in Hz |

::: scpn_phase_orchestrator.drivers.psi_informational

## Symbolic Driver

Deterministic phase sequence:

$$\Psi_S(k) = s_{k \bmod N}$$

where $s = [s_0, s_1, \ldots, s_{N-1}]$ is a pre-defined phase
sequence that repeats with period $N$.

Models symbolic/semiotic patterns — language token sequences,
musical motifs, protocol state machines, ritual rhythms. The sequence
encodes a pattern that the oscillator network is driven to reproduce.

**Parameters:**

| Parameter | Type | Constraint | Description |
|-----------|------|------------|-------------|
| `sequence` | `list[float]` | non-empty | Phase values (radians) |

**Usage:**

```python
from scpn_phase_orchestrator.drivers.psi_symbolic import SymbolicDriver

# Musical 4/4 pattern: downbeat, weak, medium, weak
pattern = [0.0, np.pi, np.pi/2, np.pi]
driver = SymbolicDriver(pattern)

# Step 0 → 0.0, step 1 → π, step 4 → 0.0 (wraps)
psi = driver.compute(step=0)
```

::: scpn_phase_orchestrator.drivers.psi_symbolic

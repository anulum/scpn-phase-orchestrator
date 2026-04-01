# Autotune

The autotune subsystem provides tools for identifying unknown system 
parameters and discovering governing dynamics from raw data.

## Phase-SINDy Symbolic Discovery

The `PhaseSINDy` module implements **Sparse Identification of Nonlinear 
Dynamics** tailored for phase oscillator networks. It allows the 
orchestrator to act as an "Autonomous Physicist," reverse-engineering 
the differential equations of a system from observed time-series data.

### Theoretical Basis

SINDy assumes that the dynamics $\dot{\theta}$ can be represented as 
a sparse linear combination of terms from a library $\Theta$:

$$ \dot{\theta} = \Theta(\theta) \Xi $$

For SPO, the library $\Theta$ includes:
1. **Constant terms:** Representing natural frequencies $\omega_i$.
2. **Coupling terms:** $\sin(\theta_j - \theta_i)$ representing Kuramoto-style interactions.

The model uses **Sequentially Thresholded Least Squares (STLSQ)** to 
discover the sparsest set of coefficients that explain the data, 
effectively filtering out noise and revealing the underlying topology.

### Use Cases
- **System Identification:** Discovering the coupling strength $K_{nm}$ 
  in a biological network where the wiring is unknown.
- **Topological Verification:** Verifying that a physical system actually 
  follows the assumed Kuramoto model before engageing control logic.
- **Anomaly Detection:** Detecting shifts in the governing equations 
  (e.g., a component failure that changes the interaction physics).

::: scpn_phase_orchestrator.autotune.sindy

## Frequency Identification

Identifies natural frequencies $\omega_i$ from phase time-series.

::: scpn_phase_orchestrator.autotune.freq_id

## Coupling Estimation

Estimates the coupling matrix $K_{nm}$ assuming a fixed interaction model.

::: scpn_phase_orchestrator.autotune.coupling_est

# Neural Network Module (nn)

Differentiable Kuramoto dynamics for neural network integration via JAX and equinox.

**Requires:** `pip install scpn-phase-orchestrator[nn]` (installs jax + equinox)

## Functional API

Pure JAX functions — JIT-compilable, vmap-compatible, fully differentiable.

::: scpn_phase_orchestrator.nn.functional

## KuramotoLayer

Equinox module wrapping Kuramoto dynamics as a learnable layer.

::: scpn_phase_orchestrator.nn.kuramoto_layer

## StuartLandauLayer

Equinox module with phase + amplitude dynamics. Learnable coupling matrices,
frequencies, and bifurcation parameters.

::: scpn_phase_orchestrator.nn.stuart_landau_layer

## BOLD Signal Generator

Balloon-Windkessel hemodynamic model for converting oscillator amplitudes
to simulated fMRI BOLD signal. Differentiable for gradient-based fMRI fitting.

::: scpn_phase_orchestrator.nn.bold

## Reservoir Computing

Kuramoto-based reservoir computing with linear readout. Universal
approximation near edge-of-bifurcation (arXiv:2407.16172, 2024).

::: scpn_phase_orchestrator.nn.reservoir

## UDE-Kuramoto (Universal Differential Equation)

Physics backbone + learned neural residual. The known sin(Δθ) coupling
plus a small MLP that handles model mismatch. First Python UDE for oscillators.

::: scpn_phase_orchestrator.nn.ude

## Inverse Kuramoto

Gradient-based inference of coupling matrix K and frequencies ω from
observed phase trajectories. Backpropagates through the ODE solver.

::: scpn_phase_orchestrator.nn.inverse

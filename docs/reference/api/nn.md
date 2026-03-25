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

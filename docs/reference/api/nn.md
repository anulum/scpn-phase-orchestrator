# Neural Network Module (nn)

Differentiable Kuramoto dynamics for neural network integration via JAX and equinox.

**Requires:** `pip install scpn-phase-orchestrator[nn]` (installs jax + equinox)

## Functional API

Pure JAX functions — JIT-compilable, vmap-compatible, fully differentiable.

::: scpn_phase_orchestrator.nn.functional

## KuramotoLayer

Equinox module wrapping Kuramoto dynamics as a learnable layer.

::: scpn_phase_orchestrator.nn.kuramoto_layer

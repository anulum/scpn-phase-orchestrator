#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SCPN Phase Orchestrator — Example: Inverse Kuramoto Pipeline
#
# Generates synthetic oscillator data with known coupling,
# then infers the coupling matrix from the observed phases.
#
# Usage: python examples/inverse_kuramoto.py
# Requires: pip install scpn-phase-orchestrator[nn]

from __future__ import annotations

import jax
import jax.numpy as jnp

from scpn_phase_orchestrator.nn import (
    coupling_correlation,
    infer_coupling,
    kuramoto_forward,
    order_parameter,
)


def main() -> None:
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    N = 6
    dt = 0.02
    n_steps = 100

    # Ground truth: random symmetric coupling + frequencies
    omegas_true = jax.random.normal(k1, (N,)) * 0.5
    raw = jax.random.normal(k2, (N, N)) * 0.3
    K_true = (raw + raw.T) / 2.0
    K_true = K_true.at[jnp.diag_indices(N)].set(0.0)

    # Generate observed data
    phases0 = jax.random.uniform(k3, (N,), maxval=2.0 * jnp.pi)
    _, trajectory = kuramoto_forward(phases0, omegas_true, K_true, dt, n_steps)
    observed = jnp.concatenate([phases0[jnp.newaxis, :], trajectory])

    print(f"Ground truth: N={N}, {n_steps} steps, dt={dt}")
    print(f"R at start: {float(order_parameter(phases0)):.3f}")
    print(f"R at end:   {float(order_parameter(trajectory[-1])):.3f}")
    print()

    # Infer coupling from observed phases
    print("Running inverse Kuramoto (200 epochs)...")
    K_inferred, omegas_inferred, losses = infer_coupling(
        observed, dt=dt, n_epochs=200, lr=0.01, l1_weight=0.001
    )

    corr = coupling_correlation(K_true, K_inferred)
    print(f"Loss: {losses[0]:.4f} → {losses[-1]:.4f}")
    print(f"Coupling correlation: {float(corr):.3f}")
    print(f"K_true max:     {float(jnp.max(jnp.abs(K_true))):.3f}")
    print(f"K_inferred max: {float(jnp.max(jnp.abs(K_inferred))):.3f}")


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Synthetic binding experiment for KuramotoLayer

"""Train coupling matrix K to solve oscillator group binding.

Setup: N oscillators in G groups. Desired behavior: oscillators in the
same group synchronize (high group R), different groups desynchronize
(low global R).

Loss: -sum(R_group^2) + R_global^2
This maximizes within-group coherence while penalizing global sync.

V2: replaced PLV loss (saturated, zero gradients) with direct phase
clustering via group order parameters.
"""

from __future__ import annotations

import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from scpn_phase_orchestrator.nn.functional import kuramoto_forward, order_parameter


def group_order_parameters(
    phases: jax.Array, labels: jax.Array, n_groups: int
) -> tuple[jax.Array, jax.Array]:
    """Compute per-group and global order parameters.

    Returns:
        (group_Rs, R_global): group_Rs is (n_groups,), R_global is scalar
    """
    R_global = order_parameter(phases)
    group_Rs = jnp.zeros(n_groups)
    for g in range(n_groups):
        mask = labels == g
        z = jnp.exp(1j * phases) * mask
        n_g = jnp.sum(mask)
        R_g = jnp.where(n_g > 0, jnp.abs(jnp.sum(z)) / n_g, 0.0)
        group_Rs = group_Rs.at[g].set(R_g)
    return group_Rs, R_global


def binding_loss(
    K: jax.Array,
    phases: jax.Array,
    omegas: jax.Array,
    labels: jax.Array,
    n_groups: int,
    dt: float,
    n_steps: int,
) -> jax.Array:
    """Loss: -sum(R_group^2) + R_global^2.

    Maximizes within-group sync while penalizing global sync.
    """
    final, _ = kuramoto_forward(phases, omegas, K, dt, n_steps)
    group_Rs, R_global = group_order_parameters(final, labels, n_groups)
    return -jnp.sum(group_Rs**2) + R_global**2


def run_experiment(
    n: int = 16,
    n_groups: int = 3,
    n_steps: int = 50,
    dt: float = 0.02,
    lr: float = 0.05,
    n_epochs: int = 300,
    seed: int = 42,
) -> dict:
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    labels = jax.random.randint(k1, (n,), 0, n_groups)
    # Spread natural frequencies by group to give gradient a starting signal
    base_omegas = jnp.array([2.0 * g for g in range(n_groups)])
    omegas = base_omegas[labels] + 0.1 * jax.random.normal(k2, (n,))
    phases_init = jax.random.uniform(k3, (n,), maxval=2.0 * jnp.pi)

    # Larger initial K to create meaningful dynamics
    raw = 0.3 * jax.random.normal(k4, (n, n))
    K = (raw + raw.T) / 2.0
    K = K.at[jnp.diag_indices(n)].set(0.0)

    loss_and_grad = jax.value_and_grad(
        lambda K_: binding_loss(K_, phases_init, omegas, labels, n_groups, dt, n_steps)
    )

    losses = []
    t0 = time.time()

    for epoch in range(n_epochs):
        loss_val, grad_K = loss_and_grad(K)
        K = K - lr * grad_K
        K = (K + K.T) / 2.0
        K = K.at[jnp.diag_indices(n)].set(0.0)

        losses.append(float(loss_val))
        if epoch % 50 == 0:
            group_Rs, R_global = group_order_parameters(
                kuramoto_forward(phases_init, omegas, K, dt, n_steps)[0],
                labels,
                n_groups,
            )
            print(
                f"  epoch {epoch:4d}  loss={loss_val:.4f}  "
                f"R_groups={[round(float(r), 3) for r in group_Rs]}  "
                f"R_global={float(R_global):.3f}"
            )

    elapsed = time.time() - t0

    # Final evaluation
    final_phases = kuramoto_forward(phases_init, omegas, K, dt, n_steps)[0]
    group_Rs, R_global = group_order_parameters(final_phases, labels, n_groups)

    # K structure analysis
    K_np = np.array(K)
    labels_np = np.array(labels)
    intra_k, inter_k = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if labels_np[i] == labels_np[j]:
                intra_k.append(float(K_np[i, j]))
            else:
                inter_k.append(float(K_np[i, j]))

    mean_intra_k = float(np.mean(intra_k)) if intra_k else 0.0
    mean_inter_k = float(np.mean(inter_k)) if inter_k else 0.0
    mean_group_R = float(jnp.mean(group_Rs))

    results = {
        "n": n,
        "n_groups": n_groups,
        "n_steps": n_steps,
        "n_epochs": n_epochs,
        "dt": dt,
        "lr": lr,
        "seed": seed,
        "elapsed_s": round(elapsed, 2),
        "initial_loss": round(float(losses[0]), 4),
        "final_loss": round(float(losses[-1]), 4),
        "mean_group_R": round(mean_group_R, 4),
        "R_global": round(float(R_global), 4),
        "group_Rs": [round(float(r), 4) for r in group_Rs],
        "mean_intra_K": round(mean_intra_k, 4),
        "mean_inter_K": round(mean_inter_k, 4),
        "K_separation": round(mean_intra_k - mean_inter_k, 4),
        "works": mean_group_R > 0.7 and float(R_global) < mean_group_R - 0.1,
    }

    print("\n--- Results ---")
    for k, v in results.items():
        print(f"  {k}: {v}")

    return results


if __name__ == "__main__":
    from pathlib import Path

    results = run_experiment()
    out = Path("experiments/nn_binding_synthetic_results.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out}")

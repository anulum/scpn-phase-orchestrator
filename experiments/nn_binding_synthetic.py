# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Synthetic binding experiment for KuramotoLayer

"""Train KuramotoLayer to solve a synthetic oscillator binding problem.

Setup: N oscillators belong to G groups. Ground truth: oscillators in the
same group should synchronize (converge to same phase), oscillators in
different groups should desynchronize (spread apart).

Loss: maximize intra-group PLV, minimize inter-group PLV.

This tests whether gradient-based optimization of the coupling matrix K
can recover group structure from random initialization.

AKOrN reference (ICLR 2025): uses Kuramoto as activation function for
adversarial robustness. Our use case is different — we learn the coupling
matrix itself, not use dynamics as a fixed nonlinearity.
"""

from __future__ import annotations

import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from scpn_phase_orchestrator.nn.functional import kuramoto_forward, order_parameter, plv


def make_ground_truth(n: int, n_groups: int, key: jax.Array) -> jax.Array:
    """Create group assignment matrix. Returns (n,) integer labels."""
    return jax.random.randint(key, (n,), 0, n_groups)


def binding_loss(
    K: jax.Array,
    phases: jax.Array,
    omegas: jax.Array,
    labels: jax.Array,
    n_groups: int,
    dt: float,
    n_steps: int,
) -> jax.Array:
    """Loss: -mean(intra-group PLV) + mean(inter-group PLV).

    Minimizing this encourages same-group sync and cross-group desync.
    """
    _, traj = kuramoto_forward(phases, omegas, K, dt, n_steps)
    P = plv(traj)

    intra_sum = jnp.float32(0.0)
    inter_sum = jnp.float32(0.0)
    intra_count = jnp.float32(0.0)
    inter_count = jnp.float32(0.0)

    for g in range(n_groups):
        mask = labels == g
        n_g = jnp.sum(mask)
        # Intra-group: PLV between members of same group
        group_plv = jnp.sum(P * jnp.outer(mask, mask)) - n_g  # exclude diagonal
        intra_sum += group_plv
        intra_count += n_g * (n_g - 1)
        # Inter-group: PLV between this group and others
        anti_mask = ~mask
        cross_plv = jnp.sum(P * jnp.outer(mask, anti_mask))
        inter_sum += cross_plv
        inter_count += n_g * jnp.sum(anti_mask)

    mean_intra = jnp.where(intra_count > 0, intra_sum / intra_count, 0.0)
    mean_inter = jnp.where(inter_count > 0, inter_sum / inter_count, 0.0)

    return -mean_intra + mean_inter


def run_experiment(
    n: int = 16,
    n_groups: int = 3,
    n_steps: int = 100,
    dt: float = 0.01,
    lr: float = 0.01,
    n_epochs: int = 200,
    seed: int = 42,
) -> dict:
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    labels = make_ground_truth(n, n_groups, k1)
    omegas = jax.random.normal(k2, (n,)) * 0.5
    phases_init = jax.random.uniform(k3, (n,), maxval=2.0 * jnp.pi)

    # Initialize K: small random symmetric
    raw = 0.05 * jax.random.normal(k4, (n, n))
    K = (raw + raw.T) / 2.0

    loss_and_grad = jax.value_and_grad(
        lambda K_: binding_loss(K_, phases_init, omegas, labels, n_groups, dt, n_steps)
    )

    losses = []
    t0 = time.time()

    for epoch in range(n_epochs):
        loss_val, grad_K = loss_and_grad(K)
        # Gradient descent with symmetric constraint
        K = K - lr * grad_K
        K = (K + K.T) / 2.0  # enforce symmetry
        K = K.at[jnp.diag_indices(n)].set(0.0)  # zero diagonal

        losses.append(float(loss_val))
        if epoch % 50 == 0:
            print(f"  epoch {epoch:4d}  loss={loss_val:.4f}")

    elapsed = time.time() - t0

    # Evaluate final K
    _, traj = kuramoto_forward(phases_init, omegas, K, dt, n_steps)
    P_final = plv(traj)
    R_final = float(order_parameter(traj[-1]))

    # Compute group-level metrics
    K_np = np.array(K)
    labels_np = np.array(labels)
    intra_k = []
    inter_k = []
    for i in range(n):
        for j in range(i + 1, n):
            if labels_np[i] == labels_np[j]:
                intra_k.append(float(K_np[i, j]))
            else:
                inter_k.append(float(K_np[i, j]))

    mean_intra_k = float(np.mean(intra_k)) if intra_k else 0.0
    mean_inter_k = float(np.mean(inter_k)) if inter_k else 0.0

    intra_plv_vals = []
    inter_plv_vals = []
    P_np = np.array(P_final)
    for i in range(n):
        for j in range(i + 1, n):
            if labels_np[i] == labels_np[j]:
                intra_plv_vals.append(float(P_np[i, j]))
            else:
                inter_plv_vals.append(float(P_np[i, j]))

    mean_intra_plv = float(np.mean(intra_plv_vals)) if intra_plv_vals else 0.0
    mean_inter_plv = float(np.mean(inter_plv_vals)) if inter_plv_vals else 0.0

    results = {
        "n": n,
        "n_groups": n_groups,
        "n_steps": n_steps,
        "n_epochs": n_epochs,
        "dt": dt,
        "lr": lr,
        "seed": seed,
        "elapsed_s": round(elapsed, 2),
        "final_loss": round(float(losses[-1]), 4),
        "initial_loss": round(float(losses[0]), 4),
        "final_R": round(R_final, 4),
        "mean_intra_K": round(mean_intra_k, 4),
        "mean_inter_K": round(mean_inter_k, 4),
        "mean_intra_PLV": round(mean_intra_plv, 4),
        "mean_inter_PLV": round(mean_inter_plv, 4),
        "K_separation": round(mean_intra_k - mean_inter_k, 4),
        "PLV_separation": round(mean_intra_plv - mean_inter_plv, 4),
        "works": mean_intra_plv > mean_inter_plv + 0.1,
    }

    print(f"\n--- Results ---")
    for k, v in results.items():
        print(f"  {k}: {v}")

    return results


if __name__ == "__main__":
    results = run_experiment()
    with open("experiments/nn_binding_synthetic_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to experiments/nn_binding_synthetic_results.json")

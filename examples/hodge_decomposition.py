#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SCPN Phase Orchestrator — Example: Hodge Decomposition of Coupling
#
# Every coupling flow decomposes into gradient (phase-locking),
# curl (rotational), and harmonic (topological) components.
# Symmetric coupling has zero curl. Asymmetric coupling reveals
# directional information flow.
#
# Usage: python examples/hodge_decomposition.py
# Requires: pip install scpn-phase-orchestrator

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.coupling.hodge import hodge_decomposition

TWO_PI = 2.0 * np.pi


def main() -> None:
    n = 5
    rng = np.random.default_rng(42)
    phases = rng.uniform(0, TWO_PI, n)

    print("Hodge Decomposition of Coupling Dynamics")
    print("=" * 50)

    # Case 1: Symmetric coupling (gradient only)
    knm_sym = rng.uniform(0.5, 2.0, (n, n))
    knm_sym = 0.5 * (knm_sym + knm_sym.T)
    np.fill_diagonal(knm_sym, 0.0)

    res = hodge_decomposition(knm_sym, phases)
    print("\nCase 1: Symmetric K (no directional bias)")
    print(f"  Gradient (phase-locking): {np.linalg.norm(res.gradient):.4f}")
    print(f"  Curl (rotational):        {np.linalg.norm(res.curl):.4f}")
    print(f"  Harmonic (topological):   {np.linalg.norm(res.harmonic):.4f}")
    print("  → Curl ≈ 0: confirmed symmetric = pure gradient flow")

    # Case 2: Asymmetric coupling (gradient + curl)
    knm_asym = rng.uniform(0.5, 2.0, (n, n))
    np.fill_diagonal(knm_asym, 0.0)

    res = hodge_decomposition(knm_asym, phases)
    print("\nCase 2: Asymmetric K (directional information flow)")
    print(f"  Gradient (phase-locking): {np.linalg.norm(res.gradient):.4f}")
    print(f"  Curl (rotational):        {np.linalg.norm(res.curl):.4f}")
    print(f"  Harmonic (topological):   {np.linalg.norm(res.harmonic):.4f}")
    print("  → Curl > 0: directional coupling creates rotational flow")

    # Case 3: One-directional chain (strong curl)
    knm_chain = np.zeros((n, n))
    for i in range(n - 1):
        knm_chain[i, i + 1] = 2.0  # forward only
    np.fill_diagonal(knm_chain, 0.0)

    res = hodge_decomposition(knm_chain, phases)
    print("\nCase 3: One-directional chain (A→B→C→D→E)")
    print(f"  Gradient: {np.linalg.norm(res.gradient):.4f}")
    print(f"  Curl:     {np.linalg.norm(res.curl):.4f}")
    print(f"  Harmonic: {np.linalg.norm(res.harmonic):.4f}")
    print("  → Maximum curl: information flows one way only")

    total = res.gradient + res.curl + res.harmonic
    direct = np.sum(
        knm_chain * np.cos(phases[np.newaxis, :] - phases[:, np.newaxis]), axis=1
    )
    print(
        f"\n  Reconstruction check: |total - direct| = "
        f"{np.linalg.norm(total - direct):.1e}"
    )


if __name__ == "__main__":
    main()

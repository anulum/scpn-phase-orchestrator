# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Network-security topology adaptation demo

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.binding import load_binding_spec, validate_binding_spec
from scpn_phase_orchestrator.monitor.lyapunov import LyapunovGuard
from scpn_phase_orchestrator.monitor.transfer_entropy import transfer_entropy_matrix
from scpn_phase_orchestrator.supervisor import (
    HigherOrderTopologySupervisor,
    TopologyMutationPolicy,
)

FloatArray: TypeAlias = NDArray[np.float64]

SPEC_PATH = Path(__file__).with_name("binding_spec.yaml")


def network_security_phase_history(steps: int = 80) -> FloatArray:
    """Return deterministic traffic/attack/defence traces with lag evidence."""
    if steps < 8:
        raise ValueError("steps must be at least 8")
    t = np.linspace(0.0, 4.0 * np.pi, steps, dtype=np.float64)
    history: FloatArray = np.vstack(
        [
            t,
            np.roll(t, 1) + 0.02,
            np.roll(t, 2) + 0.04,
            t + np.pi,
            np.roll(t, 1) + np.pi + 0.03,
            np.roll(t, 1) + 0.30,
            np.roll(t, 2) + 0.32,
            np.roll(t, 3) + 0.34,
        ]
    ).astype(np.float64)
    return history % (2.0 * np.pi)


def network_security_phase_snapshot() -> FloatArray:
    """Return low-global, locally coherent network-security phases."""
    return np.array(
        [0.00, 0.02, 0.04, np.pi, np.pi + 0.03, 0.30, 0.32, 0.34],
        dtype=np.float64,
    )


def network_security_te_supported_knm(
    history: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Build symmetric pairwise support from transfer-entropy evidence."""
    te = transfer_entropy_matrix(history, n_bins=8)
    support = np.maximum(te, te.T)
    max_support = float(np.max(support))
    if max_support > 0.0:
        support = support / max_support
    np.fill_diagonal(support, 0.0)
    knm: FloatArray = 0.20 * support
    return knm, te


def topology_lyapunov_validation(
    phases: FloatArray,
    before_knm: FloatArray,
    after_knm: FloatArray,
) -> dict[str, object]:
    """Return Lyapunov evidence for a proposed network-security mutation."""
    before = LyapunovGuard().evaluate(phases, before_knm)
    after = LyapunovGuard().evaluate(phases, after_knm)
    return {
        "before_V": before.V,
        "after_V": after.V,
        "delta_V": after.V - before.V,
        "non_increasing": after.V <= before.V,
        "before_in_basin": before.in_basin,
        "after_in_basin": after.in_basin,
        "after_max_phase_diff": after.max_phase_diff,
    }


def run_demo() -> dict[str, object]:
    """Run one TE-supported topology mutation for network security."""
    spec = load_binding_spec(SPEC_PATH)
    errors = validate_binding_spec(spec)
    if errors:
        raise ValueError(f"invalid network_security binding spec: {errors}")

    history = network_security_phase_history()
    knm, te = network_security_te_supported_knm(history)
    phases = network_security_phase_snapshot()
    policy = TopologyMutationPolicy(
        mutation_rate=0.35,
        coherence_floor=0.78,
        pairwise_threshold=0.85,
        simplex_threshold=0.995,
        max_new_simplices=3,
        max_simplex_strength=0.22,
        simplex_pairwise_support_floor=0.10,
    )
    result = HigherOrderTopologySupervisor(policy).mutate(phases, knm)
    lyapunov = topology_lyapunov_validation(phases, knm, result.knm)
    te_threshold = 0.25 * float(np.max(te))
    te_edges = int(np.count_nonzero(te > te_threshold)) if te_threshold > 0.0 else 0
    return {
        "domainpack": spec.name,
        "scenario": "lateral_movement_te_supported_topology_mutation",
        "policy": {
            "mutation_rate": policy.mutation_rate,
            "coherence_floor": policy.coherence_floor,
            "simplex_pairwise_support_floor": policy.simplex_pairwise_support_floor,
            "te_support_threshold": te_threshold,
        },
        "transfer_entropy": {
            "max": float(np.max(te)),
            "support_edges": te_edges,
        },
        "lyapunov_validation": lyapunov,
        "audit": result.to_audit_record(),
        "actuating": False,
    }


def main() -> None:
    """Print the demo audit payload as stable JSON."""
    print(json.dumps(run_demo(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

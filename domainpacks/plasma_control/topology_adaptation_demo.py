# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Plasma topology adaptation demo

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.binding import load_binding_spec, validate_binding_spec
from scpn_phase_orchestrator.coupling.knm import CouplingBuilder
from scpn_phase_orchestrator.monitor.lyapunov import LyapunovGuard
from scpn_phase_orchestrator.supervisor import (
    HigherOrderTopologySupervisor,
    TopologyMutationPolicy,
)

FloatArray: TypeAlias = NDArray[np.float64]

SPEC_PATH = Path(__file__).with_name("binding_spec.yaml")


def plasma_demo_phases() -> FloatArray:
    """Return a deterministic low-global, locally coherent plasma phase state."""
    return np.array(
        [
            0.00,
            0.02,
            0.04,
            np.pi,
            np.pi + 0.02,
            np.pi + 0.04,
            1.70,
            4.30,
        ],
        dtype=np.float64,
    )


def topology_lyapunov_validation(
    phases: FloatArray,
    before_knm: FloatArray,
    after_knm: FloatArray,
) -> dict[str, object]:
    """Return Lyapunov evidence for a proposed plasma topology mutation."""
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
    """Run one guarded higher-order topology mutation for plasma control."""
    spec = load_binding_spec(SPEC_PATH)
    errors = validate_binding_spec(spec)
    if errors:
        raise ValueError(f"invalid plasma_control binding spec: {errors}")

    coupling = CouplingBuilder().build(
        n_layers=len(spec.layers),
        base_strength=spec.coupling.base_strength,
        decay_alpha=spec.coupling.decay_alpha,
    )
    policy = TopologyMutationPolicy(
        mutation_rate=0.35,
        coherence_floor=0.78,
        simplex_threshold=0.995,
        max_new_simplices=2,
        max_simplex_strength=0.25,
        simplex_pairwise_support_floor=0.12,
    )
    phases = plasma_demo_phases()
    result = HigherOrderTopologySupervisor(policy).mutate(phases, coupling.knm)
    lyapunov = topology_lyapunov_validation(phases, coupling.knm, result.knm)
    return {
        "domainpack": spec.name,
        "policy": {
            "mutation_rate": policy.mutation_rate,
            "coherence_floor": policy.coherence_floor,
            "simplex_pairwise_support_floor": policy.simplex_pairwise_support_floor,
        },
        "lyapunov_validation": lyapunov,
        "audit": result.to_audit_record(),
    }


def main() -> None:
    """Print the demo audit payload as stable JSON."""
    print(json.dumps(run_demo(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

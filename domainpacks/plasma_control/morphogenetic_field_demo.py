# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Plasma-control morphogenetic field demo

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.binding import load_binding_spec, validate_binding_spec
from scpn_phase_orchestrator.coupling.knm import CouplingBuilder
from scpn_phase_orchestrator.supervisor import (
    MorphogeneticFieldPolicy,
    MorphogeneticTopologySupervisor,
    build_morphogenetic_field_snapshot,
)

FloatArray: TypeAlias = NDArray[np.float64]

SPEC_PATH = Path(__file__).with_name("binding_spec.yaml")


def plasma_edge_localised_demo_phases() -> FloatArray:
    """Return deterministic layer phases for an edge-localised stress replay."""
    return np.array(
        [
            2.60,
            0.50,
            2.80,
            3.05,
            0.00,
            0.03,
            0.06,
            3.10,
        ],
        dtype=np.float64,
    )


def run_demo() -> dict[str, object]:
    """Run one morphogenetic topology-field step for plasma control."""
    spec = load_binding_spec(SPEC_PATH)
    errors = validate_binding_spec(spec)
    if errors:
        raise ValueError(f"invalid plasma_control binding spec: {errors}")

    coupling = CouplingBuilder().build(
        n_layers=len(spec.layers),
        base_strength=spec.coupling.base_strength,
        decay_alpha=spec.coupling.decay_alpha,
    )
    policy = MorphogeneticFieldPolicy(
        growth_rate=0.40,
        shrink_rate=0.25,
        diffusion_rate=0.14,
        coherence_target=0.78,
        max_delta=0.06,
        max_coupling=1.0,
    )
    result = MorphogeneticTopologySupervisor(policy).step(
        plasma_edge_localised_demo_phases(),
        coupling.knm,
    )
    snapshot = build_morphogenetic_field_snapshot(result, top_k=10)
    return {
        "domainpack": spec.name,
        "scenario": "edge_localised_transport_barrier_stress",
        "policy": {
            "growth_rate": policy.growth_rate,
            "shrink_rate": policy.shrink_rate,
            "diffusion_rate": policy.diffusion_rate,
            "coherence_target": policy.coherence_target,
            "max_delta": policy.max_delta,
        },
        "audit": result.to_audit_record(),
        "snapshot": snapshot.to_audit_record(),
        "actuating": False,
    }


def main() -> None:
    """Print the demo audit payload as stable JSON."""
    print(json.dumps(run_demo(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

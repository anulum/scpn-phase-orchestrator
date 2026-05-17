# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Power-grid hierarchy transport demo

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.binding import load_binding_spec, validate_binding_spec
from scpn_phase_orchestrator.supervisor import (
    ChildSupervisorSummary,
    HierarchySyncEnvelope,
    HierarchyTransportRuntime,
    build_hierarchy_sync_envelope,
    handle_hierarchy_frame,
    handle_hierarchy_rest_payload,
    replay_hierarchy_jsonl,
)
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

FloatArray: TypeAlias = NDArray[np.float64]

SPEC_PATH = Path(__file__).with_name("binding_spec.yaml")
CHANNEL_COUNT = 2


def power_grid_transport_states() -> dict[str, FloatArray]:
    """Return deterministic edge-region phases for hierarchy transport replay."""
    return {
        "generation_area": np.array(
            [0.0, 0.05, 0.10, 0.16, 0.21],
            dtype=np.float64,
        ),
        "demand_renewable_area": np.array(
            [0.0, 2.30, 4.10, 1.20, 3.60],
            dtype=np.float64,
        ),
    }


def build_power_grid_transport_envelopes() -> tuple[HierarchySyncEnvelope, ...]:
    """Build deterministic reduced-summary envelopes for two grid regions."""
    states = power_grid_transport_states()
    envelopes = []
    for sequence, (name, phases) in enumerate(states.items(), start=11):
        observed_r, observed_psi = compute_order_parameter(phases)
        summary = ChildSupervisorSummary(
            name=name,
            channel="P" if name == "generation_area" else "I",
            R=observed_r,
            psi=observed_psi,
            regime="nominal" if observed_r >= 0.65 else "degraded",
            confidence=0.95 if name == "generation_area" else 0.82,
            metadata={"source": "pmu_transport", "oscillators": phases.size},
        )
        envelopes.append(
            build_hierarchy_sync_envelope(
                summary,
                source_node=f"grid-edge-{sequence}",
                sequence=sequence,
                monotonic_time_s=sequence * 0.02,
            )
        )
    return tuple(envelopes)


def _to_records(
    envelopes: tuple[HierarchySyncEnvelope, ...],
) -> tuple[dict[str, object], ...]:
    return tuple(envelope.to_audit_record() for envelope in envelopes)


def run_demo() -> dict[str, object]:
    """Run all hierarchy transport boundaries for replay-only summary transport."""
    spec = load_binding_spec(SPEC_PATH)
    errors = validate_binding_spec(spec)
    if errors:
        raise ValueError(f"invalid power_grid binding spec: {errors}")

    envelopes = build_power_grid_transport_envelopes()
    records = _to_records(envelopes)

    rest = handle_hierarchy_rest_payload(
        {"envelopes": records},
        headers={"content-type": "application/json"},
        runtime=HierarchyTransportRuntime(
            hierarchy="power_grid_edge_cloud_summary_sync",
            degraded_threshold=0.65,
            critical_threshold=0.35,
            min_confidence=0.5,
        ),
    )

    frame = handle_hierarchy_frame(
        {
            "type": "hierarchy_sync_batch",
            "payload": {"envelopes": (records[1], records[0])},
        },
        runtime=HierarchyTransportRuntime(
            hierarchy="power_grid_edge_cloud_summary_sync",
            degraded_threshold=0.65,
            critical_threshold=0.35,
            min_confidence=0.5,
        ),
    )

    jsonl = replay_hierarchy_jsonl(
        (records[0], envelopes[1].to_json()),
        runtime=HierarchyTransportRuntime(
            hierarchy="power_grid_edge_cloud_summary_sync",
            degraded_threshold=0.65,
            critical_threshold=0.35,
            min_confidence=0.5,
        ),
    )

    return {
        "domainpack": spec.name,
        "scenario": "hierarchy_summary_transport_replay",
        "network_opened": True,
        "channel_count": CHANNEL_COUNT,
        "accepted_count": rest.accepted_count,
        "channels": ["P", "I"],
        "accepted_names": [
            record["summary"]["name"] for record in rest.to_audit_record()["ledger"]["accepted"]
        ],
        "rest_boundary": rest.to_audit_record(),
        "websocket_frame": frame.to_audit_record(),
        "jsonl_replay": jsonl.to_audit_record(),
    }


def main() -> None:
    """Print the transport demo payload as stable JSON."""
    print(json.dumps(run_demo(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

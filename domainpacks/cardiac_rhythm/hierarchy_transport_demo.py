# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Cardiac hierarchy transport demo

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


def cardiac_transport_states() -> dict[str, FloatArray]:
    """Return deterministic conduction-axis phases for hierarchy transport replay."""
    return {
        "pacemaker_atrial_axis": np.array(
            [0.00, 0.04, 0.08, 0.54, 0.58],
            dtype=np.float64,
        ),
        "ventricular_recovery_axis": np.array(
            [2.70, 3.05, 3.40, 3.72, 4.05],
            dtype=np.float64,
        ),
    }


def build_cardiac_transport_envelopes() -> tuple[HierarchySyncEnvelope, ...]:
    """Build deterministic reduced-summary envelopes for cardiac axes."""
    states = cardiac_transport_states()
    envelopes = []
    for sequence, (name, phases) in enumerate(states.items(), start=21):
        observed_r, observed_psi = compute_order_parameter(phases)
        summary = ChildSupervisorSummary(
            name=name,
            channel="P",
            R=observed_r,
            psi=observed_psi,
            regime="nominal" if observed_r >= 0.65 else "degraded",
            confidence=0.9 if name == "pacemaker_atrial_axis" else 0.78,
            metadata={"source": "conduction_transport", "oscillators": phases.size},
        )
        envelopes.append(
            build_hierarchy_sync_envelope(
                summary,
                source_node=f"cardiac-edge-{sequence}",
                sequence=sequence,
                monotonic_time_s=sequence * 0.01,
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
        raise ValueError(f"invalid cardiac_rhythm binding spec: {errors}")

    envelopes = build_cardiac_transport_envelopes()
    records = _to_records(envelopes)

    rest = handle_hierarchy_rest_payload(
        {"envelopes": records},
        headers={"content-type": "application/json"},
        runtime=HierarchyTransportRuntime(
            hierarchy="cardiac_edge_cloud_summary_sync",
            degraded_threshold=0.65,
            critical_threshold=0.35,
            min_confidence=0.5,
        ),
    )

    frame = handle_hierarchy_frame(
        {"kind": "hierarchy_sync_batch", "payload": {"envelopes": records}},
        runtime=HierarchyTransportRuntime(
            hierarchy="cardiac_edge_cloud_summary_sync",
            degraded_threshold=0.65,
            critical_threshold=0.35,
            min_confidence=0.5,
        ),
    )

    jsonl = replay_hierarchy_jsonl(
        (records[0], envelopes[1].to_json()),
        runtime=HierarchyTransportRuntime(
            hierarchy="cardiac_edge_cloud_summary_sync",
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
            record["summary"]["name"]
            for record in rest.to_audit_record()["ledger"]["accepted"]
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

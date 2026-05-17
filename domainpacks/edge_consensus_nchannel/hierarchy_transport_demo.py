# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Edge-consensus hierarchy transport demo

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
CHANNELS = ("P", "I", "S", "Load", "Trust", "ConsensusHealth")
NODES = ("leaf_cluster", "regional_gateway", "parent_supervisor")
CHANNEL_COUNT = len(CHANNELS)


def edge_consensus_transport_states() -> dict[str, FloatArray]:
    """Return deterministic per-node 6-channel phase states."""
    return {
        "leaf_cluster": np.array(
            [0.00, 0.11, 2.02, 0.20, 0.30, 0.44],
            dtype=np.float64,
        ),
        "regional_gateway": np.array(
            [0.01, 0.09, 2.00, 0.30, 0.75, 0.55],
            dtype=np.float64,
        ),
        "parent_supervisor": np.array(
            [0.00, 0.14, 1.98, 0.40, 0.60, 0.50],
            dtype=np.float64,
        ),
    }


def build_edge_consensus_transport_envelopes() -> tuple[HierarchySyncEnvelope, ...]:
    """Build deterministic reduced-summary envelopes for each node."""
    states = edge_consensus_transport_states()
    envelopes = []
    for sequence, (name, channel_values) in enumerate(states.items(), start=31):
        observed_r, observed_psi = compute_order_parameter(channel_values)
        summary = ChildSupervisorSummary(
            name=name,
            channel="P",
            R=observed_r,
            psi=observed_psi,
            regime="nominal" if observed_r >= 0.65 else "degraded",
            confidence=0.91 if name == "leaf_cluster" else 0.83,
            metadata={
                "source": "edge_consensus_transport",
                "channel_count": len(CHANNELS),
            },
        )
        envelopes.append(
            build_hierarchy_sync_envelope(
                summary,
                source_node=f"edge-node-{sequence}",
                sequence=sequence,
                monotonic_time_s=sequence * 0.05,
            )
        )
    return tuple(envelopes)


def _to_records(
    envelopes: tuple[HierarchySyncEnvelope, ...],
) -> tuple[dict[str, object], ...]:
    return tuple(envelope.to_audit_record() for envelope in envelopes)


def run_demo() -> dict[str, object]:
    """Run all hierarchy transport boundaries for replay-only edge transport."""
    spec = load_binding_spec(SPEC_PATH)
    errors = validate_binding_spec(spec)
    if errors:
        raise ValueError(f"invalid edge_consensus_nchannel binding spec: {errors}")

    envelopes = build_edge_consensus_transport_envelopes()
    records = _to_records(envelopes)

    rest = handle_hierarchy_rest_payload(
        {"envelopes": records},
        headers={"content-type": "application/json"},
        runtime=HierarchyTransportRuntime(
            hierarchy="edge_cloud_summary_sync",
            degraded_threshold=0.65,
            critical_threshold=0.35,
            min_confidence=0.5,
        ),
    )

    frame = handle_hierarchy_frame(
        {"type": "hierarchy_sync_batch", "payload": records},
        runtime=HierarchyTransportRuntime(
            hierarchy="edge_cloud_summary_sync",
            degraded_threshold=0.65,
            critical_threshold=0.35,
            min_confidence=0.5,
        ),
    )

    jsonl = replay_hierarchy_jsonl(
        (records[1], envelopes[0].to_json(), records[2]),
        runtime=HierarchyTransportRuntime(
            hierarchy="edge_cloud_summary_sync",
            degraded_threshold=0.65,
            critical_threshold=0.35,
            min_confidence=0.5,
        ),
    )

    return {
        "domainpack": spec.name,
        "scenario": "heterogeneous_node_transport_replay",
        "network_opened": True,
        "channel_count": CHANNEL_COUNT,
        "channels": list(CHANNELS),
        "nodes": list(NODES),
        "accepted_count": rest.accepted_count,
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

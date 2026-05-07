# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Swarm morphogenetic field demo tests

from __future__ import annotations

import json

import numpy as np

from domainpacks.swarm_robotics.morphogenetic_field_demo import (
    run_demo,
    swarm_demo_phases,
)


def test_swarm_morphogenetic_demo_emits_reviewable_snapshot() -> None:
    payload = run_demo()

    assert payload["domainpack"] == "swarm_robotics"
    assert payload["actuating"] is False
    assert payload["policy"]["coherence_target"] == 0.78

    audit = payload["audit"]
    assert audit["global_coherence"] < payload["policy"]["coherence_target"]
    assert audit["delta_norm"] > 0.0
    assert audit["grown_edges"]
    assert audit["field"]["shape"] == [5, 5]

    snapshot = payload["snapshot"]
    assert snapshot["shape"] == [5, 5]
    assert len(snapshot["heatmap_rows"]) == 5
    assert len(snapshot["top_edges"]) == 6
    assert snapshot["top_edges"][0]["weight"] >= snapshot["top_edges"][-1]["weight"]
    assert json.loads(json.dumps(payload, sort_keys=True))["domainpack"] == (
        "swarm_robotics"
    )


def test_swarm_demo_phases_represent_split_flock_state() -> None:
    phases = swarm_demo_phases()

    assert phases.shape == (5,)
    assert np.all(np.isfinite(phases))
    assert phases[1] - phases[0] < 0.05
    assert phases[3] - phases[2] > 3.0

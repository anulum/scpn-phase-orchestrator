# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Traffic topology adaptation demo tests

from __future__ import annotations

import numpy as np

from domainpacks.traffic_flow.topology_adaptation_demo import (
    run_demo,
    traffic_demo_phase_history,
    transfer_entropy_supported_knm,
)


def test_traffic_topology_demo_adds_te_supported_simplices() -> None:
    payload = run_demo()

    assert payload["domainpack"] == "traffic_flow"
    assert payload["transfer_entropy"]["max"] > 0.0
    assert payload["transfer_entropy"]["support_edges"] > 0
    audit = payload["audit"]
    assert isinstance(audit, dict)
    assert audit["global_coherence"] < payload["policy"]["coherence_floor"]
    assert audit["pairwise_delta_norm"] > 0.0
    assert audit["added_simplices"]
    assert audit["hyperedge_count"] >= len(audit["added_simplices"])


def test_transfer_entropy_support_knm_is_symmetric_and_bounded() -> None:
    history = traffic_demo_phase_history()
    knm, te = transfer_entropy_supported_knm(history)

    assert history.shape == (6, 80)
    assert te.shape == (6, 6)
    assert knm.shape == (6, 6)
    assert np.allclose(knm, knm.T)
    assert np.all(np.diag(knm) == 0.0)
    assert float(np.max(knm)) <= 0.20
    assert float(np.max(te)) > 0.0


def test_traffic_phase_history_rejects_too_few_steps() -> None:
    try:
        traffic_demo_phase_history(steps=7)
    except ValueError as exc:
        assert "steps" in str(exc)
    else:
        raise AssertionError("expected ValueError for too-short history")

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Studio public facade tests

from __future__ import annotations

from math import log

import numpy as np
import pytest

import scpn_phase_orchestrator.studio as studio
from scpn_phase_orchestrator.supervisor import (
    MorphogeneticFieldState,
    evaluate_strange_loop_drift_scenarios,
    render_morphogenetic_field_svg,
)


def test_public_studio_facade_exports_passive_physics_review_panels() -> None:
    """The Studio facade exposes named passive review panels for operator UI use."""
    n_bins = 8
    integrated_record = {
        "monitor": "integrated_information",
        "phi": 0.1,
        "normalised_phi": 0.1 / log(n_bins),
        "total_integration": 0.2,
        "minimum_partition": [[0], [1]],
        "pairwise_mi": [[0.5, 0.1], [0.1, 0.5]],
        "n_bins": n_bins,
        "claim_boundary": "engineering_proxy_not_theoretical_iit",
    }
    strange_loop_records = [
        result.to_audit_record() for result in evaluate_strange_loop_drift_scenarios()
    ]
    morphogenetic_artifact = render_morphogenetic_field_svg(
        MorphogeneticFieldState(
            np.array(
                [
                    [0.0, 0.8],
                    [0.3, 0.0],
                ],
                dtype=np.float64,
            )
        ),
        top_k=2,
    ).to_audit_record()

    integrated_panel = studio.build_integrated_information_panel([integrated_record])
    strange_loop_panel = studio.build_strange_loop_studio_panel(strange_loop_records)
    morphogenetic_panel = studio.build_morphogenetic_field_studio_panel(
        morphogenetic_artifact
    )

    assert integrated_panel["claim_boundary"] == (
        "engineering_proxy_not_theoretical_iit"
    )
    assert integrated_panel["consciousness_claim_permitted"] is False
    assert strange_loop_panel["claim_boundary"] == (
        "strange_loop_drift_review_not_live_actuation"
    )
    assert strange_loop_panel["actuation_permitted"] is False
    assert morphogenetic_panel["panel_kind"] == ("studio_morphogenetic_field_panel")
    assert morphogenetic_panel["actuation_permitted"] is False
    assert morphogenetic_panel["strongest_edge"] == {
        "source": 0,
        "target": 1,
        "weight": pytest.approx(0.8),
    }
    assert "build_multiverse_counterfactual_studio_panel" in studio.__all__
    assert callable(studio.build_multiverse_counterfactual_studio_panel)
    assert "build_hybrid_order_studio_panel" in studio.__all__
    assert callable(studio.build_hybrid_order_studio_panel)

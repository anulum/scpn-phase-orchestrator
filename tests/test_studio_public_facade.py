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
from scpn_phase_orchestrator.binding.semantic import compile_symbolic_binding
from scpn_phase_orchestrator.binding.topos_semantic import (
    validate_symbolic_binding_functor,
)
from scpn_phase_orchestrator.supervisor import (
    MorphogeneticFieldState,
    build_autopoietic_lineage_replay_corpus,
    build_autopoietic_lineage_sandbox,
    build_intergenerational_policy_inheritance,
    build_intergenerational_policy_inheritance_history,
    evaluate_strange_loop_drift_scenarios,
    render_morphogenetic_field_svg,
)
from scpn_phase_orchestrator.supervisor.evolutionary_search import (
    run_offline_evolutionary_supervisor_search,
)
from scpn_phase_orchestrator.supervisor.information_geometry import (
    propose_information_geometry_control,
)
from scpn_phase_orchestrator.supervisor.policy_rules import (
    PolicyAction,
    PolicyCondition,
    PolicyRule,
)
from scpn_phase_orchestrator.supervisor.topos_policy import (
    validate_policy_composition_category,
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
    information_geometry_record = propose_information_geometry_control(
        [0.2, 0.3, 0.5],
        [0.3, 0.3, 0.4],
        max_step=0.05,
    ).to_audit_record()
    topos_artifacts = compile_symbolic_binding(
        "1-layer public facade categorical binding",
        name="studio_facade_topos",
        oscillators_per_layer=1,
        dry_run_steps=1,
    )
    topos_symbolic_report = validate_symbolic_binding_functor(
        topos_artifacts
    ).to_audit_record()
    topos_policy_report = validate_policy_composition_category(
        [
            PolicyRule(
                name="studio_facade_topos_guard",
                regimes=["DEGRADED"],
                condition=PolicyCondition(
                    metric="R",
                    layer=0,
                    op="<",
                    threshold=0.6,
                ),
                actions=[PolicyAction(knob="K", scope="global", value=0.05, ttl_s=3.0)],
            )
        ]
    ).to_audit_record()
    evolutionary_report = run_offline_evolutionary_supervisor_search(
        {"K": 0.4},
        [
            {
                "replay_id": "facade_nominal",
                "reward": 0.91,
                "safety_margin": 0.08,
                "violations": [],
            }
        ],
        stl_spec="always (R >= 0.8)",
        trace={"R": [0.9, 0.89]},
        generation_count=1,
        population_size=2,
        mutation_step=0.02,
    ).to_audit_record()
    lineage_manifest = build_autopoietic_lineage_sandbox(
        {"K": 0.42, "alpha": 0.18},
        build_autopoietic_lineage_replay_corpus(),
        child_budget=2,
        mutation_step=0.02,
        minimum_replay_reward=0.7,
        minimum_safety_margin=0.1,
    )
    inheritance_manifests = [
        build_intergenerational_policy_inheritance(
            lineage_manifest,
            child,
            signer_id="studio-facade-review-key",
            signing_key="studio-facade-local-signing-key",
        )
        for child in lineage_manifest["child_candidates"]
        if child["status"] == "accepted_for_review"
    ]
    inheritance_history = build_intergenerational_policy_inheritance_history(
        lineage_manifest,
        inheritance_manifests,
    )
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
    information_geometry_panel = studio.build_information_geometry_studio_panel(
        [information_geometry_record]
    )
    topos_panel = studio.build_topos_semantic_binding_studio_panel(
        [topos_symbolic_report],
        [topos_policy_report],
    )
    evolutionary_panel = (
        studio.build_evolutionary_supervisor_policy_search_studio_panel(
            [evolutionary_report]
        )
    )
    lineage_panel = studio.build_autopoietic_lineage_studio_panel([lineage_manifest])
    inheritance_panel = studio.build_intergenerational_inheritance_studio_panel(
        [inheritance_history]
    )
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
    assert information_geometry_panel["claim_boundary"] == (
        "information_geometry_control_not_live_actuation"
    )
    assert information_geometry_panel["actuation_permitted"] is False
    assert topos_panel["proof_boundary"] == (
        "categorical_validation_prototype_not_formal_topos_proof"
    )
    assert topos_panel["formal_proof_claim_permitted"] is False
    assert topos_panel["actuation_permitted"] is False
    assert evolutionary_panel["claim_boundary"] == (
        "offline_evolutionary_supervisor_review_not_live_actuation"
    )
    assert evolutionary_panel["actuation_permitted"] is False
    assert lineage_panel["claim_boundary"] == (
        "autopoietic_lineage_sandbox_review_not_live_merge"
    )
    assert lineage_panel["execution_disabled"] is True
    assert lineage_panel["actuation_permitted"] is False
    assert lineage_panel["replay_domain_count"] == 4
    assert "build_autopoietic_lineage_studio_panel" in studio.__all__
    assert inheritance_panel["claim_boundary"] == (
        "intergenerational_inheritance_review_not_direct_hot_patch"
    )
    assert inheritance_panel["direct_hot_patch_permitted"] is False
    assert inheritance_panel["actuation_permitted"] is False
    assert inheritance_panel["history_record_total"] == 2
    assert "build_intergenerational_inheritance_studio_panel" in studio.__all__
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
    assert "build_information_geometry_studio_panel" in studio.__all__
    assert callable(studio.build_information_geometry_studio_panel)
    assert "build_topos_semantic_binding_studio_panel" in studio.__all__
    assert callable(studio.build_topos_semantic_binding_studio_panel)
    assert "build_evolutionary_supervisor_policy_search_studio_panel" in studio.__all__
    assert callable(studio.build_evolutionary_supervisor_policy_search_studio_panel)

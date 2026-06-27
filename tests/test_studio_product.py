# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Studio product manifest tests

from __future__ import annotations

import json
import sys
from hashlib import sha256
from pathlib import Path

import pytest

import scpn_phase_orchestrator.studio as studio
from scpn_phase_orchestrator.studio.product import (
    STUDIO_REVIEW_PANEL_REGISTRY,
    PanelRecord,
    build_studio_product_manifest,
)


def test_studio_product_manifest_exposes_review_only_physics_panels() -> None:
    """Standalone Studio product manifest lists passive physics review panels."""
    manifest = build_studio_product_manifest()

    assert manifest["manifest_kind"] == "studio_product_manifest"
    assert manifest["product"] == "spo_studio"
    assert manifest["standalone_shell"] == "tools/spo_studio.py"
    assert manifest["launch_command"] == "streamlit run tools/spo_studio.py"
    assert manifest["actuation_permitted"] is False
    assert manifest["network_opened"] is False
    assert manifest["hardware_write_permitted"] is False
    assert manifest["qpu_execution_permitted"] is False
    assert manifest["operator_review_required"] is True
    assert (
        manifest["manifest_sha256"]
        == sha256(
            json.dumps(
                {
                    key: value
                    for key, value in manifest.items()
                    if key != "manifest_sha256"
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    )

    panels = manifest["review_panels"]
    panel_ids = [panel["panel_id"] for panel in panels]
    assert len(panel_ids) == len(set(panel_ids))
    assert "evolutionary_supervisor_policy_search" in panel_ids
    assert "information_geometry_control" in panel_ids
    assert "integrated_information_monitor" in panel_ids
    assert "morphogenetic_field_topology" in panel_ids
    assert "multiverse_counterfactual_rollout" in panel_ids
    assert "hybrid_order_parameters" in panel_ids
    assert "topos_semantic_binding" in panel_ids
    assert "sheaf_cohomology_control" in panel_ids
    assert "autopoietic_lineage_sandbox" in panel_ids
    assert "digital_twin_confidence_review" in panel_ids
    assert "intergenerational_policy_inheritance" in panel_ids
    assert "strange_loop_meta_orchestrator" in panel_ids

    evolutionary_panel = next(
        panel
        for panel in panels
        if panel["panel_id"] == "evolutionary_supervisor_policy_search"
    )
    assert evolutionary_panel["builder"] == (
        "build_evolutionary_supervisor_policy_search_studio_panel"
    )
    assert evolutionary_panel["claim_boundary"] == (
        "offline_evolutionary_supervisor_review_not_live_actuation"
    )
    assert evolutionary_panel["source_rows"] == (
        "Evolutionary supervisor policy search",
    )
    assert evolutionary_panel["actuation_permitted"] is False
    assert evolutionary_panel["live_merge_permitted"] is False
    assert evolutionary_panel["hot_patch_permitted"] is False
    assert evolutionary_panel["execution_disabled"] is True
    assert evolutionary_panel["operator_review_required"] is True
    assert evolutionary_panel["required_evidence"] == (
        "offline replay-search audit report",
        "candidate-level SHA-256 hashes",
        "STL robustness and replay-safety metrics",
        "optional policy-DSL mutation report",
        "optional deterministic domain examples",
    )
    lineage_panel = next(
        panel for panel in panels if panel["panel_id"] == "autopoietic_lineage_sandbox"
    )
    assert lineage_panel["builder"] == "build_autopoietic_lineage_studio_panel"
    assert lineage_panel["claim_boundary"] == (
        "autopoietic_lineage_sandbox_review_not_live_merge"
    )
    assert lineage_panel["required_evidence"] == (
        "offline lineage sandbox manifest",
        "domain-labelled replay corpus rows",
        "child-policy SHA-256 hashes",
        "accepted and rejected child evidence",
    )
    assert lineage_panel["actuation_permitted"] is False
    assert lineage_panel["live_merge_permitted"] is False
    assert lineage_panel["hot_patch_permitted"] is False
    assert lineage_panel["execution_disabled"] is True
    assert lineage_panel["operator_review_required"] is True
    sheaf_panel = next(
        panel for panel in panels if panel["panel_id"] == "sheaf_cohomology_control"
    )
    assert sheaf_panel["builder"] == "build_sheaf_cohomology_studio_panel"
    assert sheaf_panel["claim_boundary"] == (
        "sheaf_cohomology_review_not_live_actuation"
    )
    assert sheaf_panel["required_evidence"] == (
        "sheaf-Laplacian obstruction audit record",
        "residual-edge triage summary",
        "cohomology-dimension evidence",
        "review-only control proposal",
    )
    assert sheaf_panel["actuation_permitted"] is False
    assert sheaf_panel["live_merge_permitted"] is False
    assert sheaf_panel["hot_patch_permitted"] is False
    assert sheaf_panel["execution_disabled"] is True
    assert sheaf_panel["operator_review_required"] is True
    twin_confidence_panel = next(
        panel
        for panel in panels
        if panel["panel_id"] == "digital_twin_confidence_review"
    )
    assert twin_confidence_panel["builder"] == "build_twin_confidence_studio_panel"
    assert twin_confidence_panel["claim_boundary"] == (
        "digital_twin_confidence_observability_not_actuation"
    )
    assert twin_confidence_panel["required_evidence"] == (
        "TwinConfidenceScore audit records",
        "TwinConfidenceSummary audit record",
        "score and summary SHA-256 hashes",
        "calibrated status and confidence aggregates",
    )
    assert twin_confidence_panel["actuation_permitted"] is False
    assert twin_confidence_panel["live_merge_permitted"] is False
    assert twin_confidence_panel["hot_patch_permitted"] is False
    assert twin_confidence_panel["execution_disabled"] is True
    assert twin_confidence_panel["operator_review_required"] is True
    inheritance_panel = next(
        panel
        for panel in panels
        if panel["panel_id"] == "intergenerational_policy_inheritance"
    )
    assert inheritance_panel["builder"] == (
        "build_intergenerational_inheritance_studio_panel"
    )
    assert inheritance_panel["claim_boundary"] == (
        "intergenerational_inheritance_review_not_direct_hot_patch"
    )
    assert inheritance_panel["required_evidence"] == (
        "signed inheritance-history package",
        "inheritance SHA-256 hashes",
        "HMAC signature metadata",
        "multi-objective replay-fitness rows",
    )
    assert inheritance_panel["direct_hot_patch_permitted"] is False
    assert inheritance_panel["actuation_permitted"] is False
    assert inheritance_panel["live_merge_permitted"] is False
    assert inheritance_panel["hot_patch_permitted"] is False
    assert inheritance_panel["execution_disabled"] is True
    assert inheritance_panel["operator_review_required"] is True
    for panel in panels:
        builder = panel["builder"]
        assert isinstance(builder, str)
        assert callable(getattr(studio, builder))


def test_studio_product_manifest_is_public_and_streamlit_free() -> None:
    manifest = studio.build_studio_product_manifest()

    assert "build_studio_product_manifest" in studio.__all__
    assert manifest == build_studio_product_manifest()
    assert "streamlit" not in sys.modules


def test_studio_product_manifest_rejects_malformed_panel_ids() -> None:
    """The standalone manifest rejects unnamed or duplicated panel records."""
    first_panel = dict(STUDIO_REVIEW_PANEL_REGISTRY[0])
    missing_id_panel = {**first_panel, "panel_id": ""}

    duplicate_registry = (
        first_panel,
        {**dict(STUDIO_REVIEW_PANEL_REGISTRY[1]), "panel_id": first_panel["panel_id"]},
    )

    with pytest.raises(ValueError, match="non-empty panel_id"):
        build_studio_product_manifest(panel_registry=(missing_id_panel,))

    with pytest.raises(ValueError, match="panel_id values must be unique"):
        build_studio_product_manifest(panel_registry=duplicate_registry)


def test_studio_product_manifest_rejects_enabled_panel_gates() -> None:
    """The product manifest fails closed if a panel allows execution."""
    unsafe_panel: PanelRecord = {
        **dict(STUDIO_REVIEW_PANEL_REGISTRY[0]),
        "execution_disabled": False,
    }

    with pytest.raises(ValueError, match="execution_disabled invalid"):
        build_studio_product_manifest(panel_registry=(unsafe_panel,))


def test_standalone_streamlit_shell_uses_product_manifest() -> None:
    source = Path("tools/spo_studio.py").read_text(encoding="utf-8")

    assert "build_studio_product_manifest" in source
    assert "Passive review panels" in source

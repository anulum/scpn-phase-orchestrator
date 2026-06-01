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

import scpn_phase_orchestrator.studio as studio
from scpn_phase_orchestrator.studio.product import build_studio_product_manifest


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
    for panel in panels:
        builder = panel["builder"]
        assert isinstance(builder, str)
        assert callable(getattr(studio, builder))


def test_studio_product_manifest_is_public_and_streamlit_free() -> None:
    manifest = studio.build_studio_product_manifest()

    assert "build_studio_product_manifest" in studio.__all__
    assert manifest == build_studio_product_manifest()
    assert "streamlit" not in sys.modules


def test_standalone_streamlit_shell_uses_product_manifest() -> None:
    source = Path("tools/spo_studio.py").read_text(encoding="utf-8")

    assert "build_studio_product_manifest" in source
    assert "Passive review panels" in source

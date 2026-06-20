# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SPO Studio ui_helpers package wiring tests

"""Wiring contract for the ``studio.ui_helpers`` package.

Asserts that every responsibility submodule imports under its dotted path and
that the package ``__init__`` re-exports each public symbol as the *same object*
defined in its owning submodule, so the split preserves the original flat
``ui_helpers`` import surface with zero consumer churn.
"""

from __future__ import annotations

import importlib

import pytest

import scpn_phase_orchestrator.studio.ui_helpers as ui

PACKAGE = "scpn_phase_orchestrator.studio.ui_helpers"

# Full dotted paths spelled out so the module-linkage guard
# (tools/check_test_module_linkage.py) finds each submodule's import path in the
# test corpus.
SUBMODULES = (
    "scpn_phase_orchestrator.studio.ui_helpers._shared",
    "scpn_phase_orchestrator.studio.ui_helpers._state",
    "scpn_phase_orchestrator.studio.ui_helpers.canvas",
    "scpn_phase_orchestrator.studio.ui_helpers.charts",
    "scpn_phase_orchestrator.studio.ui_helpers.connectors",
    "scpn_phase_orchestrator.studio.ui_helpers.deployment",
    "scpn_phase_orchestrator.studio.ui_helpers.guidance",
    "scpn_phase_orchestrator.studio.ui_helpers.hardware",
    "scpn_phase_orchestrator.studio.ui_helpers.panel_evolutionary",
    "scpn_phase_orchestrator.studio.ui_helpers.panel_hybrid_order",
    "scpn_phase_orchestrator.studio.ui_helpers.panel_information_geometry",
    "scpn_phase_orchestrator.studio.ui_helpers.panel_lineage",
    "scpn_phase_orchestrator.studio.ui_helpers.panel_morphogenetic",
    "scpn_phase_orchestrator.studio.ui_helpers.panel_multiverse",
    "scpn_phase_orchestrator.studio.ui_helpers.panel_sheaf",
    "scpn_phase_orchestrator.studio.ui_helpers.panel_strange_loop",
    "scpn_phase_orchestrator.studio.ui_helpers.panel_topos",
    "scpn_phase_orchestrator.studio.ui_helpers.replay",
    "scpn_phase_orchestrator.studio.ui_helpers.tables",
)

# Owning submodule for each public symbol re-exported by the package __init__.
PUBLIC_SYMBOL_MODULE = {
    "StudioKnobState": "_state",
    "StudioReplayResult": "_state",
    "apply_canvas_binding_rewrite_candidate": "canvas",
    "apply_knob_update": "_state",
    "binding_spec_project_state": "guidance",
    "build_autopoietic_lineage_studio_panel": "panel_lineage",
    "build_beginner_guidance": "guidance",
    "build_canvas_binding_rewrite_candidate": "canvas",
    "build_canvas_edit_artifact": "canvas",
    "build_canvas_graph": "canvas",
    "build_canvas_interaction_state": "canvas",
    "build_canvas_layout_manifest": "canvas",
    "build_canvas_topology_patch": "canvas",
    "build_command_table": "deployment",
    "build_deployment_package": "deployment",
    "build_deployment_readiness": "deployment",
    "build_error_report": "guidance",
    "build_evolutionary_supervisor_policy_search_studio_panel": "panel_evolutionary",
    "build_export_manifests": "deployment",
    "build_hardware_target_package": "hardware",
    "build_hybrid_order_studio_panel": "panel_hybrid_order",
    "build_information_geometry_studio_panel": "panel_information_geometry",
    "build_integrated_information_panel": "charts",
    "build_intergenerational_inheritance_studio_panel": "panel_lineage",
    "build_layer_table": "tables",
    "build_live_connector_plan": "connectors",
    "build_live_connector_run_record": "connectors",
    "build_morphogenetic_field_studio_panel": "panel_morphogenetic",
    "build_multiverse_counterfactual_studio_panel": "panel_multiverse",
    "build_operator_checklist": "deployment",
    "build_oscillator_edit_artifact": "canvas",
    "build_oscillator_table": "tables",
    "build_owned_live_connector_runtime_record": "connectors",
    "build_package_materialisation_plan": "deployment",
    "build_regime_chart_payload": "charts",
    "build_runtime_snapshot": "guidance",
    "build_series_chart_payload": "charts",
    "build_service_process_manifest": "deployment",
    "build_sheaf_cohomology_studio_panel": "panel_sheaf",
    "build_strange_loop_studio_panel": "panel_strange_loop",
    "build_topos_semantic_binding_studio_panel": "panel_topos",
    "build_verified_hardware_target_package": "hardware",
    "disabled_export_reasons": "deployment",
    "discover_domainpacks": "_state",
    "run_binding_spec_replay": "replay",
}

# Private helpers and the external loader the original flat module exposed as
# attributes and which tests still reach through the package namespace.
NON_PUBLIC_REEXPORTS = {
    "_run_owned_live_adapter": "connectors",
    "_stable_json_payload": "connectors",
    "_validate_candidate_binding_yaml": "canvas",
}


@pytest.mark.parametrize("dotted_path", SUBMODULES)
def test_submodule_imports_under_dotted_path(dotted_path: str) -> None:
    assert dotted_path.startswith(f"{PACKAGE}.")
    module = importlib.import_module(dotted_path)
    assert module.__name__ == dotted_path
    assert (module.__doc__ or "").strip(), f"{dotted_path} has no module docstring"


def test_public_symbol_module_table_covers_dunder_all() -> None:
    assert set(PUBLIC_SYMBOL_MODULE) == set(ui.__all__)
    assert len(ui.__all__) == len(set(ui.__all__)), "duplicate names in __all__"


@pytest.mark.parametrize("symbol,module_name", sorted(PUBLIC_SYMBOL_MODULE.items()))
def test_public_symbol_reexport_is_owning_module_object(
    symbol: str, module_name: str
) -> None:
    owner = importlib.import_module(f"{PACKAGE}.{module_name}")
    assert getattr(ui, symbol) is getattr(owner, symbol)


@pytest.mark.parametrize("symbol,module_name", sorted(NON_PUBLIC_REEXPORTS.items()))
def test_non_public_reexport_is_owning_module_object(
    symbol: str, module_name: str
) -> None:
    owner = importlib.import_module(f"{PACKAGE}.{module_name}")
    assert getattr(ui, symbol) is getattr(owner, symbol)


def test_load_binding_spec_mirrors_binding_loader() -> None:
    from scpn_phase_orchestrator.binding.loader import load_binding_spec

    assert ui.load_binding_spec is load_binding_spec


def test_every_reexport_owner_is_a_real_submodule() -> None:
    owners = set(PUBLIC_SYMBOL_MODULE.values()) | set(NON_PUBLIC_REEXPORTS.values())
    submodule_names = {path.rsplit(".", 1)[1] for path in SUBMODULES}
    assert owners <= submodule_names

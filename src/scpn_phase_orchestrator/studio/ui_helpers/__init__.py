# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SPO Studio pure UI helpers

"""Pure helper layer for the SPO Studio Streamlit surface."""

from __future__ import annotations

from scpn_phase_orchestrator.binding.loader import (
    load_binding_spec as load_binding_spec,
)

from ._state import (
    StudioKnobState,
    StudioReplayResult,
    apply_knob_update,
    discover_domainpacks,
)
from .canvas import (
    _validate_candidate_binding_yaml as _validate_candidate_binding_yaml,
)
from .canvas import (
    apply_canvas_binding_rewrite_candidate,
    build_canvas_binding_rewrite_candidate,
    build_canvas_edit_artifact,
    build_canvas_graph,
    build_canvas_interaction_state,
    build_canvas_layout_manifest,
    build_canvas_topology_patch,
    build_oscillator_edit_artifact,
)
from .charts import (
    build_integrated_information_panel,
    build_regime_chart_payload,
    build_series_chart_payload,
)
from .connectors import (
    _run_owned_live_adapter as _run_owned_live_adapter,
)
from .connectors import (
    _stable_json_payload as _stable_json_payload,
)
from .connectors import (
    build_live_connector_plan,
    build_live_connector_run_record,
    build_owned_live_connector_runtime_record,
)
from .deployment import (
    build_command_table,
    build_deployment_package,
    build_deployment_readiness,
    build_export_manifests,
    build_operator_checklist,
    build_package_materialisation_plan,
    build_service_process_manifest,
    disabled_export_reasons,
)
from .guidance import (
    binding_spec_project_state,
    build_beginner_guidance,
    build_error_report,
    build_runtime_snapshot,
)
from .hardware import (
    build_hardware_target_package,
    build_verified_hardware_target_package,
)
from .panel_evolutionary import (
    build_evolutionary_supervisor_policy_search_studio_panel,
)
from .panel_hybrid_order import (
    build_hybrid_order_studio_panel,
)
from .panel_information_geometry import (
    build_information_geometry_studio_panel,
)
from .panel_lineage import (
    build_autopoietic_lineage_studio_panel,
    build_intergenerational_inheritance_studio_panel,
)
from .panel_morphogenetic import (
    build_morphogenetic_field_studio_panel,
)
from .panel_multiverse import (
    build_multiverse_counterfactual_studio_panel,
)
from .panel_sheaf import (
    build_sheaf_cohomology_studio_panel,
)
from .panel_strange_loop import (
    build_strange_loop_studio_panel,
)
from .panel_topos import (
    build_topos_semantic_binding_studio_panel,
)
from .panel_twin_confidence import (
    build_twin_confidence_studio_panel,
)
from .replay import (
    run_binding_spec_replay,
)
from .tables import (
    build_layer_table,
    build_oscillator_table,
)

__all__ = [
    "StudioKnobState",
    "StudioReplayResult",
    "apply_canvas_binding_rewrite_candidate",
    "apply_knob_update",
    "binding_spec_project_state",
    "build_beginner_guidance",
    "build_canvas_edit_artifact",
    "build_canvas_binding_rewrite_candidate",
    "build_canvas_graph",
    "build_canvas_interaction_state",
    "build_canvas_layout_manifest",
    "build_canvas_topology_patch",
    "build_command_table",
    "build_export_manifests",
    "build_deployment_package",
    "build_deployment_readiness",
    "build_error_report",
    "build_autopoietic_lineage_studio_panel",
    "build_intergenerational_inheritance_studio_panel",
    "build_evolutionary_supervisor_policy_search_studio_panel",
    "build_hybrid_order_studio_panel",
    "build_information_geometry_studio_panel",
    "build_integrated_information_panel",
    "build_sheaf_cohomology_studio_panel",
    "build_layer_table",
    "build_hardware_target_package",
    "build_live_connector_plan",
    "build_live_connector_run_record",
    "build_morphogenetic_field_studio_panel",
    "build_multiverse_counterfactual_studio_panel",
    "build_owned_live_connector_runtime_record",
    "build_oscillator_edit_artifact",
    "build_oscillator_table",
    "build_operator_checklist",
    "build_regime_chart_payload",
    "build_package_materialisation_plan",
    "build_runtime_snapshot",
    "build_series_chart_payload",
    "build_service_process_manifest",
    "build_strange_loop_studio_panel",
    "build_topos_semantic_binding_studio_panel",
    "build_twin_confidence_studio_panel",
    "build_verified_hardware_target_package",
    "disabled_export_reasons",
    "discover_domainpacks",
    "run_binding_spec_replay",
]

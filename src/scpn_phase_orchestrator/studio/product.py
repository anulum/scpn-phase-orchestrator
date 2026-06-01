# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Studio product manifest

"""Standalone Studio product manifest for passive physics review panels."""

from __future__ import annotations

import json
from collections.abc import Sequence
from hashlib import sha256

__all__ = [
    "STUDIO_REVIEW_PANEL_REGISTRY",
    "build_studio_product_manifest",
]

PanelRecord = dict[str, object]

STUDIO_REVIEW_PANEL_REGISTRY: tuple[PanelRecord, ...] = (
    {
        "panel_id": "integrated_information_monitor",
        "title": "Integrated-information monitor",
        "builder": "build_integrated_information_panel",
        "claim_boundary": "engineering_proxy_not_theoretical_iit",
        "source_rows": ("Integrated-information monitor",),
        "required_evidence": (
            "integrated-information audit record",
            "normalised Phi proxy metrics",
            "minimum-partition evidence",
            "optional symmetric pairwise-MI matrix",
        ),
        "actuation_permitted": False,
        "live_merge_permitted": False,
        "hot_patch_permitted": False,
        "execution_disabled": True,
        "operator_review_required": True,
    },
    {
        "panel_id": "strange_loop_meta_orchestrator",
        "title": "Strange-loop meta-orchestrator",
        "builder": "build_strange_loop_studio_panel",
        "claim_boundary": "strange_loop_drift_review_not_live_actuation",
        "source_rows": ("Strange-loop meta-orchestrator",),
        "required_evidence": (
            "deterministic drift-scenario audit records",
            "scenario SHA-256 hashes",
            "drift, oscillation, over-control, and coherence metrics",
            "expected-trigger evidence",
        ),
        "actuation_permitted": False,
        "live_merge_permitted": False,
        "hot_patch_permitted": False,
        "execution_disabled": True,
        "operator_review_required": True,
    },
    {
        "panel_id": "morphogenetic_field_topology",
        "title": "Morphogenetic field topology",
        "builder": "build_morphogenetic_field_studio_panel",
        "claim_boundary": "morphogenetic_field_svg_review_not_live_actuation",
        "source_rows": ("Morphogenetic field topology",),
        "required_evidence": (
            "passive SVG field artefact",
            "field-energy statistics",
            "off-diagonal topology-edge records",
            "fixed-width heatmap rows",
        ),
        "actuation_permitted": False,
        "live_merge_permitted": False,
        "hot_patch_permitted": False,
        "execution_disabled": True,
        "operator_review_required": True,
    },
    {
        "panel_id": "multiverse_counterfactual_rollout",
        "title": "Multiverse counterfactual rollout",
        "builder": "build_multiverse_counterfactual_studio_panel",
        "claim_boundary": "multiverse_counterfactual_review_not_live_actuation",
        "source_rows": ("Multiverse counterfactual simulator",),
        "required_evidence": (
            "passive rollout manifest",
            "matching branch-risk report",
            "branch SHA-256 hashes",
            "bounded coherence and topology-pressure metrics",
        ),
        "actuation_permitted": False,
        "live_merge_permitted": False,
        "hot_patch_permitted": False,
        "execution_disabled": True,
        "operator_review_required": True,
    },
    {
        "panel_id": "hybrid_order_parameters",
        "title": "Entanglement-aware hybrid order parameters",
        "builder": "build_hybrid_order_studio_panel",
        "claim_boundary": "quantum_cosimulation_monitor_not_qpu_execution",
        "source_rows": ("Entanglement-aware hybrid order parameters",),
        "required_evidence": (
            "hybrid order-parameter audit records",
            "local simulator backend contract",
            "bipartition entropy evidence",
            "scenario SHA-256 hashes",
        ),
        "actuation_permitted": False,
        "live_merge_permitted": False,
        "hot_patch_permitted": False,
        "execution_disabled": True,
        "operator_review_required": True,
        "qpu_execution_permitted": False,
    },
    {
        "panel_id": "information_geometry_control",
        "title": "Information-geometry control",
        "builder": "build_information_geometry_studio_panel",
        "claim_boundary": "information_geometry_control_not_live_actuation",
        "source_rows": ("Information-geometry control",),
        "required_evidence": (
            "information-geometry proposal audit record",
            "metric-tensor and diagonal positivity evidence",
            "Fisher-Rao and Wasserstein distances",
            "optional JAX/NumPy backend parity evidence",
        ),
        "actuation_permitted": False,
        "live_merge_permitted": False,
        "hot_patch_permitted": False,
        "execution_disabled": True,
        "operator_review_required": True,
    },
    {
        "panel_id": "topos_semantic_binding",
        "title": "Topos semantic binding",
        "builder": "build_topos_semantic_binding_studio_panel",
        "proof_boundary": "categorical_validation_prototype_not_formal_topos_proof",
        "source_rows": ("Topos-theoretic semantic binding",),
        "required_evidence": (
            "symbolic-binding functor validation report",
            "policy-composition category report",
            "domain obligation examples",
            "report and example SHA-256 hashes",
        ),
        "actuation_permitted": False,
        "live_merge_permitted": False,
        "hot_patch_permitted": False,
        "execution_disabled": True,
        "operator_review_required": True,
        "formal_proof_claim_permitted": False,
    },
    {
        "panel_id": "evolutionary_supervisor_policy_search",
        "title": "Evolutionary supervisor policy search",
        "builder": "build_evolutionary_supervisor_policy_search_studio_panel",
        "claim_boundary": "offline_evolutionary_supervisor_review_not_live_actuation",
        "source_rows": ("Evolutionary supervisor policy search",),
        "required_evidence": (
            "offline replay-search audit report",
            "candidate-level SHA-256 hashes",
            "STL robustness and replay-safety metrics",
            "optional policy-DSL mutation report",
            "optional deterministic domain examples",
        ),
        "actuation_permitted": False,
        "live_merge_permitted": False,
        "hot_patch_permitted": False,
        "execution_disabled": True,
        "operator_review_required": True,
    },
)


def build_studio_product_manifest(
    *,
    panel_registry: Sequence[PanelRecord] = STUDIO_REVIEW_PANEL_REGISTRY,
) -> dict[str, object]:
    """Return the standalone Studio product manifest.

    The manifest is intentionally metadata-only. It lets a packaged Studio shell
    discover passive panel contracts and required evidence without importing
    optional runtimes, executing panel builders, opening transports, or touching
    hardware.
    """
    panels = tuple(dict(panel) for panel in panel_registry)
    panel_ids = [panel.get("panel_id") for panel in panels]
    if any(not isinstance(panel_id, str) or not panel_id for panel_id in panel_ids):
        raise ValueError("panel_registry entries require non-empty panel_id values")
    if len(set(panel_ids)) != len(panel_ids):
        raise ValueError("panel_registry panel_id values must be unique")
    for panel in panels:
        _require_disabled_panel_gates(panel)
    manifest = {
        "manifest_kind": "studio_product_manifest",
        "product": "spo_studio",
        "standalone_shell": "tools/spo_studio.py",
        "launch_command": "streamlit run tools/spo_studio.py",
        "review_panel_count": len(panels),
        "review_panels": panels,
        "operator_review_required": True,
        "actuation_permitted": False,
        "live_merge_permitted": False,
        "hot_patch_permitted": False,
        "network_opened": False,
        "hardware_write_permitted": False,
        "qpu_execution_permitted": False,
    }
    manifest["manifest_sha256"] = _manifest_sha256(manifest)
    return manifest


def _require_disabled_panel_gates(panel: PanelRecord) -> None:
    for field, expected in (
        ("actuation_permitted", False),
        ("live_merge_permitted", False),
        ("hot_patch_permitted", False),
        ("execution_disabled", True),
        ("operator_review_required", True),
    ):
        if panel.get(field) is not expected:
            raise ValueError(f"{panel.get('panel_id', '<unknown>')} {field} invalid")


def _manifest_sha256(manifest: dict[str, object]) -> str:
    payload = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()

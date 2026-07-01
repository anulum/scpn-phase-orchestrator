# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — API documentation navigation regressions

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import yaml

DOCS_ROOT = Path("docs")
API_ROOT = DOCS_ROOT / "reference" / "api"
MKDOCS_CONFIG = Path("mkdocs.yml")
SOURCE_ROOT = Path("src") / "scpn_phase_orchestrator"
AUTODOC_EXCLUSIONS = {
    "scpn_phase_orchestrator.runtime.audit_logger",
    "scpn_phase_orchestrator.runtime.replay",
    "scpn_phase_orchestrator.runtime.audit_signing",
    "scpn_phase_orchestrator.runtime.audit_stream",
    "scpn_phase_orchestrator.runtime.cli",
    "scpn_phase_orchestrator.runtime.cli.binding",
    "scpn_phase_orchestrator.runtime.cli.diagnostics",
    "scpn_phase_orchestrator.runtime.cli.monitoring",
    "scpn_phase_orchestrator.runtime.cli.plugins",
    "scpn_phase_orchestrator.runtime.cli.plugins.execution",
    "scpn_phase_orchestrator.runtime.cli.plugins.storage",
    "scpn_phase_orchestrator.runtime.cli.plugins.lifecycle",
    "scpn_phase_orchestrator.runtime.cli.plugins.remediation",
    "scpn_phase_orchestrator.runtime.cli.plugins.scheduler",
    "scpn_phase_orchestrator.runtime.cli.plugins.scheduler_control",
    "scpn_phase_orchestrator.runtime.cli.plugins.revocation",
    "scpn_phase_orchestrator.runtime.cli.meta",
    "scpn_phase_orchestrator.runtime.cli.verification",
    "scpn_phase_orchestrator.runtime.cli.run",
    "scpn_phase_orchestrator.runtime.cli.assurance",
    "scpn_phase_orchestrator.runtime.cli.provenance",
    "scpn_phase_orchestrator.runtime.cli.audit",
    "scpn_phase_orchestrator.runtime.cli.digital_twin",
    "scpn_phase_orchestrator.runtime.cli.koopman_mpc",
    "scpn_phase_orchestrator.runtime.cli.queuewaves",
    "scpn_phase_orchestrator.runtime.cli.quickstart",
    "scpn_phase_orchestrator.runtime.cli.scaffold",
    "scpn_phase_orchestrator.runtime.cli.supervisor_candidate",
    "scpn_phase_orchestrator.grpc_gen",
    "scpn_phase_orchestrator.grpc_gen._spo_pb2_fallback",
    "scpn_phase_orchestrator.grpc_gen._spo_pb2_grpc_fallback",
    "scpn_phase_orchestrator.runtime.grpc_gen.spo_pb2",
    "scpn_phase_orchestrator.runtime.grpc_gen.spo_pb2_grpc",
    "scpn_phase_orchestrator.grpc_gen.spo_pb2",
    "scpn_phase_orchestrator.grpc_gen.spo_pb2_grpc",
    "scpn_phase_orchestrator.runtime.network_security",
    "scpn_phase_orchestrator.runtime.server",
    "scpn_phase_orchestrator.runtime.server_grpc",
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
    "scpn_phase_orchestrator.studio.ui_helpers.panel_twin_confidence",
    "scpn_phase_orchestrator.studio.ui_helpers.replay",
    "scpn_phase_orchestrator.studio.ui_helpers.tables",
    "scpn_phase_orchestrator.nn.supervisor.candidate_bridge",
    "scpn_phase_orchestrator.nn.supervisor.checkpoint",
    "scpn_phase_orchestrator.nn.supervisor.comparison",
    "scpn_phase_orchestrator.nn.supervisor.policy",
    "scpn_phase_orchestrator.nn.supervisor.ppo",
    "scpn_phase_orchestrator.nn.supervisor.projection",
    "scpn_phase_orchestrator.nn.supervisor.replay",
    "scpn_phase_orchestrator.nn.supervisor.rollouts",
    "scpn_phase_orchestrator.plugins.registry.manifest",
    "scpn_phase_orchestrator.plugins.registry.policy",
    "scpn_phase_orchestrator.plugins.registry.request",
    "scpn_phase_orchestrator.plugins.registry.storage",
    "scpn_phase_orchestrator.plugins.registry.revocation",
    "scpn_phase_orchestrator.plugins.registry.lifecycle",
    "scpn_phase_orchestrator.plugins.registry.runtime",
    "scpn_phase_orchestrator.plugins.registry.rust_handoff",
    "scpn_phase_orchestrator.binding.digital_twin.contract",
    "scpn_phase_orchestrator.binding.digital_twin.envelope",
    "scpn_phase_orchestrator.binding.digital_twin.evidence",
    "scpn_phase_orchestrator.binding.digital_twin.adapter_memory",
    "scpn_phase_orchestrator.binding.digital_twin.adapter_grpc",
    "scpn_phase_orchestrator.binding.digital_twin.adapter_kafka",
    "scpn_phase_orchestrator.binding.digital_twin.adapter_hardware",
    "scpn_phase_orchestrator.binding.digital_twin.adapter_rest",
    "scpn_phase_orchestrator.supervisor.formal_export.verification_package",
    "scpn_phase_orchestrator.supervisor.formal_export.runtime_certificate",
    "scpn_phase_orchestrator.supervisor.formal_export.petri_export",
    "scpn_phase_orchestrator.supervisor.formal_export.policy_export",
    "scpn_phase_orchestrator.supervisor.formal_export.stl_export",
    "scpn_phase_orchestrator.supervisor.hierarchy.boundary",
    "scpn_phase_orchestrator.supervisor.hierarchy.plan",
    "scpn_phase_orchestrator.supervisor.hierarchy.sync",
    "scpn_phase_orchestrator.supervisor.hierarchy.consensus",
    "scpn_phase_orchestrator.binding.semantic.coercion",
    "scpn_phase_orchestrator.binding.semantic.retrieval",
    "scpn_phase_orchestrator.binding.semantic.review",
    "scpn_phase_orchestrator.binding.semantic.serialization",
    "scpn_phase_orchestrator.binding.semantic.compiler",
    "scpn_phase_orchestrator.monitor.stl.monitor",
    "scpn_phase_orchestrator.monitor.stl.automaton",
    "scpn_phase_orchestrator.monitor.stl.controller",
    "scpn_phase_orchestrator.monitor.stl.projection",
    "scpn_phase_orchestrator.monitor.stl.actuation_gate",
    "scpn_phase_orchestrator.monitor.stl.closed_loop",
}


def _nav_paths(nav: object) -> Iterator[str]:
    if isinstance(nav, str):
        yield nav
        return
    if isinstance(nav, list):
        for item in nav:
            yield from _nav_paths(item)
        return
    if isinstance(nav, dict):
        for value in nav.values():
            yield from _nav_paths(value)


def test_all_api_reference_pages_are_in_mkdocs_nav() -> None:
    config_text = MKDOCS_CONFIG.read_text(encoding="utf-8")
    for tag in (
        "!!python/name:pymdownx.superfences.fence_code_format",
        "!!python/name:material.extensions.emoji.twemoji",
        "!!python/name:material.extensions.emoji.to_svg",
    ):
        config_text = config_text.replace(tag, "")
    config = yaml.safe_load(config_text)
    nav_paths = set(_nav_paths(config["nav"]))
    api_pages = {
        path.relative_to(DOCS_ROOT).as_posix() for path in API_ROOT.glob("*.md")
    }

    assert api_pages <= nav_paths


def test_public_source_modules_have_api_autodoc() -> None:
    autodoc_text = "\n".join(
        path.read_text(encoding="utf-8") for path in API_ROOT.glob("*.md")
    )
    documented_modules = {
        match.group(1)
        for match in re.finditer(
            r"^:{3,4}\s+(scpn_phase_orchestrator(?:\.[\w_]+)+)",
            autodoc_text,
            re.MULTILINE,
        )
    }

    public_modules = set()
    for path in SOURCE_ROOT.rglob("*.py"):
        relative = path.relative_to(SOURCE_ROOT)
        if relative.name == "__init__.py":
            continue
        module_parts = relative.with_suffix("").parts
        if any(part.startswith("_") for part in module_parts):
            continue
        public_modules.add("scpn_phase_orchestrator." + ".".join(module_parts))

    missing = public_modules - documented_modules - AUTODOC_EXCLUSIONS

    assert missing == set()

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Studio verified hardware package binding gate tests

"""A verified hardware package never reports an invalid binding as passed.

Studio replays a binding spec even when validation reports errors, and blocks
Docker and WASM packaging for it. The verified hardware package listed
"binding validation passed" and returned ``review_ready`` for the same spec
once the hardware evidence was complete.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from scpn_phase_orchestrator.binding import validate_binding_spec
from scpn_phase_orchestrator.binding.loader import load_binding_spec
from scpn_phase_orchestrator.studio.ui_helpers import (
    StudioKnobState,
    build_deployment_readiness,
    build_verified_hardware_target_package,
    run_binding_spec_replay,
)

ROOT = Path(__file__).resolve().parents[1]
MINIMAL_SPEC = ROOT / "domainpacks" / "minimal_domain" / "binding_spec.yaml"

EVIDENCE: dict[str, object] = {
    "generated_artifact_path": "build/hardware/minimal_domain/fpga_top.v",
    "generated_artifact_sha256": "a" * 64,
    "simulator_parity_report": "reports/minimal_domain_parity.json",
    "simulator_parity_sha256": "b" * 64,
    "simulator_parity_status": "passed",
    "target_toolchain": "yosys-nextpnr",
    "target_toolchain_version": "yosys 0.40 / nextpnr 0.7",
    "operator_signoff": True,
}


def _invalid_spec(tmp_path: Path) -> Path:
    """Write a minimal spec whose objectives name a layer that does not exist."""
    text = MINIMAL_SPEC.read_text(encoding="utf-8")
    edited = text.replace("good_layers: [0, 1]", "good_layers: [0, 7]", 1)
    assert edited != text
    spec = tmp_path / "binding_spec.yaml"
    spec.write_text(edited, encoding="utf-8")
    assert validate_binding_spec(load_binding_spec(spec))
    return spec


def test_invalid_binding_blocks_the_verified_hardware_package(tmp_path: Path) -> None:
    """Complete evidence does not make an invalid binding review-ready."""
    result = run_binding_spec_replay(
        _invalid_spec(tmp_path), steps=3, knobs=StudioKnobState(K=1.0)
    )
    assert build_deployment_readiness(result.project_state)["overall_status"] == (
        "blocked"
    )

    package = build_verified_hardware_target_package(result, evidence=EVIDENCE)

    assert package["overall_status"] == "blocked"
    assert "binding validation passed" not in package["safety_gates"]
    assert "binding validation blocked" in package["safety_gates"]
    assert package["commands"] == []
    reasons = package["blocked_reasons"]
    assert isinstance(reasons, list)
    assert any("layer index 7" in str(reason) for reason in reasons)


def test_valid_binding_with_complete_evidence_is_review_ready() -> None:
    """A valid binding with complete evidence keeps its review-ready package."""
    result = run_binding_spec_replay(
        MINIMAL_SPEC, steps=3, knobs=StudioKnobState(K=1.0)
    )

    package = build_verified_hardware_target_package(result, evidence=EVIDENCE)

    assert package["overall_status"] == "review_ready"
    assert package["blocked_reasons"] == []
    assert "binding validation passed" in package["safety_gates"]


def test_unfinished_replay_blocks_the_verified_hardware_package() -> None:
    """A replay status other than completed blocks the hardware handoff."""
    result = run_binding_spec_replay(
        MINIMAL_SPEC, steps=3, knobs=StudioKnobState(K=1.0)
    )
    runtime = replace(result.project_state.runtime, replay_status="failed")
    unfinished = replace(
        result, project_state=replace(result.project_state, runtime=runtime)
    )

    package = build_verified_hardware_target_package(unfinished, evidence=EVIDENCE)

    assert package["overall_status"] == "blocked"
    assert package["blocked_reasons"] == ["local replay has not completed"]
    assert "local replay incomplete" in package["safety_gates"]
    assert package["commands"] == []

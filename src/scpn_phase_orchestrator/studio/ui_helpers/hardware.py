# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SPO Studio hardware target package builders

"""Hardware target package builders for Studio replay results."""

from __future__ import annotations

from collections.abc import Mapping

from ._shared import (
    _connector_by_transport,
    _deployment_blocked_reasons,
    _is_sha256_digest,
    _require_non_empty_text,
    _require_sequence,
)
from ._state import StudioReplayResult


def build_hardware_target_package(result: StudioReplayResult) -> dict[str, object]:
    """Return a review-only hardware target package for Studio.

    Parameters
    ----------
    result : StudioReplayResult
        The Studio replay result.

    Returns
    -------
    dict[str, object]
        A review-only hardware target package for Studio.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
    if not isinstance(result, StudioReplayResult):
        raise ValueError("replay result must be a StudioReplayResult")
    connector_plan = result.connector_plan
    hardware_connector = _connector_by_transport(connector_plan, "hardware")
    return {
        "package_kind": "studio_hardware_target_package",
        "project_name": result.project_state.project_name,
        "overall_status": "evidence_required",
        "contract_hash": _require_non_empty_text(
            connector_plan.get("contract_hash"),
            "contract_hash",
        ),
        "hardware_write_permitted": False,
        "network_opened": False,
        "targets": ["fpga_verilog", "neuromorphic_schedule"],
        "required_evidence": [
            "generated hardware artefact path",
            "simulator parity report",
            "target toolchain version",
            "operator sign-off",
        ],
        "commands": [
            "review connector_plan.json",
            "generate FPGA Verilog with KuramotoVerilogCompiler",
            "run simulator parity before hardware handoff",
        ],
        "connector": hardware_connector,
        "export_artifacts": [
            manifest.to_audit_record() for manifest in result.export_manifests
        ],
    }


def build_verified_hardware_target_package(
    result: StudioReplayResult,
    *,
    evidence: Mapping[str, object],
) -> dict[str, object]:
    """Return a verified hardware package only when evidence is complete.

    The package is ``review_ready`` only when the evidence is complete, the
    local replay completed, and no export warning blocks deployment. A binding
    with validation errors blocks the hardware handoff as it blocks Docker and
    WASM packaging; the blocking reasons are listed in ``blocked_reasons``.

    Parameters
    ----------
    result : StudioReplayResult
        The Studio replay result.
    evidence : Mapping[str, object]
        Verification evidence mapping.

    Returns
    -------
    dict[str, object]
        A verified hardware package only when evidence is complete.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
    if not isinstance(result, StudioReplayResult):
        raise ValueError("replay result must be a StudioReplayResult")
    if not isinstance(evidence, Mapping):
        raise ValueError("hardware evidence must be a mapping")

    base_package = build_hardware_target_package(result)
    normalised, invalid_evidence = _normalise_hardware_evidence(evidence)
    verified = not invalid_evidence
    # The same export warnings that block Docker and WASM packaging block the
    # hardware handoff: hardware evidence does not stand in for a valid binding.
    blocked_reasons = list(_deployment_blocked_reasons(result.project_state.exports))
    replay_completed = result.project_state.runtime.replay_status == "completed"
    if not replay_completed:
        blocked_reasons.append("local replay has not completed")
    if blocked_reasons:
        overall_status = "blocked"
    elif verified:
        overall_status = "review_ready"
    else:
        overall_status = "evidence_required"
    return {
        "package_kind": "studio_verified_hardware_target_package",
        "project_name": result.project_state.project_name,
        "overall_status": overall_status,
        "blocked_reasons": blocked_reasons,
        "evidence_status": "verified" if verified else "blocked",
        "contract_hash": base_package["contract_hash"],
        "hardware_write_permitted": False,
        "network_opened": False,
        "targets": list(_require_sequence(base_package.get("targets"), "targets")),
        "required_evidence": list(
            _require_sequence(
                base_package.get("required_evidence"),
                "required_evidence",
            )
        ),
        "invalid_evidence": invalid_evidence,
        "evidence": normalised,
        "connector": base_package["connector"],
        "commands": (
            [
                "review verified_hardware_target_package.json",
                "compare generated artefact hash before handoff",
                "archive simulator parity report with package",
            ]
            if overall_status == "review_ready"
            else []
        ),
        "safety_gates": [
            "local replay completed" if replay_completed else "local replay incomplete",
            (
                "binding validation blocked"
                if _deployment_blocked_reasons(result.project_state.exports)
                else "binding validation passed"
            ),
            "hardware evidence verified" if verified else "hardware evidence blocked",
            "hardware output remains operator-controlled",
        ],
        "export_artifacts": list(
            _require_sequence(
                base_package.get("export_artifacts"),
                "export_artifacts",
            )
        ),
    }


def _normalise_hardware_evidence(
    evidence: Mapping[str, object],
) -> tuple[dict[str, object], list[str]]:
    """Validate hardware-loop evidence, returning the record and invalid fields."""
    invalid: list[str] = []
    normalised: dict[str, object] = {}
    for field in (
        "generated_artifact_path",
        "simulator_parity_report",
        "target_toolchain",
        "target_toolchain_version",
    ):
        value = evidence.get(field)
        if isinstance(value, str) and value.strip():
            normalised[field] = value.strip()
        else:
            invalid.append(f"{field} is required")

    for field in ("generated_artifact_sha256", "simulator_parity_sha256"):
        value = evidence.get(field)
        if _is_sha256_digest(value):
            normalised[field] = str(value).lower()
        elif value is None:
            invalid.append(f"{field} is required")
        else:
            invalid.append(f"{field} must be a SHA-256 digest")

    parity_status = evidence.get("simulator_parity_status")
    if isinstance(parity_status, str) and parity_status.strip().lower() == "passed":
        normalised["simulator_parity_status"] = "passed"
    else:
        invalid.append("simulator_parity_status must be passed")

    if evidence.get("operator_signoff") is True:
        normalised["operator_signoff"] = True
    else:
        invalid.append("operator_signoff must be true")
    return normalised, invalid

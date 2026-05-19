# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hybrid neuromorphic-quantum co-compiler

"""Deterministic hybrid co-compiler review manifests."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from hashlib import sha256

__all__ = ["build_hybrid_cocompiler_manifest"]


def build_hybrid_cocompiler_manifest(
    quantum_manifest: Mapping[str, object],
    neuromorphic_manifest: Mapping[str, object],
    *,
    n_channel_semantics: Sequence[str] = ("Q_control", "S_spike", "audit"),
) -> dict[str, object]:
    """Combine quantum and spiking manifests under one audit envelope."""
    quantum_manifest = _validate_manifest_mapping(
        quantum_manifest,
        label="quantum_manifest",
    )
    neuromorphic_manifest = _validate_manifest_mapping(
        neuromorphic_manifest,
        label="neuromorphic_manifest",
    )
    _validate_manifest_kind(
        quantum_manifest,
        expected="quantum_compiler_manifest",
        label="quantum manifest kind",
    )
    _validate_manifest_kind(
        neuromorphic_manifest,
        expected="neuromorphic_schedule_manifest",
        label="neuromorphic manifest kind",
    )
    _validate_permission_fields(
        quantum_manifest,
        label="quantum",
        fields=("qpu_execution_permitted", "actuation_permitted"),
    )
    _validate_permission_fields(
        neuromorphic_manifest,
        label="neuromorphic",
        fields=("hardware_write_permitted", "actuation_permitted"),
    )
    semantics = _normalise_semantics(n_channel_semantics)
    blocked_reasons = _blocked_reasons(quantum_manifest, neuromorphic_manifest)
    target_backends = _target_backends(quantum_manifest, neuromorphic_manifest)
    component_hashes = _component_hashes(quantum_manifest, neuromorphic_manifest)
    parity = {
        "engine": "hybrid_manifest_status_reconstruction",
        "quantum_status": quantum_manifest.get("status"),
        "neuromorphic_status": neuromorphic_manifest.get("status"),
        "quantum_term_count": _term_count(quantum_manifest.get("co_simulation_parity")),
        "neuromorphic_sample_count": _sample_count(
            neuromorphic_manifest.get("simulator_parity")
        ),
    }
    manifest: dict[str, object] = {
        "manifest_kind": "hybrid_neuromorphic_quantum_cocompiler",
        "schema_version": 1,
        "status": "blocked" if blocked_reasons else "co_simulation_parity_passed",
        "target_backends": target_backends,
        "n_channel_semantics": semantics,
        "component_hashes": component_hashes,
        "co_simulation_parity": parity,
        "blocked_reasons": blocked_reasons,
        "qpu_execution_permitted": False,
        "hardware_write_permitted": False,
        "actuation_permitted": False,
        "operator_commands": [
            "review hybrid_neuromorphic_quantum_cocompiler.json",
            "run quantum and neuromorphic simulators under the shared audit envelope",
        ],
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    manifest["hybrid_manifest_sha256"] = sha256(canonical.encode("utf-8")).hexdigest()
    return manifest


def _validate_manifest_mapping(
    manifest: Mapping[str, object],
    *,
    label: str,
) -> Mapping[str, object]:
    if not isinstance(manifest, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return manifest


def _validate_manifest_kind(
    manifest: Mapping[str, object],
    *,
    expected: str,
    label: str,
) -> None:
    if manifest.get("manifest_kind") != expected:
        raise ValueError(f"{label} must be {expected}")


def _validate_permission_fields(
    manifest: Mapping[str, object],
    *,
    label: str,
    fields: Sequence[str],
) -> None:
    for field in fields:
        if not isinstance(manifest.get(field), bool):
            raise ValueError(f"{label} {field} must be a bool")


def _normalise_semantics(n_channel_semantics: Sequence[str]) -> list[str]:
    if (
        isinstance(n_channel_semantics, str)
        or not isinstance(n_channel_semantics, Sequence)
        or not n_channel_semantics
    ):
        raise ValueError("n_channel_semantics must be a non-empty sequence")
    semantics: list[str] = []
    for channel in n_channel_semantics:
        if not isinstance(channel, str) or not channel.strip():
            raise ValueError("n_channel_semantics entries must be non-empty strings")
        semantics.append(channel.strip())
    return semantics


def _blocked_reasons(
    quantum_manifest: Mapping[str, object],
    neuromorphic_manifest: Mapping[str, object],
) -> list[str]:
    reasons: list[str] = []
    if quantum_manifest.get("status") != "co_simulation_parity_passed":
        reasons.append("quantum compiler parity must pass")
    if neuromorphic_manifest.get("status") != "simulator_parity_passed":
        reasons.append("neuromorphic simulator parity must pass")
    for permission in (
        "qpu_execution_permitted",
        "actuation_permitted",
    ):
        if quantum_manifest.get(permission) is not False:
            reasons.append(f"quantum {permission} must remain false")
    for permission in (
        "hardware_write_permitted",
        "actuation_permitted",
    ):
        if neuromorphic_manifest.get(permission) is not False:
            reasons.append(f"neuromorphic {permission} must remain false")
    return reasons


def _target_backends(
    quantum_manifest: Mapping[str, object],
    neuromorphic_manifest: Mapping[str, object],
) -> list[str]:
    return [
        *_string_list(
            quantum_manifest.get("target_backends"),
            "quantum target_backends",
        ),
        *_string_list(
            neuromorphic_manifest.get("target_backends"),
            "neuromorphic target_backends",
        ),
    ]


def _component_hashes(
    quantum_manifest: Mapping[str, object],
    neuromorphic_manifest: Mapping[str, object],
) -> dict[str, str]:
    return {
        "quantum_qasm_sha256": _hash_text(quantum_manifest, "qasm_sha256"),
        "quantum_manifest_sha256": _hash_text(quantum_manifest, "manifest_sha256"),
        "neuromorphic_schedule_sha256": _hash_text(
            neuromorphic_manifest,
            "schedule_sha256",
        ),
    }


def _hash_text(manifest: Mapping[str, object], key: str) -> str:
    value = manifest.get(key)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdefABCDEF" for ch in value)
    ):
        raise ValueError(f"{key} must be a 64-character SHA-256 hex string")
    return value


def _string_list(value: object, label: str) -> list[str]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a sequence")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ValueError(f"{label} entries must be non-empty strings")
        result.append(item)
    return result


def _term_count(value: object) -> int:
    if isinstance(value, Mapping):
        term_count = value.get("term_count")
        if isinstance(term_count, int):
            return term_count
    return 0


def _sample_count(value: object) -> int:
    if isinstance(value, Mapping):
        sample_count = value.get("sample_count")
        if isinstance(sample_count, int):
            return sample_count
    return 0

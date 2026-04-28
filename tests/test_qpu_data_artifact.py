# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — QPU data artifact tests

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.artifacts.qpu_data import (
    SCHEMA_VERSION,
    compile_domain_to_qpu_artifact,
    emit_qpu_data_artifact,
    validate_qpu_data_artifact,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _base_payload(**overrides: object) -> dict[str, Any]:
    payload = emit_qpu_data_artifact(
        domain="unit",
        source_name="unit-fixture",
        source_mode="curated",
        K_nm=np.array([[0.0, 0.4], [0.4, 0.0]]),
        omega=np.array([1.0, 1.2]),
        theta0=np.array([0.0, 0.5]),
        layer_assignments=["a", "b"],
        normalization="unit canonical scaling",
        extraction_method="unit-test",
        replay_id="unit:replay:1",
        metadata={"purpose": "contract-test"},
    )
    payload.update(overrides)
    return payload


def test_emit_curated_domainpack_artifact() -> None:
    payload = compile_domain_to_qpu_artifact(
        REPO_ROOT / "domainpacks" / "minimal_domain",
        source_mode="curated",
        replay_id="domainpack:minimal_domain:0.1.0",
    )
    artifact = validate_qpu_data_artifact(payload)

    assert payload["schema_version"] == SCHEMA_VERSION
    assert artifact.domain == "minimal_domain"
    assert artifact.n_oscillators == 4
    assert artifact.K_nm.shape == (4, 4)
    assert artifact.omega.shape == (4,)
    assert artifact.layer_assignments == ["lower", "lower", "upper", "upper"]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("K_nm", [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]], "square"),
        ("K_nm", [[0.0, 1.0], [0.2, 0.0]], "symmetric"),
        ("K_nm", [[0.0, -0.1], [-0.1, 0.0]], "non-negative"),
        ("K_nm", [[1.0, 0.1], [0.1, 0.0]], "diagonal"),
        ("omega", [1.0, 2.0, 3.0], "omega shape"),
        ("theta0", [0.0], "theta0 shape"),
        ("layer_assignments", ["only-one"], "layer_assignments length"),
    ],
)
def test_validate_rejects_invalid_artifact_fields(
    field: str, value: object, match: str
) -> None:
    payload = _base_payload()
    payload[field] = value
    with pytest.raises(ValueError, match=match):
        emit_qpu_data_artifact(
            domain=str(payload["domain"]),
            source_name=str(payload["source_name"]),
            source_mode=str(payload["source_mode"]),
            K_nm=payload["K_nm"],
            omega=payload["omega"],
            theta0=payload["theta0"],
            layer_assignments=payload["layer_assignments"],
            normalization=str(payload["normalization"]),
            extraction_method=str(payload["extraction_method"]),
            replay_id=str(payload["replay_id"]),
            metadata=payload["metadata"],
        )


def test_validate_rejects_missing_publication_provenance() -> None:
    payload = emit_qpu_data_artifact(
        domain="unit",
        source_name="unit-fixture",
        source_mode="recorded",
        K_nm=np.array([[0.0, 0.4], [0.4, 0.0]]),
        omega=np.array([1.0, 1.2]),
        normalization="unit canonical scaling",
        extraction_method="unit-test",
    )

    with pytest.raises(ValueError, match="source_timestamp or replay_id"):
        validate_qpu_data_artifact(payload)


def test_validate_rejects_synthetic_when_publication_safe_required() -> None:
    payload = emit_qpu_data_artifact(
        domain="unit",
        source_name="unit-fixture",
        source_mode="synthetic",
        K_nm=np.array([[0.0, 0.4], [0.4, 0.0]]),
        omega=np.array([1.0, 1.2]),
        normalization="unit canonical scaling",
        extraction_method="unit-test",
        replay_id="synthetic:smoke",
        metadata={"synthetic": True},
    )

    with pytest.raises(ValueError, match="synthetic artifacts"):
        validate_qpu_data_artifact(payload)
    assert validate_qpu_data_artifact(
        payload, require_publication_safe=False
    ).is_synthetic


def test_hashes_are_stable_and_verified() -> None:
    first = _base_payload()
    second = _base_payload()
    assert first["hashes"] == second["hashes"]
    assert first["artifact_sha256"] == second["artifact_sha256"]
    hashes = first["hashes"]
    assert isinstance(hashes, dict)
    assert "K_nm_sha256" in hashes
    assert "omega_sha256" in hashes
    assert "theta0_sha256" in hashes

    tampered = dict(first)
    tampered["K_nm"] = [[0.0, 0.5], [0.5, 0.0]]
    with pytest.raises(ValueError, match="K_nm_sha256"):
        validate_qpu_data_artifact(tampered)


def test_artifact_sha256_is_verified() -> None:
    payload = _base_payload(artifact_sha256="bad")
    with pytest.raises(ValueError, match="artifact_sha256"):
        validate_qpu_data_artifact(payload)

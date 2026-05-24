# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — QPU data artifact tests

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.artifacts.qpu_data import (
    SCHEMA_VERSION,
    QPUDataArtifact,
    compile_domain_to_qpu_artifact,
    emit_qpu_data_artifact,
    read_qpu_data_artifact,
    validate_qpu_data_artifact,
    write_qpu_data_artifact,
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


def test_hash_mismatch_in_manifest_is_rejected_with_precise_message() -> None:
    payload = _base_payload()
    checks = [
        ("K_nm_sha256", "not-a-real-hash"),
        ("omega_sha256", "wrong"),
    ]
    for key, bad_value in checks:
        tampered = _base_payload(
            hashes={"K_nm_sha256": payload["hashes"]["K_nm_sha256"]}
        )
        tampered["hashes"][key] = bad_value
        with pytest.raises(ValueError, match=f"{key} does not match artifact data"):
            validate_qpu_data_artifact(tampered)


def test_validate_rejects_non_mapping_payload() -> None:
    with pytest.raises(ValueError, match="artifact payload must be a mapping"):
        validate_qpu_data_artifact([1, 2, 3])


def test_metadata_and_provenance_are_normalised_and_layer_ids_strified() -> None:
    payload = emit_qpu_data_artifact(
        domain="  unit ",
        source_name="\tunit-fixture\n",
        source_mode="recorded",
        K_nm=np.array([[0.0, 0.6], [0.6, 0.0]]),
        omega=np.array([1.0, 1.2]),
        theta0=np.array([0.0, 0.5]),
        layer_assignments=[1, " two "],
        normalization=" normalized ",
        extraction_method=" test-extractor ",
        replay_id="unit:replay:normalised",
        metadata={"phase": 1},
    )

    artifact = validate_qpu_data_artifact(payload)

    assert artifact.domain == "unit"
    assert artifact.source_name == "unit-fixture"
    assert artifact.normalization == "normalized"
    assert artifact.extraction_method == "test-extractor"
    assert artifact.layer_assignments == ["1", " two "]


def test_manifest_metadata_matches_domainpack_and_manifest_shape() -> None:
    payload = compile_domain_to_qpu_artifact(
        REPO_ROOT / "domainpacks" / "minimal_domain",
        source_mode="recorded",
        source_timestamp="2026-01-01T00:00:00Z",
    )
    metadata = payload["metadata"]

    assert metadata["source_project"] == "scpn-phase-orchestrator"
    assert metadata["binding_spec"] == "binding_spec.yaml"
    assert metadata["n_oscillators"] == len(payload["layer_assignments"])
    assert metadata["coupling"]["base_strength"] >= 0.0
    assert payload["source_timestamp"] == "2026-01-01T00:00:00Z"
    assert metadata["n_layers"] == 2


def test_optional_publication_fields_are_enforced_and_skipped_when_requested() -> None:
    with_record = emit_qpu_data_artifact(
        domain="unit",
        source_name="unit-fixture",
        source_mode="recorded",
        K_nm=np.array([[0.0, 0.4], [0.4, 0.0]]),
        omega=np.array([1.0, 1.2]),
        normalization="unit canonical scaling",
        extraction_method="unit-test",
        source_timestamp="2026-01-01T00:00:00Z",
    )
    with_replay_only = emit_qpu_data_artifact(
        domain="unit",
        source_name="unit-fixture",
        source_mode="recorded",
        K_nm=np.array([[0.0, 0.4], [0.4, 0.0]]),
        omega=np.array([1.0, 1.2]),
        normalization="unit canonical scaling",
        extraction_method="unit-test",
        replay_id="unit:replay:2",
    )
    assert validate_qpu_data_artifact(with_record).source_timestamp is not None
    assert validate_qpu_data_artifact(with_replay_only).replay_id == "unit:replay:2"

    missing_both = emit_qpu_data_artifact(
        domain="unit",
        source_name="unit-fixture",
        source_mode="recorded",
        K_nm=np.array([[0.0, 0.4], [0.4, 0.0]]),
        omega=np.array([1.0, 1.2]),
        normalization="unit canonical scaling",
        extraction_method="unit-test",
    )
    with pytest.raises(ValueError, match="source_timestamp or replay_id"):
        validate_qpu_data_artifact(missing_both)
    assert (
        validate_qpu_data_artifact(
            missing_both,
            require_publication_safe=False,
        ).source_mode
        == "recorded"
    )


def test_artifact_sha256_is_verified() -> None:
    payload = _base_payload(artifact_sha256="bad")
    with pytest.raises(ValueError, match="artifact_sha256"):
        validate_qpu_data_artifact(payload)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("domain", "   ", "domain must be non-empty"),
        ("source_name", "", "source_name must be non-empty"),
        ("source_mode", "unreviewed", "source_mode must be one of"),
        ("normalization", "", "normalization must be non-empty"),
        ("extraction_method", " ", "extraction_method must be non-empty"),
    ],
)
def test_emit_rejects_missing_required_provenance_strings(
    field: str, value: object, match: str
) -> None:
    kwargs = {
        "domain": "unit",
        "source_name": "unit-fixture",
        "source_mode": "curated",
        "K_nm": np.array([[0.0, 0.4], [0.4, 0.0]]),
        "omega": np.array([1.0, 1.2]),
        "normalization": "unit canonical scaling",
        "extraction_method": "unit-test",
        "replay_id": "unit:replay:1",
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=match):
        emit_qpu_data_artifact(**kwargs)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("K_nm", [0.0, 0.4], "K_nm must be 2-D"),
        ("omega", [1.0, np.nan], "omega must contain only finite values"),
    ],
)
def test_emit_rejects_malformed_numeric_arrays(
    field: str, value: object, match: str
) -> None:
    kwargs = {
        "domain": "unit",
        "source_name": "unit-fixture",
        "source_mode": "curated",
        "K_nm": np.array([[0.0, 0.4], [0.4, 0.0]]),
        "omega": np.array([1.0, 1.2]),
        "normalization": "unit canonical scaling",
        "extraction_method": "unit-test",
        "replay_id": "unit:replay:1",
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=match):
        emit_qpu_data_artifact(**kwargs)


def test_from_dict_rejects_missing_fields_and_schema_mismatch() -> None:
    payload = _base_payload()
    missing_hash = dict(payload)
    missing_hash.pop("hashes")

    with pytest.raises(ValueError, match="artifact missing required fields"):
        QPUDataArtifact.from_dict(missing_hash)

    unsupported_schema = dict(payload)
    unsupported_schema["schema_version"] = "scpn-quantum-control.qpu-data-artifact.v0"
    with pytest.raises(
        ValueError,
        match="unsupported QPU data artifact schema version",
    ):
        QPUDataArtifact.from_dict(unsupported_schema)


def test_from_dict_rejects_non_mapping_metadata_and_hashes() -> None:
    payload = _base_payload()
    payload["metadata"] = "bad-metadata"

    with pytest.raises(ValueError, match="metadata must be a mapping"):
        QPUDataArtifact.from_dict(payload)

    payload["metadata"] = {"source": "unit"}
    payload["hashes"] = [("K_nm_sha256", "not", "a", "map")]

    with pytest.raises(ValueError, match="hashes must be a mapping"):
        QPUDataArtifact.from_dict(payload)


def test_json_roundtrip_preserves_payload_and_hashes() -> None:
    payload = _base_payload()

    artifact = QPUDataArtifact.from_json(QPUDataArtifact.from_dict(payload).to_json())

    assert artifact.to_dict() == payload
    assert artifact.hashes == payload["hashes"]
    assert artifact.replay_id == "unit:replay:1"


def test_qpu_artifact_file_io_writes_newline_and_reads_validated_payload(tmp_path):
    payload = _base_payload()
    artifact = QPUDataArtifact.from_dict(payload)
    path = tmp_path / "qpu-data-artifact.json"

    write_qpu_data_artifact(path, artifact)
    raw = path.read_text(encoding="utf-8")
    loaded = read_qpu_data_artifact(path)

    assert raw.endswith("\n")
    assert loaded.to_dict() == payload


def test_reading_tampered_json_file_reports_digest_mismatch(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["artifact_sha256"] = "not-real"
    path = tmp_path / "broken-qpu-data-artifact.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact_sha256"):
        read_qpu_data_artifact(path)


def test_read_qpu_data_artifact_rejects_non_mapping_payload(tmp_path: Path) -> None:
    path = tmp_path / "non-mapping-qpu-data-artifact.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact payload must be a mapping"):
        read_qpu_data_artifact(path)


def test_compile_domain_to_qpu_artifact_rejects_synthetic_payload_by_default() -> None:
    with pytest.raises(
        ValueError, match="synthetic artifacts are not publication-safe"
    ):
        compile_domain_to_qpu_artifact(
            REPO_ROOT / "domainpacks" / "minimal_domain",
            source_mode="synthetic",
            replay_id="unit:replay:1",
        )


def test_compile_domain_to_qpu_artifact_accepts_binding_spec_path_inputs() -> None:
    domain_pack = REPO_ROOT / "domainpacks" / "minimal_domain"
    from_dir = compile_domain_to_qpu_artifact(
        domain_pack,
        source_mode="curated",
        replay_id="domainpack:minimal_domain:0.1.0",
    )
    from_file = compile_domain_to_qpu_artifact(
        domain_pack / "binding_spec.yaml",
        source_mode="curated",
        replay_id="domainpack:minimal_domain:0.1.0",
    )
    from_string = compile_domain_to_qpu_artifact(
        str(domain_pack / "binding_spec.yaml"),
        source_mode="curated",
        replay_id="domainpack:minimal_domain:0.1.0",
    )

    assert from_dir["artifact_sha256"] == from_file["artifact_sha256"]
    assert from_dir["artifact_sha256"] == from_string["artifact_sha256"]
    assert from_dir["metadata"]["binding_spec"] == "binding_spec.yaml"
    assert from_file["metadata"]["binding_spec"] == "binding_spec.yaml"
    assert from_string["metadata"]["binding_spec"] == "binding_spec.yaml"


def test_compile_domain_to_qpu_artifact_allows_skip_publication_safety_checks() -> None:
    payload = compile_domain_to_qpu_artifact(
        REPO_ROOT / "domainpacks" / "minimal_domain",
        source_mode="recorded",
        require_publication_safe=False,
    )

    assert payload["source_timestamp"] is None
    assert payload["replay_id"] is None
    validated = validate_qpu_data_artifact(payload, require_publication_safe=False)
    assert validated.domain == "minimal_domain"


def test_metadata_order_does_not_change_manifest_digest() -> None:
    first = emit_qpu_data_artifact(
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
        metadata={"x": 1, "y": 2},
    )
    second = emit_qpu_data_artifact(
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
        metadata={"y": 2, "x": 1},
    )

    assert first["artifact_sha256"] == second["artifact_sha256"]
    assert first["hashes"] == second["hashes"]


def test_validate_rejects_non_finite_theta0() -> None:
    with pytest.raises(ValueError, match="theta0 must contain only finite values"):
        emit_qpu_data_artifact(
            domain="unit",
            source_name="unit-fixture",
            source_mode="curated",
            K_nm=[[0.0, 0.4], [0.4, 0.0]],
            omega=[1.0, 1.2],
            theta0=[0.0, np.nan],
            layer_assignments=["a", "b"],
            normalization="unit canonical scaling",
            extraction_method="unit-test",
            replay_id="unit:replay:1",
        )


def test_read_qpu_data_artifact_rejects_malformed_json_file(tmp_path: Path) -> None:
    path = tmp_path / "malformed-qpu-data-artifact.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        read_qpu_data_artifact(path)


def test_read_qpu_data_artifact_rejects_non_finite_json_constants(
    tmp_path: Path,
) -> None:
    path = tmp_path / "non-finite-qpu-data-artifact.json"
    path.write_text('{"metadata": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="finite JSON"):
        read_qpu_data_artifact(path)


def test_emit_qpu_data_artifact_rejects_non_finite_metadata() -> None:
    with pytest.raises(ValueError, match="finite JSON"):
        emit_qpu_data_artifact(
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
            metadata={"quality_score": float("nan")},
        )


def test_emit_qpu_data_artifact_rejects_non_serialisable_metadata() -> None:
    with pytest.raises(TypeError, match="not JSON serializable"):
        emit_qpu_data_artifact(
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
            metadata={"bad": {1, 2}},
        )

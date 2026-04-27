# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — QPU data artifact emitter

"""QPU-ready oscillator artifacts for SCPN Quantum Control.

Phase Orchestrator owns domain-to-oscillator compilation. This module
serialises that output into the canonical JSON payload accepted by
SCPN Quantum Control's QPU data artifact validator.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.binding.loader import load_binding_spec
from scpn_phase_orchestrator.binding.types import BindingSpec
from scpn_phase_orchestrator.coupling.knm import CouplingBuilder

SCHEMA_VERSION = "scpn-quantum-control.qpu-data-artifact.v1"
REAL_SOURCE_MODES = frozenset({"recorded", "replay", "curated", "derived"})
SYNTHETIC_SOURCE_MODES = frozenset({"synthetic", "simulation", "fixture"})
ALL_SOURCE_MODES = REAL_SOURCE_MODES | SYNTHETIC_SOURCE_MODES

_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "domain",
        "source_name",
        "source_mode",
        "K_nm",
        "omega",
        "theta0",
        "layer_assignments",
        "normalization",
        "extraction_method",
        "source_timestamp",
        "replay_id",
        "metadata",
        "hashes",
        "artifact_sha256",
    }
)


def _array_sha256(array: NDArray[np.float64]) -> str:
    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _finite_float_array(name: str, value: Any, *, ndim: int) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _hashes_with_verified_values(
    hashes: Mapping[str, str],
    *,
    K_nm: NDArray[np.float64],
    omega: NDArray[np.float64],
    theta0: NDArray[np.float64] | None,
) -> dict[str, str]:
    result = dict(hashes)
    expected = {
        "K_nm_sha256": _array_sha256(K_nm),
        "omega_sha256": _array_sha256(omega),
    }
    if theta0 is not None:
        expected["theta0_sha256"] = _array_sha256(theta0)
    for key, value in expected.items():
        if key in result and result[key] != value:
            raise ValueError(f"{key} does not match artifact data")
        result[key] = value
    return result


@dataclass(frozen=True)
class QPUDataArtifact:
    """Validated oscillator data ready for quantum-control compilation."""

    domain: str
    source_name: str
    source_mode: str
    K_nm: NDArray[np.float64]
    omega: NDArray[np.float64]
    theta0: NDArray[np.float64] | None = None
    layer_assignments: list[str] = field(default_factory=list)
    normalization: str = ""
    extraction_method: str = ""
    source_timestamp: str | None = None
    replay_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    hashes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        domain = str(self.domain).strip()
        source_name = str(self.source_name).strip()
        source_mode = str(self.source_mode).strip()
        normalization = str(self.normalization).strip()
        extraction_method = str(self.extraction_method).strip()
        K_nm = _finite_float_array("K_nm", self.K_nm, ndim=2)
        omega = _finite_float_array("omega", self.omega, ndim=1)
        theta0 = (
            None
            if self.theta0 is None
            else _finite_float_array("theta0", self.theta0, ndim=1)
        )
        layer_assignments = [str(item) for item in self.layer_assignments]

        if not domain:
            raise ValueError("domain must be non-empty")
        if not source_name:
            raise ValueError("source_name must be non-empty")
        if source_mode not in ALL_SOURCE_MODES:
            raise ValueError(f"source_mode must be one of {sorted(ALL_SOURCE_MODES)}")
        if not normalization:
            raise ValueError("normalization must be non-empty")
        if not extraction_method:
            raise ValueError("extraction_method must be non-empty")
        if K_nm.shape[0] != K_nm.shape[1]:
            raise ValueError("K_nm must be square")
        if omega.shape != (K_nm.shape[0],):
            msg = f"omega shape must be ({K_nm.shape[0]},), got {omega.shape}"
            raise ValueError(msg)
        if theta0 is not None and theta0.shape != omega.shape:
            raise ValueError(f"theta0 shape must match omega shape, got {theta0.shape}")
        if layer_assignments and len(layer_assignments) != K_nm.shape[0]:
            raise ValueError("layer_assignments length must match K_nm dimension")
        if not np.allclose(np.diag(K_nm), 0.0, atol=1e-12):
            raise ValueError("K_nm diagonal must be zero")
        if np.any(K_nm < -1e-12):
            raise ValueError("K_nm must be non-negative; encode lags in metadata")
        if not np.allclose(K_nm, K_nm.T, atol=1e-12):
            raise ValueError("K_nm must be symmetric for current Kuramoto-XY circuits")

        hashes = _hashes_with_verified_values(
            self.hashes,
            K_nm=K_nm,
            omega=omega,
            theta0=theta0,
        )

        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "source_name", source_name)
        object.__setattr__(self, "source_mode", source_mode)
        object.__setattr__(self, "K_nm", K_nm)
        object.__setattr__(self, "omega", omega)
        object.__setattr__(self, "theta0", theta0)
        object.__setattr__(self, "layer_assignments", layer_assignments)
        object.__setattr__(self, "normalization", normalization)
        object.__setattr__(self, "extraction_method", extraction_method)
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "hashes", hashes)

    @property
    def n_oscillators(self) -> int:
        """Number of oscillators encoded by the artifact."""
        return int(self.K_nm.shape[0])

    @property
    def is_synthetic(self) -> bool:
        """Whether the artifact is non-publication synthetic data."""
        return self.source_mode in SYNTHETIC_SOURCE_MODES

    def require_publication_safe(self) -> None:
        """Reject synthetic or insufficiently traceable artifacts."""
        if self.is_synthetic:
            raise ValueError("synthetic artifacts are not publication-safe")
        if not (self.source_timestamp or self.replay_id):
            raise ValueError(
                "publication artifacts require source_timestamp or replay_id"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the canonical JSON-compatible payload."""
        payload = {
            "schema_version": SCHEMA_VERSION,
            "domain": self.domain,
            "source_name": self.source_name,
            "source_mode": self.source_mode,
            "K_nm": self.K_nm.tolist(),
            "omega": self.omega.tolist(),
            "theta0": None if self.theta0 is None else self.theta0.tolist(),
            "layer_assignments": list(self.layer_assignments),
            "normalization": self.normalization,
            "extraction_method": self.extraction_method,
            "source_timestamp": self.source_timestamp,
            "replay_id": self.replay_id,
            "metadata": dict(self.metadata),
            "hashes": dict(self.hashes),
        }
        payload["artifact_sha256"] = _json_sha256(payload)
        return payload

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialise to JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> QPUDataArtifact:
        """Load and validate an artifact mapping."""
        missing = _REQUIRED_FIELDS - data.keys()
        if missing:
            raise ValueError(f"artifact missing required fields: {sorted(missing)}")
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported QPU data artifact schema version")

        parsed = cls(
            domain=str(data["domain"]),
            source_name=str(data["source_name"]),
            source_mode=str(data["source_mode"]),
            K_nm=np.asarray(data["K_nm"], dtype=np.float64),
            omega=np.asarray(data["omega"], dtype=np.float64),
            theta0=(
                None
                if data.get("theta0") is None
                else np.asarray(data["theta0"], dtype=np.float64)
            ),
            layer_assignments=list(data.get("layer_assignments", [])),
            normalization=str(data["normalization"]),
            extraction_method=str(data["extraction_method"]),
            source_timestamp=data.get("source_timestamp"),
            replay_id=data.get("replay_id"),
            metadata=dict(data.get("metadata", {})),
            hashes=dict(data.get("hashes", {})),
        )
        expected = parsed.to_dict()["artifact_sha256"]
        if data.get("artifact_sha256") != expected:
            raise ValueError("artifact_sha256 does not match artifact payload")
        return parsed

    @classmethod
    def from_json(cls, payload: str) -> QPUDataArtifact:
        """Load and validate an artifact from JSON."""
        return cls.from_dict(json.loads(payload))


def _layer_assignments(spec: BindingSpec) -> list[str]:
    assignments: list[str] = []
    for layer in spec.layers:
        assignments.extend([layer.name] * len(layer.oscillator_ids))
    return assignments


def _domain_metadata(spec: BindingSpec, domainpack_path: Path) -> dict[str, Any]:
    return {
        "source_project": "scpn-phase-orchestrator",
        "binding_spec": domainpack_path.name,
        "binding_version": spec.version,
        "safety_tier": spec.safety_tier,
        "sample_period_s": spec.sample_period_s,
        "control_period_s": spec.control_period_s,
        "n_layers": len(spec.layers),
        "n_oscillators": sum(len(layer.oscillator_ids) for layer in spec.layers),
        "coupling": {
            "base_strength": spec.coupling.base_strength,
            "decay_alpha": spec.coupling.decay_alpha,
        },
    }


def emit_qpu_data_artifact(
    *,
    domain: str,
    source_name: str,
    source_mode: str,
    K_nm: NDArray[np.float64] | Sequence[Sequence[float]],
    omega: NDArray[np.float64] | Sequence[float],
    normalization: str,
    extraction_method: str,
    theta0: NDArray[np.float64] | Sequence[float] | None = None,
    layer_assignments: Sequence[str] = (),
    source_timestamp: str | None = None,
    replay_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Emit a validated QPU artifact payload from oscillator arrays."""
    artifact = QPUDataArtifact(
        domain=domain,
        source_name=source_name,
        source_mode=source_mode,
        K_nm=np.asarray(K_nm, dtype=np.float64),
        omega=np.asarray(omega, dtype=np.float64),
        theta0=None if theta0 is None else np.asarray(theta0, dtype=np.float64),
        layer_assignments=list(layer_assignments),
        normalization=normalization,
        extraction_method=extraction_method,
        source_timestamp=source_timestamp,
        replay_id=replay_id,
        metadata=dict(metadata or {}),
    )
    return artifact.to_dict()


def compile_domain_to_qpu_artifact(
    domain_pack: str | Path,
    *,
    source_mode: str,
    source_name: str | None = None,
    replay_id: str | None = None,
    source_timestamp: str | None = None,
    theta0: Sequence[float] | NDArray[np.float64] | None = None,
    require_publication_safe: bool = True,
) -> dict[str, Any]:
    """Compile a binding spec or domainpack directory into a QPU artifact."""
    path = Path(domain_pack)
    binding_path = path / "binding_spec.yaml" if path.is_dir() else path
    spec = load_binding_spec(binding_path)
    n_oscillators = sum(len(layer.oscillator_ids) for layer in spec.layers)
    coupling = CouplingBuilder().build(
        n_oscillators,
        spec.coupling.base_strength,
        spec.coupling.decay_alpha,
    )
    payload = emit_qpu_data_artifact(
        domain=spec.name,
        source_name=source_name or spec.name,
        source_mode=source_mode,
        K_nm=coupling.knm,
        omega=np.asarray(spec.get_omegas(), dtype=np.float64),
        theta0=theta0,
        layer_assignments=_layer_assignments(spec),
        normalization="binding_spec CouplingBuilder.build exponential K_nm",
        extraction_method="scpn_phase_orchestrator.binding_spec.v1",
        source_timestamp=source_timestamp,
        replay_id=replay_id,
        metadata=_domain_metadata(spec, binding_path),
    )
    if require_publication_safe:
        validate_qpu_data_artifact(payload, require_publication_safe=True)
    return payload


def validate_qpu_data_artifact(
    artifact: QPUDataArtifact | Mapping[str, Any],
    *,
    require_publication_safe: bool = True,
) -> QPUDataArtifact:
    """Validate a QPU artifact and optionally enforce publication safety."""
    parsed = (
        artifact
        if isinstance(artifact, QPUDataArtifact)
        else QPUDataArtifact.from_dict(artifact)
    )
    if require_publication_safe:
        parsed.require_publication_safe()
    return parsed


def read_qpu_data_artifact(path: str | Path) -> QPUDataArtifact:
    """Read a QPU data artifact from JSON."""
    return QPUDataArtifact.from_json(Path(path).read_text(encoding="utf-8"))


def write_qpu_data_artifact(path: str | Path, artifact: QPUDataArtifact) -> None:
    """Write a QPU data artifact to JSON."""
    Path(path).write_text(artifact.to_json() + "\n", encoding="utf-8")


__all__ = [
    "ALL_SOURCE_MODES",
    "QPUDataArtifact",
    "REAL_SOURCE_MODES",
    "SCHEMA_VERSION",
    "SYNTHETIC_SOURCE_MODES",
    "compile_domain_to_qpu_artifact",
    "emit_qpu_data_artifact",
    "read_qpu_data_artifact",
    "validate_qpu_data_artifact",
    "write_qpu_data_artifact",
]

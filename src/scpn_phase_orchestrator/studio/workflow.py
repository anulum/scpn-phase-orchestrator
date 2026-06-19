# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Studio workflow state core

"""Pure-Python workflow state objects for SPO Studio audit artefacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from math import isfinite
from types import MappingProxyType
from typing import Literal, TypeAlias

__all__ = [
    "BindingProposal",
    "ExportManifest",
    "ImportedSourceSummary",
    "RuntimeSnapshot",
    "StudioProjectState",
]

SafetyPosture = Literal["review_artifact", "deployable"]
JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def _required_mapping(value: object, *, field_name: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _required_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _string_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        raise ValueError(f"{field_name} must be a sequence of strings")
    return tuple(_required_string(item, field_name=field_name) for item in value)


def _non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be a non-negative int")
    if value < 0:
        raise ValueError(f"{field_name} must be a non-negative int")
    return value


def _stable_sha256(payload: bytes | str) -> str:
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return sha256(payload).hexdigest()


def _sha256_digest(value: object, *, field_name: str) -> str:
    digest = _required_string(value, field_name=field_name)
    if len(digest) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in digest
    ):
        raise ValueError(
            f"{field_name} must be a 64-character hexadecimal SHA-256 digest"
        )
    return digest


def _json_safe_value(value: object, *, field_name: str) -> JsonValue:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{field_name} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        return _json_safe_mapping(value, field_name=field_name)
    if isinstance(value, list | tuple):
        return [_json_safe_value(item, field_name=field_name) for item in value]
    raise ValueError(
        f"{field_name} contains non-JSON-serialisable value {type(value).__name__}"
    )


def _json_safe_mapping(
    value: object,
    *,
    field_name: str,
) -> dict[str, JsonValue]:
    mapping = _required_mapping(value, field_name=field_name)
    safe: dict[str, JsonValue] = {}
    for key, item in mapping.items():
        if not isinstance(key, str):
            raise ValueError(f"{field_name} contains a non-string key")
        safe[key] = _json_safe_value(item, field_name=field_name)
    return safe


def _freeze_json_value(value: JsonValue) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json_value(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _frozen_json_mapping(
    value: object,
    *,
    field_name: str,
) -> Mapping[str, object]:
    safe = _json_safe_mapping(value, field_name=field_name)
    return MappingProxyType(
        {key: _freeze_json_value(item) for key, item in safe.items()}
    )


def _finite_float(value: object, *, field_name: str) -> int | float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} contains a bool where a float is required")
    if not isinstance(value, int | float):
        raise ValueError(f"{field_name} must be a finite int or float")
    if not isfinite(value):
        raise ValueError(f"{field_name} contains a non-finite float")
    return value


def _finite_float_mapping(
    value: object,
    *,
    field_name: str,
) -> dict[str, int | float]:
    mapping = _required_mapping(value, field_name=field_name)
    safe: dict[str, int | float] = {}
    for key, item in mapping.items():
        if not isinstance(key, str):
            raise ValueError(f"{field_name} contains a non-string key")
        safe[key] = _finite_float(item, field_name=field_name)
    return safe


def _frozen_finite_float_mapping(
    value: object,
    *,
    field_name: str,
) -> Mapping[str, int | float]:
    return MappingProxyType(_finite_float_mapping(value, field_name=field_name))


def _layer_metric_name(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("layer_metrics names must be non-empty strings")
    return value


def _layer_metrics(
    value: object,
) -> tuple[tuple[str, int | float], ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError("layer_metrics must be a sequence of 2-item entries")

    metrics: list[tuple[str, int | float]] = []
    for entry in value:
        if (
            isinstance(entry, str | bytes)
            or not isinstance(entry, Sequence)
            or len(entry) != 2
        ):
            raise ValueError("layer_metrics entries must contain layer and value")
        layer, metric = entry
        metrics.append(
            (
                _layer_metric_name(layer),
                _finite_float(metric, field_name="layer_metrics"),
            )
        )
    return tuple(metrics)


def _hierarchy_watermarks(
    value: object,
) -> dict[str, int]:
    mapping = _required_mapping(value, field_name="hierarchy_watermarks")
    watermarks: dict[str, int] = {}
    for node, watermark in mapping.items():
        if not isinstance(node, str):
            raise ValueError("hierarchy_watermarks contains a non-string node")
        if isinstance(watermark, bool) or not isinstance(watermark, int):
            raise ValueError("hierarchy_watermarks values must be non-negative ints")
        if watermark < 0:
            raise ValueError("hierarchy_watermarks values must be non-negative ints")
        watermarks[node] = watermark
    return watermarks


def _frozen_hierarchy_watermarks(
    value: object,
) -> Mapping[str, int]:
    return MappingProxyType(_hierarchy_watermarks(value))


@dataclass(frozen=True, slots=True)
class ImportedSourceSummary:
    """Audit-ready summary of a source imported into SPO Studio."""

    source_kind: str
    sha256: str
    byte_count: int
    channel_count: int
    sample_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_kind",
            _required_string(self.source_kind, field_name="source_kind"),
        )
        object.__setattr__(
            self,
            "sha256",
            _sha256_digest(self.sha256, field_name="sha256"),
        )
        for field_name in ("byte_count", "channel_count", "sample_count"):
            object.__setattr__(
                self,
                field_name,
                _non_negative_int(getattr(self, field_name), field_name=field_name),
            )

    @classmethod
    def from_payload(
        cls,
        *,
        source_kind: str,
        payload: bytes,
        channel_count: int,
        sample_count: int,
    ) -> ImportedSourceSummary:
        """Create a stable source summary from raw imported bytes.

        Parameters
        ----------
        source_kind : str
            Kind of imported source data.
        payload : bytes
            The payload mapping or bytes.
        channel_count : int
            Number of channels in the source.
        sample_count : int
            Number of samples in the source.

        Returns
        -------
        ImportedSourceSummary
            A stable source summary from raw imported bytes.
        """
        return cls(
            source_kind=source_kind,
            sha256=_stable_sha256(payload),
            byte_count=len(payload),
            channel_count=channel_count,
            sample_count=sample_count,
        )

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-ready audit record.

        Returns
        -------
        dict[str, object]
            A JSON-ready audit record.
        """
        return {
            "source_kind": self.source_kind,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "channel_count": self.channel_count,
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True, slots=True)
class BindingProposal:
    """Candidate binding specification and its validation context."""

    yaml_text: str
    validation_errors: tuple[str, ...] = ()
    inferred_channels: tuple[str, ...] = ()
    confidence_factors: dict[str, float] = field(default_factory=dict)
    provenance: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "yaml_text",
            _required_string(self.yaml_text, field_name="yaml_text"),
        )
        object.__setattr__(
            self,
            "validation_errors",
            _string_tuple(self.validation_errors, field_name="validation_errors"),
        )
        object.__setattr__(
            self,
            "inferred_channels",
            _string_tuple(self.inferred_channels, field_name="inferred_channels"),
        )
        object.__setattr__(
            self,
            "confidence_factors",
            _frozen_finite_float_mapping(
                self.confidence_factors,
                field_name="confidence_factors",
            ),
        )
        object.__setattr__(
            self,
            "provenance",
            _frozen_json_mapping(self.provenance, field_name="provenance"),
        )

    @property
    def yaml_sha256(self) -> str:
        """Stable digest for the proposed binding YAML.

        Returns
        -------
        str
            Stable digest for the proposed binding YAML.
        """
        return _stable_sha256(self.yaml_text)

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-ready audit record.

        Returns
        -------
        dict[str, object]
            A JSON-ready audit record.
        """
        return {
            "yaml_sha256": self.yaml_sha256,
            "validation_errors": list(self.validation_errors),
            "inferred_channels": list(self.inferred_channels),
            "confidence_factors": _finite_float_mapping(
                self.confidence_factors,
                field_name="confidence_factors",
            ),
            "provenance": _json_safe_mapping(
                self.provenance,
                field_name="provenance",
            ),
        }


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    """Current runtime telemetry exposed to Studio review workflows."""

    R: float
    Psi: float
    K: float
    alpha: float
    zeta: float
    regime: str
    layer_metrics: tuple[tuple[str, float], ...] = ()
    hierarchy_watermarks: dict[str, int] = field(default_factory=dict)
    replay_status: str = "not_started"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "regime",
            _required_string(self.regime, field_name="regime"),
        )
        object.__setattr__(
            self,
            "replay_status",
            _required_string(self.replay_status, field_name="replay_status"),
        )
        for field_name in ("R", "Psi", "K", "alpha", "zeta"):
            object.__setattr__(
                self,
                field_name,
                _finite_float(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "layer_metrics",
            _layer_metrics(self.layer_metrics),
        )
        object.__setattr__(
            self,
            "hierarchy_watermarks",
            _frozen_hierarchy_watermarks(self.hierarchy_watermarks),
        )

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-ready audit record.

        Returns
        -------
        dict[str, object]
            A JSON-ready audit record.
        """
        return {
            "R": _finite_float(self.R, field_name="R"),
            "Psi": _finite_float(self.Psi, field_name="Psi"),
            "K": _finite_float(self.K, field_name="K"),
            "alpha": _finite_float(self.alpha, field_name="alpha"),
            "zeta": _finite_float(self.zeta, field_name="zeta"),
            "regime": self.regime,
            "layer_metrics": [
                {
                    "layer": _layer_metric_name(layer),
                    "value": _finite_float(value, field_name="layer_metrics"),
                }
                for layer, value in self.layer_metrics
            ],
            "hierarchy_watermarks": _hierarchy_watermarks(self.hierarchy_watermarks),
            "replay_status": self.replay_status,
        }


@dataclass(frozen=True, slots=True)
class ExportManifest:
    """Manifest for an exportable Studio artefact."""

    target_kind: str
    file_name: str
    payload: str
    command: str
    safety_posture: SafetyPosture
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_kind",
            _required_string(self.target_kind, field_name="target_kind"),
        )
        object.__setattr__(
            self,
            "file_name",
            _required_string(self.file_name, field_name="file_name"),
        )
        object.__setattr__(
            self,
            "payload",
            _required_string(self.payload, field_name="payload"),
        )
        object.__setattr__(
            self,
            "command",
            _required_string(self.command, field_name="command"),
        )
        object.__setattr__(
            self,
            "warnings",
            _string_tuple(self.warnings, field_name="warnings"),
        )
        if self.safety_posture not in ("review_artifact", "deployable"):
            raise ValueError("safety_posture must be review_artifact or deployable")
        if self.safety_posture == "deployable" and not self.warnings:
            raise ValueError("deployable exports require warnings")

    @classmethod
    def review_artifact(
        cls,
        *,
        target_kind: str,
        file_name: str,
        payload: str,
        command: str,
        warnings: tuple[str, ...] = (),
    ) -> ExportManifest:
        """Build a manifest for review-only artefacts.

        Parameters
        ----------
        target_kind : str
            Target export kind.
        file_name : str
            Destination file name.
        payload : str
            The payload mapping or bytes.
        command : str
            The command string.
        warnings : tuple[str, ...]
            Warning messages.

        Returns
        -------
        ExportManifest
            A manifest for review-only artefacts.
        """
        return cls(
            target_kind=target_kind,
            file_name=file_name,
            payload=payload,
            command=command,
            safety_posture="review_artifact",
            warnings=warnings,
        )

    @property
    def payload_sha256(self) -> str:
        """Stable digest for the exported payload.

        Returns
        -------
        str
            Stable digest for the exported payload.
        """
        return _stable_sha256(self.payload)

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-ready audit record.

        Returns
        -------
        dict[str, object]
            A JSON-ready audit record.
        """
        return {
            "target_kind": self.target_kind,
            "file_name": self.file_name,
            "payload_sha256": self.payload_sha256,
            "command": self.command,
            "safety_posture": self.safety_posture,
            "warnings": list(self.warnings),
        }


def _export_tuple(value: object) -> tuple[ExportManifest, ...]:
    if not isinstance(value, list | tuple):
        raise ValueError("exports must be a sequence of ExportManifest entries")
    exports = tuple(value)
    for manifest in exports:
        if not isinstance(manifest, ExportManifest):
            raise ValueError("exports entries must be ExportManifest instances")
    return exports


@dataclass(frozen=True, slots=True)
class StudioProjectState:
    """Complete SPO Studio state for serialisation into audit logs."""

    project_name: str
    source: ImportedSourceSummary
    binding: BindingProposal
    runtime: RuntimeSnapshot
    exports: tuple[ExportManifest, ...] = ()
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "project_name",
            _required_string(self.project_name, field_name="project_name"),
        )
        if not isinstance(self.source, ImportedSourceSummary):
            raise ValueError("source must be an ImportedSourceSummary")
        if not isinstance(self.binding, BindingProposal):
            raise ValueError("binding must be a BindingProposal")
        if not isinstance(self.runtime, RuntimeSnapshot):
            raise ValueError("runtime must be a RuntimeSnapshot")
        object.__setattr__(
            self,
            "exports",
            _export_tuple(self.exports),
        )
        object.__setattr__(
            self,
            "metadata",
            _frozen_json_mapping(self.metadata, field_name="metadata"),
        )

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-ready project-state audit record.

        Returns
        -------
        dict[str, object]
            A JSON-ready project-state audit record.
        """
        return {
            "project_name": self.project_name,
            "source": self.source.to_audit_record(),
            "binding": self.binding.to_audit_record(),
            "runtime": self.runtime.to_audit_record(),
            "exports": [manifest.to_audit_record() for manifest in self.exports],
            "metadata": _json_safe_mapping(self.metadata, field_name="metadata"),
        }

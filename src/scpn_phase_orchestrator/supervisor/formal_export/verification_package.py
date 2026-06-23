# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Formal verification package assembly

"""Verification package assembly: safety properties, checker commands, artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from scpn_phase_orchestrator.exceptions import PolicyError

from ._shared import PrismExport, TLAExport

_PACKAGE_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")


_SUPPORTED_CHECKERS = {"prism", "tlc", "spin", "smt"}


_SUPPORTED_TEXT_ARTIFACT_TYPES = {"promela", "smt2"}


@dataclass(frozen=True)
class FormalTextArtifact:
    """Reviewed external proof artefact text for package manifests.

    This object lets operators add already-reviewed Promela or SMT-LIB artefacts
    to the same deterministic package contract as generated PRISM/TLA exports.
    It records text only; it does not generate, write, or execute external
    checker inputs.
    """

    artifact_type: str
    text: str

    def __post_init__(self) -> None:
        if self.artifact_type not in _SUPPORTED_TEXT_ARTIFACT_TYPES:
            raise PolicyError("formal text artifact type must be 'promela' or 'smt2'")
        if not isinstance(self.text, str) or not self.text.strip():
            raise PolicyError("formal text artifact must contain non-empty text")
        for char in self.text:
            if ord(char) < 32 and char not in {"\n", "\r", "\t"}:
                raise PolicyError(
                    "formal text artifact must not contain unsafe control characters"
                )


@dataclass(frozen=True)
class FormalSafetyProperty:
    """Named model-checking property bound to one exported artefact."""

    name: str
    artifact_name: str
    checker: str
    expression: str
    description: str = ""
    required: bool = True

    def __post_init__(self) -> None:
        _require_package_identifier(self.name, "property name")
        _require_package_identifier(self.artifact_name, "property artifact_name")
        if self.checker not in _SUPPORTED_CHECKERS:
            raise PolicyError(
                "property checker must be 'prism', 'tlc', 'spin', or 'smt'"
            )
        if not isinstance(self.expression, str) or not self.expression.strip():
            raise PolicyError("property expression must be a non-empty string")
        if any(ord(char) < 32 for char in self.expression):
            raise PolicyError("property expression must not contain control characters")
        if any(ord(char) < 32 for char in self.description):
            raise PolicyError(
                "property description must not contain control characters"
            )

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe formal property record.

        Returns
        -------
        dict[str, object]
            Return a JSON-safe formal property record.
        """
        return {
            "name": self.name,
            "artifact_name": self.artifact_name,
            "checker": self.checker,
            "expression": self.expression,
            "description": self.description,
            "required": self.required,
        }


@dataclass(frozen=True)
class FormalCheckerCommand:
    """External model-checker command manifest for one property."""

    property_name: str
    checker: str
    artifact_name: str
    command: tuple[str, ...]
    execution_permitted: bool = False

    def __post_init__(self) -> None:
        _require_package_identifier(self.property_name, "checker property_name")
        if self.checker not in _SUPPORTED_CHECKERS:
            raise PolicyError(
                "checker command checker must be 'prism', 'tlc', 'spin', or 'smt'"
            )
        _require_package_identifier(self.artifact_name, "checker artifact_name")
        if not self.command:
            raise PolicyError("checker command must not be empty")
        for part in self.command:
            if not isinstance(part, str) or not part.strip():
                raise PolicyError("checker command parts must be non-empty strings")
            if any(ord(char) < 32 for char in part):
                raise PolicyError(
                    "checker command parts must not contain control characters"
                )
        if self.execution_permitted:
            raise PolicyError("formal checker command execution must stay disabled")

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe external checker command record.

        Returns
        -------
        dict[str, object]
            Return a JSON-safe external checker command record.
        """
        return {
            "property_name": self.property_name,
            "checker": self.checker,
            "artifact_name": self.artifact_name,
            "command": list(self.command),
            "execution_permitted": self.execution_permitted,
        }


@dataclass(frozen=True)
class FormalVerificationPackage:
    """Deterministic bundle for external formal-verification workflows."""

    package_name: str
    artifact_hashes: dict[str, str]
    artifact_types: dict[str, str]
    properties: tuple[FormalSafetyProperty, ...]
    checker_commands: tuple[FormalCheckerCommand, ...]
    package_hash: str

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe package manifest.

        Returns
        -------
        dict[str, object]
            Return a JSON-safe package manifest.
        """
        return {
            "package_name": self.package_name,
            "artifact_hashes": dict(sorted(self.artifact_hashes.items())),
            "artifact_types": dict(sorted(self.artifact_types.items())),
            "properties": [item.to_audit_record() for item in self.properties],
            "checker_commands": [
                command.to_audit_record() for command in self.checker_commands
            ],
            "package_hash": self.package_hash,
        }


def _require_package_identifier(value: object, field_name: str) -> str:
    """Return the validated verification-package identifier, else raise."""
    if not isinstance(value, str) or not _PACKAGE_NAME_RE.fullmatch(value):
        raise PolicyError(
            f"{field_name} must start with a letter and contain only letters, "
            "digits, underscore, dot, or hyphen"
        )
    return value


def _artifact_text(export: PrismExport | TLAExport | FormalTextArtifact) -> str:
    """Return the text content of a verification artifact."""
    if isinstance(export, PrismExport):
        return export.model
    if isinstance(export, FormalTextArtifact):
        return export.text
    return export.module


def _artifact_type(export: PrismExport | TLAExport | FormalTextArtifact) -> str:
    """Return the type label of a verification artifact."""
    if isinstance(export, PrismExport):
        return "prism"
    if isinstance(export, FormalTextArtifact):
        return export.artifact_type
    return "tla"


def _checker_command(property_: FormalSafetyProperty) -> FormalCheckerCommand:
    """Return the checker command for a verification artifact."""
    command: tuple[str, ...]
    if property_.checker == "prism":
        command = (
            "prism",
            f"{property_.artifact_name}.prism",
            "-pf",
            property_.expression,
        )
    else:
        if property_.checker == "spin":
            command = (
                "spin",
                "-run",
                f"{property_.artifact_name}.pml",
            )
        elif property_.checker == "smt":
            command = (
                "z3",
                f"{property_.artifact_name}.smt2",
            )
        else:
            command = (
                "tlc2.TLC",
                f"{property_.artifact_name}.tla",
                "-config",
                f"{property_.artifact_name}.cfg",
            )
        return FormalCheckerCommand(
            property_name=property_.name,
            checker=property_.checker,
            artifact_name=property_.artifact_name,
            command=command,
        )
    return FormalCheckerCommand(
        property_name=property_.name,
        checker=property_.checker,
        artifact_name=property_.artifact_name,
        command=command,
    )


def _checker_matches_artifact(
    property_: FormalSafetyProperty,
    artifact_type: str,
) -> bool:
    """Return whether a checker matches an artifact type."""
    if property_.checker == "prism":
        return artifact_type == "prism"
    if property_.checker == "tlc":
        return artifact_type == "tla"
    if property_.checker == "spin":
        return artifact_type == "promela"
    return artifact_type == "smt2"


def build_formal_verification_package(
    artifacts: Mapping[str, PrismExport | TLAExport | FormalTextArtifact],
    properties: Sequence[FormalSafetyProperty],
    *,
    package_name: str = "spo-formal-verification",
) -> FormalVerificationPackage:
    """Build a deterministic manifest for external model-checker execution.

    The package records exported artefact hashes, property-library entries, and
    exact checker commands. It never writes files or invokes external tools;
    CI or operators can materialise the package and run the recorded commands
    in a controlled environment.

    Parameters
    ----------
    artifacts : Mapping[str, PrismExport | TLAExport | FormalTextArtifact]
        Mapping of artefact name to its formal export.
    properties : Sequence[FormalSafetyProperty]
        The formal safety properties.
    package_name : str
        Name for the verification package.

    Returns
    -------
    FormalVerificationPackage
        The formal verification package manifest.

    Raises
    ------
    PolicyError
        If the artefacts or properties fail policy checks.
    """
    _require_package_identifier(package_name, "package_name")
    if not artifacts:
        raise PolicyError("formal verification package requires artifacts")
    if not properties:
        raise PolicyError("formal verification package requires properties")

    artifact_hashes: dict[str, str] = {}
    artifact_types: dict[str, str] = {}
    for artifact_name, export in sorted(artifacts.items()):
        _require_package_identifier(artifact_name, "artifact name")
        if not isinstance(export, PrismExport | TLAExport | FormalTextArtifact):
            raise PolicyError(
                "formal artifacts must be PrismExport, TLAExport, or FormalTextArtifact"
            )
        artifact_text = _artifact_text(export)
        if not artifact_text.strip():
            raise PolicyError(f"formal artifact {artifact_name!r} is empty")
        artifact_hashes[artifact_name] = hashlib.sha256(
            artifact_text.encode("utf-8")
        ).hexdigest()
        artifact_types[artifact_name] = _artifact_type(export)

    property_names: set[str] = set()
    commands: list[FormalCheckerCommand] = []
    for property_ in properties:
        if property_.name in property_names:
            raise PolicyError(f"duplicate formal property {property_.name!r}")
        property_names.add(property_.name)
        if property_.artifact_name not in artifact_hashes:
            raise PolicyError(
                f"formal property {property_.name!r} references unknown artifact "
                f"{property_.artifact_name!r}"
            )
        if not _checker_matches_artifact(
            property_,
            artifact_types[property_.artifact_name],
        ):
            raise PolicyError(
                f"formal property {property_.name!r} checker does not match "
                f"artifact {property_.artifact_name!r}"
            )
        commands.append(_checker_command(property_))

    package_seed = {
        "package_name": package_name,
        "artifact_hashes": dict(sorted(artifact_hashes.items())),
        "artifact_types": dict(sorted(artifact_types.items())),
        "properties": [item.to_audit_record() for item in properties],
        "checker_commands": [command.to_audit_record() for command in commands],
    }
    package_hash = hashlib.sha256(
        json.dumps(package_seed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return FormalVerificationPackage(
        package_name=package_name,
        artifact_hashes=artifact_hashes,
        artifact_types=artifact_types,
        properties=tuple(properties),
        checker_commands=tuple(commands),
        package_hash=package_hash,
    )

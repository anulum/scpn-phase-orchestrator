# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Formal runtime control certificate

"""Checker availability auditing and runtime control certificate construction."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

from scpn_phase_orchestrator.exceptions import PolicyError

from .verification_package import (
    _SUPPORTED_CHECKERS,
    FormalVerificationPackage,
    _require_package_identifier,
)

_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")


@dataclass(frozen=True)
class FormalCheckerAvailability:
    """Non-executing readiness record for one external checker command."""

    property_name: str
    checker: str
    artifact_name: str
    executable: str
    command: tuple[str, ...]
    available: bool
    resolved_path: str | None = None
    status: str = "missing_executable"
    execution_permitted: bool = False

    def __post_init__(self) -> None:
        """Validate the non-executing checker availability contract."""
        _require_package_identifier(self.property_name, "availability property_name")
        if self.checker not in _SUPPORTED_CHECKERS:
            raise PolicyError(
                "availability checker must be 'prism', 'tlc', 'spin', or 'smt'"
            )
        _require_package_identifier(self.artifact_name, "availability artifact_name")
        if not isinstance(self.executable, str) or not self.executable.strip():
            raise PolicyError("availability executable must be a non-empty string")
        if any(ord(char) < 32 for char in self.executable):
            raise PolicyError(
                "availability executable must not contain control characters"
            )
        if self.resolved_path is not None and (
            not isinstance(self.resolved_path, str)
            or not self.resolved_path.strip()
            or any(ord(char) < 32 for char in self.resolved_path)
        ):
            raise PolicyError(
                "availability resolved_path must be None or a non-empty safe string"
            )
        if self.status not in {"ready_not_executed", "missing_executable"}:
            raise PolicyError("availability status is unsupported")
        if self.available != (self.status == "ready_not_executed"):
            raise PolicyError("availability status must match available flag")
        if not self.command:
            raise PolicyError("availability command must not be empty")
        for part in self.command:
            if not isinstance(part, str) or not part.strip():
                raise PolicyError(
                    "availability command parts must be non-empty strings"
                )
            if any(ord(char) < 32 for char in part):
                raise PolicyError(
                    "availability command parts must not contain control characters"
                )
        if self.execution_permitted:
            raise PolicyError("formal checker availability must not permit execution")

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe non-executing checker readiness record.

        Returns
        -------
        dict[str, object]
            Return a JSON-safe non-executing checker readiness record.
        """
        return {
            "property_name": self.property_name,
            "checker": self.checker,
            "artifact_name": self.artifact_name,
            "executable": self.executable,
            "command": list(self.command),
            "available": self.available,
            "resolved_path": self.resolved_path,
            "status": self.status,
            "execution_permitted": self.execution_permitted,
        }


@dataclass(frozen=True)
class FormalCheckerResult:
    """Reviewed external checker result bound to one package hash.

    Results are audit records supplied by CI or a human-reviewed verification
    workflow after materialising the package outside this library. The
    constructor validates identity, checker kind, package hash, and result hash;
    it never executes checkers and never grants actuation.
    """

    property_name: str
    checker: str
    artifact_name: str
    package_hash: str
    result_hash: str
    status: str
    passed: bool
    detail: str = ""
    execution_permitted: bool = False

    def __post_init__(self) -> None:
        """Validate reviewed checker result identity and fail-closed status."""
        _require_package_identifier(self.property_name, "checker result property_name")
        if self.checker not in _SUPPORTED_CHECKERS:
            raise PolicyError(
                "checker result checker must be 'prism', 'tlc', 'spin', or 'smt'"
            )
        _require_package_identifier(self.artifact_name, "checker result artifact_name")
        _require_sha256(self.package_hash, "checker result package_hash")
        _require_sha256(self.result_hash, "checker result_hash")
        if self.status not in {"passed", "failed", "not_run"}:
            raise PolicyError("checker result status is unsupported")
        if not isinstance(self.passed, bool):
            raise PolicyError("checker result passed flag must be a boolean")
        if self.passed != (self.status == "passed"):
            raise PolicyError("checker result status must match passed flag")
        if any(ord(char) < 32 for char in self.detail):
            raise PolicyError(
                "checker result detail must not contain control characters"
            )
        if self.execution_permitted:
            raise PolicyError("checker result execution must stay disabled")

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe external checker result record.

        Returns
        -------
        dict[str, object]
            Return a JSON-safe external checker result record.
        """
        return {
            "property_name": self.property_name,
            "checker": self.checker,
            "artifact_name": self.artifact_name,
            "package_hash": self.package_hash,
            "result_hash": self.result_hash,
            "status": self.status,
            "passed": self.passed,
            "detail": self.detail,
            "execution_permitted": self.execution_permitted,
        }


@dataclass(frozen=True)
class FormalRuntimeCertificate:
    """Fail-closed runtime certificate for formal supervisor evidence.

    A certificate binds a formal verification package, finite runtime bounds,
    checker readiness, and externally supplied checker results into one
    deterministic hash. A verified certificate is still non-actuating; it is an
    auditable precondition for operator review or a separate runtime monitor.
    """

    certificate_name: str
    package_name: str
    package_hash: str
    runtime_bounds: dict[str, float]
    checker_availability: tuple[FormalCheckerAvailability, ...]
    checker_results: tuple[FormalCheckerResult, ...]
    required_property_count: int
    passed_required_count: int
    missing_required_properties: tuple[str, ...]
    failed_required_properties: tuple[str, ...]
    unavailable_checker_properties: tuple[str, ...]
    status: str
    certificate_hash: str
    actuation_permitted: bool = False

    def __post_init__(self) -> None:
        """Validate certificate integrity and non-actuating runtime status."""
        _require_package_identifier(self.certificate_name, "certificate_name")
        _require_package_identifier(self.package_name, "certificate package_name")
        _require_sha256(self.package_hash, "certificate package_hash")
        _validate_runtime_bounds(self.runtime_bounds)
        for availability in self.checker_availability:
            if not isinstance(availability, FormalCheckerAvailability):
                raise PolicyError(
                    "certificate checker_availability must contain availability records"
                )
        for result in self.checker_results:
            if not isinstance(result, FormalCheckerResult):
                raise PolicyError(
                    "certificate checker_results must contain result records"
                )
        for field_name, value in (
            ("required_property_count", self.required_property_count),
            ("passed_required_count", self.passed_required_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PolicyError(f"certificate {field_name} must be non-negative")
        if self.passed_required_count > self.required_property_count:
            raise PolicyError(
                "certificate passed_required_count must not exceed required count"
            )
        _validate_identifier_tuple(
            self.missing_required_properties,
            "certificate missing_required_properties",
        )
        _validate_identifier_tuple(
            self.failed_required_properties,
            "certificate failed_required_properties",
        )
        _validate_identifier_tuple(
            self.unavailable_checker_properties,
            "certificate unavailable_checker_properties",
        )
        if self.status not in {"verified_non_actuating", "blocked"}:
            raise PolicyError("certificate status is unsupported")
        if self.status == "verified_non_actuating" and (
            self.missing_required_properties
            or self.failed_required_properties
            or self.unavailable_checker_properties
            or self.passed_required_count != self.required_property_count
        ):
            raise PolicyError("verified certificate must have complete passed evidence")
        _require_sha256(self.certificate_hash, "certificate_hash")
        if self.actuation_permitted:
            raise PolicyError("formal runtime certificate must remain non-actuating")

    def to_audit_record(self) -> dict[str, object]:
        """Return a deterministic JSON-safe runtime certificate.

        Returns
        -------
        dict[str, object]
            Return a deterministic JSON-safe runtime certificate.
        """
        return {
            "certificate_name": self.certificate_name,
            "package_name": self.package_name,
            "package_hash": self.package_hash,
            "runtime_bounds": dict(sorted(self.runtime_bounds.items())),
            "checker_availability": [
                item.to_audit_record() for item in self.checker_availability
            ],
            "checker_results": [
                item.to_audit_record() for item in self.checker_results
            ],
            "required_property_count": self.required_property_count,
            "passed_required_count": self.passed_required_count,
            "missing_required_properties": list(self.missing_required_properties),
            "failed_required_properties": list(self.failed_required_properties),
            "unavailable_checker_properties": list(self.unavailable_checker_properties),
            "status": self.status,
            "certificate_hash": self.certificate_hash,
            "actuation_permitted": self.actuation_permitted,
        }


def _require_sha256(value: object, field_name: str) -> str:
    """Return ``value`` as a SHA-256 hex digest, else raise."""
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PolicyError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _validate_identifier_tuple(values: tuple[str, ...], field_name: str) -> None:
    """Return the validated tuple of identifiers, else raise."""
    if not isinstance(values, tuple):
        raise PolicyError(f"{field_name} must be a tuple")
    seen: set[str] = set()
    for value in values:
        _require_package_identifier(value, field_name)
        if value in seen:
            raise PolicyError(f"{field_name} must not contain duplicates")
        seen.add(value)


def _validate_runtime_bounds(bounds: Mapping[str, object]) -> dict[str, float]:
    """Return the validated runtime bounds, else raise."""
    if not isinstance(bounds, Mapping) or not bounds:
        raise PolicyError("runtime bounds must be a non-empty mapping")
    parsed: dict[str, float] = {}
    for key, value in bounds.items():
        bound_key = _require_package_identifier(key, "runtime bound name")
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise PolicyError("runtime bounds must be finite real numbers")
        bound_value = float(value)
        if not isfinite(bound_value):
            raise PolicyError("runtime bounds must be finite real numbers")
        parsed[bound_key] = bound_value
    if len(parsed) != len(bounds):
        raise PolicyError("runtime bounds must not contain duplicate names")
    return parsed


def audit_formal_checker_availability(
    package: FormalVerificationPackage,
    *,
    executable_paths: Mapping[str, str | None] | None = None,
) -> tuple[FormalCheckerAvailability, ...]:
    """Return non-executing external-checker readiness records.

    The audit resolves only the first command token for each package checker
    command. It never materialises artefacts, writes files, launches subprocesses,
    or changes the package execution policy. Tests and CI may inject
    ``executable_paths`` for deterministic readiness checks; production callers
    can omit it to use ``shutil.which`` against the current host.

    Parameters
    ----------
    package : FormalVerificationPackage
        The formal verification package.
    executable_paths : Mapping[str, str | None] | None
        Mapping of checker name to executable path, or ``None``.

    Returns
    -------
    tuple[FormalCheckerAvailability, ...]
        The non-executing external-checker readiness records.

    Raises
    ------
    PolicyError
        If the package fails its fail-closed policy checks.
    """
    if not isinstance(package, FormalVerificationPackage):
        raise PolicyError("checker availability audit requires a formal package")
    records: list[FormalCheckerAvailability] = []
    for command in package.checker_commands:
        executable = command.command[0]
        if executable_paths is None:
            resolved_path = shutil.which(executable)
        elif executable in executable_paths:
            resolved_path = executable_paths[executable]
        else:
            resolved_path = None
        available = resolved_path is not None
        records.append(
            FormalCheckerAvailability(
                property_name=command.property_name,
                checker=command.checker,
                artifact_name=command.artifact_name,
                executable=executable,
                command=command.command,
                available=available,
                resolved_path=resolved_path,
                status="ready_not_executed" if available else "missing_executable",
                execution_permitted=False,
            )
        )
    return tuple(records)


def build_runtime_control_certificate(
    package: FormalVerificationPackage,
    checker_availability: Sequence[FormalCheckerAvailability],
    checker_results: Sequence[FormalCheckerResult],
    runtime_bounds: Mapping[str, object],
    *,
    certificate_name: str = "spo-runtime-control-certificate",
) -> FormalRuntimeCertificate:
    """Build a fail-closed runtime certificate from formal evidence.

    The certificate is verified only when every required package property has a
    matching available checker and a passed external result bound to the current
    package hash. Missing, failed, stale, or unavailable evidence produces a
    blocked certificate. The returned record never permits actuation.

    Parameters
    ----------
    package : FormalVerificationPackage
        The formal verification package.
    checker_availability : Sequence[FormalCheckerAvailability]
        External-checker readiness records.
    checker_results : Sequence[FormalCheckerResult]
        External-checker result records.
    runtime_bounds : Mapping[str, object]
        Finite runtime bounds for the certificate.
    certificate_name : str
        Name for the emitted certificate.

    Returns
    -------
    FormalRuntimeCertificate
        The fail-closed runtime control certificate.

    Raises
    ------
    PolicyError
        If the formal evidence fails the fail-closed policy.
    """
    if not isinstance(package, FormalVerificationPackage):
        raise PolicyError("runtime certificate requires a formal package")
    _require_package_identifier(certificate_name, "certificate_name")
    parsed_bounds = _validate_runtime_bounds(runtime_bounds)

    property_by_name = {property_.name: property_ for property_ in package.properties}
    availability_by_property: dict[str, FormalCheckerAvailability] = {}
    for record in checker_availability:
        if not isinstance(record, FormalCheckerAvailability):
            raise PolicyError("checker availability records are required")
        if record.property_name not in property_by_name:
            raise PolicyError("checker availability references unknown property")
        if record.property_name in availability_by_property:
            raise PolicyError("duplicate checker availability property")
        expected = property_by_name[record.property_name]
        if record.checker != expected.checker or record.artifact_name != (
            expected.artifact_name
        ):
            raise PolicyError("checker availability does not match package property")
        availability_by_property[record.property_name] = record

    result_by_property: dict[str, FormalCheckerResult] = {}
    for result in checker_results:
        if not isinstance(result, FormalCheckerResult):
            raise PolicyError("checker result records are required")
        if result.property_name not in property_by_name:
            raise PolicyError("checker result references unknown property")
        if result.property_name in result_by_property:
            raise PolicyError("duplicate checker result property")
        expected = property_by_name[result.property_name]
        if result.checker != expected.checker or result.artifact_name != (
            expected.artifact_name
        ):
            raise PolicyError("checker result does not match package property")
        if result.package_hash != package.package_hash:
            raise PolicyError("checker result package_hash does not match package")
        result_by_property[result.property_name] = result

    required_names = tuple(
        property_.name for property_ in package.properties if property_.required
    )
    missing_required = tuple(
        name for name in required_names if name not in result_by_property
    )
    failed_required = tuple(
        name
        for name in required_names
        if name in result_by_property and not result_by_property[name].passed
    )
    unavailable_required = tuple(
        name
        for name in required_names
        if name not in availability_by_property
        or not availability_by_property[name].available
    )
    passed_required_count = sum(
        int(
            name in result_by_property
            and result_by_property[name].passed
            and name in availability_by_property
            and availability_by_property[name].available
        )
        for name in required_names
    )
    status = (
        "verified_non_actuating"
        if not missing_required and not failed_required and not unavailable_required
        else "blocked"
    )
    sorted_availability = tuple(
        availability_by_property[name] for name in sorted(availability_by_property)
    )
    sorted_results = tuple(
        result_by_property[name] for name in sorted(result_by_property)
    )
    certificate_seed = {
        "certificate_name": certificate_name,
        "package_name": package.package_name,
        "package_hash": package.package_hash,
        "runtime_bounds": dict(sorted(parsed_bounds.items())),
        "checker_availability": [
            item.to_audit_record() for item in sorted_availability
        ],
        "checker_results": [item.to_audit_record() for item in sorted_results],
        "required_property_count": len(required_names),
        "passed_required_count": passed_required_count,
        "missing_required_properties": list(missing_required),
        "failed_required_properties": list(failed_required),
        "unavailable_checker_properties": list(unavailable_required),
        "status": status,
        "actuation_permitted": False,
    }
    certificate_hash = hashlib.sha256(
        json.dumps(
            certificate_seed,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return FormalRuntimeCertificate(
        certificate_name=certificate_name,
        package_name=package.package_name,
        package_hash=package.package_hash,
        runtime_bounds=parsed_bounds,
        checker_availability=sorted_availability,
        checker_results=sorted_results,
        required_property_count=len(required_names),
        passed_required_count=passed_required_count,
        missing_required_properties=tuple(sorted(missing_required)),
        failed_required_properties=tuple(sorted(failed_required)),
        unavailable_checker_properties=tuple(sorted(unavailable_required)),
        status=status,
        certificate_hash=certificate_hash,
        actuation_permitted=False,
    )

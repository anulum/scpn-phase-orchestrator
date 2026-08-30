# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Reactor semantic vocabulary

"""Closed semantic primitives for the U0 reactor contract.

The reactor configuration itself is deliberately not an enum. Configurations
are registry entries so new reactor topologies can be added without changing
the carrier algebra or pretending that they are tokamaks.
"""

from __future__ import annotations

import math
import re
from enum import Enum, StrEnum
from numbers import Real
from typing import TypeVar

U0_SCHEMA_VERSION = "1.0.0"
SEMANTIC_OWNER = "SCPN-PHASE-ORCHESTRATOR"
ACTION_OWNER = "SCPN-CONTROL"
PLANT_TRUTH_OWNERS = frozenset({"SCPN-FUSION-CORE", "SCPN-MIF-CORE"})
REVIEW_ONLY_AUTHORITY = "review_only"

_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:\-]{0,127}$")
_SEMVER_RE = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

EnumType = TypeVar("EnumType", bound=Enum)


class SemanticCarrier(StrEnum):
    """Mathematical carrier of a semantic record."""

    CYCLIC_PHASE = "cyclic_phase"
    COMPLEX_MODE = "complex_mode"
    FIELD_PHASE = "field_phase"
    EVENT_CYCLE = "event_cycle"
    BOUNDED_FEATURE = "bounded_feature"
    CATEGORICAL_STATE = "categorical_state"
    PROTOCOL_PHASE = "protocol_phase"
    NUMERICAL_PHASE = "numerical_phase"


class ConfinementFamily(StrEnum):
    """Broad physical confinement family, independent of reactor geometry."""

    MAGNETIC_CLOSED = "magnetic_closed"
    MAGNETIC_OPEN = "magnetic_open"
    SELF_MAGNETIC = "self_magnetic"
    INERTIAL = "inertial"
    MAGNETO_INERTIAL = "magneto_inertial"
    ELECTROSTATIC = "electrostatic"
    BEAM_TARGET = "beam_target"
    HYBRID = "hybrid"
    EXTENSION = "extension"


class DriverKind(StrEnum):
    """Energy, confinement, or compression driver."""

    EXTERNAL_MAGNETIC_COILS = "external_magnetic_coils"
    PLASMA_CURRENT = "plasma_current"
    NEUTRAL_BEAM = "neutral_beam"
    RADIOFREQUENCY_OR_MICROWAVE = "radiofrequency_or_microwave"
    LASER = "laser"
    ION_BEAM = "ion_beam"
    ELECTRON_BEAM = "electron_beam"
    PULSED_POWER = "pulsed_power"
    SOLID_OR_LIQUID_LINER = "solid_or_liquid_liner"
    PROJECTILE = "projectile"
    PLASMA_JET = "plasma_jet"
    ELECTROSTATIC_POTENTIAL = "electrostatic_potential"
    COMBINED = "combined"


class OperatingCadence(StrEnum):
    """Temporal operating form of a device or experiment."""

    STEADY = "steady"
    QUASI_STEADY = "quasi_steady"
    LONG_PULSE = "long_pulse"
    PULSED_SHOT = "pulsed_shot"
    REPETITIVE_TARGET = "repetitive_target"
    SINGLE_EXPERIMENT = "single_experiment"


class ClockKind(StrEnum):
    """Declared time basis; domains of different kinds are never implicit peers."""

    PLANT_MONOTONIC = "plant_monotonic"
    SIMULATION_MONOTONIC = "simulation_monotonic"
    SHOT_RELATIVE = "shot_relative"
    FACILITY_SYNCHRONIZED = "facility_synchronized"
    WALL_CLOCK = "wall_clock"
    MODEL_TICK = "model_tick"
    UNKNOWN = "unknown"


class ReactionKind(StrEnum):
    """Fusion reaction or fuel class, separate from confinement."""

    DEUTERIUM_TRITIUM = "deuterium_tritium"
    DEUTERIUM_DEUTERIUM = "deuterium_deuterium"
    DEUTERIUM_HELIUM3 = "deuterium_helium3"
    PROTON_BORON11 = "proton_boron11"
    ADVANCED_OR_EXTENSION = "advanced_or_extension"


class ConversionKind(StrEnum):
    """Energy-conversion or experimental boundary."""

    THERMAL_BLANKET = "thermal_blanket"
    DIRECT_CONVERSION = "direct_conversion"
    SUBCRITICAL_FISSION_BLANKET = "subcritical_fission_blanket"
    EXPERIMENTAL_NO_POWER_CONVERSION = "experimental_no_power_conversion"
    EXTENSION = "extension"


class EvidenceClass(StrEnum):
    """Maturity and epistemic source of a claim."""

    OBSERVED = "O"
    EXPERIMENTAL = "X"
    SIMULATION = "S"
    DEVELOPING = "D"
    CONCEPT = "C"
    SCAFFOLD = "G"
    REVIEW_HYPOTHESIS = "R"
    UNKNOWN = "U"


class ValidityState(StrEnum):
    """Fail-closed validity state for a semantic object."""

    VALID = "valid"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"
    STALE = "stale"
    OUT_OF_DISTRIBUTION = "out_of_distribution"
    UNOBSERVABLE = "unobservable"
    INVALID = "invalid"


class QualityState(StrEnum):
    """Measurement or estimator quality."""

    VALID = "valid"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"
    INVALID = "invalid"


class PhaseRelationType(StrEnum):
    """Declared relationship between two phase records."""

    SAME_MODE = "same_mode"
    HARMONIC = "harmonic"
    DRIVEN = "driven"
    EVENT_LOCKED = "event_locked"
    CAUSAL_CANDIDATE = "causal_candidate"
    REVIEW_HYPOTHESIS = "review_hypothesis"


class RelationInterpretation(StrEnum):
    """Operational interpretation of a phase relationship."""

    HEALTHY = "healthy"
    PATHOLOGICAL = "pathological"
    AMBIGUOUS = "ambiguous"
    CONTEXT_DEPENDENT = "context_dependent"


class RegimeState(StrEnum):
    """Top-level operational state without erasing regime axes."""

    NOMINAL = "nominal"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    RECOVERY = "recovery"
    UNKNOWN = "unknown"


def require_text(value: object, *, field: str) -> str:
    """Return a stripped non-empty string.

    Parameters
    ----------
    value : object
        Candidate value.
    field : str
        Field name used in diagnostics.

    Returns
    -------
    str
        Validated text.

    Raises
    ------
    ValueError
        If the value is not a non-empty string.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def require_enum(
    value: object,
    enum_type: type[EnumType],
    *,
    field: str,
) -> EnumType:
    """Return an enum member without coercing untyped public input.

    Parameters
    ----------
    value : object
        Candidate enum member.
    enum_type : type[EnumType]
        Required enum class.
    field : str
        Field name used in diagnostics.

    Returns
    -------
    EnumType
        The validated enum member.

    Raises
    ------
    ValueError
        If ``value`` is not a member of ``enum_type``.
    """
    if not isinstance(value, enum_type):
        raise ValueError(f"{field} must be a {enum_type.__name__} member")
    return value


def require_identifier(value: object, *, field: str) -> str:
    """Return a validated identifier.

    Identifiers support namespaces separated by a colon, allowing extension
    registries without changing the core configuration vocabulary.

    Parameters
    ----------
    value : object
        Candidate identifier.
    field : str
        Field name used in diagnostics.

    Returns
    -------
    str
        The validated identifier.

    Raises
    ------
    ValueError
        If ``value`` is empty, not text, or violates identifier syntax.
    """
    text = require_text(value, field=field)
    if _IDENTIFIER_RE.fullmatch(text) is None:
        raise ValueError(f"{field} must be a namespaced identifier")
    return text


def require_semver(value: object, *, field: str) -> str:
    """Return a strict MAJOR.MINOR.PATCH version.

    Parameters
    ----------
    value : object
        Candidate semantic version.
    field : str
        Field name used in diagnostics.

    Returns
    -------
    str
        The validated semantic version.

    Raises
    ------
    ValueError
        If ``value`` is not non-empty text in MAJOR.MINOR.PATCH form.
    """
    text = require_text(value, field=field)
    if _SEMVER_RE.fullmatch(text) is None:
        raise ValueError(f"{field} must use MAJOR.MINOR.PATCH")
    return text


def require_u0_schema(value: object) -> str:
    """Return the supported U0 schema version or refuse compatibility.

    Parameters
    ----------
    value : object
        Candidate U0 schema version.

    Returns
    -------
    str
        The exact supported U0 schema version.

    Raises
    ------
    ValueError
        If the version is malformed, historical, or forward-incompatible.
    """
    version = require_semver(value, field="schema_version")
    supported_major, supported_minor, _ = _version_parts(U0_SCHEMA_VERSION)
    major, minor, _ = _version_parts(version)
    if major != supported_major:
        raise ValueError(
            f"unsupported schema major {major}; U0 supports {supported_major}"
        )
    if minor > supported_minor:
        raise ValueError(
            f"unsupported forward schema {version}; U0 supports {U0_SCHEMA_VERSION}"
        )
    if version != U0_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported historical schema {version}; migrate to {U0_SCHEMA_VERSION}"
        )
    return version


def require_sha256(value: object, *, field: str) -> str:
    """Return one lowercase hexadecimal SHA-256 digest.

    Parameters
    ----------
    value : object
        Candidate digest.
    field : str
        Field name used in diagnostics.

    Returns
    -------
    str
        The validated 64-character digest.

    Raises
    ------
    ValueError
        If ``value`` is not a lowercase hexadecimal SHA-256 digest.
    """
    text = require_text(value, field=field)
    if _SHA256_RE.fullmatch(text) is None:
        raise ValueError(f"{field} must contain 64 lowercase hexadecimal characters")
    return text


def finite_real(value: object, *, field: str) -> float:
    """Return a finite real scalar while rejecting booleans.

    Parameters
    ----------
    value : object
        Candidate scalar.
    field : str
        Field name used in diagnostics.

    Returns
    -------
    float
        The validated finite scalar.

    Raises
    ------
    ValueError
        If ``value`` is boolean, non-real, or non-finite.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite real")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be a finite real")
    return parsed


def non_negative_real(value: object, *, field: str) -> float:
    """Return a finite non-negative real scalar.

    Parameters
    ----------
    value : object
        Candidate scalar.
    field : str
        Field name used in diagnostics.

    Returns
    -------
    float
        The validated non-negative scalar.

    Raises
    ------
    ValueError
        If ``value`` is not finite and real or is negative.
    """
    parsed = finite_real(value, field=field)
    if parsed < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return parsed


def probability(value: object, *, field: str) -> float:
    """Return a probability in the closed interval zero to one.

    Parameters
    ----------
    value : object
        Candidate probability.
    field : str
        Field name used in diagnostics.

    Returns
    -------
    float
        The validated probability.

    Raises
    ------
    ValueError
        If ``value`` is not finite and real or lies outside ``[0, 1]``.
    """
    parsed = finite_real(value, field=field)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return parsed


def non_negative_integer(value: object, *, field: str) -> int:
    """Return a non-negative integer while rejecting booleans.

    Parameters
    ----------
    value : object
        Candidate integer.
    field : str
        Field name used in diagnostics.

    Returns
    -------
    int
        The validated non-negative integer.

    Raises
    ------
    ValueError
        If ``value`` is boolean, non-integral, or negative.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def require_exact_keys(
    payload: object,
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    field: str,
) -> dict[str, object]:
    """Return a mapping only when its key set matches the contract.

    Parameters
    ----------
    payload : object
        Candidate mapping.
    required : frozenset[str]
        Keys that must be present.
    optional : frozenset[str]
        Additional permitted keys.
    field : str
        Field name used in diagnostics.

    Returns
    -------
    dict[str, object]
        The validated input mapping.

    Raises
    ------
    ValueError
        If the input is not a string-keyed mapping or its key set differs.
    """
    if not isinstance(payload, dict) or any(
        not isinstance(key, str) for key in payload
    ):
        raise ValueError(f"{field} must be an object with string keys")
    keys = frozenset(payload)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        raise ValueError(f"{field} is missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"{field} has unknown fields: {', '.join(sorted(unknown))}")
    return payload


def _version_parts(value: str) -> tuple[int, int, int]:
    """Return parts of a semantic version already validated by the caller."""
    major, minor, patch = value.split(".")
    return (int(major), int(minor), int(patch))


__all__ = [
    "ACTION_OWNER",
    "PLANT_TRUTH_OWNERS",
    "REVIEW_ONLY_AUTHORITY",
    "SEMANTIC_OWNER",
    "U0_SCHEMA_VERSION",
    "ConfinementFamily",
    "ClockKind",
    "ConversionKind",
    "DriverKind",
    "EvidenceClass",
    "OperatingCadence",
    "PhaseRelationType",
    "QualityState",
    "ReactionKind",
    "RegimeState",
    "RelationInterpretation",
    "SemanticCarrier",
    "ValidityState",
    "finite_real",
    "non_negative_integer",
    "non_negative_real",
    "probability",
    "require_exact_keys",
    "require_enum",
    "require_identifier",
    "require_semver",
    "require_sha256",
    "require_text",
    "require_u0_schema",
]

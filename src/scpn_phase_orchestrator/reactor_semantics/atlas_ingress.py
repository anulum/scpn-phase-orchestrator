# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Reactor technology atlas ingress

"""Bounded, sealed ingress for the review-only reactor technology atlas."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from importlib.resources import files
from typing import Final, cast

from jsonschema import Draft202012Validator, FormatChecker

MAX_ATLAS_JSON_BYTES: Final = 256 * 1024
MAX_ATLAS_JSON_DEPTH: Final = 32
MAX_ATLAS_CONTAINER_ITEMS: Final = 4096

_RFC3339_TIMESTAMP = re.compile(
    r"[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])"
    r"T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]"
    r"(?:\.[0-9]{1,6})?(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])"
)


class AtlasIngressError(ValueError):
    """Report a bounded atlas refusal with a stable machine-readable code.

    Parameters
    ----------
    code : str
        Refusal category; no untrusted input is interpolated into the message.
    limit : int | None, optional
        Applicable documented resource ceiling, when the refusal is a limit.
    """

    def __init__(self, code: str, *, limit: int | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.limit = limit


@dataclass(frozen=True, slots=True)
class ReactorTechnologyAtlas:
    """Validated immutable atlas bytes and their observation-time view.

    Attributes
    ----------
    canonical_json : bytes
        Canonical serialization of the accepted envelope, without rewriting
        the sealed payload or its original timestamp offset.
    observed_at_utc : str
        Observation time normalized to UTC with ``Z``; fractional seconds are
        rendered at six digits when nonzero.
    source_sha256 : str
        SHA-256 of the exact input bytes, including any original whitespace.
    """

    canonical_json: bytes
    observed_at_utc: str
    source_sha256: str

    def to_record(self) -> dict[str, object]:
        """Return an independent mutable copy of the validated envelope.

        Returns
        -------
        dict[str, object]
            Atlas record; mutations cannot alter this validated snapshot.
        """
        return cast(dict[str, object], json.loads(self.canonical_json))


def _parse_observed_at(value: object) -> datetime:
    if not isinstance(value, str) or not _RFC3339_TIMESTAMP.fullmatch(value):
        raise AtlasIngressError("invalid_observed_at")
    if value.endswith("-00:00"):
        raise AtlasIngressError("invalid_observed_at")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone(UTC)
    except (OverflowError, ValueError) as exc:
        raise AtlasIngressError("invalid_observed_at") from exc


def _valid_atlas_datetime(value: object) -> bool:
    """Apply the same RFC3339 policy to JSON Schema format validation."""
    try:
        _parse_observed_at(value)
    except AtlasIngressError:
        return False
    return True


atlas_format_checker = FormatChecker()
atlas_format_checker.checks("date-time")(_valid_atlas_datetime)


@lru_cache(maxsize=1)
def _atlas_validator() -> Draft202012Validator:
    resource = files("scpn_phase_orchestrator.reactor_semantics").joinpath(
        "data/reactor_technology_diagnostic_atlas.schema.json"
    )
    schema = cast(dict[str, object], json.loads(resource.read_text(encoding="utf-8")))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=atlas_format_checker)


def _check_structure(data: bytes) -> None:
    """Reject excessive nesting and container cardinality before JSON allocation."""
    depth_counts: list[int] = []
    in_string = False
    escaped = False
    for byte in data:
        if in_string:
            if escaped:
                escaped = False
            elif byte == 92:  # backslash
                escaped = True
            elif byte == 34:  # double quote
                in_string = False
            continue
        if byte == 34:
            in_string = True
        elif byte in (91, 123):  # array or object
            depth_counts.append(1)
            if len(depth_counts) > MAX_ATLAS_JSON_DEPTH:
                raise AtlasIngressError(
                    "depth_limit_exceeded", limit=MAX_ATLAS_JSON_DEPTH
                )
        elif byte in (93, 125):
            if depth_counts:
                depth_counts.pop()
        elif byte == 44 and depth_counts:  # comma
            depth_counts[-1] += 1
            if depth_counts[-1] > MAX_ATLAS_CONTAINER_ITEMS:
                raise AtlasIngressError(
                    "cardinality_limit_exceeded", limit=MAX_ATLAS_CONTAINER_ITEMS
                )


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    record: dict[str, object] = {}
    for key, value in pairs:
        if key in record:
            raise AtlasIngressError("duplicate_json_key")
        record[key] = value
    return record


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise AtlasIngressError("non_finite_number")
    return parsed


def _reject_constant(_value: str) -> float:
    raise AtlasIngressError("non_finite_number")


def reactor_technology_atlas_from_json(data: bytes) -> ReactorTechnologyAtlas:
    """Validate an atlas envelope through its public serialized ingress.

    The atlas observation timestamp is metadata about when the literature
    snapshot was observed. It is not a physical measurement or validity time.
    RFC3339 input requires seconds, an explicit known offset or ``Z``, at most
    six fractional digits, a real Gregorian date, and a UTC conversion within
    years 0001–9999. Leap seconds and ``-00:00`` unknown offsets are refused.

    Parameters
    ----------
    data : bytes
        UTF-8 JSON atlas envelope; at most 256 KiB, depth 32 and 4096 items
        per array/object. Duplicate keys and non-finite numbers are refused.

    Returns
    -------
    ReactorTechnologyAtlas
        Immutable canonical envelope and separate normalized observation time.

    Raises
    ------
    AtlasIngressError
        If limits, JSON, schema, time policy or payload seal are invalid. The
        ``code`` and optional ``limit`` fields give a bounded refusal reason.
    """
    if not isinstance(data, bytes) or not data:
        raise AtlasIngressError("invalid_input")
    if len(data) > MAX_ATLAS_JSON_BYTES:
        raise AtlasIngressError("byte_limit_exceeded", limit=MAX_ATLAS_JSON_BYTES)
    _check_structure(data)
    try:
        raw = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except UnicodeDecodeError as exc:
        raise AtlasIngressError("invalid_utf8") from exc
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        if isinstance(exc, AtlasIngressError):
            raise
        raise AtlasIngressError("invalid_json") from exc
    if not isinstance(raw, dict):
        raise AtlasIngressError("invalid_schema")
    record = cast(dict[str, object], raw)
    payload = record.get("payload")
    if not isinstance(payload, dict):
        raise AtlasIngressError("invalid_schema")
    observed_utc = _parse_observed_at(payload.get("observed_at"))
    if next(_atlas_validator().iter_errors(record), None) is not None:
        raise AtlasIngressError("invalid_schema")
    canonical_payload = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if record["payload_sha256"] != hashlib.sha256(canonical_payload).hexdigest():
        raise AtlasIngressError("payload_digest_mismatch")
    canonical_record = json.dumps(
        record,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return ReactorTechnologyAtlas(
        canonical_json=canonical_record,
        observed_at_utc=observed_utc.isoformat().replace("+00:00", "Z"),
        source_sha256=hashlib.sha256(data).hexdigest(),
    )


__all__ = [
    "AtlasIngressError",
    "MAX_ATLAS_CONTAINER_ITEMS",
    "MAX_ATLAS_JSON_BYTES",
    "MAX_ATLAS_JSON_DEPTH",
    "ReactorTechnologyAtlas",
    "atlas_format_checker",
    "reactor_technology_atlas_from_json",
]

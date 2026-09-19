# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Reactor atlas ingress tests

"""Exercise bounded and temporal atlas admission through the public facade."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest
from jsonschema import Draft202012Validator

from scpn_phase_orchestrator.reactor_semantics import (
    MAX_ATLAS_CONTAINER_ITEMS,
    MAX_ATLAS_JSON_BYTES,
    MAX_ATLAS_JSON_DEPTH,
    AtlasIngressError,
    atlas_format_checker,
    atlas_ingress,
    reactor_technology_atlas_from_json,
)

ATLAS_PATH = Path("docs/reference/data/reactor_technology_diagnostic_atlas.v1.json")
SCHEMA_PATH = Path("docs/specs/reactor_technology_diagnostic_atlas.schema.json")
PACKAGED_SCHEMA_PATH = Path(
    "src/scpn_phase_orchestrator/reactor_semantics/data/"
    "reactor_technology_diagnostic_atlas.schema.json"
)


def _sealed_atlas(observed_at: object) -> bytes:
    """Change only observation time and recompute the genuine payload seal."""
    record = cast(dict[str, object], json.loads(ATLAS_PATH.read_bytes()))
    payload = cast(dict[str, object], record["payload"])
    payload["observed_at"] = observed_at
    record["payload_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8")


def _refusal(data: bytes, code: str, *, limit: int | None = None) -> None:
    """Assert a bounded, classified refusal through the public decoder."""
    with pytest.raises(AtlasIngressError) as caught:
        reactor_technology_atlas_from_json(data)
    assert caught.value.code == code
    assert str(caught.value) == code
    assert caught.value.limit == limit


def test_real_atlas_is_sealed_and_does_not_expose_mutable_state() -> None:
    """Preserve exact source provenance and independent validated records."""
    data = ATLAS_PATH.read_bytes()
    atlas = atlas_ingress.reactor_technology_atlas_from_json(data)
    assert atlas.observed_at_utc == "2026-09-04T09:50:00Z"
    assert atlas.source_sha256 == hashlib.sha256(data).hexdigest()
    record = atlas.to_record()
    payload = cast(dict[str, object], record["payload"])
    payload["observed_at"] = "1900-01-01T00:00:00Z"
    assert cast(dict[str, object], atlas.to_record()["payload"])["observed_at"] == (
        "2026-09-04T09:50:00Z"
    )


@pytest.mark.parametrize(
    ("timestamp", "expected"),
    [
        ("2026-09-04T11:50:00+02:00", "2026-09-04T09:50:00Z"),
        ("2026-09-04T04:50:00-05:00", "2026-09-04T09:50:00Z"),
        ("2026-09-04T09:50:00.1Z", "2026-09-04T09:50:00.100000Z"),
        ("2026-09-04T09:50:00.123456Z", "2026-09-04T09:50:00.123456Z"),
    ],
)
def test_offsets_preserve_the_instant_without_rewriting_sealed_bytes(
    timestamp: str, expected: str
) -> None:
    """Normalize only the returned observation-time view, not the payload."""
    data = _sealed_atlas(timestamp)
    atlas = reactor_technology_atlas_from_json(data)
    assert atlas.observed_at_utc == expected
    assert cast(dict[str, object], atlas.to_record()["payload"])["observed_at"] == (
        timestamp
    )


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-02-29T09:50:00Z",
        "2024-04-31T09:50:00Z",
        "2026-13-04T09:50:00Z",
        "2026-09-04T24:00:00Z",
        "2026-09-04T09:50:60Z",
        "2026-09-04T09:50:00",
        "2026-09-04T09:50:00-00:00",
        "2026-09-04T09:50:00.1234567Z",
        "2026-09-04t09:50:00z",
        "0000-09-04T09:50:00Z",
        "0001-01-01T00:00:00+01:00",
        None,
        True,
    ],
)
def test_invalid_observation_time_is_refused_by_public_and_schema_paths(
    timestamp: object,
) -> None:
    """Reject malformed dates, unknown zones, aliases and precision drift."""
    data = _sealed_atlas(timestamp)
    _refusal(data, "invalid_observed_at")
    schema = json.loads(SCHEMA_PATH.read_bytes())
    record = json.loads(data)
    validator = Draft202012Validator(schema, format_checker=atlas_format_checker)
    assert any(
        list(error.path) == ["payload", "observed_at"]
        for error in validator.iter_errors(record)
    )


def test_leap_day_and_precision_boundary_are_accepted() -> None:
    """Admit a real leap day at the documented microsecond precision."""
    atlas = reactor_technology_atlas_from_json(
        _sealed_atlas("2024-02-29T23:59:59.999999Z")
    )
    assert atlas.observed_at_utc == "2024-02-29T23:59:59.999999Z"


def test_duplicate_keys_nonfinite_numbers_and_malformed_utf8_are_refused() -> None:
    """Reject ambiguous or non-JSON bytes before schema interpretation."""
    data = ATLAS_PATH.read_bytes()
    _refusal(
        data.replace(b'"observed_at":', b'"observed_at": null, "observed_at":', 1),
        "duplicate_json_key",
    )
    _refusal(
        data.replace(b'"observed_at":', b'"extra": NaN, "observed_at":', 1),
        "non_finite_number",
    )
    _refusal(
        data.replace(b'"observed_at":', b'"extra": 1e999, "observed_at":', 1),
        "non_finite_number",
    )
    _refusal(b"\xff", "invalid_utf8")
    _refusal(b"{", "invalid_json")
    _refusal(b"[]", "invalid_schema")


def test_structural_scanner_and_numeric_decoder_refuse_unsealed_documents() -> None:
    """Exercise escaped JSON strings, stray closers and finite decimals at ingress."""
    _refusal(b'{"escaped":"quote: \\" and slash: \\\\"}', "invalid_schema")
    _refusal(b"]", "invalid_json")
    _refusal(b'{"decimal":1.25}', "invalid_schema")


def test_empty_wrong_type_and_missing_payload_are_classified() -> None:
    """Keep malformed root input separate from schema and missing-payload refusal."""
    _refusal(b"", "invalid_input")
    _refusal(cast(bytes, "not bytes"), "invalid_input")
    _refusal(b"{}", "invalid_schema")


def test_payload_hash_and_full_schema_remain_authoritative() -> None:
    """Reject a changed seal or an otherwise sealed unknown authority field."""
    data = ATLAS_PATH.read_bytes()
    _refusal(
        data.replace(b'"payload_sha256": "', b'"payload_sha256": "0', 1),
        "invalid_schema",
    )
    record = cast(dict[str, object], json.loads(data))
    record["payload_sha256"] = "0" * 64
    _refusal(json.dumps(record).encode(), "payload_digest_mismatch")
    record = cast(dict[str, object], json.loads(data))
    payload = cast(dict[str, object], record["payload"])
    payload["unknown_authority"] = True
    record["payload_sha256"] = hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode()
    ).hexdigest()
    _refusal(json.dumps(record).encode(), "invalid_schema")


def test_byte_limit_accepts_exact_boundary_and_refuses_one_more() -> None:
    """Count raw UTF-8 bytes, including harmless trailing JSON whitespace."""
    data = ATLAS_PATH.read_bytes()
    at_limit = data + b" " * (MAX_ATLAS_JSON_BYTES - len(data))
    assert len(at_limit) == MAX_ATLAS_JSON_BYTES
    assert reactor_technology_atlas_from_json(at_limit).to_record() == (
        reactor_technology_atlas_from_json(data).to_record()
    )
    _refusal(at_limit + b" ", "byte_limit_exceeded", limit=MAX_ATLAS_JSON_BYTES)


def test_depth_limit_refuses_boundary_plus_one_before_schema() -> None:
    """Allow the resource boundary but reject one extra nested container."""
    at_limit = b"[" * MAX_ATLAS_JSON_DEPTH + b"0" + b"]" * MAX_ATLAS_JSON_DEPTH
    _refusal(at_limit, "invalid_schema")
    _refusal(
        b"[" + at_limit + b"]",
        "depth_limit_exceeded",
        limit=MAX_ATLAS_JSON_DEPTH,
    )


def test_cardinality_limit_refuses_boundary_plus_one_before_schema() -> None:
    """Bound flat collection size before allocating a JSON object graph."""
    at_limit = b"[" + b",".join([b"0"] * MAX_ATLAS_CONTAINER_ITEMS) + b"]"
    _refusal(at_limit, "invalid_schema")
    _refusal(
        at_limit[:-1] + b",0]",
        "cardinality_limit_exceeded",
        limit=MAX_ATLAS_CONTAINER_ITEMS,
    )


def test_packaged_schema_is_the_exact_public_schema() -> None:
    """Prevent installation from silently weakening the documented rules."""
    assert PACKAGED_SCHEMA_PATH.read_bytes() == SCHEMA_PATH.read_bytes()
    Draft202012Validator.check_schema(json.loads(PACKAGED_SCHEMA_PATH.read_bytes()))

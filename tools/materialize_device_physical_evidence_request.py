# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — physical-evidence request materialiser
"""Materialise a device request from exact local producer fixture bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import NoReturn, cast

from scpn_phase_orchestrator.reactor_semantics import (
    device_diagnostic_plan_review_from_producer_bytes,
    device_physical_evidence_request_digest,
    device_physical_evidence_request_from_plan_review,
    device_physical_evidence_request_to_bytes,
)


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant: {value}")


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(
        path.read_bytes().decode("utf-8"),
        object_pairs_hook=_reject_duplicates,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return cast(dict[str, object], value)


def _compact(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _manifest_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def materialize(
    *,
    fixture_dir: Path,
    configuration: str,
    source_revision: str,
    source_artifact_sha256: str,
) -> bytes:
    """Return one canonical request without importing producer code."""
    manifest_path = fixture_dir / "reactor-domain.json"
    fixture_path = fixture_dir / "plan_envelope_fixture.json"
    manifest = _load_object(manifest_path)
    document = _load_object(fixture_path)
    if set(document) != {"envelope", "plan"}:
        raise ValueError("plan fixture must contain exactly envelope and plan")
    envelope = document["envelope"]
    plan = document["plan"]
    if not isinstance(envelope, dict) or not isinstance(plan, dict):
        raise ValueError("plan fixture envelope and plan must be JSON objects")
    plan_identifier = plan.get("identifier")
    if not isinstance(plan_identifier, str) or not plan_identifier:
        raise ValueError("plan identifier must be non-empty text")

    manifest_bytes = _manifest_bytes(manifest)
    plan_bytes = _compact(plan)
    canonical_envelope = dict(envelope)
    canonical_envelope["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    canonical_envelope["plan_identifier"] = plan_identifier
    canonical_envelope["plan_sha256"] = hashlib.sha256(plan_bytes).hexdigest()

    review = device_diagnostic_plan_review_from_producer_bytes(
        source_revision=source_revision,
        source_artifact_sha256=source_artifact_sha256,
        manifest_bytes=manifest_bytes,
        envelope_bytes=_compact(canonical_envelope),
        plan_bytes=plan_bytes,
    )
    request = device_physical_evidence_request_from_plan_review(
        review,
        configuration=configuration,
    )
    encoded = device_physical_evidence_request_to_bytes(request)
    if hashlib.sha256(encoded).hexdigest() != device_physical_evidence_request_digest(
        request
    ):
        raise RuntimeError("request digest differs from canonical bytes")
    return encoded


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary.write(data)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    temporary_path.chmod(0o644)
    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialise a sealed SPO physical-evidence request from exact local "
            "producer fixtures without importing or executing producer code."
        )
    )
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-artifact-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail unless the output already equals the derived canonical bytes",
    )
    return parser


def main() -> int:
    """Run the host-independent materialisation command."""
    args = _parser().parse_args()
    encoded = materialize(
        fixture_dir=args.fixture_dir,
        configuration=args.configuration,
        source_revision=args.source_revision,
        source_artifact_sha256=args.source_artifact_sha256,
    )
    if args.check:
        if not args.output.is_file() or args.output.read_bytes() != encoded:
            raise SystemExit(f"stale or missing materialised request: {args.output}")
    else:
        _write_atomic(args.output, encoded)
    document = _load_object(args.output)
    payload = document["payload"]
    if not isinstance(payload, dict):
        raise RuntimeError("materialised request payload is not an object")
    print(
        json.dumps(
            {
                "configuration": payload["configuration"],
                "envelope_sha256": hashlib.sha256(encoded).hexdigest(),
                "output": args.output.as_posix(),
                "request_id": payload["request_id"],
                "source_review_id": payload["source_review_id"],
                "source_review_sha256": payload["source_review_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

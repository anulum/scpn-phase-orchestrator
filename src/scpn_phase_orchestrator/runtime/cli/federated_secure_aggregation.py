# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Federated secure aggregation preflight CLI

"""CLI command for review-only federated secure-aggregation preflight evidence.

The command consumes newline-delimited secure-aggregation node commitment records
plus a deployment declaration, builds the deterministic secure-aggregation
manifest, then a review-only deployment preflight manifest, and emits a
non-actuating preflight bundle. It validates the same production supervisor
surface as ``supervisor.federated_secure_aggregation`` and never opens sockets,
runs aggregation, or permits live secure-aggregation execution.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import click

from scpn_phase_orchestrator.runtime.cli._app import main
from scpn_phase_orchestrator.runtime.cli._payloads import _load_json_file, _record_hash
from scpn_phase_orchestrator.supervisor.federated_secure_aggregation import (
    FederatedSecureAggregationManifest,
    FederatedSecureAggregationPreflightManifest,
    build_federated_secure_aggregation_manifest,
    build_federated_secure_aggregation_preflight_manifest,
)


@main.command("federated-secure-aggregation-preflight")
@click.argument(
    "node_commitments_jsonl",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "deployment_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path for the deterministic preflight bundle JSON.",
)
def federated_secure_aggregation_preflight(
    node_commitments_jsonl: Path,
    deployment_json: Path,
    output: Path | None,
) -> None:
    """Emit deterministic review evidence for a secure-aggregation deployment.

    Parameters
    ----------
    node_commitments_jsonl : Path
        JSONL file containing secure-aggregation node commitment records.
    deployment_json : Path
        JSON file describing the aggregation policy and deployment preflight
        inputs (quorum evidence, custody records, operator approval).
    output : Path | None
        Optional path to write the emitted bundle JSON.

    Raises
    ------
    ClickException
        If inputs are malformed or the secure-aggregation preflight fails closed.
    """
    commitments = _load_node_commitment_jsonl(node_commitments_jsonl)
    deployment = _load_json_file(
        deployment_json,
        artifact="federated secure aggregation deployment",
    )
    try:
        manifest = _build_manifest(commitments, deployment)
        preflight = _build_preflight(manifest, deployment)
    except (TypeError, ValueError) as exc:
        raise click.ClickException(
            f"federated secure aggregation preflight failed: {exc}"
        ) from exc

    bundle = _preflight_bundle_payload(
        manifest=manifest.to_audit_record(),
        preflight=preflight.to_audit_record(),
    )
    rendered = json.dumps(bundle, indent=2, sort_keys=True)
    if output is not None:
        try:
            output.write_text(rendered + "\n", encoding="utf-8")
        except OSError as exc:
            raise click.ClickException(
                "cannot write federated secure aggregation preflight bundle "
                f"{output!s}: {exc}"
            ) from exc
    click.echo(rendered)


def _build_manifest(
    commitments: Sequence[Mapping[str, object]],
    deployment: Mapping[str, object],
) -> FederatedSecureAggregationManifest:
    """Build the secure-aggregation manifest from commitments and policy config.

    Parameters
    ----------
    commitments : Sequence[Mapping[str, object]]
        Validated node commitment records.
    deployment : Mapping[str, object]
        Deployment declaration; its optional ``aggregation`` object carries the
        secure-aggregation policy parameters.

    Returns
    -------
    FederatedSecureAggregationManifest
        The deterministic secure-aggregation manifest.

    Raises
    ------
    ValueError
        If the aggregation policy block or commitments are invalid.
    """
    aggregation = _aggregation_config(deployment)
    required_policy_keys = aggregation.get("required_policy_keys")
    keys = (
        _string_sequence(required_policy_keys, "aggregation.required_policy_keys")
        if required_policy_keys is not None
        else None
    )
    return build_federated_secure_aggregation_manifest(
        commitments,
        required_policy_keys=keys,
        clipping_norm=_float_field(aggregation, "clipping_norm", default=1.0),
        min_node_count=_int_field(aggregation, "min_node_count", default=3),
        epsilon=_float_field(aggregation, "epsilon", default=3.0),
        delta=_float_field(aggregation, "delta", default=1e-6),
    )


def _build_preflight(
    manifest: FederatedSecureAggregationManifest,
    deployment: Mapping[str, object],
) -> FederatedSecureAggregationPreflightManifest:
    """Build the review-only deployment preflight manifest.

    Parameters
    ----------
    manifest : FederatedSecureAggregationManifest
        The secure-aggregation manifest to preflight.
    deployment : Mapping[str, object]
        Deployment declaration carrying quorum evidence, custody records, and
        operator approval fields.

    Returns
    -------
    FederatedSecureAggregationPreflightManifest
        The deterministic deployment preflight manifest.

    Raises
    ------
    TypeError
        If a declared field has the wrong type.
    ValueError
        If the manifest, quorum evidence, or custody records are invalid.
    """
    return build_federated_secure_aggregation_preflight_manifest(
        manifest,
        quorum_evidence=_mapping_sequence(deployment, "quorum_evidence"),
        custody_rotation_policy=_text_field(deployment, "custody_rotation_policy"),
        custody_records=_mapping_sequence(deployment, "custody_records"),
        accepted_node_threshold=_int_field(
            deployment, "accepted_node_threshold", default=None
        ),
        operator_approved=_bool_field(deployment, "operator_approved"),
        operator_id=_text_field(deployment, "operator_id"),
        service_owner=_text_field(deployment, "service_owner"),
    )


def _load_node_commitment_jsonl(path: Path) -> tuple[Mapping[str, object], ...]:
    """Load secure-aggregation node commitment records from a JSONL file.

    Parameters
    ----------
    path : Path
        Path to newline-delimited JSON node commitment records.

    Returns
    -------
    tuple[Mapping[str, object], ...]
        Parsed node commitment records.

    Raises
    ------
    ClickException
        If the file is unreadable, empty, or contains malformed rows.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise click.ClickException(
            f"cannot read node-commitment JSONL {path!s}: {exc}"
        ) from exc

    records: list[Mapping[str, object]] = []
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload: object = json.loads(line)
        except json.JSONDecodeError as exc:
            raise click.ClickException(
                f"malformed node-commitment JSONL at line {line_number}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise click.ClickException(
                f"node-commitment JSONL line {line_number} must be a JSON object"
            )
        records.append(payload)

    if not records:
        raise click.ClickException(
            "node-commitment JSONL must contain at least one record"
        )
    return tuple(records)


def _preflight_bundle_payload(
    *,
    manifest: dict[str, object],
    preflight: dict[str, object],
) -> dict[str, object]:
    """Return the deterministic CLI bundle payload.

    Parameters
    ----------
    manifest : dict[str, object]
        Secure-aggregation manifest audit record.
    preflight : dict[str, object]
        Deployment preflight manifest audit record.

    Returns
    -------
    dict[str, object]
        JSON-safe CLI bundle with a deterministic ``bundle_hash``.
    """
    bundle: dict[str, object] = {
        "schema": "scpn_federated_secure_aggregation_preflight_bundle_v1",
        "version": "1.0.0",
        "accepted_node_count": preflight["accepted_node_count"],
        "accepted_node_threshold": preflight["accepted_node_threshold"],
        "custody_rotation_policy": preflight["custody_rotation_policy"],
        "operator_id": preflight["operator_id"],
        "service_owner": preflight["service_owner"],
        "secure_aggregation_execution_permitted": False,
        "raw_data_export_permitted": False,
        "operator_review_required": True,
        "non_actuating": True,
        "secure_aggregation_manifest": manifest,
        "preflight_manifest": preflight,
    }
    bundle["bundle_hash"] = _record_hash(bundle)
    return bundle


def _aggregation_config(deployment: Mapping[str, object]) -> Mapping[str, object]:
    """Return the optional ``aggregation`` policy block as a mapping."""
    block = deployment.get("aggregation")
    if block is None:
        return {}
    if not isinstance(block, Mapping):
        raise ValueError("aggregation must be a JSON object")
    return block


def _mapping_sequence(
    deployment: Mapping[str, object],
    field: str,
) -> tuple[Mapping[str, object], ...]:
    """Return a required field as a sequence of JSON-object mappings."""
    value = deployment.get(field)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field} must be a JSON array")
    records: list[Mapping[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"{field}[{index}] must be a JSON object")
        records.append(item)
    return tuple(records)


def _string_sequence(value: object, field: str) -> tuple[str, ...]:
    """Return ``value`` as a tuple of strings, else raise."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field} must be a JSON array of strings")
    items: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"{field}[{index}] must be a string")
        items.append(item)
    return tuple(items)


def _text_field(deployment: Mapping[str, object], field: str) -> str:
    """Return a required non-empty string field, else raise."""
    value = deployment.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _bool_field(deployment: Mapping[str, object], field: str) -> bool:
    """Return a required boolean field, else raise."""
    value = deployment.get(field)
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _int_field(
    config: Mapping[str, object],
    field: str,
    *,
    default: int | None,
) -> int:
    """Return an integer field, falling back to ``default`` when absent.

    Parameters
    ----------
    config : Mapping[str, object]
        Source mapping.
    field : str
        Field name to read.
    default : int | None
        Value used when the field is absent. ``None`` makes the field required.

    Returns
    -------
    int
        The validated integer value.

    Raises
    ------
    ValueError
        If the field is missing without a default or is not an integer.
    """
    if field not in config:
        if default is None:
            raise ValueError(f"{field} is required")
        return default
    value = config[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _float_field(
    config: Mapping[str, object],
    field: str,
    *,
    default: float,
) -> float:
    """Return a float field, falling back to ``default`` when absent.

    Parameters
    ----------
    config : Mapping[str, object]
        Source mapping.
    field : str
        Field name to read.
    default : float
        Value used when the field is absent.

    Returns
    -------
    float
        The validated float value.

    Raises
    ------
    ValueError
        If the field is present but not a real number.
    """
    if field not in config:
        return default
    value = config[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    return float(value)

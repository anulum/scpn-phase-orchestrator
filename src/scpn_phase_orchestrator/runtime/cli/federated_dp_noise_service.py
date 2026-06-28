# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Federated DP noise-service preflight CLI

"""CLI command for review-only federated DP noise-service preflight evidence.

The command consumes a DP-noise request declaration plus a deployment
declaration, builds the deterministic DP-noise request and response manifests,
then a review-only deployment preflight manifest, and emits a non-actuating
preflight bundle. It validates the same production supervisor surface as
``supervisor.federated_dp_noise_service`` and never opens sockets, generates live
noise, or permits live DP noise-service execution. Missing deployment
prerequisites are reported as a not-ready readiness verdict rather than an error;
only malformed inputs fail closed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import click

from scpn_phase_orchestrator.runtime.cli._app import main
from scpn_phase_orchestrator.runtime.cli._payloads import _load_json_file, _record_hash
from scpn_phase_orchestrator.supervisor.federated_dp_noise_service import (
    DpNoiseNodePrivacyBudget,
    DpNoiseServiceRequestManifest,
    build_dp_noise_service_deployment_preflight_manifest,
    build_dp_noise_service_manifest,
)


@main.command("federated-dp-noise-service-preflight")
@click.argument(
    "request_json",
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
def federated_dp_noise_service_preflight(
    request_json: Path,
    deployment_json: Path,
    output: Path | None,
) -> None:
    """Emit deterministic review evidence for a DP noise-service deployment.

    Parameters
    ----------
    request_json : Path
        JSON file describing the DP-noise request (privacy parameters, seed
        commitment, policy keys, and per-node privacy budgets).
    deployment_json : Path
        JSON file describing the deployment preflight inputs (mechanism, custody,
        accountant, budget issuer, service endpoint, and operator approval).
    output : Path | None
        Optional path to write the emitted bundle JSON.

    Raises
    ------
    ClickException
        If inputs are malformed or the DP noise-service preflight fails closed.
        A merely not-ready deployment is reported in the bundle, not raised.
    """
    request_payload = _load_json_file(
        request_json,
        artifact="federated DP noise-service request",
    )
    deployment = _load_json_file(
        deployment_json,
        artifact="federated DP noise-service deployment",
    )
    try:
        request = _build_request(request_payload)
        response = build_dp_noise_service_manifest(request)
        preflight = build_dp_noise_service_deployment_preflight_manifest(
            request,
            response,
            mechanism_label=_string_field(deployment, "mechanism_label"),
            privacy_accountant_owner=_string_field(
                deployment, "privacy_accountant_owner"
            ),
            seed_custody_label=_string_field(deployment, "seed_custody_label"),
            budget_issuer_label=_string_field(deployment, "budget_issuer_label"),
            service_endpoint_label=_string_field(deployment, "service_endpoint_label"),
            operator_approved=_bool_field(deployment, "operator_approved"),
        )
    except (TypeError, ValueError) as exc:
        raise click.ClickException(
            f"federated DP noise-service preflight failed: {exc}"
        ) from exc

    bundle = _preflight_bundle_payload(
        deployment_ready=preflight.deployment_readiness.ready,
        deployment_reason=preflight.deployment_readiness.reason,
        request_hash=preflight.request_hash,
        response_hash=preflight.response_hash,
        request=request.to_audit_record(),
        response=response.to_audit_record(),
        preflight=preflight.to_audit_record(),
    )
    rendered = json.dumps(bundle, indent=2, sort_keys=True)
    if output is not None:
        try:
            output.write_text(rendered + "\n", encoding="utf-8")
        except OSError as exc:
            raise click.ClickException(
                "cannot write federated DP noise-service preflight bundle "
                f"{output!s}: {exc}"
            ) from exc
    click.echo(rendered)


def _build_request(payload: Mapping[str, object]) -> DpNoiseServiceRequestManifest:
    """Build a validated DP-noise request manifest from a JSON declaration.

    Parameters
    ----------
    payload : Mapping[str, object]
        JSON object carrying the DP-noise request fields.

    Returns
    -------
    DpNoiseServiceRequestManifest
        The validated request manifest.

    Raises
    ------
    ValueError
        If a required field is absent or has the wrong type, or the request
        violates the secure DP-noise policy.
    """
    return DpNoiseServiceRequestManifest(
        epsilon=_float_field(payload, "epsilon"),
        delta=_float_field(payload, "delta"),
        sensitivity=_float_field(payload, "sensitivity"),
        noise_multiplier=_float_field(payload, "noise_multiplier"),
        node_count=_int_field(payload, "node_count"),
        seed_hash=_text_field(payload, "seed_hash"),
        policy_keys=_string_tuple(payload, "policy_keys"),
        node_budgets=_build_node_budgets(payload),
    )


def _build_node_budgets(
    payload: Mapping[str, object],
) -> tuple[DpNoiseNodePrivacyBudget, ...]:
    """Build per-node privacy budgets from the request declaration.

    Parameters
    ----------
    payload : Mapping[str, object]
        JSON object whose ``node_budgets`` array carries per-node spend records.

    Returns
    -------
    tuple[DpNoiseNodePrivacyBudget, ...]
        The per-node privacy-budget records.

    Raises
    ------
    ValueError
        If ``node_budgets`` or any record is malformed.
    """
    value = payload.get("node_budgets")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("node_budgets must be a JSON array")
    budgets: list[DpNoiseNodePrivacyBudget] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"node_budgets[{index}] must be a JSON object")
        budgets.append(
            DpNoiseNodePrivacyBudget(
                node_id=_text_field(item, "node_id"),
                epsilon_spent=_float_field(item, "epsilon_spent"),
            )
        )
    return tuple(budgets)


def _preflight_bundle_payload(
    *,
    deployment_ready: bool,
    deployment_reason: str,
    request_hash: str,
    response_hash: str,
    request: dict[str, object],
    response: dict[str, object],
    preflight: dict[str, object],
) -> dict[str, object]:
    """Return the deterministic CLI bundle payload.

    Parameters
    ----------
    deployment_ready : bool
        Whether the deployment preflight reported a ready verdict.
    deployment_reason : str
        Readiness reason from the deployment preflight.
    request_hash, response_hash : str
        Hash linkage of the request and response manifests.
    request : dict[str, object]
        DP-noise request manifest audit record.
    response : dict[str, object]
        DP-noise response manifest audit record.
    preflight : dict[str, object]
        Deployment preflight manifest audit record.

    Returns
    -------
    dict[str, object]
        JSON-safe CLI bundle with a deterministic ``bundle_hash``.
    """
    bundle: dict[str, object] = {
        "schema": "scpn_federated_dp_noise_service_preflight_bundle_v1",
        "version": "1.0.0",
        "deployment_ready": deployment_ready,
        "deployment_reason": deployment_reason,
        "request_hash": request_hash,
        "response_hash": response_hash,
        "service_execution_permitted": False,
        "raw_data_export_permitted": False,
        "operator_review_required": True,
        "non_actuating": True,
        "request_manifest": request,
        "response_manifest": response,
        "preflight_manifest": preflight,
    }
    bundle["bundle_hash"] = _record_hash(bundle)
    return bundle


def _string_tuple(payload: Mapping[str, object], field: str) -> tuple[str, ...]:
    """Return a required field as a tuple of strings, else raise."""
    value = payload.get(field)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field} must be a JSON array of strings")
    items: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"{field}[{index}] must be a string")
        items.append(item)
    return tuple(items)


def _string_field(deployment: Mapping[str, object], field: str) -> str:
    """Return a required string field, allowing an empty value.

    An empty value is preserved so the supervisor surface can report it as a
    not-ready deployment prerequisite rather than failing closed.

    Raises
    ------
    ValueError
        If the field is absent or not a string.
    """
    if field not in deployment:
        raise ValueError(f"{field} is required")
    value = deployment[field]
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _text_field(payload: Mapping[str, object], field: str) -> str:
    """Return a required non-empty string field, else raise."""
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _bool_field(payload: Mapping[str, object], field: str) -> bool:
    """Return a required boolean field, else raise."""
    value = payload.get(field)
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _int_field(payload: Mapping[str, object], field: str) -> int:
    """Return a required integer field, else raise."""
    if field not in payload:
        raise ValueError(f"{field} is required")
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _float_field(payload: Mapping[str, object], field: str) -> float:
    """Return a required real-number field as a float, else raise."""
    if field not in payload:
        raise ValueError(f"{field} is required")
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    return float(value)

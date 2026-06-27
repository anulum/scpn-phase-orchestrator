# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Federated transport preflight CLI

"""CLI command for review-only federated transport preflight evidence.

The command consumes newline-delimited federated node update audit records plus a
transport declaration, then builds signed/hash-linked envelopes, replays the
batch, and emits a deterministic non-actuating preflight bundle. It validates
the same production supervisor transport surface as
``supervisor.federated_transport`` and never opens sockets or permits live
transport execution.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import click

from scpn_phase_orchestrator.runtime.cli._app import main
from scpn_phase_orchestrator.runtime.cli._payloads import _load_json_file, _record_hash
from scpn_phase_orchestrator.supervisor.federated_transport import (
    build_signed_transport_envelopes,
    build_transport_deployment_preflight_manifest,
    replay_federated_transport_batch,
)


@main.command("federated-transport-preflight")
@click.argument(
    "node_updates_jsonl",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "transport_declaration_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path for the deterministic preflight bundle JSON.",
)
def federated_transport_preflight(
    node_updates_jsonl: Path,
    transport_declaration_json: Path,
    output: Path | None,
) -> None:
    """Emit deterministic review evidence for a federated transport batch.

    Parameters
    ----------
    node_updates_jsonl : Path
        JSONL file containing node update audit records.
    transport_declaration_json : Path
        JSON file describing the intended transport boundary.
    output : Path | None
        Optional path to write the emitted bundle JSON.

    Raises
    ------
    ClickException
        If inputs are malformed or the transport preflight fails closed.
    """
    updates = _load_node_update_jsonl(node_updates_jsonl)
    declaration = _load_json_file(
        transport_declaration_json,
        artifact="federated transport declaration",
    )
    try:
        envelopes = build_signed_transport_envelopes(updates)
        replay_ledger = replay_federated_transport_batch(envelopes)
        preflight_manifest = build_transport_deployment_preflight_manifest(
            declaration,
            replay_ledger=replay_ledger,
        )
    except ValueError as exc:
        raise click.ClickException(
            f"federated transport preflight failed: {exc}"
        ) from exc

    bundle = _preflight_bundle_payload(
        envelopes=[envelope.to_audit_record() for envelope in envelopes],
        replay_ledger=replay_ledger.to_audit_record(),
        preflight_manifest=preflight_manifest.to_audit_record(),
    )
    rendered = json.dumps(bundle, indent=2, sort_keys=True)
    if output is not None:
        try:
            output.write_text(rendered + "\n", encoding="utf-8")
        except OSError as exc:
            raise click.ClickException(
                f"cannot write federated transport preflight bundle {output!s}: {exc}"
            ) from exc
    click.echo(rendered)


def _load_node_update_jsonl(path: Path) -> tuple[Mapping[str, object], ...]:
    """Load node update audit records from a JSONL file.

    Parameters
    ----------
    path : Path
        Path to newline-delimited JSON node update records.

    Returns
    -------
    tuple[Mapping[str, object], ...]
        Parsed node update records.

    Raises
    ------
    ClickException
        If the file is unreadable, empty, or contains malformed rows.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise click.ClickException(
            f"cannot read node-update JSONL {path!s}: {exc}"
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
                f"malformed node-update JSONL at line {line_number}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise click.ClickException(
                f"node-update JSONL line {line_number} must be a JSON object"
            )
        records.append(payload)

    if not records:
        raise click.ClickException("node-update JSONL must contain at least one record")
    return tuple(records)


def _preflight_bundle_payload(
    *,
    envelopes: list[dict[str, object]],
    replay_ledger: dict[str, object],
    preflight_manifest: dict[str, object],
) -> dict[str, object]:
    """Return the deterministic CLI bundle payload.

    Parameters
    ----------
    envelopes : list[dict[str, object]]
        Signed transport envelope audit records.
    replay_ledger : dict[str, object]
        Replay ledger audit record.
    preflight_manifest : dict[str, object]
        Deployment preflight manifest audit record.

    Returns
    -------
    dict[str, object]
        JSON-safe CLI bundle with a deterministic ``bundle_hash``.
    """
    bundle: dict[str, object] = {
        "schema": "scpn_federated_transport_preflight_bundle_v1",
        "version": "1.0.0",
        "envelope_count": len(envelopes),
        "batch_id": str(preflight_manifest["batch_id"]),
        "transport": str(preflight_manifest["transport"]),
        "transport_endpoint": str(preflight_manifest["transport_endpoint"]),
        "transport_execution_permitted": False,
        "raw_data_export_permitted": False,
        "operator_review_required": True,
        "non_actuating": True,
        "envelopes": envelopes,
        "replay_ledger": replay_ledger,
        "preflight_manifest": preflight_manifest,
    }
    bundle["bundle_hash"] = _record_hash(bundle)
    return bundle

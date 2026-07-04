# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI power-grid PRC audit bundle command

"""Command-line assembly for review-only power-grid PRC assessor bundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import NoReturn

import click

from scpn_phase_orchestrator.assurance.power_grid_prc_bundle import (
    DVOC_DAMPING_ROLE,
    IBR_RIDE_THROUGH_ROLE,
    PMU_RINGDOWN_ROLE,
    PowerGridPRCInputArtifact,
    build_power_grid_prc_audit_bundle,
)
from scpn_phase_orchestrator.runtime.cli._app import main


@main.command("power-grid-prc-bundle")
@click.option("--bundle-id", required=True, help="Operator bundle identifier")
@click.option("--created-at", required=True, help="Bundle creation timestamp")
@click.option("--operator-context", required=True, help="Assessor review context")
@click.option(
    "--dvoc-evidence",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="dVOC oscillation-damping evidence JSON",
)
@click.option(
    "--pmu-ringdown",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="PMU ringdown PRC evidence JSON",
)
@click.option(
    "--ibr-ride-through",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="IBR PRC-029 ride-through evidence JSON",
)
@click.option(
    "--output",
    default=None,
    type=click.Path(),
    help="Write the sealed assessor bundle JSON here",
)
def power_grid_prc_bundle(
    bundle_id: str,
    created_at: str,
    operator_context: str,
    dvoc_evidence: str,
    pmu_ringdown: str,
    ibr_ride_through: str,
    output: str | None,
) -> None:
    """Bundle power-grid PRC evidence files for qualified review.

    Parameters
    ----------
    bundle_id : str
        Operator-assigned bundle identifier.
    created_at : str
        Timestamp stamped into the bundle.
    operator_context : str
        Human-readable assessor review context.
    dvoc_evidence : str
        dVOC oscillation-damping evidence JSON path.
    pmu_ringdown : str
        PMU ringdown evidence JSON path.
    ibr_ride_through : str
        IBR ride-through evidence JSON path.
    output : str | None
        Optional destination for the sealed bundle JSON.

    Raises
    ------
    ClickException
        If any evidence JSON or bundle validation step fails.
    """
    try:
        artifacts = (
            _artifact_from_json_path(DVOC_DAMPING_ROLE, Path(dvoc_evidence)),
            _artifact_from_json_path(PMU_RINGDOWN_ROLE, Path(pmu_ringdown)),
            _artifact_from_json_path(IBR_RIDE_THROUGH_ROLE, Path(ibr_ride_through)),
        )
        bundle = build_power_grid_prc_audit_bundle(
            bundle_id=bundle_id,
            created_at=created_at,
            operator_context=operator_context,
            artifacts=artifacts,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo("=== Power-grid PRC assessor bundle ===")
    click.echo(f"bundle: {bundle.bundle_id}  artifacts={len(bundle.artifacts)}")
    click.echo(f"claim boundary: {bundle.claim_boundary}")
    click.echo(f"content hash: {bundle.content_hash}")

    if output is not None:
        Path(output).write_text(
            json.dumps(bundle.to_audit_record(), indent=2), encoding="utf-8"
        )
        click.echo(f"\nbundle written to {output}")


def _artifact_from_json_path(role: str, path: Path) -> PowerGridPRCInputArtifact:
    """Return a bundle input artifact from a strict evidence JSON file."""
    raw = path.read_bytes()
    try:
        parsed = json.loads(raw.decode("utf-8"), parse_constant=_reject_json_constant)
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path.name} must be UTF-8 JSON") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name} must be valid JSON: {exc.msg}") from exc
    except ValueError as exc:
        raise ValueError(f"{path.name} must be strict JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    record: dict[str, object] = dict(parsed)
    return PowerGridPRCInputArtifact(
        role=role,
        source_name=path.name,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        record=record,
    )


def _reject_json_constant(value: str) -> NoReturn:
    """Reject Python's non-standard JSON constants during evidence loading."""
    raise ValueError(f"evidence JSON must be strict JSON, got {value}")

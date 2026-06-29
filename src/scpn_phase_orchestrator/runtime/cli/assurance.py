# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI assurance-case bundle command

"""Assemble a review-only assurance-case evidence bundle from the command line.

The command collects SPO runtime evidence (optionally deriving an audit-chain
integrity record from a log path, plus any evidence records supplied as JSON),
maps it to the EU AI Act / ISO 42001 / UL 4600 clause catalogue, and writes a
deterministic, hash-sealed bundle. The bundle is review-only and carries a
disclaimer that it is a technical evidence aid, not a conformity assessment.
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from scpn_phase_orchestrator.assurance import (
    AUDIT_LOGGING,
    EVIDENCE_CATEGORIES,
    REPLAY_DETERMINISM,
    EvidenceItem,
    build_assurance_case_bundle,
    build_certification_evidence_package,
    build_evidence_item,
    build_formal_verification_evidence,
    build_run_evidence,
    render_conformity_report,
    render_conformity_report_pdf,
)
from scpn_phase_orchestrator.runtime.cli._app import main
from scpn_phase_orchestrator.runtime.replay import ReplayEngine
from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine


def _audit_log_evidence(log_path: str) -> EvidenceItem:
    """Return the audit-log evidence for the assurance bundle."""
    engine = ReplayEngine(log_path)
    entries = engine.load()
    integrity_ok, verified = engine.verify_integrity(entries)
    return build_evidence_item(
        evidence_id="audit-chain-integrity",
        category=AUDIT_LOGGING,
        summary=f"Audit-chain integrity over {log_path}",
        record={
            "integrity_ok": integrity_ok,
            "verified_records": verified,
            "entry_count": len(entries),
        },
    )


def _replay_determinism_evidence(log_path: str) -> EvidenceItem:
    """Return replay-determinism evidence by re-executing the logged run."""
    engine_replay = ReplayEngine(log_path)
    entries = engine_replay.load()
    header = engine_replay.load_header(entries)
    if header is None:
        raise click.ClickException(
            f"{log_path} has no header record for determinism replay"
        )
    engine = engine_replay.build_engine(header)
    if isinstance(engine, StuartLandauEngine):
        deterministic, verified = engine_replay.verify_determinism_sl_chained(
            engine, entries
        )
    else:
        deterministic, verified = engine_replay.verify_determinism_chained(
            engine, entries
        )
    return build_evidence_item(
        evidence_id="replay-determinism",
        category=REPLAY_DETERMINISM,
        summary=f"Deterministic re-execution over {log_path}",
        record={"deterministic": deterministic, "verified_transitions": verified},
    )


def _evidence_from_file(path: str) -> list[EvidenceItem]:
    """Load assurance evidence from a file, else raise."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else [payload]
    items: list[EvidenceItem] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise click.ClickException(f"{path}[{index}] must be a JSON object")
        try:
            items.append(
                build_evidence_item(
                    evidence_id=str(row["evidence_id"]),
                    category=str(row["category"]),
                    summary=str(row["summary"]),
                    record=dict(row["record"]),
                )
            )
        except KeyError as exc:
            raise click.ClickException(
                f"{path}[{index}] is missing required field {exc}"
            ) from exc
        except (ValueError, TypeError) as exc:
            raise click.ClickException(f"{path}[{index}] is invalid: {exc}") from exc
    return items


def _run_result_evidence(path: str) -> list[EvidenceItem]:
    """Return assurance evidence derived from a simulation run record."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise click.ClickException(f"{path} must be a SimulationResult JSON object")
    items = list(build_run_evidence(payload))
    if not items:
        raise click.ClickException(
            f"{path} carries no audit-integrity or conformal-gate evidence"
        )
    return items


def _formal_package_evidence(path: str) -> EvidenceItem:
    """Return formal-verification evidence from a verification-package manifest."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise click.ClickException(
            f"{path} must be a FormalVerificationPackage JSON manifest"
        )
    try:
        return build_formal_verification_evidence(payload)
    except (ValueError, TypeError) as exc:
        raise click.ClickException(f"{path} is not a valid manifest: {exc}") from exc


def _collect_evidence(
    audit_log: str | None,
    evidence_files: tuple[str, ...],
    run_results: tuple[str, ...] = (),
    formal_packages: tuple[str, ...] = (),
    *,
    verify_determinism: bool = False,
) -> list[EvidenceItem]:
    """Return evidence loaded from CLI inputs."""
    evidence: list[EvidenceItem] = []
    if audit_log is not None:
        evidence.append(_audit_log_evidence(audit_log))
        if verify_determinism:
            evidence.append(_replay_determinism_evidence(audit_log))
    for path in run_results:
        evidence.extend(_run_result_evidence(path))
    for path in formal_packages:
        evidence.append(_formal_package_evidence(path))
    for path in evidence_files:
        evidence.extend(_evidence_from_file(path))
    if not evidence:
        raise click.ClickException(
            "no evidence supplied; pass --audit-log, --run-result, "
            "--formal-package, and/or "
            f"--evidence-file (categories: {sorted(EVIDENCE_CATEGORIES)})"
        )
    return evidence


@main.command(name="assurance-case")
@click.option("--system", "system_name", required=True, help="System name")
@click.option(
    "--audit-log",
    "audit_log",
    default=None,
    type=click.Path(exists=True),
    help="Audit log path; adds an audit-chain integrity evidence item",
)
@click.option(
    "--evidence-file",
    "evidence_files",
    multiple=True,
    type=click.Path(exists=True),
    help="JSON file with evidence record(s); repeatable",
)
@click.option(
    "--verify-determinism",
    "verify_determinism",
    is_flag=True,
    help="With --audit-log, add replay-determinism evidence (re-runs determinism)",
)
@click.option(
    "--run-result",
    "run_results",
    multiple=True,
    type=click.Path(exists=True),
    help="SimulationResult JSON; auto-derives run evidence; repeatable",
)
@click.option(
    "--formal-package",
    "formal_packages",
    multiple=True,
    type=click.Path(exists=True),
    help="FormalVerificationPackage manifest JSON; adds formal evidence; repeatable",
)
@click.option("--output", default=None, type=click.Path(), help="Output JSON file")
@click.option(
    "--report-out",
    "report_out",
    default=None,
    type=click.Path(),
    help="Also write a human-readable Markdown conformity report to this path",
)
@click.option(
    "--report-pdf-out",
    "report_pdf_out",
    default=None,
    type=click.Path(),
    help="Also write a deterministic text-PDF conformity report to this path",
)
def assurance_case(
    system_name: str,
    audit_log: str | None,
    evidence_files: tuple[str, ...],
    run_results: tuple[str, ...],
    formal_packages: tuple[str, ...],
    verify_determinism: bool,
    output: str | None,
    report_out: str | None,
    report_pdf_out: str | None,
) -> None:
    """Assemble an assurance-case evidence bundle.

    Parameters
    ----------
    system_name : str
        Name of the system the bundle describes.
    audit_log : str | None
        Optional audit log path; an audit-chain integrity item is added.
    evidence_files : tuple[str, ...]
        Optional JSON files, each a record or list of records with
        ``evidence_id``, ``category``, ``summary``, and ``record`` fields.
    run_results : tuple[str, ...]
        Optional ``SimulationResult`` JSON files; audit-integrity and
        conformal-gate evidence is auto-derived from each.
    formal_packages : tuple[str, ...]
        Optional ``FormalVerificationPackage`` manifest JSON files; a
        formal-verification evidence item is derived from each.
    verify_determinism : bool
        When set with ``--audit-log``, re-run the determinism check and add a
        replay-determinism evidence item.
    output : str | None
        Optional output path; the bundle JSON is printed to stdout otherwise.
    report_out : str | None
        Optional path for a human-readable Markdown conformity report rendered
        from the same sealed bundle.
    report_pdf_out : str | None
        Optional path for a deterministic text-PDF conformity report rendered
        from the same sealed bundle.
    """
    evidence = _collect_evidence(
        audit_log,
        evidence_files,
        run_results,
        formal_packages,
        verify_determinism=verify_determinism,
    )
    bundle = build_assurance_case_bundle(system_name, evidence)
    serialised = json.dumps(bundle.to_audit_record(), indent=2, sort_keys=True)
    if output is not None:
        Path(output).write_text(serialised + "\n", encoding="utf-8")
        click.echo(f"Wrote assurance-case bundle to {output} ({bundle.bundle_hash})")
    else:
        click.echo(serialised)
    if report_out is not None:
        Path(report_out).write_text(render_conformity_report(bundle), encoding="utf-8")
        click.echo(f"Wrote conformity report to {report_out} ({bundle.bundle_hash})")
    if report_pdf_out is not None:
        Path(report_pdf_out).write_bytes(render_conformity_report_pdf(bundle))
        click.echo(
            f"Wrote conformity report PDF to {report_pdf_out} ({bundle.bundle_hash})"
        )


@main.command(name="certification-evidence")
@click.option("--system", "system_name", required=True, help="System name")
@click.option(
    "--audit-log",
    "audit_log",
    default=None,
    type=click.Path(exists=True),
    help="Audit log path; adds an audit-chain integrity evidence item",
)
@click.option(
    "--evidence-file",
    "evidence_files",
    multiple=True,
    type=click.Path(exists=True),
    help="JSON file with evidence record(s); repeatable",
)
@click.option(
    "--verify-determinism",
    "verify_determinism",
    is_flag=True,
    help="With --audit-log, add replay-determinism evidence (re-runs determinism)",
)
@click.option(
    "--run-result",
    "run_results",
    multiple=True,
    type=click.Path(exists=True),
    help="SimulationResult JSON; auto-derives run evidence; repeatable",
)
@click.option(
    "--formal-package",
    "formal_packages",
    multiple=True,
    type=click.Path(exists=True),
    help="FormalVerificationPackage manifest JSON; adds formal evidence; repeatable",
)
@click.option(
    "--output-dir",
    "output_dir",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True),
    help="Directory where the review package files will be written",
)
def certification_evidence(
    system_name: str,
    audit_log: str | None,
    evidence_files: tuple[str, ...],
    run_results: tuple[str, ...],
    formal_packages: tuple[str, ...],
    verify_determinism: bool,
    output_dir: str,
) -> None:
    """Assemble a standards-shaped certification evidence package.

    Parameters
    ----------
    system_name : str
        Name of the system the package describes.
    audit_log : str | None
        Optional audit log path; an audit-chain integrity item is added.
    evidence_files : tuple[str, ...]
        Optional JSON files, each a record or list of records with
        ``evidence_id``, ``category``, ``summary``, and ``record`` fields.
    run_results : tuple[str, ...]
        Optional ``SimulationResult`` JSON files; audit-integrity and
        conformal-gate evidence is auto-derived from each.
    formal_packages : tuple[str, ...]
        Optional ``FormalVerificationPackage`` manifest JSON files; a
        formal-verification evidence item is derived from each.
    verify_determinism : bool
        When set with ``--audit-log``, re-run the determinism check and add a
        replay-determinism evidence item.
    output_dir : str
        Output directory for ``manifest.json``, ``assurance_bundle.json``,
        ``conformity_report.md``, ``conformity_report.pdf``, and
        ``test_vectors.json``.
    """
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise click.ClickException(f"output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    package = build_certification_evidence_package(
        system_name,
        _collect_evidence(
            audit_log,
            evidence_files,
            run_results,
            formal_packages,
            verify_determinism=verify_determinism,
        ),
    )
    for relative_path, payload in package.to_files().items():
        (destination / relative_path).write_bytes(payload)
    click.echo(
        "Wrote certification evidence package to "
        f"{destination} ({package.manifest['package_hash']})"
    )

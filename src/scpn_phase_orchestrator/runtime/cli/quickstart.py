# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — `spo quickstart` golden-path demo command

"""One-command golden-path demo: validate, run, replay and report a domain.

`spo quickstart <domain>` runs the whole supported workflow on a bundled,
research-tier binding so a new user reaches a real result in one command — the
"5-minute golden path". It composes the same public APIs the individual
``validate`` / ``run`` / ``replay`` / ``report`` commands use; it adds no new
modelling. The bundled binding is research-tier so the local runtime may execute
it; the production domainpack of the same domain still requires the formal-export
and certified-controller pipeline.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import click

from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash
from scpn_phase_orchestrator.binding import (
    load_binding_spec,
    validate_binding_spec,
)
from scpn_phase_orchestrator.evaluation.auditor import audit_detector
from scpn_phase_orchestrator.reporting.explainability import (
    build_explainability_report,
    render_markdown,
)
from scpn_phase_orchestrator.reporting.summary import build_audit_report_summary
from scpn_phase_orchestrator.runtime.audit_logger import AuditLogger
from scpn_phase_orchestrator.runtime.cli._app import main
from scpn_phase_orchestrator.runtime.replay import ReplayEngine
from scpn_phase_orchestrator.runtime.simulation import simulate

_ASSET_ROOT = Path(__file__).resolve().parent / "_quickstart_assets"
_DOMAINS = ("power", "eeg")
#: Positional targets the ``quickstart`` command accepts. ``power``/``eeg`` run
#: the simulation golden path; ``evidence`` re-verifies a committed real sealed
#: record; ``auditor`` runs the detector audit on bundled demonstration scores.
_QUICKSTART_TARGETS = (*_DOMAINS, "evidence", "auditor")
#: Bundled synthetic demonstration scores for the ``auditor`` variant.
_AUDITOR_SCORES = _ASSET_ROOT / "auditor" / "scores.json"
#: Reproducible audit parameters for the ``auditor`` demonstration.
_AUDITOR_TARGET_FALSE_ALARM = 0.1
_AUDITOR_PERMUTATIONS = 500
_AUDITOR_SEED = 7
_AUDITOR_ALPHA = 0.05
#: The committed, non-synthetic sealed record the ``evidence`` variant re-checks.
#: Located relative to the repository root (this command's onboarding target is a
#: clone), it is the single canonical copy guarded by
#: ``tests/test_iso_ne_case1_real_evidence.py``.
_EVIDENCE_RECORD = (
    Path(__file__).resolve().parents[4]
    / "examples"
    / "real_data"
    / "iso_ne_case1"
    / "pmu_ringdown_prc_evidence.json"
)


@main.command("quickstart")
@click.argument("domain", type=click.Choice(_QUICKSTART_TARGETS))
@click.option("--steps", default=250, type=int, help="Simulation steps")
@click.option("--seed", default=42, type=int, help="Deterministic RNG seed")
@click.option(
    "--output",
    default=None,
    type=click.Path(),
    help="Write the Markdown report here instead of printing it",
)
def quickstart(domain: str, steps: int, seed: int, output: str | None) -> None:
    """Run the validate → run → replay → report golden path for a domain.

    Parameters
    ----------
    domain : str
        The target — ``"power"`` or ``"eeg"`` runs the simulation golden path;
        ``"evidence"`` re-verifies a committed real sealed record; ``"auditor"``
        runs the detector audit on bundled demonstration scores.
    steps : int
        Number of simulation steps (simulation targets only).
    seed : int
        Seed for the deterministic RNG (simulation targets only).
    output : str | None
        Optional path for the report; printed to stdout if omitted.

    Raises
    ------
    SystemExit
        If the bundled binding fails validation or produces no step records, or
        if a sealed evidence record fails re-verification.
    ClickException
        If the bundled binding asset or the sealed evidence record is missing.
    """
    if domain == "evidence":
        _run_evidence_quickstart(output)
        return
    if domain == "auditor":
        _run_auditor_quickstart(output)
        return

    binding_path = _ASSET_ROOT / domain / "binding_spec.yaml"
    if not binding_path.exists():
        raise click.ClickException(f"quickstart asset not found: {binding_path}")

    spec = load_binding_spec(binding_path)
    errors = validate_binding_spec(spec)
    if errors:
        for error in errors:
            click.echo(f"ERROR: {error}", err=True)
        raise SystemExit(1)

    oscillators = sum(len(layer.oscillator_ids) for layer in spec.layers)
    click.echo(f"=== SPO quickstart: {domain} ===")
    click.echo(
        f"binding {spec.name} v{spec.version} "
        f"(safety_tier={spec.safety_tier}) — {oscillators} oscillators, "
        f"{len(spec.layers)} layers"
    )
    click.echo("[1/4] validate: OK")

    with tempfile.TemporaryDirectory() as tmp_dir:
        audit_path = Path(tmp_dir) / "quickstart_audit.jsonl"
        audit_logger = AuditLogger(audit_path)
        try:
            result = simulate(
                spec,
                steps=steps,
                seed=seed,
                policy_enabled=True,
                audit_logger=audit_logger,
                binding_spec_path=binding_path,
            )
        finally:
            audit_logger.close()

        amplitude = (
            f"  mean_amplitude={result.mean_amplitude:.4f}"
            if result.mean_amplitude is not None
            else ""
        )
        click.echo(
            f"[2/4] run: {result.steps} steps  R_good={result.r_good:.4f}  "
            f"R_bad={result.r_bad:.4f}  separation={result.separation:.4f}  "
            f"regime={result.final_regime}{amplitude}"
        )

        replay_engine = ReplayEngine(str(audit_path))
        entries = replay_engine.load()
        integrity_ok, n_verified = ReplayEngine.verify_integrity(entries)
        if not integrity_ok:
            click.echo("ERROR: audit hash chain failed verification", err=True)
            raise SystemExit(1)
        click.echo(f"[3/4] replay: audit hash chain verified ({n_verified} records)")

        summary = build_audit_report_summary(
            entries,
            hash_chain_ok=integrity_ok,
            hash_chain_verified=n_verified,
        )
        explanation = build_explainability_report(entries, max_actions=8)
        markdown = render_markdown(explanation)

    regimes = summary.get("regime_counts")
    if isinstance(regimes, dict) and regimes:
        spread = "  ".join(f"{name}={count}" for name, count in sorted(regimes.items()))
        click.echo(f"[4/4] report: regimes {spread}")
    else:
        click.echo("[4/4] report: generated")

    if output is not None:
        Path(output).write_text(markdown, encoding="utf-8")
        click.echo(f"\nMarkdown report written to {output}")
    else:
        click.echo("\n" + markdown)


def _verify_evidence_seals(record: Mapping[str, Any]) -> tuple[bool, bool]:
    """Recompute the record's cryptographic seals.

    Parameters
    ----------
    record : Mapping
        The sealed evidence record.

    Returns
    -------
    tuple[bool, bool]
        ``(top_level_ok, nested_prc_ok)`` — whether the top-level content hash
        and the nested PRC-evidence hash both recompute to their sealed values.
    """
    top_payload = {key: value for key, value in record.items() if key != "content_hash"}
    top_ok = canonical_record_hash(top_payload) == record.get("content_hash")

    prc = record.get("prc_evidence")
    if not isinstance(prc, Mapping):
        return top_ok, False
    prc_payload = {key: value for key, value in prc.items() if key != "content_hash"}
    nested_ok = canonical_record_hash(prc_payload) == prc.get(
        "content_hash"
    ) and record.get("prc_evidence_hash") == prc.get("content_hash")
    return top_ok, nested_ok


def _run_evidence_quickstart(output: str | None) -> None:
    """Re-verify the committed real sealed evidence record and print the verdict.

    Loads the non-synthetic ISO-NE PMU-ringdown PRC record, recomputes both its
    seals so the reader need not take the project's word for it, and prints the
    honest, review-only verdict. A broken seal is a hard failure.

    Parameters
    ----------
    output : str | None
        Optional path for the verdict text; printed to stdout if omitted.

    Raises
    ------
    SystemExit
        If either seal fails to recompute.
    ClickException
        If the sealed evidence record is missing.
    """
    if not _EVIDENCE_RECORD.exists():
        raise click.ClickException(
            f"sealed evidence record not found: {_EVIDENCE_RECORD} "
            "(the evidence quickstart runs from a repository clone)"
        )
    record: dict[str, Any] = json.loads(_EVIDENCE_RECORD.read_text(encoding="utf-8"))
    top_ok, nested_ok = _verify_evidence_seals(record)
    prc = record.get("prc_evidence", {})
    modes = prc.get("mode_family_counts", {})
    worst_damping = float(prc.get("worst_damping_ratio", 0.0))

    lines = [
        "=== SPO quickstart: evidence ===",
        f"record {_EVIDENCE_RECORD.name} (schema {record.get('schema')})",
        f"source {record.get('source_name')} "
        f"sha256 {str(record.get('source_sha256'))[:16]}… (real, non-synthetic)",
        f"[1/3] top-level seal: {'VERIFIED' if top_ok else 'FAILED'}",
        f"[2/3] nested PRC seal: {'VERIFIED' if nested_ok else 'FAILED'}",
        f"[3/3] verdict: {prc.get('verdict')} — {prc.get('flagged_count')} mode(s) "
        f"flagged, worst damping ratio {worst_damping:.4f}",
        "       modes: "
        + "  ".join(f"{name}={count}" for name, count in sorted(modes.items())),
        f"       standard: {prc.get('standard')}",
        f"review-only: {record.get('claim_boundary')} "
        f"(review_only={record.get('review_only')}) — an offline screening record, "
        "not a live-actuation claim",
    ]
    text = "\n".join(lines)

    if output is not None:
        Path(output).write_text(text + "\n", encoding="utf-8")
        click.echo(f"Evidence verdict written to {output}")
    else:
        click.echo(text)

    if not (top_ok and nested_ok):
        click.echo("ERROR: a committed evidence seal failed to recompute", err=True)
        raise SystemExit(1)


def _run_auditor_quickstart(output: str | None) -> None:
    """Audit a detector on bundled demonstration scores and print the verdict.

    Runs the event-vs-null skill audit on a committed synthetic scores fixture
    (clearly not real detector output) so the onboarding path shows the audit
    computing a reproducible, honest verdict: the false-alarm-controlled
    detection rate and the permutation p-value.

    Parameters
    ----------
    output : str | None
        Optional path for the verdict text; printed to stdout if omitted.

    Raises
    ------
    ClickException
        If the bundled scores fixture is missing or malformed.
    """
    if not _AUDITOR_SCORES.exists():
        raise click.ClickException(
            f"bundled auditor scores not found: {_AUDITOR_SCORES}"
        )
    spec: dict[str, Any] = json.loads(_AUDITOR_SCORES.read_text(encoding="utf-8"))
    detector_name = str(spec.get("detector_name", "detector"))
    try:
        event_scores = [float(value) for value in spec["event_scores"]]
        null_scores = [float(value) for value in spec["null_scores"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise click.ClickException(
            f"bundled auditor scores are malformed: {exc}"
        ) from exc

    audit = audit_detector(
        event_scores=event_scores,
        null_scores=null_scores,
        detector_name=detector_name,
        target_false_alarm=_AUDITOR_TARGET_FALSE_ALARM,
        n_permutations=_AUDITOR_PERMUTATIONS,
        seed=_AUDITOR_SEED,
        alpha=_AUDITOR_ALPHA,
    )
    lines = [
        "=== SPO quickstart: auditor ===",
        f"scores: bundled demonstration ({detector_name}) — "
        f"{len(event_scores)} event, {len(null_scores)} null segments",
        f"[1/2] audit: detection_rate={audit.detection_rate:.4f}  "
        f"achieved_false_alarm={audit.achieved_false_alarm:.4f}  "
        f"(target ≤ {_AUDITOR_TARGET_FALSE_ALARM})",
        f"[2/2] significance: permutation p={audit.p_value:.4f} "
        f"({audit.significance.n_permutations} perms, seed {_AUDITOR_SEED})  "
        f"beats_chance={audit.beats_chance} (alpha={_AUDITOR_ALPHA})",
        "note: synthetic onboarding fixture, not real detector output",
    ]
    text = "\n".join(lines)

    if output is not None:
        Path(output).write_text(text + "\n", encoding="utf-8")
        click.echo(f"Auditor verdict written to {output}")
    else:
        click.echo(text)

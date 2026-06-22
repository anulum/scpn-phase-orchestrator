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

import tempfile
from pathlib import Path

import click

from scpn_phase_orchestrator.binding import (
    load_binding_spec,
    validate_binding_spec,
)
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
_DOMAINS = ("power",)


@main.command("quickstart")
@click.argument("domain", type=click.Choice(_DOMAINS))
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
        The bundled demo domain (currently ``"power"``).
    steps : int
        Number of simulation steps.
    seed : int
        Seed for the deterministic RNG.
    output : str | None
        Optional path for the Markdown report; printed to stdout if omitted.

    Raises
    ------
    SystemExit
        If the bundled binding fails validation or produces no step records.
    ClickException
        If the bundled binding asset is missing.
    """
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

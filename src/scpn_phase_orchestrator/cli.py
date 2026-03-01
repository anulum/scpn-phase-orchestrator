# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

import re
from pathlib import Path

import click
import numpy as np

from scpn_phase_orchestrator.audit.replay import ReplayEngine
from scpn_phase_orchestrator.binding.loader import (
    load_binding_spec,
    validate_binding_spec,
)
from scpn_phase_orchestrator.coupling.knm import CouplingBuilder
from scpn_phase_orchestrator.upde.engine import UPDEEngine


@click.group()
def main() -> None:
    """SCPN Phase Orchestrator CLI."""


@main.command()
@click.argument("binding_spec", type=click.Path(exists=True))
def validate(binding_spec: str) -> None:
    """Validate a binding specification file."""
    spec = load_binding_spec(Path(binding_spec))
    errors = validate_binding_spec(spec)
    if errors:
        for e in errors:
            click.echo(f"ERROR: {e}", err=True)
        raise SystemExit(1)
    click.echo("Valid")


@main.command()
@click.argument("binding_spec", type=click.Path(exists=True))
@click.option("--steps", default=100, type=int, help="Simulation steps")
def run(binding_spec: str, steps: int) -> None:
    """Run simulation from a binding spec."""
    spec = load_binding_spec(Path(binding_spec))
    errors = validate_binding_spec(spec)
    if errors:
        for e in errors:
            click.echo(f"ERROR: {e}", err=True)
        raise SystemExit(1)

    n_osc = sum(len(layer.oscillator_ids) for layer in spec.layers)
    builder = CouplingBuilder()
    coupling = builder.build(
        n_osc, spec.coupling.base_strength, spec.coupling.decay_alpha
    )
    engine = UPDEEngine(n_osc, dt=spec.sample_period_s)

    phases = np.random.default_rng(42).uniform(0, 2 * np.pi, n_osc)
    omegas = np.ones(n_osc)

    for _ in range(steps):
        phases = engine.step(phases, omegas, coupling.knm, 0.0, 0.0, coupling.alpha)

    good_mask = np.zeros(n_osc, dtype=bool)
    bad_mask = np.zeros(n_osc, dtype=bool)
    for idx in spec.objectives.good_layers:
        if idx < n_osc:
            good_mask[idx] = True
    for idx in spec.objectives.bad_layers:
        if idx < n_osc:
            bad_mask[idx] = True

    r_good, _ = (
        engine.compute_order_parameter(phases[good_mask])
        if good_mask.any()
        else (0.0, 0.0)
    )
    r_bad, _ = (
        engine.compute_order_parameter(phases[bad_mask])
        if bad_mask.any()
        else (0.0, 0.0)
    )

    click.echo(f"R_good={r_good:.4f}  R_bad={r_bad:.4f}")


@main.command()
@click.argument("log_path", type=click.Path(exists=True))
@click.option("--output", default=None, type=click.Path(), help="Output file")
def replay(log_path: str, output: str | None) -> None:
    """Replay an audit log and print summary."""
    re = ReplayEngine(log_path)
    entries = re.load()
    step_entries = [e for e in entries if "step" in e]
    event_entries = [e for e in entries if "event" in e]
    click.echo(f"Steps logged: {len(step_entries)}")
    click.echo(f"Events logged: {len(event_entries)}")
    if step_entries:
        last = step_entries[-1]
        click.echo(f"Final regime: {last.get('regime', 'unknown')}")
        click.echo(f"Final stability: {last.get('stability', 0.0):.4f}")


@main.command()
@click.argument("log_path", type=click.Path(exists=True))
def report(log_path: str) -> None:
    """Generate coherence report from audit log."""
    click.echo("Report generation planned for v0.3")


@main.command()
@click.argument("domain_name")
def scaffold(domain_name: str) -> None:
    """Create a domainpack directory structure with template files."""
    if not re.match(r"^[a-zA-Z0-9_-]+$", domain_name):
        raise click.BadParameter(
            f"domain_name must match [a-zA-Z0-9_-]+, got {domain_name!r}"
        )
    base = Path(f"domainpacks/{domain_name}")
    base.mkdir(parents=True, exist_ok=True)
    spec_file = base / "binding_spec.yaml"
    if not spec_file.exists():
        spec_file.write_text(
            f"# Binding spec for {domain_name}\n"
            f"name: {domain_name}\n"
            "version: '0.1'\n"
            "safety_tier: advisory\n"
            "sample_period_s: 0.01\n"
            "control_period_s: 0.1\n"
            "layers: []\n"
            "oscillator_families: {}\n"
            "coupling:\n"
            "  base_strength: 0.45\n"
            "  decay_alpha: 0.3\n"
            "  templates: {}\n"
            "drivers:\n"
            "  physical: {}\n"
            "  informational: {}\n"
            "  symbolic: {}\n"
            "objectives:\n"
            "  good_layers: []\n"
            "  bad_layers: []\n"
            "boundaries: []\n"
            "actuators: []\n",
            encoding="utf-8",
        )
    readme = base / "README.md"
    if not readme.exists():
        readme.write_text(f"# {domain_name} domainpack\n", encoding="utf-8")
    click.echo(f"Scaffolded domainpack at {base}")

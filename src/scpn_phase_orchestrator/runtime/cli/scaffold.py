# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI scaffold, generate and demo commands

"""Command-line entry point for validation, replay, export, and review workflows.

The CLI wraps public SPO APIs behind explicit commands for binding validation,
inspection, auto-binding proposals, coupling estimation, formal export, replay,
plugin catalogs, scaffolding, and selected runtime utilities. Commands validate
local inputs and emit text or JSON review artifacts; they do not push commits,
start network services, or perform live actuation unless an explicit subcommand
is invoked for that runtime path.
"""

from __future__ import annotations

import csv
import http.client
import io
import json
import re
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import click
import numpy as np

from scpn_phase_orchestrator.autotune.binding_proposal import (
    propose_binding_from_time_series_csv,
)
from scpn_phase_orchestrator.binding import (
    compile_symbolic_binding,
    load_binding_spec,
)
from scpn_phase_orchestrator.runtime.cli._app import (
    _PHYSIONET_HEARTBEAT_CITATION,
    _PHYSIONET_HEARTBEAT_URL,
    main,
)
from scpn_phase_orchestrator.scaffold.llm import (
    LLMScaffoldProvider,
    StaticJSONScaffoldProvider,
    configured_llm_scaffold_provider,
    propose_domainpack_from_description,
)


@main.command()
@click.argument("domain_name")
@click.option(
    "--llm",
    "use_llm",
    is_flag=True,
    help="Generate the binding spec from a natural-language description.",
)
@click.option(
    "--description",
    default=None,
    help="Natural-language domain description for --llm mode.",
)
@click.option(
    "--llm-response-json",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Offline JSON response file for deterministic LLM scaffold review.",
)
def scaffold(
    domain_name: str,
    use_llm: bool,
    description: str | None,
    llm_response_json: str | None,
) -> None:
    """Create a domainpack directory structure with template files.

    Parameters
    ----------
    domain_name : str
        Name of the domainpack to scaffold.
    use_llm : bool
        Whether to use the LLM-assisted scaffolder.
    description : str | None
        Domain description text, or ``None``.
    llm_response_json : str | None
        Path to a cached LLM response JSON, or ``None``.

    Raises
    ------
    BadParameter
        If a CLI argument is invalid.
    ClickException
        If the inputs are invalid or the operation fails.
    """
    if not re.match(r"^[a-zA-Z0-9_-]+$", domain_name):
        raise click.BadParameter(
            f"domain_name must match [a-zA-Z0-9_-]+, got {domain_name!r}"
        )
    base = Path(f"domainpacks/{domain_name}")
    if use_llm:
        if not description:
            raise click.BadParameter("--description is required with --llm")
        provider: LLMScaffoldProvider
        if llm_response_json:
            provider = StaticJSONScaffoldProvider(
                Path(llm_response_json).read_text(encoding="utf-8")
            )
        else:
            try:
                provider = configured_llm_scaffold_provider()
            except RuntimeError as exc:
                raise click.ClickException(str(exc)) from exc
        try:
            proposal = propose_domainpack_from_description(
                description,
                project_name=domain_name,
                provider=provider,
            )
        except (RuntimeError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        base.mkdir(parents=True, exist_ok=True)
        (base / "binding_spec.yaml").write_text(
            proposal.yaml_text,
            encoding="utf-8",
        )
        readme = base / "README.md"
        if not readme.exists():
            readme.write_text(
                f"# {domain_name} domainpack\n\n"
                "LLM-assisted domainpack scaffold. Review the generated "
                "binding_spec.yaml, llm_scaffold_audit.json, boundaries, "
                "actuators, and oscillator mappings before production use.\n",
                encoding="utf-8",
            )
        (base / "llm_scaffold_audit.json").write_text(
            json.dumps(proposal.to_audit_record(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        click.echo(f"Scaffolded LLM-assisted domainpack at {base}")
        return

    base.mkdir(parents=True, exist_ok=True)
    spec_file = base / "binding_spec.yaml"
    if not spec_file.exists():
        spec_file.write_text(
            f"name: {domain_name}\n"
            "version: '0.1.0'\n"
            "safety_tier: research\n"
            "sample_period_s: 0.01\n"
            "control_period_s: 0.1\n"
            "layers:\n"
            "  - name: default\n"
            "    index: 0\n"
            "    oscillator_ids: [osc_0]\n"
            "oscillator_families:\n"
            "  default:\n"
            "    channel: P\n"
            "    extractor_type: physical\n"
            "coupling:\n"
            "  base_strength: 0.45\n"
            "  decay_alpha: 0.3\n"
            "drivers:\n"
            "  physical: {}\n"
            "  informational: {}\n"
            "  symbolic: {}\n"
            "objectives:\n"
            "  good_layers: [0]\n"
            "  bad_layers: []\n"
            "boundaries: []\n"
            "actuators: []\n",
            encoding="utf-8",
        )
    readme = base / "README.md"
    if not readme.exists():
        readme.write_text(f"# {domain_name} domainpack\n", encoding="utf-8")
    click.echo(f"Scaffolded domainpack at {base}")


@main.command("generate")
@click.argument("intent")
@click.option(
    "--name",
    default="generated_domain",
    help="Generated domainpack name.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default=None,
    help=(
        "Directory for binding_spec.yaml, policy.yaml, README.md, "
        "review_notebook.ipynb, and audit.json."
    ),
)
@click.option(
    "--oscillators-per-layer",
    default=8,
    show_default=True,
    help="Oscillators assigned to each inferred layer.",
)
@click.option(
    "--dry-run-steps",
    default=8,
    show_default=True,
    help="Validation simulation steps before artefacts are emitted.",
)
def generate(
    intent: str,
    name: str,
    output_dir: str | None,
    oscillators_per_layer: int,
    dry_run_steps: int,
) -> None:
    """Generate reviewable binding artefacts from symbolic domain intent.

    Parameters
    ----------
    intent : str
        Symbolic domain intent text.
    name : str
        The span or resource name.
    output_dir : str | None
        Destination directory, or ``None``.
    oscillators_per_layer : int
        Number of oscillators per generated layer.
    dry_run_steps : int
        Number of integration steps for the embedded dry run.
    """
    artefacts = compile_symbolic_binding(
        intent,
        name=name,
        oscillators_per_layer=oscillators_per_layer,
        dry_run_steps=dry_run_steps,
    )
    output_path = Path("domainpacks") / name if output_dir is None else Path(output_dir)
    artefacts.write_domainpack(output_path)
    click.echo(f"Generated domainpack at {output_path}")
    click.echo(f"schema_valid={artefacts.schema_valid}")
    click.echo(f"confidence={artefacts.audit_record['confidence']:.3f}")
    click.echo(f"retrieval_matches={len(artefacts.retrieval_evidence)}")
    click.echo(f"dry_run_R={artefacts.dry_run_order_parameter:.6f}")


@main.command()
@click.option(
    "--domain",
    default="minimal_domain",
    help="Domainpack to demo (default: minimal_domain).",
)
@click.option(
    "--dataset",
    default=None,
    help="Real-data demo dataset alias/path/URL. Use heartbeat.csv for PhysioNet HRB.",
)
@click.option(
    "--target",
    default="coherence",
    type=click.Choice(["coherence"]),
    help="Review target for real-data demo.",
)
@click.option("--steps", default=100, help="Number of simulation steps.")
@click.option("--port", default=8000, help="Server port.")
def demo(domain: str, dataset: str | None, target: str, steps: int, port: int) -> None:
    """Run a self-contained demo: simulate + print live coherence.

    Parameters
    ----------
    domain : str
        Domain label.
    dataset : str | None
        Dataset name, or ``None``.
    target : str
        Target metric or channel.
    steps : int
        Number of simulation steps to run.
    port : int
        Port to bind.

    Raises
    ------
    SystemExit
        If the command fails; the error is reported and the process exits non-zero.
    """
    if dataset is not None:
        _run_real_data_demo(dataset=dataset, target=target, steps=steps, port=port)
        return

    domainpack_dir = Path(__file__).parent.parent.parent / "domainpacks"
    spec_path = _contained_domainpack_spec(domainpack_dir, domain)
    if not spec_path.exists():
        # Try relative to cwd.
        spec_path = _contained_domainpack_spec(Path("domainpacks"), domain)
    if not spec_path.exists():
        listing_root = (
            domainpack_dir if domainpack_dir.exists() else Path("domainpacks")
        )
        available = sorted(
            d.name
            for d in listing_root.iterdir()
            if d.is_dir() and (d / "binding_spec.yaml").exists()
        )
        click.echo(f"Domainpack '{domain}' not found.", err=True)
        click.echo(f"Available: {', '.join(available)}", err=True)
        raise SystemExit(1)

    spec = load_binding_spec(spec_path)
    click.echo(f"SPO Demo — {spec.name}")
    click.echo(f"  Oscillators: {sum(len(ly.oscillator_ids) for ly in spec.layers)}")
    click.echo(f"  Layers: {len(spec.layers)}")
    click.echo(f"  Steps: {steps}")
    click.echo("-" * 40)

    from scpn_phase_orchestrator.runtime.server import SimulationState

    sim = SimulationState(spec)
    for step in range(1, steps + 1):
        state = sim.step()
        if step % max(1, steps // 10) == 0 or step == steps:
            R = state["R_global"]
            regime = state["regime"]
            click.echo(f"  Step {step:>5d}: R={R:.3f} [{regime}]")

    click.echo("-" * 40)
    click.echo(f"Final R={state['R_global']:.3f}, regime={state['regime']}")
    click.echo("\nTo serve with full stack:")
    click.echo("  cd deploy && docker compose up")
    click.echo("  Open http://localhost:8000 (dashboard)")
    click.echo("  Open http://localhost:3000 (Grafana)")
    click.echo("  Open http://localhost:9090 (Prometheus)")


def _run_real_data_demo(*, dataset: str, target: str, steps: int, port: int) -> None:
    """Run the real-data scaffolding demo."""
    if steps < 1:
        raise click.BadParameter("steps must be positive")
    if target != "coherence":
        raise click.BadParameter("only target=coherence is supported")
    csv_text, source = _load_demo_dataset(dataset)
    proposal = propose_binding_from_time_series_csv(
        csv_text,
        sample_rate_hz=None,
        project_name="heartbeat_coherence_demo",
    )
    record = cast(dict[str, Any], proposal.to_audit_record())
    binding_record = cast(dict[str, Any], record["binding"])
    source_record = cast(dict[str, Any], record["source"])
    metadata_record = cast(dict[str, Any], record["metadata"])
    runtime_record = cast(dict[str, Any], record["runtime"])
    provenance = cast(dict[str, Any], binding_record["provenance"])
    click.echo("SPO Real-Data Demo — heartbeat coherence")
    click.echo(f"  Dataset: {dataset}")
    click.echo(f"  Source: {source}")
    click.echo(f"  Citation: {_PHYSIONET_HEARTBEAT_CITATION}")
    click.echo(f"  Target: {target}")
    click.echo(f"  Rows used: {source_record['sample_count']}")
    click.echo(f"  Sample rate: {provenance['sample_rate_hz']:.6g} Hz")
    click.echo(f"  Inferred channels: {', '.join(binding_record['inferred_channels'])}")
    click.echo(f"  Proposal mode: {metadata_record['proposal_mode']}")
    click.echo(f"  Replay status: {runtime_record['replay_status']}")
    click.echo(f"  Initial R: {runtime_record['R']:.6f}")
    click.echo(f"  Initial K: {runtime_record['K']:.6f}")
    click.echo("-" * 40)
    click.echo("Review-only binding YAML:")
    click.echo(proposal.binding.yaml_text, nl=False)
    click.echo("-" * 40)
    click.echo("Dashboard/replay path:")
    click.echo(
        "  spo auto-bind time-series-csv heartbeat.csv "
        "--project-name heartbeat_coherence_demo --json-out"
    )
    click.echo(f"  spo demo --dataset heartbeat.csv --target coherence --steps {steps}")
    click.echo("  cd deploy && docker compose up")
    click.echo(f"  Open http://localhost:{port} (dashboard)")


def _load_demo_dataset(dataset: str) -> tuple[str, str]:
    """Load the demo dataset, else raise."""
    if dataset == "heartbeat.csv":
        raw = _download_text(_PHYSIONET_HEARTBEAT_URL, max_bytes=512_000)
        return _normalise_heartbeat_csv(raw, max_rows=256), _PHYSIONET_HEARTBEAT_URL
    path = Path(dataset)
    if path.exists() and path.is_file():
        return path.read_text(encoding="utf-8"), str(path)
    if dataset.startswith(("https://", "http://")):
        return _download_text(dataset, max_bytes=512_000), dataset
    raise click.BadParameter(
        "dataset must be heartbeat.csv, an existing local CSV path, or an http(s) URL"
    )


def _download_text(url: str, *, max_bytes: int) -> str:
    """Download text content from a URL, else raise."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise click.ClickException("dataset URL must be an absolute HTTPS URL")
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    connection = http.client.HTTPSConnection(
        parsed.hostname,
        parsed.port,
        timeout=20,
    )
    try:
        connection.request(
            "GET",
            path,
            headers={"User-Agent": "scpn-phase-orchestrator-demo/1"},
        )
        response = connection.getresponse()
        if response.status < 200 or response.status >= 300:
            raise click.ClickException(
                f"demo dataset download failed with HTTP {response.status}"
            )
        payload = response.read(max_bytes + 1)
    finally:
        connection.close()
    if len(payload) > max_bytes:
        raise click.ClickException("demo dataset is too large")
    return payload.decode("utf-8")


def _normalise_heartbeat_csv(raw: str, *, max_rows: int) -> str:
    """Return the normalised heartbeat-CSV rows."""
    reader = csv.DictReader(io.StringIO(raw))
    required = {"rr_ms", "hr_bpm"}
    if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
        raise click.ClickException("heartbeat dataset must include rr_ms and hr_bpm")
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["time", "rr_ms", "hr_bpm"])
    writer.writeheader()
    samples: list[tuple[float, float]] = []
    for row in reader:
        rr_ms = _finite_csv_float(row.get("rr_ms"), "rr_ms")
        hr_bpm = _finite_csv_float(row.get("hr_bpm"), "hr_bpm")
        samples.append((rr_ms, hr_bpm))
        if len(samples) >= max_rows:
            break
    if len(samples) < 3:
        raise click.ClickException("heartbeat dataset must contain at least 3 rows")
    sample_period_s = float(np.median([rr_ms for rr_ms, _hr_bpm in samples])) / 1000.0
    if not np.isfinite(sample_period_s) or sample_period_s <= 0.0:
        raise click.ClickException("heartbeat dataset has invalid RR interval timing")
    for index, (rr_ms, hr_bpm) in enumerate(samples):
        writer.writerow(
            {
                "time": f"{index * sample_period_s:.6f}",
                "rr_ms": f"{rr_ms:.9g}",
                "hr_bpm": f"{hr_bpm:.9g}",
            }
        )
    return output.getvalue()


def _finite_csv_float(value: object, field: str) -> float:
    """Return a CSV field as a finite float, else raise."""
    if not isinstance(value, str):
        raise click.ClickException(f"heartbeat dataset has non-numeric {field}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise click.ClickException(
            f"heartbeat dataset has non-numeric {field}"
        ) from exc
    if not np.isfinite(number):
        raise click.ClickException(f"heartbeat dataset has non-finite {field}")
    return number


def _contained_domainpack_spec(domainpack_root: Path, domain: str) -> Path:
    """Return the validated contained domainpack spec, else raise."""
    if not isinstance(domain, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", domain):
        raise click.BadParameter("domain must match [A-Za-z0-9_-]+")
    root = domainpack_root.resolve()
    spec_path = (root / domain / "binding_spec.yaml").resolve()
    try:
        spec_path.relative_to(root)
    except ValueError as exc:
        raise click.BadParameter("domain resolves outside domainpack root") from exc
    return spec_path

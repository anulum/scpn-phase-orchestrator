# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI replay, watch, report and explain commands

"""Command-line entry point for validation, replay, export, and review workflows.

The CLI wraps public SPO APIs behind explicit commands for binding validation,
inspection, auto-binding proposals, coupling estimation, formal export, replay,
plugin catalogs, scaffolding, and selected runtime utilities. Commands validate
local inputs and emit text or JSON review artifacts; they do not push commits,
start network services, or perform live actuation unless an explicit subcommand
is invoked for that runtime path.
"""

from __future__ import annotations

import click

from scpn_phase_orchestrator.reporting.summary import build_audit_report_summary
from scpn_phase_orchestrator.runtime.audit_stream import (
    AuditStreamEvent,
    iter_event_stream,
    read_event_stream,
    verify_event_stream_integrity,
)
from scpn_phase_orchestrator.runtime.cli._app import (
    main,
)
from scpn_phase_orchestrator.runtime.cli._shared import (
    _count_dict,
    _float_list,
    _float_value,
    _int_value,
    _string_list,
)
from scpn_phase_orchestrator.runtime.replay import ReplayEngine
from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine


@main.command()
@click.argument("log_path", type=click.Path(exists=True))
@click.option("--output", default=None, type=click.Path(), help="Output file")
@click.option("--verify", is_flag=True, help="Verify determinism via re-execution")
def replay(log_path: str, output: str | None, verify: bool) -> None:
    """Replay an audit log and print summary.

    Parameters
    ----------
    log_path : str
        Filesystem path to the audit log.
    output : str | None
        Destination path, or ``None`` for stdout.
    verify : bool
        Whether to verify hash-chain integrity.

    Raises
    ------
    SystemExit
        If the command fails; the error is reported and the process exits non-zero.
    """
    replay_engine = ReplayEngine(log_path)
    entries = replay_engine.load()
    step_data = [e for e in entries if "step" in e]
    event_data = [e for e in entries if "event" in e]
    click.echo(f"Steps logged: {len(step_data)}")
    click.echo(f"Events logged: {len(event_data)}")
    if step_data:
        last = step_data[-1]
        click.echo(f"Final regime: {last.get('regime', 'unknown')}")
        click.echo(f"Final stability: {last.get('stability', 0.0):.4f}")
    if verify:
        integrity_ok, n_integrity = ReplayEngine.verify_integrity(entries)
        if not integrity_ok:
            click.echo(
                f"ERROR: audit integrity FAILED after {n_integrity} records",
                err=True,
            )
            raise SystemExit(1)
        header = replay_engine.load_header(entries)
        if header is None:
            click.echo("ERROR: no header record in log", err=True)
            raise SystemExit(1)
        engine = replay_engine.build_engine(header)
        if isinstance(engine, StuartLandauEngine):
            passed, n = replay_engine.verify_determinism_sl_chained(engine, entries)
        else:
            passed, n = replay_engine.verify_determinism_chained(engine, entries)
        if passed:
            click.echo(f"Determinism verified: {n} transitions OK")
        else:
            click.echo(f"Determinism FAILED at transition {n}", err=True)
            raise SystemExit(1)


def _watch_line(event: AuditStreamEvent) -> str:
    """Process one line of the watched audit stream."""
    payload = event.payload
    if event.event_type == "step":
        step = _int_value(payload.get("step"))
        regime = str(payload.get("regime", "unknown"))
        stability = _float_value(payload.get("stability"))
        return (
            f"#{event.sequence} step step={step} regime={regime} "
            f"stability={stability:.4f} hash={event.event_hash[:12]}"
        )
    if event.event_type == "header":
        n_osc = _int_value(payload.get("n_oscillators"))
        dt = _float_value(payload.get("dt"))
        return (
            f"#{event.sequence} header n_oscillators={n_osc} "
            f"dt={dt:.6g} hash={event.event_hash[:12]}"
        )
    step_value = payload.get("step")
    suffix = f" step={step_value}" if isinstance(step_value, int) else ""
    return f"#{event.sequence} {event.event_type}{suffix} hash={event.event_hash[:12]}"


@main.command()
@click.argument("stream_path", type=click.Path(exists=True))
@click.option(
    "--format",
    "stream_format",
    type=click.Choice(["protobuf"]),
    default="protobuf",
    show_default=True,
    help="Audit stream encoding.",
)
@click.option("--from-start", is_flag=True, help="Replay existing events first")
@click.option("--max-events", default=None, type=int, help="Stop after N events")
@click.option("--poll-interval", default=0.2, type=float, help="Tail poll interval")
def watch(
    stream_path: str,
    stream_format: str,
    from_start: bool,
    max_events: int | None,
    poll_interval: float,
) -> None:
    """Tail and replay the live audit event stream.

    Parameters
    ----------
    stream_path : str
        Filesystem path to the audit event stream.
    stream_format : str
        Audit stream format.
    from_start : bool
        Whether to replay from the start of the stream.
    max_events : int | None
        Maximum number of events to read, or ``None``.
    poll_interval : float
        Poll interval in seconds.

    Raises
    ------
    SystemExit
        If the command fails; the error is reported and the process exits non-zero.
    """
    if max_events is not None and max_events < 1:
        click.echo("ERROR: --max-events must be >= 1", err=True)
        raise SystemExit(1)
    if poll_interval <= 0.0:
        click.echo("ERROR: --poll-interval must be positive", err=True)
        raise SystemExit(1)
    if stream_format != "protobuf":
        click.echo("ERROR: unsupported stream format", err=True)
        raise SystemExit(1)

    events: list[AuditStreamEvent] = []
    try:
        if from_start and max_events is None:
            events = read_event_stream(stream_path)
            for event in events:
                click.echo(_watch_line(event))
        else:
            for event in iter_event_stream(
                stream_path,
                from_start=from_start,
                poll_interval_s=poll_interval,
            ):
                events.append(event)
                click.echo(_watch_line(event))
                if max_events is not None and len(events) >= max_events:
                    break
    except ValueError as exc:
        click.echo(f"ERROR: {exc}", err=True)
        raise SystemExit(1) from exc

    ok, verified = verify_event_stream_integrity(events)
    status = "OK" if ok else "FAILED"
    click.echo(f"stream integrity: {status} ({verified} events)")
    if not ok:
        raise SystemExit(1)


@main.command()
@click.argument("log_path", type=click.Path(exists=True))
@click.option("--json-out", is_flag=True, help="Output JSON instead of text")
def report(log_path: str, json_out: bool) -> None:
    """Generate coherence report from audit log.

    Parameters
    ----------
    log_path : str
        Filesystem path to the audit log.
    json_out : bool
        Whether to print machine-readable JSON output.

    Raises
    ------
    SystemExit
        If the command fails; the error is reported and the process exits non-zero.
    """
    import json as _json

    replay_engine = ReplayEngine(log_path)
    entries = replay_engine.load()
    steps = [e for e in entries if "step" in e and "layers" in e]

    if not steps:
        click.echo("ERROR: no step records in log", err=True)
        raise SystemExit(1)

    integrity_ok, n_verified = ReplayEngine.verify_integrity(entries)
    summary = build_audit_report_summary(
        entries,
        hash_chain_ok=integrity_ok,
        hash_chain_verified=n_verified,
    )

    if json_out:
        click.echo(_json.dumps(summary, indent=2))
        return

    n_steps = _int_value(summary["steps"])
    n_layers = _int_value(summary["layers"])
    layer_r_mean = _float_list(summary.get("layer_r_mean"))
    layer_r_final = _float_list(summary.get("layer_r_final"))
    regime_counts = _count_dict(summary.get("regime_counts"))
    action_counts = _count_dict(summary.get("action_counts"))

    click.echo(f"Steps: {n_steps}  Layers: {n_layers}")
    mode = "Stuart-Landau" if summary["amplitude_mode"] else "Kuramoto"
    click.echo(f"Mode: {mode}")
    click.echo(f"Final regime: {summary['final_regime']}")
    final_stability = _float_value(summary["final_stability"])
    click.echo(f"Final stability: {final_stability:.4f}")
    click.echo()
    for i in range(n_layers):
        click.echo(
            f"  L{i}: R_mean={layer_r_mean[i]:.4f}  R_final={layer_r_final[i]:.4f}"
        )
    channel_algebra = summary.get("channel_algebra")
    if isinstance(channel_algebra, dict):
        required = _string_list(channel_algebra.get("required_channels"))
        optional = _string_list(channel_algebra.get("optional_channels"))
        derived = _string_list(channel_algebra.get("derived_channels"))
        delayed = _string_list(channel_algebra.get("delayed_channels"))
        uncertain = _string_list(channel_algebra.get("uncertain_channels"))
        missing = _string_list(channel_algebra.get("missing_required_channels"))
        click.echo()
        click.echo(
            "Channel algebra: "
            f"required={len(required)} optional={len(optional)} "
            f"derived={len(derived)} delayed={len(delayed)} "
            f"uncertain={len(uncertain)}"
        )
        if missing:
            click.echo(f"  Missing required channels: {', '.join(missing)}")
    integrated_information = summary.get("integrated_information")
    if isinstance(integrated_information, dict):
        records = _int_value(integrated_information.get("records", 0))
        latest_phi = _float_value(integrated_information.get("latest_phi", 0.0))
        latest_normalised = _float_value(
            integrated_information.get("latest_normalised_phi", 0.0)
        )
        total_integration = _float_value(
            integrated_information.get("latest_total_integration", 0.0)
        )
        click.echo()
        click.echo(
            "Integrated information: "
            f"records={records} phi={latest_phi:.4f} "
            f"normalised_phi={latest_normalised:.4f} "
            f"total_integration={total_integration:.4f}"
        )
    click.echo()
    click.echo("Regime distribution:")
    for regime, count in sorted(regime_counts.items()):
        pct = 100.0 * count / n_steps
        click.echo(f"  {regime}: {count} ({pct:.1f}%)")
    if action_counts:
        click.echo()
        click.echo("Actions fired:")
        for knob, count in sorted(action_counts.items()):
            click.echo(f"  {knob}: {count}")
    click.echo()
    status = "OK" if integrity_ok else "FAILED"
    click.echo(f"Hash chain: {status} ({n_verified} records verified)")


@main.command()
@click.argument("log_path", type=click.Path(exists=True))
@click.option("--markdown-out", default=None, type=click.Path(), help="Write Markdown")
@click.option("--pdf-out", default=None, type=click.Path(), help="Write text PDF")
@click.option("--max-actions", default=12, type=int, help="Maximum action explanations")
def explain(
    log_path: str,
    markdown_out: str | None,
    pdf_out: str | None,
    max_actions: int,
) -> None:
    """Generate a human-readable explanation report from an audit log.

    Parameters
    ----------
    log_path : str
        Filesystem path to the audit log.
    markdown_out : str | None
        Destination Markdown path, or ``None``.
    pdf_out : str | None
        Destination PDF path, or ``None``.
    max_actions : int
        Maximum number of actions to include.

    Raises
    ------
    SystemExit
        If the command fails; the error is reported and the process exits non-zero.
    """
    from scpn_phase_orchestrator.reporting.explainability import (
        build_explainability_report,
        render_markdown,
        write_markdown,
        write_pdf,
    )

    if max_actions < 1:
        click.echo("ERROR: --max-actions must be >= 1", err=True)
        raise SystemExit(1)

    replay_engine = ReplayEngine(log_path)
    entries = replay_engine.load()
    try:
        explanation = build_explainability_report(entries, max_actions=max_actions)
    except ValueError as exc:
        click.echo(f"ERROR: {exc}", err=True)
        raise SystemExit(1) from exc

    wrote = False
    if markdown_out is not None:
        write_markdown(explanation, markdown_out)
        click.echo(f"Markdown report written: {markdown_out}")
        wrote = True
    if pdf_out is not None:
        write_pdf(explanation, pdf_out)
        click.echo(f"PDF report written: {pdf_out}")
        wrote = True
    if not wrote:
        click.echo(render_markdown(explanation), nl=False)

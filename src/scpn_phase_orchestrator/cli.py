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

from scpn_phase_orchestrator.actuation.constraints import ActionProjector
from scpn_phase_orchestrator.audit.logger import AuditLogger
from scpn_phase_orchestrator.audit.replay import ReplayEngine
from scpn_phase_orchestrator.binding import load_binding_spec, validate_binding_spec
from scpn_phase_orchestrator.coupling.geometry_constraints import (
    GeometryConstraint,
    NonNegativeConstraint,
    SymmetryConstraint,
    project_knm,
)
from scpn_phase_orchestrator.coupling.knm import CouplingBuilder, CouplingState
from scpn_phase_orchestrator.drivers.psi_informational import InformationalDriver
from scpn_phase_orchestrator.drivers.psi_physical import PhysicalDriver
from scpn_phase_orchestrator.drivers.psi_symbolic import SymbolicDriver
from scpn_phase_orchestrator.imprint.state import ImprintState
from scpn_phase_orchestrator.imprint.update import ImprintModel
from scpn_phase_orchestrator.monitor.boundaries import BoundaryObserver
from scpn_phase_orchestrator.supervisor.events import EventBus
from scpn_phase_orchestrator.supervisor.petri_adapter import PetriNetAdapter
from scpn_phase_orchestrator.supervisor.petri_net import (
    Arc,
    Marking,
    PetriNet,
    Place,
    Transition,
    parse_guard,
)
from scpn_phase_orchestrator.supervisor.policy import SupervisorPolicy
from scpn_phase_orchestrator.supervisor.policy_rules import (
    PolicyEngine,
    load_policy_rules,
)
from scpn_phase_orchestrator.supervisor.regimes import RegimeManager
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState
from scpn_phase_orchestrator.upde.order_params import (
    compute_order_parameter,
    compute_plv,
)
from scpn_phase_orchestrator.upde.pac import modulation_index
from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine


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
@click.option("--audit", default=None, type=click.Path(), help="Audit log (JSONL)")
@click.option("--seed", default=42, type=int, help="RNG seed")
def run(binding_spec: str, steps: int, audit: str | None, seed: int) -> None:
    """Run simulation from a binding spec."""
    spec = load_binding_spec(Path(binding_spec))
    errors = validate_binding_spec(spec)
    if errors:
        for e in errors:
            click.echo(f"ERROR: {e}", err=True)
        raise SystemExit(1)

    n_osc = sum(len(layer.oscillator_ids) for layer in spec.layers)
    if n_osc == 0:
        click.echo("ERROR: no oscillators defined in layers", err=True)
        raise SystemExit(1)

    builder = CouplingBuilder()
    amplitude_mode = spec.amplitude is not None
    sl_engine: StuartLandauEngine | None = None
    upde_engine: UPDEEngine | None = None
    mu: np.ndarray | None = None

    if amplitude_mode:
        amp = spec.amplitude
        assert amp is not None  # nosec B101 — narrowing for mypy
        coupling = builder.build_with_amplitude(
            n_osc,
            spec.coupling.base_strength,
            spec.coupling.decay_alpha,
            amp.amp_coupling_strength,
            amp.amp_coupling_decay,
        )
        sl_engine = StuartLandauEngine(n_osc, dt=spec.sample_period_s)
        mu = np.full(n_osc, amp.mu)
    else:
        coupling = builder.build(
            n_osc,
            spec.coupling.base_strength,
            spec.coupling.decay_alpha,
        )
        upde_engine = UPDEEngine(n_osc, dt=spec.sample_period_s)
    event_bus = EventBus()
    boundary_observer = BoundaryObserver(spec.boundaries)
    boundary_observer.set_event_bus(event_bus)
    regime_manager = RegimeManager(event_bus=event_bus)

    petri_adapter: PetriNetAdapter | None = None
    if spec.protocol_net is not None:
        pn = spec.protocol_net
        places = [Place(name) for name in pn.places]
        transitions = []
        for ts in pn.transitions:
            guard = parse_guard(ts.guard) if ts.guard else None
            transitions.append(
                Transition(
                    name=ts.name,
                    inputs=[Arc(a["place"], a.get("weight", 1)) for a in ts.inputs],
                    outputs=[Arc(a["place"], a.get("weight", 1)) for a in ts.outputs],
                    guard=guard,
                )
            )
        net = PetriNet(places, transitions)
        petri_adapter = PetriNetAdapter(
            net,
            Marking(tokens=dict(pn.initial)),
            pn.place_regime,
            event_bus=event_bus,
        )

    supervisor = SupervisorPolicy(regime_manager, petri_adapter=petri_adapter)

    # ActionProjector — clip outputs to safe bounds
    projector = ActionProjector(
        rate_limits={"K": 0.1, "zeta": 0.2, "alpha": 0.1, "Psi": 0.5},
        value_bounds={"K": (-0.5, 0.5), "zeta": (0.0, 0.5), "alpha": (-1.0, 1.0)},
    )
    prev_values: dict[str, float] = {"K": 0.0, "zeta": 0.0, "alpha": 0.0, "Psi": 0.0}

    # Policy rules from domainpack (optional)
    policy_engine: PolicyEngine | None = None
    spec_path = Path(binding_spec)
    policy_path = spec_path.parent / "policy.yaml"
    if policy_path.exists():
        rules = load_policy_rules(policy_path)
        if rules:
            policy_engine = PolicyEngine(rules)

    imprint_model = None
    imprint_state = None
    if spec.imprint_model is not None:
        imprint_model = ImprintModel(
            spec.imprint_model.decay_rate, spec.imprint_model.saturation
        )
        imprint_state = ImprintState(m_k=np.zeros(n_osc), last_update=0.0)

    geo_constraints: list[GeometryConstraint] = []
    if spec.geometry_prior is not None:
        ct = spec.geometry_prior.constraint_type.lower()
        if "symmetric" in ct:
            geo_constraints.append(SymmetryConstraint())
        if "non_negative" in ct or "nonneg" in ct:
            geo_constraints.append(NonNegativeConstraint())

    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, 2 * np.pi, n_osc)
    omegas = np.array(
        [1.0 + 0.1 * layer.index for layer in spec.layers for _ in layer.oscillator_ids]
    )

    # Stuart-Landau state: (2N,) = [phases, amplitudes]
    if amplitude_mode and mu is not None:
        r_init = np.sqrt(np.maximum(mu, 0.0))
        sl_state = np.concatenate([phases, r_init])
        phases_history: list[np.ndarray] = []
        amps_history: list[np.ndarray] = []

    layer_osc_ranges: dict[int, list[int]] = {}
    osc_idx = 0
    for layer in spec.layers:
        n_layer = len(layer.oscillator_ids)
        layer_osc_ranges[layer.index] = list(range(osc_idx, osc_idx + n_layer))
        osc_idx += n_layer

    # Initialise drive parameters from spec.drivers
    zeta = max(
        spec.drivers.physical.get("zeta", 0.0),
        spec.drivers.informational.get("zeta", 0.0),
        spec.drivers.symbolic.get("zeta", 0.0),
    )
    zeta_ttl = 0
    psi_target = spec.drivers.physical.get("psi", 0.0)

    psi_driver: PhysicalDriver | InformationalDriver | SymbolicDriver | None = None
    if "frequency" in spec.drivers.physical:
        psi_driver = PhysicalDriver(
            frequency=spec.drivers.physical["frequency"],
            amplitude=spec.drivers.physical.get("amplitude", 1.0),
        )
    elif "cadence_hz" in spec.drivers.informational:
        psi_driver = InformationalDriver(
            cadence_hz=spec.drivers.informational["cadence_hz"],
        )
    elif "sequence" in spec.drivers.symbolic:
        psi_driver = SymbolicDriver(
            sequence=spec.drivers.symbolic["sequence"],
        )
    audit_logger = AuditLogger(audit) if audit else None
    if audit_logger is not None:
        audit_logger.log_header(
            n_oscillators=n_osc,
            dt=spec.sample_period_s,
            seed=seed,
            amplitude_mode=amplitude_mode,
        )
    try:
        for step_idx in range(steps):
            if zeta_ttl > 0:
                zeta_ttl -= 1
                if zeta_ttl == 0:
                    zeta = 0.0

            if psi_driver is not None:
                t = step_idx * spec.sample_period_s
                if isinstance(psi_driver, SymbolicDriver):
                    psi_target = psi_driver.compute(step_idx)
                else:
                    psi_target = psi_driver.compute(t)

            eff_knm = coupling.knm
            eff_alpha = coupling.alpha
            if imprint_model is not None and imprint_state is not None:
                eff_knm = imprint_model.modulate_coupling(eff_knm, imprint_state)
                eff_alpha = imprint_model.modulate_lag(eff_alpha, imprint_state)
            if geo_constraints:
                eff_knm = project_knm(eff_knm, geo_constraints)

            input_phases = phases.copy()
            if amplitude_mode and sl_engine is not None and mu is not None:
                assert coupling.knm_r is not None  # nosec B101
                eff_mu = mu
                if imprint_model is not None and imprint_state is not None:
                    eff_mu = imprint_model.modulate_mu(mu, imprint_state)
                sl_state = sl_engine.step(
                    sl_state,
                    omegas,
                    eff_mu,
                    eff_knm,
                    coupling.knm_r,
                    zeta,
                    psi_target,
                    eff_alpha,
                    epsilon=spec.amplitude.epsilon,  # type: ignore[union-attr]
                )
                phases = sl_state[:n_osc]
                amplitudes = sl_state[n_osc:]
                phases_history.append(phases.copy())
                amps_history.append(amplitudes.copy())
            else:
                assert upde_engine is not None  # nosec B101
                phases = upde_engine.step(
                    phases, omegas, eff_knm, zeta, psi_target, eff_alpha
                )

            layer_states = []
            for layer in spec.layers:
                osc_ids = layer_osc_ranges[layer.index]
                if osc_ids:
                    r, psi_l = compute_order_parameter(phases[osc_ids])
                else:
                    r, psi_l = 0.0, 0.0
                ls_kwargs: dict = {"R": r, "psi": psi_l}
                if amplitude_mode:
                    layer_r = amplitudes[osc_ids] if osc_ids else np.array([])
                    if layer_r.size > 0:
                        ls_kwargs["mean_amplitude"] = float(np.mean(layer_r))
                        mean_r = float(np.mean(layer_r))
                        if mean_r > 0:
                            ls_kwargs["amplitude_spread"] = float(
                                np.std(layer_r) / mean_r
                            )
                layer_states.append(LayerState(**ls_kwargs))

            n_layers = len(spec.layers)
            cla = np.zeros((n_layers, n_layers))
            for li in range(n_layers):
                for lj in range(li + 1, n_layers):
                    ids_i = layer_osc_ranges[spec.layers[li].index]
                    ids_j = layer_osc_ranges[spec.layers[lj].index]
                    if ids_i and ids_j:
                        pi, pj = phases[ids_i], phases[ids_j]
                        min_len = min(len(pi), len(pj))
                        plv = compute_plv(pi[:min_len], pj[:min_len])
                        cla[li, lj] = plv
                        cla[lj, li] = plv

            mean_r_val = (
                float(np.mean([ls.R for ls in layer_states])) if layer_states else 0.0
            )
            state_kwargs: dict = {
                "layers": layer_states,
                "cross_layer_alignment": cla,
                "stability_proxy": mean_r_val,
                "regime_id": regime_manager.current_regime.value,
            }
            if amplitude_mode:
                state_kwargs["mean_amplitude"] = float(np.mean(amplitudes))
                sub_count = int(np.sum(amplitudes < 0.1))
                state_kwargs["subcritical_fraction"] = (
                    sub_count / n_osc if n_osc > 0 else 0.0
                )
                if len(phases_history) >= 20:
                    recent_ph = np.array(phases_history[-20:])
                    recent_am = np.array(amps_history[-20:])
                    pac_vals = [
                        modulation_index(recent_ph[:, i], recent_am[:, i])
                        for i in range(n_osc)
                    ]
                    state_kwargs["pac_max"] = float(max(pac_vals))

            upde_state = UPDEState(**state_kwargs)
            obs_values: dict[str, float] = {"R": upde_state.stability_proxy}
            if amplitude_mode:
                obs_values["mean_amplitude"] = upde_state.mean_amplitude
                obs_values["pac_max"] = upde_state.pac_max
                obs_values["subcritical_fraction"] = upde_state.subcritical_fraction
            for i, ls in enumerate(layer_states):
                obs_values[f"R_{i}"] = ls.R
            boundary_state = boundary_observer.observe(obs_values, step=step_idx)
            actions = supervisor.decide(
                upde_state, boundary_state, petri_ctx=obs_values
            )

            if policy_engine is not None:
                actions.extend(
                    policy_engine.evaluate(
                        regime_manager.current_regime,
                        upde_state,
                        spec.objectives.good_layers,
                        spec.objectives.bad_layers,
                    )
                )

            actions = [
                projector.project(a, prev_values.get(a.knob, 0.0)) for a in actions
            ]

            for act in actions:
                if act.knob == "zeta":
                    zeta = min(zeta + act.value, 0.5)
                    zeta_ttl = int(act.ttl_s / spec.sample_period_s)
                elif act.knob == "K":
                    if act.scope == "global":
                        coupling = CouplingState(
                            knm=coupling.knm * (1.0 + act.value),
                            alpha=coupling.alpha,
                            active_template=coupling.active_template,
                        )
                    elif act.scope.startswith("layer_"):
                        idx = int(act.scope.split("_", 1)[1])
                        new_knm = coupling.knm.copy()
                        new_knm[idx, :] *= 1.0 + act.value
                        new_knm[:, idx] *= 1.0 + act.value
                        new_knm[idx, idx] = 0.0
                        coupling = CouplingState(
                            knm=new_knm,
                            alpha=coupling.alpha,
                            active_template=coupling.active_template,
                        )
                elif act.knob == "Psi":
                    psi_target = act.value
                prev_values[act.knob] = act.value

            if imprint_model is not None and imprint_state is not None:
                exposure = np.array(
                    [
                        layer_states[i].R
                        for i, layer in enumerate(spec.layers)
                        for _ in layer.oscillator_ids
                    ]
                )
                imprint_state = imprint_model.update(
                    imprint_state, exposure, spec.sample_period_s
                )

            if audit_logger is not None:
                audit_logger.log_step(
                    step_idx,
                    upde_state,
                    actions,
                    phases=input_phases,
                    omegas=omegas,
                    knm=eff_knm,
                    alpha=eff_alpha,
                    zeta=zeta,
                    psi_drive=psi_target,
                )

        # Final coherence
        good_phases = [
            phases[i]
            for idx in spec.objectives.good_layers
            for i in layer_osc_ranges.get(idx, [])
        ]
        bad_phases = [
            phases[i]
            for idx in spec.objectives.bad_layers
            for i in layer_osc_ranges.get(idx, [])
        ]

        r_good = (
            compute_order_parameter(np.array(good_phases))[0] if good_phases else 0.0
        )
        r_bad = compute_order_parameter(np.array(bad_phases))[0] if bad_phases else 0.0

        regime = regime_manager.current_regime.value
        msg = f"R_good={r_good:.4f}  R_bad={r_bad:.4f}  regime={regime}"
        if amplitude_mode:
            mean_a = float(np.mean(amplitudes))
            msg += f"  mean_amplitude={mean_a:.4f}"
        click.echo(msg)
    finally:
        if audit_logger is not None:
            for evt in event_bus.history:
                audit_logger.log_event(
                    evt.kind, {"step": evt.step, "detail": evt.detail}
                )
            audit_logger.close()


@main.command()
@click.argument("log_path", type=click.Path(exists=True))
@click.option("--output", default=None, type=click.Path(), help="Output file")
@click.option("--verify", is_flag=True, help="Verify determinism via re-execution")
def replay(log_path: str, output: str | None, verify: bool) -> None:
    """Replay an audit log and print summary."""
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
        header = replay_engine.load_header(entries)
        if header is None:
            click.echo("ERROR: no header record in log", err=True)
            raise SystemExit(1)
        engine = replay_engine.build_engine(header)
        if isinstance(engine, StuartLandauEngine):
            click.echo("Stuart-Landau chained replay not yet supported")
            raise SystemExit(0)
        passed, n = replay_engine.verify_determinism_chained(engine, entries)
        if passed:
            click.echo(f"Determinism verified: {n} transitions OK")
        else:
            click.echo(f"Determinism FAILED at transition {n}", err=True)
            raise SystemExit(1)


@main.command()
@click.argument("log_path", type=click.Path(exists=True))
def report(log_path: str) -> None:
    """Generate coherence report from audit log."""
    click.echo("Report generation planned for v0.3")


@main.group()
def queuewaves() -> None:
    """QueueWaves — real-time cascade failure detector."""


main.add_command(queuewaves)


@queuewaves.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=8080, type=int)
def serve(config_path: str, host: str, port: int) -> None:
    """Start QueueWaves server."""
    from scpn_phase_orchestrator.apps.queuewaves.server import run_server

    run_server(config_path, host=host, port=port)


@queuewaves.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
def check(config_path: str) -> None:
    """One-shot: scrape → analyze → exit 0 (ok) or 1 (anomalies)."""
    from pathlib import Path as _Path

    from scpn_phase_orchestrator.apps.queuewaves.config import load_config
    from scpn_phase_orchestrator.apps.queuewaves.detector import AnomalyDetector
    from scpn_phase_orchestrator.apps.queuewaves.pipeline import PhaseComputePipeline

    cfg = load_config(_Path(config_path))
    pipeline = PhaseComputePipeline(cfg)

    # Run a few ticks with empty buffers to initialise phases
    import numpy as _np

    rng = _np.random.default_rng(0)
    buffers = {svc.name: rng.standard_normal(cfg.buffer_length) for svc in cfg.services}
    snap = pipeline.tick(buffers)
    detector = AnomalyDetector(cfg.thresholds)
    anomalies = detector.detect(snap)

    click.echo(
        f"R_good={snap.r_good:.4f}  R_bad={snap.r_bad:.4f}  regime={snap.regime}"
    )
    if anomalies:
        for a in anomalies:
            click.echo(f"  [{a.severity}] {a.message}")
        raise SystemExit(1)
    click.echo("No anomalies detected.")


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

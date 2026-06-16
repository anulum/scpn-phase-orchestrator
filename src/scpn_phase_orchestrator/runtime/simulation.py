# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — shared non-actuating simulation core

"""Single-source closed/open-loop simulation core for binding specs.

The CLI ``spo run`` command and the public ``Orchestrator.evaluate`` API both
call :func:`simulate`, so a binding spec is advanced by exactly one
implementation — there is no second loop to drift out of fidelity.

The core is non-actuating: it never writes to hardware, opens a network
connection, or enforces a safety tier (tier enforcement is a caller policy, kept
in the CLI). It supports both the Kuramoto (UPDE) and amplitude (Stuart-Landau)
engines, optional Hebbian imprint plasticity, geometry-prior projection,
exogenous physical/informational/symbolic drivers, Petri-net protocol gating,
boundary observation, and the supervisor + domainpack policy control loop.

``policy_enabled`` is the open/closed-loop switch:

* ``True`` (closed loop) — the supervisor and domainpack policy evaluate the
  state every control interval and their bounded, projected actions feed back
  into coupling, lag, damping, and drive. This is what ``spo run`` uses.
* ``False`` (open loop) — the same exogenous drivers and intrinsic plasticity
  still run, but no control feedback is applied. This is the baseline against
  which the closed-loop orchestration uplift is measured on the same seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.actuation.constraints import ActionProjector
from scpn_phase_orchestrator.binding import (
    ChannelRuntimeExecutor,
    resolved_binding_config,
)
from scpn_phase_orchestrator.binding.types import ProtocolNetSpec
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

if TYPE_CHECKING:
    from pathlib import Path

    from scpn_phase_orchestrator.binding.types import BindingSpec
    from scpn_phase_orchestrator.runtime.audit_logger import AuditLogger

FloatArray = NDArray[np.float64]

__all__ = ["SimulationResult", "simulate", "petri_net_from_protocol"]


def petri_net_from_protocol(protocol: ProtocolNetSpec) -> tuple[PetriNet, Marking]:
    """Build a Petri net and initial marking from a protocol-net spec."""

    places = [Place(name) for name in protocol.places]
    transitions = []
    for ts in protocol.transitions:
        guard = parse_guard(ts.guard) if ts.guard else None
        transitions.append(
            Transition(
                name=ts.name,
                inputs=[Arc(a["place"], a.get("weight", 1)) for a in ts.inputs],
                outputs=[Arc(a["place"], a.get("weight", 1)) for a in ts.outputs],
                guard=guard,
            )
        )
    return PetriNet(places, transitions), Marking(tokens=dict(protocol.initial))


@dataclass(frozen=True)
class SimulationResult:
    """Outcome of one :func:`simulate` run.

    Attributes:
        spec_name: Binding-spec name.
        steps: Number of steps advanced.
        policy_enabled: Whether the closed-loop control feedback was active.
        amplitude_mode: Whether the Stuart-Landau (amplitude) engine was used.
        final_phases: Final oscillator phases, shape ``(n,)``.
        final_amplitudes: Final amplitudes for amplitude mode, else ``None``.
        r_good: Final order parameter over the objective good layers.
        r_bad: Final order parameter over the objective bad layers.
        separation: ``r_good - r_bad`` (the coherence objective margin).
        final_regime: Final regime label.
        mean_amplitude: Final mean amplitude for amplitude mode, else ``None``.
        r_good_history / r_bad_history: Per-step good/bad order parameters.
        boundary_violation_total: Summed boundary violations across steps.
        action_total: Total projected control actions applied (0 when open loop).
    """

    spec_name: str
    steps: int
    policy_enabled: bool
    amplitude_mode: bool
    final_phases: FloatArray
    final_amplitudes: FloatArray | None
    r_good: float
    r_bad: float
    separation: float
    final_regime: str
    mean_amplitude: float | None
    r_good_history: tuple[float, ...]
    r_bad_history: tuple[float, ...]
    boundary_violation_total: int
    action_total: int

    def to_record(self) -> dict[str, object]:
        """Return a deterministic JSON-serialisable summary (history omitted)."""

        return {
            "spec_name": self.spec_name,
            "steps": self.steps,
            "policy_enabled": self.policy_enabled,
            "amplitude_mode": self.amplitude_mode,
            "r_good": self.r_good,
            "r_bad": self.r_bad,
            "separation": self.separation,
            "final_regime": self.final_regime,
            "mean_amplitude": self.mean_amplitude,
            "boundary_violation_total": self.boundary_violation_total,
            "action_total": self.action_total,
        }


def _objective_r(
    phases: FloatArray,
    layer_indices: list[int],
    layer_osc_ranges: dict[int, list[int]],
) -> float:
    selected = [
        phases[i] for idx in layer_indices for i in layer_osc_ranges.get(idx, [])
    ]
    if not selected:
        return 0.0
    return float(compute_order_parameter(np.array(selected))[0])


def simulate(
    spec: BindingSpec,
    *,
    steps: int = 100,
    seed: int = 42,
    policy_enabled: bool = True,
    audit_logger: AuditLogger | None = None,
    binding_spec_path: Path | None = None,
) -> SimulationResult:
    """Advance a binding spec for ``steps`` and return the simulation outcome.

    Args:
        spec: A validated binding spec.
        steps: Number of integration steps.
        seed: RNG seed for the initial phases.
        policy_enabled: Closed-loop control feedback on (``True``) or open-loop
            baseline (``False``).
        audit_logger: Optional logger; when given, the header, per-step records,
            and events are written. The caller owns its lifecycle (close).
        binding_spec_path: Optional path to the spec, used only to locate an
            adjacent ``policy.yaml``. When ``None``, no domainpack policy rules
            are loaded (the supervisor policy still runs when ``policy_enabled``).

    Returns:
        A :class:`SimulationResult`.

    Raises:
        ValueError: If the spec declares no oscillators.
    """

    n_osc = sum(len(layer.oscillator_ids) for layer in spec.layers)
    if n_osc == 0:
        raise ValueError("no oscillators defined in layers")

    binding_summary = resolved_binding_config(spec)
    builder = CouplingBuilder()
    amplitude_mode = spec.amplitude is not None
    sl_engine: StuartLandauEngine | None = None
    upde_engine: UPDEEngine | None = None
    mu: FloatArray | None = None

    if amplitude_mode:
        amp = spec.amplitude
        assert amp is not None  # nosec B101
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
    channel_runtime = ChannelRuntimeExecutor.from_spec(spec)

    petri_adapter: PetriNetAdapter | None = None
    if spec.protocol_net is not None:
        net, marking = petri_net_from_protocol(spec.protocol_net)
        petri_adapter = PetriNetAdapter(
            net,
            marking,
            spec.protocol_net.place_regime,
            event_bus=event_bus,
        )

    supervisor = SupervisorPolicy(regime_manager, petri_adapter=petri_adapter)
    projector = ActionProjector.from_actuator_mappings(spec.actuators)
    prev_values: dict[str, float] = {"K": 0.0, "zeta": 0.0, "alpha": 0.0, "Psi": 0.0}

    policy_engine: PolicyEngine | None = None
    if binding_spec_path is not None:
        policy_path = binding_spec_path.parent / "policy.yaml"
        if policy_path.exists():
            rules = load_policy_rules(policy_path)
            if rules:
                policy_engine = PolicyEngine(rules)

    imprint_model: ImprintModel | None = None
    imprint_state: ImprintState | None = None
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
    omegas = np.array(spec.get_omegas(), dtype=np.float64)

    amplitudes = np.array([], dtype=np.float64)
    input_amplitudes = np.array([], dtype=np.float64)
    sl_state = np.array([], dtype=np.float64)
    phases_history: list[FloatArray] = []
    amps_history: list[FloatArray] = []
    if amplitude_mode and mu is not None:
        r_init = np.sqrt(np.maximum(mu, 0.0))
        sl_state = np.concatenate([phases, r_init])

    layer_osc_ranges: dict[int, list[int]] = {}
    osc_idx = 0
    for layer in spec.layers:
        n_layer = len(layer.oscillator_ids)
        layer_osc_ranges[layer.index] = list(range(osc_idx, osc_idx + n_layer))
        osc_idx += n_layer

    zeta = max(
        (cfg.get("zeta", 0.0) for cfg in spec.drivers.all_channel_configs().values()),
        default=0.0,
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

    control_interval = max(1, round(spec.control_period_s / spec.sample_period_s))

    if audit_logger is not None:
        audit_logger.log_header(
            n_oscillators=n_osc,
            dt=spec.sample_period_s,
            seed=seed,
            amplitude_mode=amplitude_mode,
            binding_config=binding_summary,
        )

    r_good_history: list[float] = []
    r_bad_history: list[float] = []
    boundary_violation_total = 0
    action_total = 0
    eff_mu = mu

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
        logged_zeta = zeta
        logged_psi = psi_target
        if amplitude_mode and sl_engine is not None and mu is not None:
            assert coupling.knm_r is not None  # nosec B101
            eff_mu = mu
            if imprint_model is not None and imprint_state is not None:
                eff_mu = imprint_model.modulate_mu(mu, imprint_state)
            # Capture the pre-step amplitudes so the audit record pairs them with
            # the pre-step phases. Logging post-step amplitudes alongside pre-step
            # phases yields a mixed state that no single step produced, which is
            # what broke `spo replay --verify` for amplitude-driven specs.
            input_amplitudes = sl_state[n_osc:].copy()
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
                        ls_kwargs["amplitude_spread"] = float(np.std(layer_r) / mean_r)
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

        runtime_execution = channel_runtime.execute(layer_states)
        executed_layer_states = list(runtime_execution.layers)

        mean_r_val = (
            float(np.mean([ls.R for ls in executed_layer_states]))
            if executed_layer_states
            else 0.0
        )
        state_kwargs: dict = {
            "layers": executed_layer_states,
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

        if imprint_state is not None:
            state_kwargs["imprint_mean"] = float(np.mean(imprint_state.m_k))

        obs_values: dict[str, float] = {"R": state_kwargs["stability_proxy"]}
        if amplitude_mode:
            obs_values["mean_amplitude"] = state_kwargs.get("mean_amplitude", 0.0)
            obs_values["pac_max"] = state_kwargs.get("pac_max", 0.0)
            obs_values["subcritical_fraction"] = state_kwargs.get(
                "subcritical_fraction", 0.0
            )
        for i, ls in enumerate(executed_layer_states):
            obs_values[f"R_{i}"] = ls.R
        boundary_state = boundary_observer.observe(obs_values, step=step_idx)
        state_kwargs["boundary_violation_count"] = len(boundary_state.violations)
        boundary_violation_total += len(boundary_state.violations)
        upde_state = UPDEState(**state_kwargs)

        actions: list = []
        if policy_enabled and step_idx % control_interval == 0:
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
        action_total += len(actions)

        for act in actions:
            if act.knob == "zeta":
                zeta = max(0.0, min(zeta + act.value, 0.5))
                zeta_ttl = int(act.ttl_s / spec.sample_period_s)
            elif act.knob == "K":
                if act.scope == "global":
                    coupling = CouplingState(
                        knm=coupling.knm * (1.0 + act.value),
                        alpha=coupling.alpha,
                        active_template=coupling.active_template,
                        knm_r=coupling.knm_r,
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
                        knm_r=coupling.knm_r,
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
            log_kwargs: dict = {
                "phases": input_phases,
                "omegas": omegas,
                "knm": eff_knm,
                "alpha": eff_alpha,
                "zeta": logged_zeta,
                "psi_drive": logged_psi,
                "channel_runtime": runtime_execution.to_audit_record(),
            }
            if amplitude_mode:
                log_kwargs["amplitudes"] = input_amplitudes
                log_kwargs["mu"] = eff_mu
                log_kwargs["knm_r"] = coupling.knm_r
                log_kwargs["epsilon"] = spec.amplitude.epsilon  # type: ignore[union-attr]
            audit_logger.log_step(step_idx, upde_state, actions, **log_kwargs)

        r_good_history.append(
            _objective_r(phases, spec.objectives.good_layers, layer_osc_ranges)
        )
        r_bad_history.append(
            _objective_r(phases, spec.objectives.bad_layers, layer_osc_ranges)
        )

    if audit_logger is not None:
        for evt in event_bus.history:
            audit_logger.log_event(evt.kind, {"step": evt.step, "detail": evt.detail})

    r_good = r_good_history[-1] if r_good_history else 0.0
    r_bad = r_bad_history[-1] if r_bad_history else 0.0
    mean_amplitude = (
        float(np.mean(amplitudes)) if amplitude_mode and amplitudes.size > 0 else None
    )

    return SimulationResult(
        spec_name=spec.name,
        steps=steps,
        policy_enabled=policy_enabled,
        amplitude_mode=amplitude_mode,
        final_phases=phases,
        final_amplitudes=amplitudes if amplitude_mode else None,
        r_good=r_good,
        r_bad=r_bad,
        separation=r_good - r_bad,
        final_regime=regime_manager.current_regime.value,
        mean_amplitude=mean_amplitude,
        r_good_history=tuple(r_good_history),
        r_bad_history=tuple(r_bad_history),
        boundary_violation_total=boundary_violation_total,
        action_total=action_total,
    )

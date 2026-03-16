# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — QueueWaves phase compute pipeline

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.apps.queuewaves.config import (
    ConfigCompiler,
    QueueWavesConfig,
)
from scpn_phase_orchestrator.binding.types import BindingSpec
from scpn_phase_orchestrator.coupling.knm import CouplingBuilder, CouplingState
from scpn_phase_orchestrator.imprint.state import ImprintState
from scpn_phase_orchestrator.imprint.update import ImprintModel
from scpn_phase_orchestrator.monitor.boundaries import BoundaryObserver
from scpn_phase_orchestrator.oscillators.physical import PhysicalExtractor
from scpn_phase_orchestrator.supervisor.policy import SupervisorPolicy
from scpn_phase_orchestrator.supervisor.regimes import RegimeManager
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState
from scpn_phase_orchestrator.upde.order_params import (
    compute_order_parameter,
    compute_plv,
)

__all__ = ["PipelineSnapshot", "PhaseComputePipeline"]

TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class ServiceSnapshot:
    name: str
    layer: str
    phase: float
    omega: float
    amplitude: float
    imprint: float


@dataclass(frozen=True)
class PipelineSnapshot:
    tick: int
    timestamp: float
    r_good: float
    r_bad: float
    regime: str
    services: list[ServiceSnapshot]
    plv_matrix: list[list[float]]
    layer_states: list[dict]
    boundary_violations: list[str]
    actions: list[dict]

    def to_dict(self) -> dict:
        return {
            "tick": self.tick,
            "timestamp": self.timestamp,
            "r_good": self.r_good,
            "r_bad": self.r_bad,
            "regime": self.regime,
            "services": [
                {
                    "name": s.name,
                    "layer": s.layer,
                    "phase": s.phase,
                    "omega": s.omega,
                    "amplitude": s.amplitude,
                    "imprint": s.imprint,
                }
                for s in self.services
            ],
            "plv_matrix": self.plv_matrix,
            "layer_states": self.layer_states,
            "boundary_violations": self.boundary_violations,
            "actions": self.actions,
        }


class PhaseComputePipeline:
    """Wraps SPO core components into a single tick() call for QueueWaves."""

    def __init__(self, cfg: QueueWavesConfig):
        self._cfg = cfg
        compiler = ConfigCompiler()
        self._spec: BindingSpec = compiler.compile(cfg)

        self._n_osc = sum(len(layer.oscillator_ids) for layer in self._spec.layers)
        self._service_names = [
            oid for layer in self._spec.layers for oid in layer.oscillator_ids
        ]

        builder = CouplingBuilder()
        self._coupling: CouplingState = builder.build(
            self._n_osc,
            self._spec.coupling.base_strength,
            self._spec.coupling.decay_alpha,
        )
        self._engine = UPDEEngine(self._n_osc, dt=self._spec.sample_period_s)
        self._boundary_observer = BoundaryObserver(self._spec.boundaries)
        self._regime_manager = RegimeManager()
        self._supervisor = SupervisorPolicy(self._regime_manager)

        self._imprint_model = ImprintModel(decay_rate=0.01, saturation=5.0)
        self._imprint_state = ImprintState(m_k=np.zeros(self._n_osc), last_update=0.0)

        self._extractor = PhysicalExtractor()

        rng = np.random.default_rng(42)
        self._phases = rng.uniform(0, TWO_PI, self._n_osc)
        self._omegas = np.ones(self._n_osc, dtype=np.float64)

        self._layer_osc_ranges: dict[int, list[int]] = {}
        osc_idx = 0
        for layer in self._spec.layers:
            n_layer = len(layer.oscillator_ids)
            self._layer_osc_ranges[layer.index] = list(
                range(osc_idx, osc_idx + n_layer)
            )
            osc_idx += n_layer

        self._service_layer_map: dict[str, str] = {}
        for svc in cfg.services:
            self._service_layer_map[svc.name] = svc.layer

        self._tick_count = 0

    @property
    def tick_count(self) -> int:
        return self._tick_count

    @property
    def regime(self) -> str:
        return self._regime_manager.current_regime.value

    @property
    def imprint_levels(self) -> NDArray:
        return self._imprint_state.m_k

    def tick(self, buffers: dict[str, NDArray]) -> PipelineSnapshot:
        """Run one full pipeline cycle.

        Args:
            buffers: mapping service_name -> 1-D signal array (ring buffer contents).
        """
        self._tick_count += 1

        # 1. Extract phases from raw signals via Hilbert
        for i, svc_name in enumerate(self._service_names):
            signal = buffers.get(svc_name)
            if signal is None or len(signal) < 4:
                continue
            states = self._extractor.extract(signal, 1.0 / self._cfg.scrape_interval_s)
            if states:
                self._phases[i] = states[0].theta
                self._omegas[i] = max(states[0].omega, 0.01)

        # 2. Imprint-modulated coupling
        eff_knm = self._imprint_model.modulate_coupling(
            self._coupling.knm, self._imprint_state
        )

        # 3. UPDE step
        self._phases = self._engine.step(
            self._phases, self._omegas, eff_knm, 0.0, 0.0, self._coupling.alpha
        )

        # 4. Order parameters per layer
        layer_states: list[LayerState] = []
        for layer in self._spec.layers:
            osc_ids = self._layer_osc_ranges[layer.index]
            if osc_ids:
                r, psi = compute_order_parameter(self._phases[osc_ids])
            else:
                r, psi = 0.0, 0.0
            layer_states.append(LayerState(R=r, psi=psi))

        # 5. R_good, R_bad
        good_phases = [
            self._phases[i]
            for idx in self._spec.objectives.good_layers
            for i in self._layer_osc_ranges.get(idx, [])
        ]
        bad_phases = [
            self._phases[i]
            for idx in self._spec.objectives.bad_layers
            for i in self._layer_osc_ranges.get(idx, [])
        ]
        r_good = (
            compute_order_parameter(np.array(good_phases))[0] if good_phases else 0.0
        )
        r_bad = compute_order_parameter(np.array(bad_phases))[0] if bad_phases else 0.0

        # 6. PLV matrix
        n_layers = len(self._spec.layers)
        plv_mat = np.zeros((n_layers, n_layers))
        for li in range(n_layers):
            for lj in range(li + 1, n_layers):
                ids_i = self._layer_osc_ranges[self._spec.layers[li].index]
                ids_j = self._layer_osc_ranges[self._spec.layers[lj].index]
                if ids_i and ids_j:
                    min_len = min(len(ids_i), len(ids_j))
                    plv = compute_plv(
                        self._phases[ids_i[:min_len]],
                        self._phases[ids_j[:min_len]],
                    )
                    plv_mat[li, lj] = plv
                    plv_mat[lj, li] = plv

        # 7. Boundary observation
        mean_r = float(np.mean([ls.R for ls in layer_states])) if layer_states else 0.0
        upde_state = UPDEState(
            layers=layer_states,
            cross_layer_alignment=plv_mat,
            stability_proxy=mean_r,
            regime_id=self._regime_manager.current_regime.value,
        )
        obs_values = {"R": mean_r, "R_bad": r_bad}
        for i, ls in enumerate(layer_states):
            obs_values[f"R_{i}"] = ls.R
        boundary_state = self._boundary_observer.observe(obs_values)

        # 8. Supervisor actions
        actions = self._supervisor.decide(upde_state, boundary_state)

        # 9. Imprint update
        exposure = np.array(
            [
                layer_states[i].R
                for i, layer in enumerate(self._spec.layers)
                for _ in layer.oscillator_ids
            ]
        )
        self._imprint_state = self._imprint_model.update(
            self._imprint_state, exposure, self._spec.sample_period_s
        )

        # 10. Build snapshot
        svc_snapshots = []
        for i, svc_name in enumerate(self._service_names):
            svc_snapshots.append(
                ServiceSnapshot(
                    name=svc_name,
                    layer=self._service_layer_map.get(svc_name, "unknown"),
                    phase=float(self._phases[i]),
                    omega=float(self._omegas[i]),
                    amplitude=1.0,
                    imprint=float(self._imprint_state.m_k[i]),
                )
            )

        return PipelineSnapshot(
            tick=self._tick_count,
            timestamp=time.time(),
            r_good=r_good,
            r_bad=r_bad,
            regime=self._regime_manager.current_regime.value,
            services=svc_snapshots,
            plv_matrix=plv_mat.tolist(),
            layer_states=[{"R": ls.R, "psi": ls.psi} for ls in layer_states],
            boundary_violations=boundary_state.violations,
            actions=[
                {"knob": a.knob, "scope": a.scope, "value": a.value} for a in actions
            ],
        )

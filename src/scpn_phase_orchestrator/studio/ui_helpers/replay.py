# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SPO Studio binding-spec replay driver

"""Binding-spec replay driver assembling a StudioReplayResult."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.binding.loader import load_binding_spec
from scpn_phase_orchestrator.coupling.knm import CouplingState

from ._shared import _finite_number, _require_non_empty_text
from ._state import StudioReplayResult
from .canvas import build_canvas_graph
from .connectors import build_live_connector_plan
from .guidance import binding_spec_project_state, build_runtime_snapshot
from .tables import build_layer_table, build_oscillator_table

if TYPE_CHECKING:
    from ._state import StudioKnobState


class _ReplaySimulationState(Protocol):
    """Simulation-state surface required by the Studio replay helper."""

    coupling: CouplingState
    omegas: NDArray[np.float64]

    def step(self) -> dict[str, object]:
        """Advance the replay and return a JSON-like runtime snapshot."""

    def snapshot(self) -> dict[str, object]:
        """Return a JSON-like runtime snapshot without advancing replay."""


def run_binding_spec_replay(
    spec_path: Path,
    *,
    steps: int,
    knobs: StudioKnobState,
) -> StudioReplayResult:
    """Run a local binding-spec replay and return Studio-ready payloads.

    Parameters
    ----------
    spec_path : Path
        Filesystem path to the binding-spec file.
    steps : int
        Number of replay steps.
    knobs : StudioKnobState
        The Studio knob state.

    Returns
    -------
    StudioReplayResult
        A local binding-spec replay and return Studio-ready payloads.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    from scpn_phase_orchestrator.runtime.server import SimulationState

    spec = load_binding_spec(spec_path)
    sim: _ReplaySimulationState = SimulationState(spec)
    _apply_replay_knobs(sim, knobs)

    r_history: list[float] = []
    regime_history: list[str] = []
    final_state: Mapping[str, object] = sim.snapshot()
    for _ in range(steps):
        final_state = sim.step()
        r_history.append(_finite_number(final_state["R_global"], "R_global"))
        regime_history.append(_require_non_empty_text(final_state["regime"], "regime"))

    runtime = build_runtime_snapshot(
        final_state=final_state,
        knobs=knobs,
        replay_status="completed",
    )
    project_state = binding_spec_project_state(
        project_name=spec.name,
        spec_path=spec_path,
        knobs=knobs,
        runtime=runtime,
    )
    return StudioReplayResult(
        project_state=project_state,
        r_history=tuple(r_history),
        regime_history=tuple(regime_history),
        layer_table=build_layer_table(spec),
        oscillator_table=build_oscillator_table(spec),
        canvas_graph=build_canvas_graph(spec),
        connector_plan=build_live_connector_plan(spec),
        export_manifests=project_state.exports,
    )


def _apply_replay_knobs(
    sim: _ReplaySimulationState,
    knobs: StudioKnobState,
) -> None:
    """Apply the Studio knobs to a simulation's coupling and natural frequencies.

    Scales the coupling matrix (and its reverse partner) by ``K``, adds the
    ``alpha`` phase lag off-diagonal, and shifts the natural frequencies by
    ``zeta * Psi``, replacing ``sim.coupling`` with the studio-replay variant.
    """
    scaled_knm = np.asarray(sim.coupling.knm, dtype=np.float64) * knobs.K
    alpha = np.asarray(sim.coupling.alpha, dtype=np.float64).copy()
    if knobs.alpha:
        alpha = alpha + knobs.alpha
        np.fill_diagonal(alpha, 0.0)
    knm_r = None
    if sim.coupling.knm_r is not None:
        knm_r = np.asarray(sim.coupling.knm_r, dtype=np.float64) * knobs.K
    sim.coupling = CouplingState(
        knm=scaled_knm,
        alpha=alpha,
        active_template=f"{sim.coupling.active_template}:studio_replay",
        knm_r=knm_r,
    )
    if knobs.zeta or knobs.Psi:
        sim.omegas = np.asarray(sim.omegas, dtype=np.float64) + knobs.zeta * knobs.Psi

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — UPDE Julia runtime contracts

"""UPDE optional Julia backend runtime-availability contracts."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

import pytest

from scpn_phase_orchestrator.upde import _run as run_mod
from scpn_phase_orchestrator.upde import (
    basin_stability,
    delay,
    envelope,
    geometric,
    hypergraph,
    inertial,
    market,
    order_params,
    pac,
    reduction,
    simplicial,
    splitting,
    swarmalator,
)
from scpn_phase_orchestrator.upde._julia_runtime import require_juliacall_main


@dataclass(frozen=True)
class JuliaLoaderCase:
    """Named UPDE Julia loader case."""

    name: str
    loader: Callable[[], object]


_UPDE_JULIA_BRIDGE_MODULES = (
    "scpn_phase_orchestrator.experimental.accelerators.upde._basin_stability_julia",
    "scpn_phase_orchestrator.experimental.accelerators.upde._delay_julia",
    "scpn_phase_orchestrator.experimental.accelerators.upde._engine_julia",
    "scpn_phase_orchestrator.experimental.accelerators.upde._envelope_julia",
    "scpn_phase_orchestrator.experimental.accelerators.upde._geometric_julia",
    "scpn_phase_orchestrator.experimental.accelerators.upde._hypergraph_julia",
    "scpn_phase_orchestrator.experimental.accelerators.upde._inertial_julia",
    "scpn_phase_orchestrator.experimental.accelerators.upde._market_julia",
    "scpn_phase_orchestrator.experimental.accelerators.upde._order_params_julia",
    "scpn_phase_orchestrator.experimental.accelerators.upde._pac_julia",
    "scpn_phase_orchestrator.experimental.accelerators.upde._reduction_julia",
    "scpn_phase_orchestrator.experimental.accelerators.upde._simplicial_julia",
    "scpn_phase_orchestrator.experimental.accelerators.upde._splitting_julia",
    "scpn_phase_orchestrator.experimental.accelerators.upde._swarmalator_julia",
)

_UPDE_JULIA_LOADERS = (
    JuliaLoaderCase("engine-step", run_mod._load_julia_fn),
    JuliaLoaderCase("engine-schedule", run_mod._load_julia_schedule_fn),
    JuliaLoaderCase("basin-stability", basin_stability._load_julia_fn),
    JuliaLoaderCase("delay", delay._load_julia_fn),
    JuliaLoaderCase("envelope", envelope._load_julia_fns),
    JuliaLoaderCase("geometric", geometric._load_julia_fn),
    JuliaLoaderCase("hypergraph", hypergraph._load_julia_fn),
    JuliaLoaderCase("inertial", inertial._load_julia_fn),
    JuliaLoaderCase("market", market._load_julia_fn),
    JuliaLoaderCase("order-params", order_params._load_julia_fns),
    JuliaLoaderCase("pac", pac._load_julia_fns),
    JuliaLoaderCase("reduction", reduction._load_julia_fn),
    JuliaLoaderCase("simplicial", simplicial._load_julia_fn),
    JuliaLoaderCase("splitting", splitting._load_julia_fn),
    JuliaLoaderCase("swarmalator", swarmalator._load_julia_fn),
)


@pytest.fixture(autouse=True)
def _clear_cached_julia_bridges(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force loader tests through the same import path production uses."""
    order_params._BACKEND_CACHE.clear()
    for module_name in _UPDE_JULIA_BRIDGE_MODULES:
        monkeypatch.delitem(sys.modules, module_name, raising=False)


@pytest.mark.parametrize(
    "case",
    _UPDE_JULIA_LOADERS,
    ids=lambda case: case.name,
)
def test_upde_julia_loaders_reject_runtime_without_main(
    monkeypatch: pytest.MonkeyPatch,
    case: JuliaLoaderCase,
) -> None:
    """UPDE Julia loaders must not advertise a partial Julia runtime."""
    stub = ModuleType("juliacall")
    monkeypatch.setitem(sys.modules, "juliacall", stub)

    with pytest.raises(ImportError, match="juliacall.Main unavailable"):
        case.loader()


def test_runtime_probe_returns_complete_juliacall_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A complete ``juliacall`` module must pass through unchanged."""
    stub = ModuleType("juliacall")
    stub.Main = object()
    monkeypatch.setitem(sys.modules, "juliacall", stub)

    assert require_juliacall_main() is stub

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — direct experimental Julia runtime contracts

"""Direct experimental Julia bridge runtime-availability contracts."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from types import ModuleType
from typing import cast

import pytest

JuliaLoader = Callable[[], object]

DIRECT_JULIA_MODULES: tuple[str, ...] = (
    "scpn_phase_orchestrator.experimental.accelerators.coupling._attnres_julia",
    "scpn_phase_orchestrator.experimental.accelerators.coupling._hodge_julia",
    "scpn_phase_orchestrator.experimental.accelerators.coupling._spatial_modulator_julia",
    "scpn_phase_orchestrator.experimental.accelerators.coupling._spectral_julia",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._chimera_julia",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._dimension_julia",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._embedding_julia",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._entropy_prod_julia",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._itpc_julia",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._koopman_edmd_julia",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._lyapunov_julia",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._npe_julia",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._opt_entropy_julia",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._pid_julia",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._poincare_julia",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._psychedelic_julia",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._recurrence_julia",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._te_julia",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._twin_confidence_julia",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._winding_julia",
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


def _direct_loader(module: ModuleType) -> JuliaLoader:
    """Return the private direct Julia runtime loader for ``module``."""
    loader = vars(module).get("_ensure")
    if loader is None:
        loader = vars(module)["_ensure_julia_loaded"]
    return cast(JuliaLoader, loader)


@pytest.mark.parametrize(
    "module_name",
    DIRECT_JULIA_MODULES,
    ids=[name.rsplit(".", maxsplit=1)[-1] for name in DIRECT_JULIA_MODULES],
)
def test_direct_julia_bridges_reject_runtime_without_main(
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct bridge loaders must reject partial ``juliacall`` runtimes."""
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "_JULIA_MODULE", None, raising=False)
    monkeypatch.setitem(sys.modules, "juliacall", ModuleType("juliacall"))

    with pytest.raises(ImportError, match="juliacall\\.Main unavailable"):
        _direct_loader(module)()


def test_direct_runtime_probe_returns_main_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The direct runtime probe must return the injected ``Main`` symbol."""
    runtime_module = importlib.import_module(
        "scpn_phase_orchestrator.experimental.accelerators._julia_runtime"
    )
    expected_main = object()
    juliacall = ModuleType("juliacall")
    juliacall.__dict__["Main"] = expected_main
    monkeypatch.setitem(sys.modules, "juliacall", juliacall)

    probe = cast(Callable[[], object], runtime_module.require_julia_main)

    assert probe() is expected_main

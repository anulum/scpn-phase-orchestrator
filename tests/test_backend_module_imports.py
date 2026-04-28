# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Optional backend module import tests

from __future__ import annotations

import importlib

import pytest

BACKEND_MODULES = (
    "scpn_phase_orchestrator.coupling._attnres_go",
    "scpn_phase_orchestrator.coupling._attnres_julia",
    "scpn_phase_orchestrator.coupling._attnres_mojo",
    "scpn_phase_orchestrator.coupling._hodge_go",
    "scpn_phase_orchestrator.coupling._hodge_julia",
    "scpn_phase_orchestrator.coupling._hodge_mojo",
    "scpn_phase_orchestrator.coupling._spectral_go",
    "scpn_phase_orchestrator.coupling._spectral_julia",
    "scpn_phase_orchestrator.coupling._spectral_mojo",
    "scpn_phase_orchestrator.monitor._chimera_go",
    "scpn_phase_orchestrator.monitor._chimera_julia",
    "scpn_phase_orchestrator.monitor._chimera_mojo",
    "scpn_phase_orchestrator.monitor._dimension_go",
    "scpn_phase_orchestrator.monitor._dimension_julia",
    "scpn_phase_orchestrator.monitor._dimension_mojo",
    "scpn_phase_orchestrator.monitor._embedding_go",
    "scpn_phase_orchestrator.monitor._embedding_julia",
    "scpn_phase_orchestrator.monitor._embedding_mojo",
    "scpn_phase_orchestrator.monitor._entropy_prod_go",
    "scpn_phase_orchestrator.monitor._entropy_prod_julia",
    "scpn_phase_orchestrator.monitor._entropy_prod_mojo",
    "scpn_phase_orchestrator.monitor._itpc_go",
    "scpn_phase_orchestrator.monitor._itpc_julia",
    "scpn_phase_orchestrator.monitor._itpc_mojo",
    "scpn_phase_orchestrator.monitor._lyapunov_go",
    "scpn_phase_orchestrator.monitor._lyapunov_julia",
    "scpn_phase_orchestrator.monitor._lyapunov_mojo",
    "scpn_phase_orchestrator.monitor._npe_go",
    "scpn_phase_orchestrator.monitor._npe_julia",
    "scpn_phase_orchestrator.monitor._npe_mojo",
    "scpn_phase_orchestrator.monitor._poincare_go",
    "scpn_phase_orchestrator.monitor._poincare_julia",
    "scpn_phase_orchestrator.monitor._poincare_mojo",
    "scpn_phase_orchestrator.monitor._psychedelic_go",
    "scpn_phase_orchestrator.monitor._psychedelic_julia",
    "scpn_phase_orchestrator.monitor._psychedelic_mojo",
    "scpn_phase_orchestrator.monitor._recurrence_go",
    "scpn_phase_orchestrator.monitor._recurrence_julia",
    "scpn_phase_orchestrator.monitor._recurrence_mojo",
    "scpn_phase_orchestrator.monitor._te_go",
    "scpn_phase_orchestrator.monitor._te_julia",
    "scpn_phase_orchestrator.monitor._te_mojo",
    "scpn_phase_orchestrator.monitor._winding_go",
    "scpn_phase_orchestrator.monitor._winding_julia",
    "scpn_phase_orchestrator.monitor._winding_mojo",
    "scpn_phase_orchestrator.upde._basin_stability_go",
    "scpn_phase_orchestrator.upde._basin_stability_julia",
    "scpn_phase_orchestrator.upde._basin_stability_mojo",
    "scpn_phase_orchestrator.upde._engine_go",
    "scpn_phase_orchestrator.upde._engine_julia",
    "scpn_phase_orchestrator.upde._engine_mojo",
    "scpn_phase_orchestrator.upde._envelope_go",
    "scpn_phase_orchestrator.upde._envelope_julia",
    "scpn_phase_orchestrator.upde._envelope_mojo",
    "scpn_phase_orchestrator.upde._geometric_go",
    "scpn_phase_orchestrator.upde._geometric_julia",
    "scpn_phase_orchestrator.upde._geometric_mojo",
    "scpn_phase_orchestrator.upde._hypergraph_go",
    "scpn_phase_orchestrator.upde._hypergraph_julia",
    "scpn_phase_orchestrator.upde._hypergraph_mojo",
    "scpn_phase_orchestrator.upde._inertial_go",
    "scpn_phase_orchestrator.upde._inertial_julia",
    "scpn_phase_orchestrator.upde._inertial_mojo",
    "scpn_phase_orchestrator.upde._market_go",
    "scpn_phase_orchestrator.upde._market_julia",
    "scpn_phase_orchestrator.upde._market_mojo",
    "scpn_phase_orchestrator.upde._order_params_go",
    "scpn_phase_orchestrator.upde._order_params_julia",
    "scpn_phase_orchestrator.upde._order_params_mojo",
    "scpn_phase_orchestrator.upde._pac_go",
    "scpn_phase_orchestrator.upde._pac_julia",
    "scpn_phase_orchestrator.upde._pac_mojo",
    "scpn_phase_orchestrator.upde._reduction_go",
    "scpn_phase_orchestrator.upde._reduction_julia",
    "scpn_phase_orchestrator.upde._reduction_mojo",
    "scpn_phase_orchestrator.upde._ref_kernel",
    "scpn_phase_orchestrator.upde._run",
    "scpn_phase_orchestrator.upde._simplicial_go",
    "scpn_phase_orchestrator.upde._simplicial_julia",
    "scpn_phase_orchestrator.upde._simplicial_mojo",
    "scpn_phase_orchestrator.upde._splitting_go",
    "scpn_phase_orchestrator.upde._splitting_julia",
    "scpn_phase_orchestrator.upde._splitting_mojo",
    "scpn_phase_orchestrator.upde._swarmalator_go",
    "scpn_phase_orchestrator.upde._swarmalator_julia",
    "scpn_phase_orchestrator.upde._swarmalator_mojo",
)


@pytest.mark.parametrize("module_name", BACKEND_MODULES)
def test_optional_backend_module_import_surface(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module.__name__ == module_name

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — direct experimental Go runtime contracts

"""Direct experimental Go bridge shared-library availability contracts."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

GoLoader = Callable[[], object]

DIRECT_GO_MODULES: tuple[str, ...] = (
    "scpn_phase_orchestrator.experimental.accelerators.coupling._attnres_go",
    "scpn_phase_orchestrator.experimental.accelerators.coupling._hodge_go",
    "scpn_phase_orchestrator.experimental.accelerators.coupling._spatial_modulator_go",
    "scpn_phase_orchestrator.experimental.accelerators.coupling._spectral_go",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._chimera_go",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._dimension_go",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._embedding_go",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._entropy_prod_go",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._itpc_go",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._koopman_edmd_go",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._lyapunov_go",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._npe_go",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._opt_entropy_go",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._pid_go",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._poincare_go",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._psychedelic_go",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._recurrence_go",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._te_go",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._twin_confidence_go",
    "scpn_phase_orchestrator.experimental.accelerators.monitor._winding_go",
    "scpn_phase_orchestrator.experimental.accelerators.upde._basin_stability_go",
    "scpn_phase_orchestrator.experimental.accelerators.upde._delay_go",
    "scpn_phase_orchestrator.experimental.accelerators.upde._engine_go",
    "scpn_phase_orchestrator.experimental.accelerators.upde._envelope_go",
    "scpn_phase_orchestrator.experimental.accelerators.upde._geometric_go",
    "scpn_phase_orchestrator.experimental.accelerators.upde._hypergraph_go",
    "scpn_phase_orchestrator.experimental.accelerators.upde._inertial_go",
    "scpn_phase_orchestrator.experimental.accelerators.upde._market_go",
    "scpn_phase_orchestrator.experimental.accelerators.upde._order_params_go",
    "scpn_phase_orchestrator.experimental.accelerators.upde._pac_go",
    "scpn_phase_orchestrator.experimental.accelerators.upde._reduction_go",
    "scpn_phase_orchestrator.experimental.accelerators.upde._simplicial_go",
    "scpn_phase_orchestrator.experimental.accelerators.upde._splitting_go",
    "scpn_phase_orchestrator.experimental.accelerators.upde._swarmalator_go",
)


def _direct_loader(module: ModuleType) -> GoLoader:
    """Return the private direct Go shared-library loader for ``module``."""
    return cast(GoLoader, vars(module)["_load_lib"])


@pytest.mark.parametrize(
    "module_name",
    DIRECT_GO_MODULES,
    ids=[name.rsplit(".", maxsplit=1)[-1] for name in DIRECT_GO_MODULES],
)
def test_direct_go_bridges_reject_unloadable_shared_library(
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Direct Go loaders must demote loader ``OSError`` to ``ImportError``."""
    module = importlib.import_module(module_name)
    fake_library = tmp_path / "unloadable.so"
    fake_library.write_bytes(b"not an ELF shared library")

    def fail_cdll(_path: str) -> object:
        raise OSError("file too short")

    monkeypatch.setattr(module, "_LIB", None)
    monkeypatch.setattr(module, "_LIB_PATH", fake_library)
    monkeypatch.setattr(module.ctypes, "CDLL", fail_cdll)

    with pytest.raises(ImportError, match="could not be loaded"):
        _direct_loader(module)()


def test_direct_go_runtime_loader_returns_cdll_handle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The shared Go runtime loader must return the loaded library handle."""
    runtime_module = importlib.import_module(
        "scpn_phase_orchestrator.experimental.accelerators._go_runtime"
    )
    expected_handle = object()
    library_path = tmp_path / "loadable.so"

    def load_cdll(path: str) -> object:
        assert path == str(library_path)
        return expected_handle

    monkeypatch.setattr(runtime_module.ctypes, "CDLL", load_cdll)
    loader = cast(Callable[[Path], object], runtime_module.load_go_library)

    assert loader(library_path) is expected_handle

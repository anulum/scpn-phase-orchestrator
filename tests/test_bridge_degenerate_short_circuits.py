# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — degenerate-input contracts of the native bridges

"""Native bridges answer degenerate inputs without loading their runtime.

The public monitors short-circuit these inputs before dispatch, so only a
direct bridge call reaches the bridge-level early returns. Each test points
the bridge at a missing build artefact: any attempt to load Julia, a Go
library or a Mojo executable would then raise ``ImportError``, so a returned
value proves the short-circuit. The value is compared with the public NumPy
result for the same input wherever the public API accepts that input.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor.embedding import (
    mutual_information,
    nearest_neighbor_distances,
)
from scpn_phase_orchestrator.monitor.entropy_prod import entropy_production_rate
from scpn_phase_orchestrator.monitor.itpc import compute_itpc, itpc_persistence
from scpn_phase_orchestrator.monitor.npe import compute_npe, phase_distance_matrix
from scpn_phase_orchestrator.monitor.transfer_entropy import phase_transfer_entropy

_MONITOR = "scpn_phase_orchestrator.experimental.accelerators.monitor"


def _without_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    module_name: str,
) -> ModuleType:
    """Import a bridge and point it at a build artefact that does not exist."""
    module = importlib.import_module(f"{_MONITOR}.{module_name}")
    names = vars(module)
    if "_JULIA_FILE" in names:
        monkeypatch.setattr(module, "_JULIA_MODULE", None)
        monkeypatch.setattr(module, "_JULIA_FILE", tmp_path / "missing.jl")
    elif "_LIB_PATH" in names:
        monkeypatch.setattr(module, "_LIB", None)
        monkeypatch.setattr(module, "_LIB_PATH", tmp_path / "missing.so")
    else:
        monkeypatch.setattr(module, "_EXE_PATH", tmp_path / "missing_exe")
    return module


def _loader(module: ModuleType) -> Callable[[], object]:
    """Return the bridge's runtime loader."""
    for name in ("_ensure", "_load_lib", "_ensure_exe"):
        candidate = vars(module).get(name)
        if callable(candidate):
            loader: Callable[[], object] = candidate
            return loader
    raise AssertionError(f"{module.__name__} has no runtime loader")


_BRIDGES = (
    "_embedding_julia",
    "_embedding_go",
    "_embedding_mojo",
    "_itpc_julia",
    "_itpc_go",
    "_itpc_mojo",
    "_npe_mojo",
    "_entropy_prod_mojo",
    "_te_mojo",
)


@pytest.mark.parametrize("module_name", _BRIDGES)
def test_missing_artefact_makes_every_runtime_load_fail(
    module_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The configuration used below cannot load a runtime by accident."""
    module = _without_runtime(monkeypatch, tmp_path, module_name)
    with pytest.raises(ImportError):
        _loader(module)()


_EMBEDDING = ("julia", "go", "mojo")


@pytest.mark.parametrize("owner", _EMBEDDING)
@pytest.mark.parametrize("lag", [3, 5])
def test_mutual_information_without_overlap_matches_public_zero(
    owner: str,
    lag: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lag at or beyond the signal length has no pairs and returns 0.0."""
    module = _without_runtime(monkeypatch, tmp_path, f"_embedding_{owner}")
    signal = np.linspace(0.0, 1.0, 3)
    result = getattr(module, f"mutual_information_{owner}")(signal, lag, 4)
    assert result == mutual_information(signal, lag, 4, backend="python") == 0.0


@pytest.mark.parametrize("owner", _EMBEDDING)
def test_nearest_neighbours_of_no_rows_match_public_empty_pair(
    owner: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No embedded rows give empty float distances and int64 indices."""
    module = _without_runtime(monkeypatch, tmp_path, f"_embedding_{owner}")
    distances, indices = getattr(module, f"nearest_neighbor_distances_{owner}")(
        np.zeros(0, dtype=np.float64), 0, 2
    )
    public_distances, public_indices = nearest_neighbor_distances(
        np.zeros((0, 2), dtype=np.float64), backend="python"
    )
    assert distances.dtype == public_distances.dtype == np.float64
    assert indices.dtype == public_indices.dtype == np.int64
    assert distances.shape == public_distances.shape == (0,)
    assert indices.shape == public_indices.shape == (0,)


_ITPC = ("julia", "go", "mojo")


@pytest.mark.parametrize("owner", _ITPC)
def test_itpc_of_no_trials_keeps_the_bridge_length_contract(
    owner: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero trials give ``n_tp`` zeros, the bridge output length contract.

    The public ``compute_itpc`` returns an empty array for zero trials before
    it dispatches, so this bridge-level value never reaches a caller.
    """
    module = _without_runtime(monkeypatch, tmp_path, f"_itpc_{owner}")
    result = getattr(module, f"compute_itpc_{owner}")(np.zeros(0), 0, 3)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, np.zeros(3, dtype=np.float64))


@pytest.mark.parametrize("owner", _ITPC)
def test_itpc_of_no_timepoints_matches_public_empty(
    owner: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero time points give the same empty array as the public monitor."""
    module = _without_runtime(monkeypatch, tmp_path, f"_itpc_{owner}")
    result = getattr(module, f"compute_itpc_{owner}")(np.zeros(0), 2, 0)
    public = compute_itpc(np.zeros((2, 0)), backend="python")
    assert result.dtype == public.dtype == np.float64
    assert result.shape == public.shape == (0,)


@pytest.mark.parametrize("owner", _ITPC)
def test_itpc_persistence_without_pause_indices_matches_public_zero(
    owner: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No pause indices give the public persistence value 0.0."""
    module = _without_runtime(monkeypatch, tmp_path, f"_itpc_{owner}")
    phases = np.linspace(0.0, 1.0, 6)
    no_indices = np.zeros(0, dtype=np.int64)
    result = getattr(module, f"itpc_persistence_{owner}")(phases, 2, 3, no_indices)
    public = itpc_persistence(phases.reshape(2, 3), no_indices, backend="python")
    assert result == public == 0.0


def test_mojo_phase_distances_of_no_phases_match_public_empty_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No phases give an empty flat result that the public monitor accepts."""
    module = _without_runtime(monkeypatch, tmp_path, "_npe_mojo")
    result = module.phase_distance_matrix_mojo(np.zeros(0))
    public = phase_distance_matrix(np.zeros(0), backend="python")
    assert result.dtype == public.dtype == np.float64
    assert result.size == public.size == 0
    assert public.shape == (0, 0)


def test_mojo_npe_of_one_phase_matches_public_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single phase has no pair and the entropy is 0.0."""
    module = _without_runtime(monkeypatch, tmp_path, "_npe_mojo")
    phases = np.array([0.7])
    assert module.compute_npe_mojo(phases, 1.0) == compute_npe(
        phases, 1.0, backend="python"
    )
    assert module.compute_npe_mojo(phases, 1.0) == 0.0


@pytest.mark.parametrize(
    ("n", "dt"),
    [(0, 0.01), (2, 0.0)],
    ids=["no_oscillators", "zero_step"],
)
def test_mojo_entropy_production_degenerate_inputs_match_public_zero(
    n: int,
    dt: float,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No oscillators, or a zero time step, produce no entropy."""
    module = _without_runtime(monkeypatch, tmp_path, "_entropy_prod_mojo")
    phases = np.linspace(0.0, 1.0, n)
    omegas = np.ones(n)
    knm = np.full((n, n), 0.5)
    result = module.entropy_production_rate_mojo(phases, omegas, knm, 0.0, dt)
    public = entropy_production_rate(phases, omegas, knm, 0.0, dt, backend="python")
    assert result == public == 0.0


def test_mojo_transfer_entropy_of_two_samples_matches_public_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fewer than three samples leave no history window and return 0.0."""
    module = _without_runtime(monkeypatch, tmp_path, "_te_mojo")
    source = np.array([0.1, 0.4])
    target = np.array([0.2, 0.3])
    result = module.phase_te_mojo(source, target, 4)
    public = phase_transfer_entropy(source, target, 4, backend="python")
    assert result == public == 0.0

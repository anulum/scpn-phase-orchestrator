# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Connectome Python fallback contracts

"""Fallback and optional-dependency contracts for connectome coupling loaders."""

from __future__ import annotations

import builtins
import sys
import types

import numpy as np
import pytest

from scpn_phase_orchestrator.coupling import connectome


def test_connectome_python_fallback_preserves_brain_network_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(connectome, "_HAS_RUST", False)

    knm = connectome.load_hcp_connectome(16, seed=7)

    assert knm.shape == (16, 16)
    np.testing.assert_allclose(knm, knm.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(knm), 0.0)
    assert np.all(knm >= 0.0)

    left = knm[:8, :8]
    cross = knm[:8, 8:]
    assert float(left[left > 0.0].mean()) > float(cross[cross > 0.0].mean())

    dmn = [int(f * 8) for f in (0.15, 0.45, 0.65, 0.85)]
    dmn += [node + 8 for node in dmn]
    dmn_weights = knm[np.ix_(dmn, dmn)]
    nonzero_dmn = dmn_weights[dmn_weights > 0.0]
    assert float(nonzero_dmn.mean()) > float(knm[knm > 0.0].mean())


def test_neurolib_hcp_loader_validates_and_slices_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = np.arange(80 * 80, dtype=np.float64).reshape(80, 80)
    matrix = raw + raw.T
    np.fill_diagonal(matrix, 0.0)

    class Dataset:
        def __init__(self, name: str) -> None:
            assert name == "hcp"
            self.Cmat = matrix

    neurolib = types.ModuleType("neurolib")
    utils = types.ModuleType("neurolib.utils")
    load_data = types.ModuleType("neurolib.utils.loadData")
    load_data.Dataset = Dataset
    monkeypatch.setitem(sys.modules, "neurolib", neurolib)
    monkeypatch.setitem(sys.modules, "neurolib.utils", utils)
    monkeypatch.setitem(sys.modules, "neurolib.utils.loadData", load_data)

    loaded = connectome.load_neurolib_hcp(5)

    assert loaded.shape == (5, 5)
    np.testing.assert_allclose(np.diag(loaded), 0.0)
    assert np.all(loaded >= 0.0)
    with pytest.raises(ValueError, match=">= 2"):
        connectome.load_neurolib_hcp(1)
    with pytest.raises(TypeError, match="n_regions must be an integer"):
        connectome.load_neurolib_hcp(True)
    with pytest.raises(ValueError, match="<= 80"):
        connectome.load_neurolib_hcp(81)


def test_neurolib_hcp_loader_rejects_invalid_dataset_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = np.ones((80, 80), dtype=np.float64)
    matrix[0, 1] = np.nan

    class Dataset:
        def __init__(self, name: str) -> None:
            assert name == "hcp"
            self.Cmat = matrix

    neurolib = types.ModuleType("neurolib")
    utils = types.ModuleType("neurolib.utils")
    load_data = types.ModuleType("neurolib.utils.loadData")
    load_data.Dataset = Dataset
    monkeypatch.setitem(sys.modules, "neurolib", neurolib)
    monkeypatch.setitem(sys.modules, "neurolib.utils", utils)
    monkeypatch.setitem(sys.modules, "neurolib.utils.loadData", load_data)

    with pytest.raises(ValueError, match="finite"):
        connectome.load_neurolib_hcp(5)


def test_neurolib_hcp_loader_reports_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def block_neurolib(
        name: str,
        globals_: object | None = None,
        locals_: object | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name.startswith("neurolib"):
            raise ModuleNotFoundError(name)
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", block_neurolib)

    with pytest.raises(ImportError, match="neurolib is required"):
        connectome.load_neurolib_hcp(2)

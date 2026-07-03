# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SSGF cost input and Rust-output validation guards

from __future__ import annotations

import importlib
import sys
from typing import Any

import numpy as np
import pytest

import scpn_phase_orchestrator.coupling.spectral as spectral_mod
import scpn_phase_orchestrator.ssgf.costs as costs_mod
from scpn_phase_orchestrator.ssgf.costs import (
    _validate_rust_costs,
    compute_ssgf_costs,
)

_W = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
_PHASES = np.array([0.1, 0.2], dtype=np.float64)
_WEIGHTS = (1.0, 0.5, 0.1, 0.1)


class TestComputeSsgfCostsGuards:
    def test_valid_round_trips(self) -> None:
        costs = compute_ssgf_costs(_W, _PHASES, _WEIGHTS)
        assert 0.0 <= costs.c1_sync <= 1.0

    @pytest.mark.parametrize(
        ("W", "phases", "weights", "match"),
        [
            (_W, _PHASES, (1.0,), "tuple of four"),
            (_W, _PHASES, (-1.0, 0.5, 0.1, 0.1), "finite non-negative"),
            (_W, np.array([True, False]), _WEIGHTS, "must not contain boolean"),
            (_W, np.array([0.1 + 1j, 0.2]), _WEIGHTS, "phases must be real-valued"),
            (_W, np.zeros((2, 2)), _WEIGHTS, "phases must be a one-dimensional"),
            (_W, np.array([]), _WEIGHTS, "at least one oscillator"),
            (_W, np.array([0.1, np.inf]), _WEIGHTS, "phases must contain only finite"),
            (
                np.array([[True, False], [False, True]]),
                _PHASES,
                _WEIGHTS,
                "W must not contain boolean",
            ),
            (np.zeros((2, 3)), _PHASES, _WEIGHTS, "W must be a square matrix"),
            (
                np.array([[0.0, np.inf], [np.inf, 0.0]]),
                _PHASES,
                _WEIGHTS,
                "W must contain only finite",
            ),
            (_W, np.array([0.1, 0.2, 0.3]), _WEIGHTS, "phases length must match"),
        ],
    )
    def test_rejects_corrupt_input(
        self, W: Any, phases: Any, weights: Any, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            compute_ssgf_costs(W, phases, weights)


def _rust_costs(c1: float, c2: float, c3: float, c4: float) -> tuple[float, ...]:
    total = _WEIGHTS[0] * c1 + _WEIGHTS[1] * c2 + _WEIGHTS[2] * c3 + _WEIGHTS[3] * c4
    return (c1, c2, c3, c4, total)


class TestValidateRustCosts:
    def test_valid_round_trips(self) -> None:
        result = _validate_rust_costs(
            _rust_costs(0.5, -0.3, 0.1, 0.0), weights=_WEIGHTS
        )
        assert result.c1_sync == pytest.approx(0.5)

    def test_rejects_wrong_term_count(self) -> None:
        with pytest.raises(ValueError, match="five cost terms"):
            _validate_rust_costs((0.5, -0.3, 0.1), weights=_WEIGHTS)

    def test_rejects_sync_deficit_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="synchronisation deficit must stay"):
            _validate_rust_costs(_rust_costs(1.5, -0.3, 0.1, 0.0), weights=_WEIGHTS)

    def test_rejects_positive_spectral_cost(self) -> None:
        with pytest.raises(ValueError, match="spectral cost must be non-positive"):
            _validate_rust_costs(_rust_costs(0.5, 0.3, 0.1, 0.0), weights=_WEIGHTS)

    def test_rejects_negative_sparsity_cost(self) -> None:
        with pytest.raises(ValueError, match="sparsity cost must be non-negative"):
            _validate_rust_costs(_rust_costs(0.5, -0.3, -0.1, 0.0), weights=_WEIGHTS)

    def test_rejects_negative_symmetry_cost(self) -> None:
        with pytest.raises(ValueError, match="symmetry cost must be non-negative"):
            _validate_rust_costs(_rust_costs(0.5, -0.3, 0.1, -0.1), weights=_WEIGHTS)

    def test_rejects_inconsistent_weighted_total(self) -> None:
        terms = (0.5, -0.3, 0.1, 0.0, 99.0)
        with pytest.raises(ValueError, match="weighted total must equal"):
            _validate_rust_costs(terms, weights=_WEIGHTS)


class TestNumpyFallback:
    def test_python_path_computes_cost_terms(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(costs_mod, "_HAS_RUST", False)

        costs = compute_ssgf_costs(_W, _PHASES, _WEIGHTS)

        assert 0.0 <= costs.c1_sync <= 1.0
        assert costs.c2_spectral_gap <= 1.0e-9  # -lambda2 is non-positive
        assert costs.c3_sparsity >= 0.0
        assert costs.c4_symmetry == pytest.approx(0.0)  # W is symmetric
        assert np.isfinite(costs.u_total)

    def test_module_marks_rust_absent_when_kernel_cannot_import(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Importing without ``spo_kernel`` drives the ImportError fallback.

        A sandboxed copy of the module is executed with ``spo_kernel`` blocked
        (a reload of the real module would change the ``SSGFCosts`` class
        identity held by other test modules). The copy must mark the Rust
        backend absent, keep the loader slot ``None``, and still compute costs
        through the NumPy floor that agree with the reference values from the
        untouched module.
        """
        original_has_rust = costs_mod._HAS_RUST
        original_rust_costs = costs_mod._rust_costs
        reference_costs = compute_ssgf_costs(_W, _PHASES, _WEIGHTS)
        monkeypatch.setitem(sys.modules, "spo_kernel", None)
        spec = importlib.util.spec_from_file_location(
            "ssgf_costs_rustless_copy", costs_mod.__file__
        )
        assert spec is not None and spec.loader is not None
        rustless = importlib.util.module_from_spec(spec)
        # The dataclass decorator resolves cls.__module__ through sys.modules,
        # so the sandboxed copy must be registered before its body executes.
        monkeypatch.setitem(sys.modules, "ssgf_costs_rustless_copy", rustless)

        spec.loader.exec_module(rustless)

        assert rustless._HAS_RUST is False
        assert rustless._rust_costs is None
        monkeypatch.setattr(spectral_mod, "ACTIVE_BACKEND", "python")
        monkeypatch.setattr(spectral_mod, "AVAILABLE_BACKENDS", ["python"])
        monkeypatch.setattr(spectral_mod, "_RUST_CACHE", None)
        python_costs = rustless.compute_ssgf_costs(_W, _PHASES, _WEIGHTS)
        assert costs_mod._HAS_RUST is original_has_rust  # real module is untouched
        assert costs_mod._rust_costs is original_rust_costs
        assert python_costs.c1_sync == pytest.approx(reference_costs.c1_sync)
        assert python_costs.c2_spectral_gap == pytest.approx(
            reference_costs.c2_spectral_gap
        )
        assert python_costs.c3_sparsity == pytest.approx(reference_costs.c3_sparsity)
        assert python_costs.c4_symmetry == pytest.approx(reference_costs.c4_symmetry)
        assert python_costs.u_total == pytest.approx(reference_costs.u_total)


class TestBooleanAliasDetector:
    def test_object_dtype_ndarray_with_boolean_is_an_alias(self) -> None:
        """A boolean nested in an object-dtype ndarray must be detected."""
        value = np.array([0.1, True], dtype=object)

        assert costs_mod._contains_boolean_alias(value) is True

    def test_object_dtype_ndarray_without_boolean_is_clean(self) -> None:
        """A purely numeric object-dtype ndarray passes the alias scan."""
        value = np.array([0.1, 0.2], dtype=object)

        assert costs_mod._contains_boolean_alias(value) is False

    def test_plain_python_list_with_boolean_is_an_alias(self) -> None:
        """A non-ndarray sequence goes through the object scan directly."""
        assert costs_mod._contains_boolean_alias([0.1, True]) is True
        assert costs_mod._contains_boolean_alias([0.1, 0.2]) is False

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Psychedelic monitor validation and fallback guards

from __future__ import annotations

import builtins
import importlib
from types import ModuleType
from typing import Any, cast

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor import psychedelic as psychedelic_mod
from scpn_phase_orchestrator.monitor.psychedelic import (
    _contains_boolean_alias,
    _contains_complex_alias,
    _contains_numeric_string_alias,
    _has_complex_payload,
    _is_numeric_string_alias,
    _validate_entropy_value,
    _validate_reduced_coupling,
    entropy_from_phases,
    reduce_coupling,
    simulate_psychedelic_trajectory,
)
from scpn_phase_orchestrator.upde.engine import UPDEEngine

_PHASES = np.array([0.1, 0.2], dtype=np.float64)
_OMEGAS = np.zeros(2, dtype=np.float64)
_KNM = np.array([[0.0, 0.4], [0.4, 0.0]], dtype=np.float64)
_ALPHA = np.zeros((2, 2), dtype=np.float64)


class _ArrayProtocolFailure:
    def __array__(self, dtype: object = None) -> np.ndarray:
        raise TypeError("array protocol unavailable")


class TestModuleLoadersAndHelpers:
    """Defensive loader and scalar-alias helper branch coverage."""

    def test_rust_reduce_import_failure_demotes_to_numpy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original_import = builtins.__import__

        def fake_import(
            name: str,
            globals: dict[str, object] | None = None,
            locals: dict[str, object] | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> ModuleType:
            if name == "spo_kernel" and "reduce_coupling_rust" in fromlist:
                raise ImportError("forced missing spo_kernel")
            return cast(
                ModuleType,
                original_import(name, globals, locals, fromlist, level),
            )

        monkeypatch.setattr(builtins, "__import__", fake_import)
        try:
            reloaded = importlib.reload(psychedelic_mod)
            assert reloaded._HAS_RUST_REDUCE is False
        finally:
            monkeypatch.setattr(builtins, "__import__", original_import)
            importlib.reload(psychedelic_mod)

    def test_load_julia_fn_returns_entropy_callable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(psychedelic_mod, "require_juliacall_main", lambda: None)

        loaded = psychedelic_mod._load_julia_fn()

        assert callable(loaded)

    def test_dispatch_returns_none_when_every_non_python_candidate_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        previous_backend = psychedelic_mod.ACTIVE_BACKEND
        previous_available = list(psychedelic_mod.AVAILABLE_BACKENDS)
        previous_loader = psychedelic_mod._LOADERS["go"]
        psychedelic_mod.ACTIVE_BACKEND = "go"
        psychedelic_mod.AVAILABLE_BACKENDS = []
        psychedelic_mod._BACKEND_FN_CACHE.clear()
        monkeypatch.setitem(
            psychedelic_mod._LOADERS,
            "go",
            lambda: (_ for _ in ()).throw(RuntimeError("unavailable")),
        )
        try:
            assert psychedelic_mod._dispatch() is None
        finally:
            psychedelic_mod.ACTIVE_BACKEND = previous_backend
            psychedelic_mod.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(psychedelic_mod._LOADERS, "go", previous_loader)
            psychedelic_mod._BACKEND_FN_CACHE.clear()

    def test_alias_helpers_tolerate_array_protocol_failures(self) -> None:
        failing = _ArrayProtocolFailure()

        assert not _contains_boolean_alias(failing)
        assert not _contains_complex_alias(failing)
        assert not _contains_numeric_string_alias(failing)
        assert not _has_complex_payload(failing)

    def test_numeric_string_helper_keeps_generic_string_errors_generic(self) -> None:
        assert not _is_numeric_string_alias(1.0)
        assert not _contains_numeric_string_alias(
            np.array([1.0, "not-a-number"], dtype=object)
        )
        assert _contains_numeric_string_alias(np.array([1.0, "2.0"], dtype=object))


class TestReduceCoupling:
    def test_rejects_non_coercible_matrix(self) -> None:
        knm = np.array([["a", "b"], ["c", "d"]], dtype=object)
        with pytest.raises(ValueError, match="finite 2-D matrix"):
            reduce_coupling(knm, 0.5)

    def test_rejects_non_square_matrix(self) -> None:
        with pytest.raises(ValueError, match="must be square"):
            reduce_coupling(np.zeros((2, 3)), 0.5)

    def test_numpy_fallback_scales_matrix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(psychedelic_mod, "_HAS_RUST_REDUCE", False)

        reduced = reduce_coupling(_KNM, 0.25)

        np.testing.assert_allclose(reduced, _KNM * 0.75)


class TestEntropyFromPhases:
    def test_rejects_non_coercible_phases(self) -> None:
        phases = np.array(["a", "b"], dtype=object)
        with pytest.raises(ValueError, match="finite 1-D phase vector"):
            entropy_from_phases(phases)

    def test_falls_back_to_numpy_when_backend_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _failing(_phases: object, _bins: int) -> float:
            raise RuntimeError("simulated backend runtime failure")

        monkeypatch.setattr(psychedelic_mod, "_dispatch", lambda: _failing)

        entropy = entropy_from_phases(np.linspace(0.0, 6.0, 12))

        assert np.isfinite(entropy)
        assert entropy >= 0.0

    def test_returns_zero_when_histogram_backend_reports_zero_total(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def zero_histogram(
            *_args: Any, **_kwargs: Any
        ) -> tuple[np.ndarray, np.ndarray]:
            return np.zeros(4, dtype=np.int64), np.linspace(0.0, 1.0, 5)

        monkeypatch.setattr(psychedelic_mod, "_dispatch", lambda: None)
        monkeypatch.setattr(psychedelic_mod.np, "histogram", zero_histogram)

        assert entropy_from_phases(np.array([0.1], dtype=np.float64), n_bins=4) == 0.0


class TestSimulateTrajectoryGuards:
    def _engine(self) -> UPDEEngine:
        return UPDEEngine(2, dt=0.01)

    def test_rejects_knm_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match=r"shape \(2, 2\)"):
            simulate_psychedelic_trajectory(
                self._engine(),
                _PHASES,
                _OMEGAS,
                np.zeros((3, 3)),
                _ALPHA,
                [0.0],
            )

    def test_rejects_boolean_reduction_schedule(self) -> None:
        with pytest.raises(TypeError, match="reduction_schedule must be a 1-D"):
            simulate_psychedelic_trajectory(
                self._engine(),
                _PHASES,
                _OMEGAS,
                _KNM,
                _ALPHA,
                True,  # type: ignore[arg-type]  # invalid on purpose
            )

    def test_rejects_reduction_schedule_array_protocol_failure(self) -> None:
        with pytest.raises(TypeError, match="reduction_schedule must be a 1-D"):
            simulate_psychedelic_trajectory(
                self._engine(),
                _PHASES,
                _OMEGAS,
                _KNM,
                _ALPHA,
                _ArrayProtocolFailure(),  # type: ignore[arg-type]
            )

    def test_rejects_negative_step_count(self) -> None:
        with pytest.raises(ValueError, match="non-negative integer"):
            simulate_psychedelic_trajectory(
                self._engine(),
                _PHASES,
                _OMEGAS,
                _KNM,
                _ALPHA,
                [0.0],
                n_steps_per_level=-1,
            )


class TestOutputContracts:
    """Direct validation of untrusted reduced-coupling and entropy output."""

    def test_reduced_coupling_rejects_non_coercible_output(self) -> None:
        bad = np.array(["a", "b", "c", "d"], dtype=object)
        with pytest.raises(ValueError, match="must be numeric"):
            _validate_reduced_coupling(bad, expected_shape=(2, 2))

    def test_reduced_coupling_rejects_numeric_string_output(self) -> None:
        bad = np.array(["0.0", "1.0", "1.0", "0.0"], dtype=object)
        with pytest.raises(ValueError, match="numeric-string"):
            _validate_reduced_coupling(bad, expected_shape=(2, 2))

    def test_entropy_value_rejects_non_coercible_output(self) -> None:
        with pytest.raises(ValueError, match="entropy output must be numeric"):
            _validate_entropy_value("not-a-number", n_bins=36)

    def test_entropy_value_rejects_numeric_string_output(self) -> None:
        with pytest.raises(ValueError, match="numeric-string"):
            _validate_entropy_value("0.5", n_bins=36)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Fractal-dimension result and output guards

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor import dimension as dimension_mod
from scpn_phase_orchestrator.monitor.dimension import (
    CorrelationDimensionResult,
    _validate_ci_values,
    _validate_ky_dimension,
    correlation_dimension,
    kaplan_yorke_dimension,
)


def _result(slope: object) -> CorrelationDimensionResult:
    return CorrelationDimensionResult(
        D2=1.0,
        epsilons=np.array([0.1, 0.2, 0.3], dtype=np.float64),
        C_eps=np.array([0.1, 0.2, 0.3], dtype=np.float64),
        slope=slope,  # type: ignore[arg-type]  # corrupted on purpose
        scaling_range=(0.1, 0.3),
    )


class TestCorrelationDimensionResultGuards:
    def test_rejects_boolean_slope(self) -> None:
        with pytest.raises(ValueError, match="slope must not contain boolean values"):
            _result(np.array([True, False]))

    def test_rejects_non_coercible_slope(self) -> None:
        with pytest.raises(ValueError, match="slope must be a finite one-dimensional"):
            _result(np.array(["a", "b"], dtype=object))


class TestOutputContracts:
    """Direct validation of untrusted compute-backend output."""

    def test_correlation_integral_rejects_non_coercible_output(self) -> None:
        bad = np.array(["a", "b"], dtype=object)
        with pytest.raises(ValueError, match="must be numeric"):
            _validate_ci_values(bad, expected_size=2)

    def test_kaplan_yorke_rejects_complex_output(self) -> None:
        with pytest.raises(ValueError, match="must be real-valued"):
            _validate_ky_dimension(complex(1.0, 1.0), n_exponents=3)


class TestKaplanYorkeBackendFallback:
    def test_falls_back_to_exact_reference_when_backend_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _failing(_name: str) -> object:
            def _fn(_values: object) -> float:
                raise RuntimeError("simulated backend runtime failure")

            return _fn

        monkeypatch.setattr(dimension_mod, "_dispatch", _failing)

        result = kaplan_yorke_dimension(np.array([1.0, -2.0], dtype=np.float64))

        assert np.isfinite(result)
        assert result >= 0.0


class TestCorrelationBackendFallback:
    def test_correlation_dimension_uses_numpy_when_backend_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _failing(_name: str) -> object:
            def _fn(*_args: object, **_kwargs: object) -> object:
                raise RuntimeError("simulated backend runtime failure")

            return _fn

        monkeypatch.setattr(dimension_mod, "_dispatch", _failing)

        angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
        trajectory = np.column_stack([np.sin(angles), np.cos(angles)])

        result = correlation_dimension(trajectory, n_epsilons=12, max_pairs=500, seed=7)

        assert np.isfinite(result.D2)
        assert result.D2 >= 0.0

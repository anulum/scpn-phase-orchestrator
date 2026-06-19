# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — UPDE engine guards and NumPy integrator paths

from __future__ import annotations

import numpy as np
import pytest

import scpn_phase_orchestrator.upde.engine as engine_module
from scpn_phase_orchestrator.upde.engine import UPDEEngine

_TWO_PI = 2.0 * np.pi
_PHASES = np.array([0.1, 0.2], dtype=np.float64)
_OMEGA = np.array([0.0, 0.0], dtype=np.float64)
_KNM = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
_ALPHA = np.zeros((2, 2), dtype=np.float64)


def _python_engine(monkeypatch: pytest.MonkeyPatch, method: str) -> UPDEEngine:
    """Build an engine whose stepper is the NumPy reference (no Rust kernel)."""
    monkeypatch.setattr(engine_module, "_HAS_RUST", False)
    engine = UPDEEngine(2, dt=0.01, method=method)
    assert engine._rust is None
    return engine


class TestRequiredArgumentGuards:
    """step and run require an explicit coupling matrix and phase-lag matrix."""

    def test_step_requires_knm(self) -> None:
        engine = UPDEEngine(2, dt=0.01)
        with pytest.raises(ValueError, match="knm is required"):
            engine.step(_PHASES, _OMEGA, None, alpha=_ALPHA)

    def test_step_requires_alpha(self) -> None:
        engine = UPDEEngine(2, dt=0.01)
        with pytest.raises(ValueError, match="alpha is required"):
            engine.step(_PHASES, _OMEGA, _KNM, alpha=None)

    def test_run_requires_knm(self) -> None:
        engine = UPDEEngine(2, dt=0.01)
        with pytest.raises(ValueError, match="knm is required"):
            engine.run(_PHASES, _OMEGA, None, alpha=_ALPHA)

    def test_run_requires_alpha(self) -> None:
        engine = UPDEEngine(2, dt=0.01)
        with pytest.raises(ValueError, match="alpha is required"):
            engine.run(_PHASES, _OMEGA, _KNM, alpha=None)


class TestRustOutputContract:
    """``_validate_rust_output`` guards untrusted kernel output."""

    def test_rejects_non_real_output(self) -> None:
        engine = UPDEEngine(2, dt=0.01)
        with pytest.raises(ValueError, match="real numeric array"):
            engine._validate_rust_output(np.array(["a", "b"]))


class TestNumpyIntegratorPaths:
    """The NumPy euler/rk4/rk45 steppers run when the Rust kernel is absent."""

    @pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
    def test_step_advances_through_numpy_path(
        self, monkeypatch: pytest.MonkeyPatch, method: str
    ) -> None:
        engine = _python_engine(monkeypatch, method)

        result = engine.step(_PHASES, _OMEGA, _KNM, alpha=_ALPHA)

        assert result.shape == (2,)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0.0) and np.all(result < _TWO_PI)

    def test_external_drive_term_is_applied(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _python_engine(monkeypatch, "euler")

        driven = engine.step(_PHASES, _OMEGA, _KNM, zeta=0.5, psi=1.0, alpha=_ALPHA)
        engine_undriven = _python_engine(monkeypatch, "euler")
        undriven = engine_undriven.step(_PHASES, _OMEGA, _KNM, alpha=_ALPHA)

        assert not np.allclose(driven, undriven)

    def test_rk45_accepts_small_step(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _python_engine(monkeypatch, "rk45")

        result = engine.step(_PHASES, _OMEGA, _KNM, alpha=_ALPHA)

        assert result.shape == (2,)
        assert engine.last_dt > 0.0
        assert np.all(np.isfinite(result))

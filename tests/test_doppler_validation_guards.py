# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Doppler contract, dispatch and engine guards

from __future__ import annotations

import numpy as np
import pytest

import scpn_phase_orchestrator.upde.doppler as doppler_module
from scpn_phase_orchestrator.upde.doppler import (
    DopplerEngine,
    doppler_run,
    doppler_run_python,
    doppler_term,
    scalarise_velocities,
    validate_doppler_backend_inputs,
)

_KNM = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)


class TestDopplerTermInputGuards:
    """``doppler_term`` rejects non-real, mismatched or non-positive inputs."""

    def test_rejects_object_dtype_knm(self) -> None:
        knm = np.array([["a", "b"], ["c", "d"]], dtype=object)
        with pytest.raises(ValueError, match="object dtype"):
            doppler_term(np.zeros(2), knm)

    def test_rejects_complex_knm(self) -> None:
        knm = np.array([[0.0, 1.0j], [1.0j, 0.0]])
        with pytest.raises(ValueError, match="must be real-valued"):
            doppler_term(np.zeros(2), knm)

    def test_rejects_string_array_knm(self) -> None:
        knm = np.array([["1", "2"], ["3", "4"]])
        with pytest.raises(ValueError, match="must be numeric"):
            doppler_term(np.zeros(2), knm)

    def test_rejects_non_finite_knm(self) -> None:
        knm = np.array([[0.0, np.inf], [np.inf, 0.0]])
        with pytest.raises(ValueError, match="NaN/Inf"):
            doppler_term(np.zeros(2), knm)

    def test_rejects_non_square_knm(self) -> None:
        with pytest.raises(ValueError, match=r"knm shape must be \(n, n\)"):
            doppler_term(np.zeros(2), np.zeros((2, 3)))

    def test_rejects_boolean_doppler_strength(self) -> None:
        with pytest.raises(ValueError, match="doppler_strength must be a finite real"):
            doppler_term(np.zeros(2), _KNM, doppler_strength=True)

    def test_rejects_non_numeric_doppler_strength(self) -> None:
        with pytest.raises(ValueError, match="doppler_strength must be a finite real"):
            doppler_term(np.zeros(2), _KNM, doppler_strength="fast")

    def test_rejects_non_positive_doppler_epsilon(self) -> None:
        with pytest.raises(ValueError, match="doppler_epsilon must be positive"):
            doppler_term(np.zeros(2), _KNM, doppler_epsilon=0.0)


class TestScalariseVelocities:
    """Velocity scalarisation guards for scalar, vector and axis projection."""

    def test_rejects_unsupported_velocity_shape(self) -> None:
        with pytest.raises(ValueError, match=r"shape \(n,\) or \(n, d\)"):
            scalarise_velocities(np.zeros((2, 3, 1)), n=2)

    def test_rejects_axis_shape_mismatch(self) -> None:
        velocities = np.zeros((2, 3))
        with pytest.raises(ValueError, match="velocity_axis shape"):
            scalarise_velocities(velocities, n=2, velocity_axis=np.zeros(2))

    def test_rejects_zero_norm_axis(self) -> None:
        velocities = np.zeros((2, 3))
        with pytest.raises(ValueError, match="non-zero finite norm"):
            scalarise_velocities(velocities, n=2, velocity_axis=np.zeros(3))


def _valid_backend_inputs(**overrides: object) -> dict[str, object]:
    """Return a self-consistent two-oscillator Doppler schedule contract."""
    inputs: dict[str, object] = {
        "phases": np.zeros(2),
        "omega_schedule": np.zeros((1, 2), dtype=np.float64),
        "knm": _KNM,
        "alpha": np.zeros((2, 2), dtype=np.float64),
        "velocity_schedule": np.zeros((1, 2), dtype=np.float64),
        "doppler_strength": 1.0,
        "doppler_epsilon": 1.0e-9,
        "zeta": 0.0,
        "psi": 0.0,
        "dt": 0.01,
        "method": "rk4",
        "n_substeps": 1,
        "atol": 1.0e-6,
        "rtol": 1.0e-3,
    }
    inputs.update(overrides)
    return inputs


class TestBackendInputContract:
    """Single-field corruption of the validated Doppler backend contract."""

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"phases": np.zeros((2, 2))}, "phases must be a non-empty vector"),
            ({"omega_schedule": np.zeros(2)}, "two-dimensional matrix"),
            ({"omega_schedule": np.zeros((0, 2))}, "at least one step"),
            (
                {"velocity_schedule": np.zeros((2, 2))},
                "velocity_schedule step count must match",
            ),
            ({"knm": np.zeros((3, 3))}, r"knm shape must be \(n, n\)"),
            ({"alpha": np.zeros((3, 3))}, r"alpha shape must be \(n, n\)"),
            ({"doppler_epsilon": 0.0}, "doppler_epsilon must be positive"),
            ({"dt": 0.0}, "dt must be positive"),
            ({"atol": 0.0}, "atol and rtol must be positive"),
            ({"method": "leapfrog"}, "method must be"),
            ({"n_substeps": True}, "n_substeps must be a positive integer"),
            ({"n_substeps": 1.5}, "n_substeps must be a positive integer"),
            ({"n_substeps": 0}, "n_substeps must be a positive integer"),
        ],
    )
    def test_rejects_corrupt_contract_field(
        self, overrides: dict[str, object], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            validate_doppler_backend_inputs(**_valid_backend_inputs(**overrides))


def _run_args() -> tuple[object, ...]:
    return (
        np.zeros(2),
        np.zeros((1, 2)),
        _KNM,
        np.zeros((2, 2)),
        np.zeros((1, 2)),
    )


class TestDopplerRunDispatch:
    """Backend-dispatch guards in the multi-backend Doppler runner."""

    def test_rejects_unavailable_named_backend(self) -> None:
        with pytest.raises(ImportError, match="is not available"):
            doppler_run(*_run_args(), backend="nonexistent_backend")

    def test_rejects_backend_output_outside_principal_branch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _bad(*_args: object, **_kwargs: object) -> np.ndarray:
            return np.array([-1.0, 1.0], dtype=np.float64)

        monkeypatch.setattr(doppler_module, "_backend_map", lambda: {"python": _bad})
        with pytest.raises(ValueError, match=r"phases must be in \[0, 2\*pi\)"):
            doppler_run(*_run_args(), backend="python")

    def test_named_backend_failure_propagates_last_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _failing(*_args: object, **_kwargs: object) -> np.ndarray:
            raise ImportError("simulated backend runtime failure")

        monkeypatch.setattr(
            doppler_module,
            "_backend_map",
            lambda: {"python": doppler_run_python, "go": _failing},
        )
        with pytest.raises(ImportError, match="simulated backend runtime failure"):
            doppler_run(*_run_args(), backend="go")

    def test_auto_dispatch_falls_through_to_python_when_others_fail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _failing(*_args: object, **_kwargs: object) -> np.ndarray:
            raise ImportError("simulated backend runtime failure")

        monkeypatch.setattr(doppler_module, "_backend_map", lambda: {"go": _failing})
        out = doppler_run(*_run_args(), backend="auto")

        assert out.shape == (2,)
        assert np.all(np.isfinite(out))

    def test_rust_backend_runs_end_to_end(self) -> None:
        if "rust" not in doppler_module._backend_map():
            pytest.skip("rust backend not built in this environment")
        out = doppler_run(*_run_args(), backend="rust")

        assert out.shape == (2,)
        assert np.all(np.isfinite(out))


def _engine(**overrides: object) -> DopplerEngine:
    params: dict[str, object] = {
        "n": 2,
        "omega": np.zeros(2),
        "k_nm": _KNM,
        "alpha": 0.0,
        "dt": 0.01,
        "velocities": np.array([1.0, -1.0], dtype=np.float64),
        "solver": "euler",
    }
    params.update(overrides)
    return DopplerEngine(**params)


class TestDopplerEngine:
    def test_requires_velocities(self) -> None:
        with pytest.raises(ValueError, match="velocities are required"):
            _engine(velocities=None)

    def test_rejects_phases_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match=r"phases shape must be \(n,\)"):
            _engine(phases=np.zeros(3))

    def test_run_rejects_omega_shape_mismatch(self) -> None:
        engine = _engine()
        with pytest.raises(ValueError, match=r"omegas shape must be \(n,\)"):
            engine.run(omegas=np.zeros(3))

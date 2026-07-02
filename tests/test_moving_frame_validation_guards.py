# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — moving-frame contract and backend-dispatch guards

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import scpn_phase_orchestrator.upde.moving_frame as moving_frame_module
from scpn_phase_orchestrator.coupling import SpatialCouplingModulator
from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _moving_frame_go as moving_frame_go_module,
)
from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _moving_frame_julia as moving_frame_julia_module,
)
from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _moving_frame_mojo as moving_frame_mojo_module,
)
from scpn_phase_orchestrator.upde.moving_frame import (
    MovingFrameUPDEEngine,
    _validate_backend_output,
    moving_frame_run,
    moving_frame_run_python,
    validate_moving_frame_backend_inputs,
)


def _valid_inputs(**overrides: object) -> dict[str, object]:
    """Return a self-consistent two-oscillator backend-input contract."""
    inputs: dict[str, object] = {
        "phases": np.zeros(2),
        "positions": np.array([-1.0, 1.0], dtype=np.float64),
        "omega_schedule": np.zeros((1, 2), dtype=np.float64),
        "knm": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
        "alpha": np.zeros((2, 2), dtype=np.float64),
        "velocity_schedule": np.zeros((1, 2), dtype=np.float64),
        "spatial_k_base": 1.0,
        "spatial_decay_form": "exponential",
        "spatial_decay_exponent": 1.0,
        "spatial_decay_length_scale": 1.0,
        "spatial_epsilon": 1.0e-12,
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


def _direct_backend_args() -> tuple[object, ...]:
    """Return a valid direct moving-frame backend call payload."""
    inputs = _valid_inputs(spatial_decay_form=1)
    return (
        inputs["phases"],
        inputs["positions"],
        inputs["omega_schedule"],
        inputs["knm"],
        inputs["alpha"],
        inputs["velocity_schedule"],
        inputs["spatial_k_base"],
        inputs["spatial_decay_form"],
        inputs["spatial_decay_exponent"],
        inputs["spatial_decay_length_scale"],
        inputs["spatial_epsilon"],
        inputs["doppler_strength"],
        inputs["doppler_epsilon"],
        inputs["zeta"],
        inputs["psi"],
        inputs["dt"],
        inputs["method"],
        inputs["n_substeps"],
        inputs["atol"],
        inputs["rtol"],
    )


class _CorruptMovingFrameGoSymbol:
    """Fake Go symbol that writes an invalid position into the output buffer."""

    restype: object
    argtypes: object

    def __call__(
        self, _phases_ptr: object, positions_ptr: object, *_args: object
    ) -> int:
        cast(Any, positions_ptr)[0] = 99.0
        return 0


class _CorruptMovingFrameGoLibrary:
    """Fake Go library exposing the moving-frame schedule symbol."""

    def __init__(self) -> None:
        self.UPDERunMovingFrameSchedule = _CorruptMovingFrameGoSymbol()


class _FailingMovingFrameGoSymbol:
    """Fake Go symbol returning a non-zero moving-frame backend status."""

    restype: object
    argtypes: object

    def __call__(self, *_args: object) -> int:
        return 9


class _FailingMovingFrameGoLibrary:
    """Fake Go library exposing a failing moving-frame schedule symbol."""

    def __init__(self) -> None:
        self.UPDERunMovingFrameSchedule = _FailingMovingFrameGoSymbol()


class TestBackendInputContract:
    """Single-field corruption of the validated backend-input contract."""

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"spatial_decay_form": "warp"}, "spatial_decay_form must be one of"),
            ({"spatial_decay_form": True}, "decay name or code"),
            ({"spatial_decay_form": 1.5}, "code must be 0, 1, 2, or 3"),
            ({"spatial_decay_form": 5}, "code must be 0, 1, 2, or 3"),
            (
                {"spatial_decay_exponent": 0.0},
                "spatial_decay_exponent must be positive",
            ),
            ({"phases": np.zeros((2, 2))}, "phases must be a non-empty vector"),
            (
                {"velocity_schedule": np.zeros((2, 2))},
                "velocity_schedule step count must match",
            ),
        ],
    )
    def test_rejects_corrupt_contract_field(
        self, overrides: dict[str, object], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            validate_moving_frame_backend_inputs(**_valid_inputs(**overrides))


class TestSpatialDecayWeights:
    """Each spatial-decay law produces a finite modulated coupling field."""

    @pytest.mark.parametrize("decay_code", [1, 2, 3])
    def test_each_decay_law_runs_through_python_backend(self, decay_code: int) -> None:
        out = moving_frame_run_python(**_valid_inputs(spatial_decay_form=decay_code))

        assert out.shape == (4,)
        assert np.all(np.isfinite(out))


class TestBackendOutputContract:
    """``_validate_backend_output`` guards untrusted compute-backend output."""

    def test_rejects_wrong_length(self) -> None:
        with pytest.raises(ValueError, match=r"shape must be \(2\*n,\)"):
            _validate_backend_output(np.zeros(3), n=2)

    def test_rejects_phases_outside_principal_branch(self) -> None:
        value = np.array([-1.0, 1.0, 0.0, 0.0], dtype=np.float64)
        with pytest.raises(ValueError, match=r"phases must be in \[0, 2\*pi\)"):
            _validate_backend_output(value, n=2)

    def test_rejects_expected_positions_shape_mismatch(self) -> None:
        value = np.array([0.1, 0.2, 0.0, 0.0], dtype=np.float64)
        with pytest.raises(ValueError, match="expected_positions must match"):
            _validate_backend_output(value, n=2, expected_positions=np.zeros(3))

    def test_rejects_ballistic_kinematics_violation(self) -> None:
        value = np.array([0.1, 0.2, 5.0, 6.0], dtype=np.float64)
        with pytest.raises(ValueError, match="violate ballistic kinematics"):
            _validate_backend_output(value, n=2, expected_positions=np.zeros(2))


class TestDirectMovingFrameAdapterOutputContracts:
    """Direct polyglot adapters validate untrusted moving-frame outputs."""

    def test_go_adapter_rejects_kinematic_output_violation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            moving_frame_go_module,
            "_load_lib",
            lambda: _CorruptMovingFrameGoLibrary(),
        )

        with pytest.raises(ValueError, match="violate ballistic kinematics"):
            moving_frame_go_module.moving_frame_run_go(*_direct_backend_args())

    def test_go_adapter_rejects_missing_symbol_and_nonzero_status(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            moving_frame_go_module,
            "_load_lib",
            lambda: SimpleNamespace(),
        )
        with pytest.raises(ImportError, match="UPDERunMovingFrameSchedule"):
            moving_frame_go_module.moving_frame_run_go(*_direct_backend_args())

        monkeypatch.setattr(
            moving_frame_go_module,
            "_load_lib",
            lambda: _FailingMovingFrameGoLibrary(),
        )
        with pytest.raises(ValueError, match="rc=9"):
            moving_frame_go_module.moving_frame_run_go(*_direct_backend_args())

    def test_julia_adapter_rejects_kinematic_output_violation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            moving_frame_julia_module,
            "_ensure",
            lambda: SimpleNamespace(
                upde_run_moving_frame_schedule=lambda *_args: np.array(
                    [0.0, 0.0, 99.0, 1.0], dtype=np.float64
                )
            ),
        )

        with pytest.raises(ValueError, match="violate ballistic kinematics"):
            moving_frame_julia_module.moving_frame_run_julia(*_direct_backend_args())

    def test_julia_adapter_rejects_missing_schedule_symbol(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            moving_frame_julia_module,
            "_ensure",
            lambda: SimpleNamespace(),
        )

        with pytest.raises(ImportError, match="upde_run_moving_frame_schedule"):
            moving_frame_julia_module.moving_frame_run_julia(*_direct_backend_args())

    def test_mojo_adapter_rejects_kinematic_output_violation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            moving_frame_mojo_module,
            "_run",
            lambda *_args, **_kwargs: [0.0, 0.0, 99.0, 1.0],
        )

        with pytest.raises(ValueError, match="violate ballistic kinematics"):
            moving_frame_mojo_module.moving_frame_run_mojo(*_direct_backend_args())


class TestMovingFrameRunDispatch:
    """Dispatch-level guards in the multi-backend moving-frame runner."""

    def test_rejects_custom_distance_kernel_modulator(self) -> None:
        modulator = SpatialCouplingModulator(
            K_base=1.0,
            distance_fn=lambda left, right: np.abs(
                np.asarray(left, float) - np.asarray(right, float)
            ).reshape(2, 2),
        )
        with pytest.raises(ValueError, match="default axial SpatialCouplingModulator"):
            moving_frame_run(
                np.zeros(2),
                np.array([-1.0, 1.0]),
                np.zeros((1, 2)),
                np.array([[0.0, 1.0], [1.0, 0.0]]),
                np.zeros((2, 2)),
                np.zeros((1, 2)),
                modulator,
                backend="python",
            )

    def test_rejects_unavailable_named_backend(self) -> None:
        with pytest.raises(ImportError, match="is not available"):
            moving_frame_run(
                np.zeros(2),
                np.array([-1.0, 1.0]),
                np.zeros((1, 2)),
                np.array([[0.0, 1.0], [1.0, 0.0]]),
                np.zeros((2, 2)),
                np.zeros((1, 2)),
                SpatialCouplingModulator(K_base=1.0),
                backend="nonexistent_backend",
            )

    def test_named_backend_failure_propagates_last_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _failing(*_args: object, **_kwargs: object) -> np.ndarray:
            raise ImportError("simulated backend runtime failure")

        monkeypatch.setattr(
            moving_frame_module,
            "_backend_map",
            lambda: {"python": moving_frame_run_python, "go": _failing},
        )
        with pytest.raises(ImportError, match="simulated backend runtime failure"):
            moving_frame_run(
                np.zeros(2),
                np.array([-1.0, 1.0]),
                np.zeros((1, 2)),
                np.array([[0.0, 1.0], [1.0, 0.0]]),
                np.zeros((2, 2)),
                np.zeros((1, 2)),
                SpatialCouplingModulator(K_base=1.0),
                backend="go",
            )

    def test_auto_dispatch_falls_through_to_python_when_others_fail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _failing(*_args: object, **_kwargs: object) -> np.ndarray:
            raise ImportError("simulated backend runtime failure")

        monkeypatch.setattr(
            moving_frame_module,
            "_backend_map",
            lambda: {"go": _failing},
        )
        out = moving_frame_run(
            np.zeros(2),
            np.array([-1.0, 1.0]),
            np.zeros((1, 2)),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.zeros((2, 2)),
            np.zeros((1, 2)),
            SpatialCouplingModulator(K_base=1.0),
            backend="auto",
        )

        assert out.shape == (4,)
        assert np.all(np.isfinite(out))

    def test_rust_backend_runs_end_to_end(self) -> None:
        if "rust" not in moving_frame_module._backend_map():
            pytest.skip("rust backend not built in this environment")
        out = moving_frame_run(
            np.zeros(2),
            np.array([-1.0, 1.0]),
            np.zeros((1, 2)),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.zeros((2, 2)),
            np.zeros((1, 2)),
            SpatialCouplingModulator(K_base=1.0),
            backend="rust",
        )

        assert out.shape == (4,)
        assert np.all(np.isfinite(out))


def _engine(**overrides: object) -> MovingFrameUPDEEngine:
    params: dict[str, object] = {
        "n": 2,
        "omega": np.zeros(2),
        "k_nm": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
        "alpha": 0.0,
        "dt": 0.01,
        "positions_t0": np.array([-1.0, 1.0], dtype=np.float64),
        "velocities": np.array([1.0, -1.0], dtype=np.float64),
        "spatial_modulator": SpatialCouplingModulator(K_base=1.0),
        "solver": "euler",
    }
    params.update(overrides)
    return MovingFrameUPDEEngine(**params)


class TestMovingFrameEngine:
    def test_requires_positions_t0(self) -> None:
        with pytest.raises(ValueError, match="positions_t0 is required"):
            _engine(positions_t0=None)

    def test_requires_spatial_modulator(self) -> None:
        with pytest.raises(ValueError, match="spatial_modulator is required"):
            _engine(spatial_modulator=None)

    def test_diagnostic_properties_expose_state(self) -> None:
        engine = _engine()

        assert engine.knm_effective.shape == (2, 2)
        assert engine.max_abs_velocity_m_per_s == pytest.approx(1.0)
        assert engine.path_length_max_m == pytest.approx(0.0)

    def test_custom_distance_kernel_modulator_is_applied_at_construction(self) -> None:
        modulator = SpatialCouplingModulator(
            K_base=1.0,
            distance_fn=lambda left, right: np.abs(
                np.asarray(left, float) - np.asarray(right, float)
            ).reshape(2, 2),
        )
        engine = _engine(spatial_modulator=modulator)

        assert engine.knm_effective.shape == (2, 2)
        np.testing.assert_array_equal(np.diag(engine.knm_effective), np.zeros(2))

    def test_single_step_advances_positions_ballistically(self) -> None:
        engine = _engine()

        engine.step()

        np.testing.assert_allclose(
            engine.positions, np.array([-1.0 + 0.01, 1.0 - 0.01]), atol=1.0e-12
        )
        assert engine.kinematic_residual_max_m <= 1.0e-12
        assert engine.max_abs_velocity_m_per_s == pytest.approx(1.0)
        assert engine.path_length_max_m == pytest.approx(0.01)

    def test_single_step_run_uses_initial_positions_as_last_start(self) -> None:
        engine = _engine()

        engine.run(n_steps=1)

        assert engine.knm_effective.shape == (2, 2)
        assert engine.time == pytest.approx(0.01)

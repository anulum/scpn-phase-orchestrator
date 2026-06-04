# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Šotek. All rights reserved.
# © Code 2020-2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — moving-frame UPDE tests

"""Behavioural tests for the PHA-C.3 moving-frame UPDE contract."""

from __future__ import annotations

import numpy as np
import pytest

import scpn_phase_orchestrator.upde.engine as engine_module
import scpn_phase_orchestrator.upde.moving_frame as moving_frame_module
from benchmarks.upde_moving_frame_benchmark import (
    benchmark_upde_moving_frame_polyglot_gate,
)
from scpn_phase_orchestrator.coupling import SpatialCouplingModulator
from scpn_phase_orchestrator.upde import MovingFrameUPDEEngine as ExportedMovingFrame
from scpn_phase_orchestrator.upde._ref_kernel import upde_run_omega_schedule_python
from scpn_phase_orchestrator.upde.doppler import doppler_term
from scpn_phase_orchestrator.upde.moving_frame import (
    MovingFrameUPDEEngine,
    moving_frame_run_python,
)


def _force_python(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_module, "ACTIVE_BACKEND", "python")
    monkeypatch.setattr(
        moving_frame_module,
        "_backend_map",
        lambda: {"python": moving_frame_run_python},
    )


def _two_body_knm(k: float = 1.0) -> np.ndarray:
    return np.array([[0.0, k], [k, 0.0]], dtype=np.float64)


def _zero_alpha(n: int = 2) -> np.ndarray:
    return np.zeros((n, n), dtype=np.float64)


def test_public_lazy_export_exposes_moving_frame_engine() -> None:
    assert ExportedMovingFrame is MovingFrameUPDEEngine


def test_zero_coupling_position_update_is_ballistic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_python(monkeypatch)
    engine = MovingFrameUPDEEngine(
        2,
        omega=np.zeros(2),
        k_nm=np.zeros((2, 2), dtype=np.float64),
        alpha=0.0,
        dt=0.25,
        positions_t0=np.zeros(2),
        velocities=np.array([2.0, -3.0], dtype=np.float64),
        spatial_modulator=SpatialCouplingModulator(K_base=1.0),
        solver="euler",
    )

    phases = engine.run(n_steps=4)

    np.testing.assert_allclose(phases, np.zeros(2), atol=1.0e-12)
    np.testing.assert_allclose(engine.positions, np.array([2.0, -3.0]), atol=1.0e-12)
    np.testing.assert_allclose(engine.distance_to_reference, np.array([2.0, 3.0]))
    assert engine.time == pytest.approx(1.0)


def test_collision_predicate_detects_near_exact_and_crossing_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_python(monkeypatch)
    engine = MovingFrameUPDEEngine(
        2,
        omega=np.zeros(2),
        k_nm=np.zeros((2, 2), dtype=np.float64),
        dt=1.0e-3,
        positions_t0=np.array([-1.0e-3, 1.0e-3], dtype=np.float64),
        velocities=np.array([1.0, -1.0], dtype=np.float64),
        spatial_modulator=SpatialCouplingModulator(K_base=1.0),
        solver="euler",
    )

    assert engine.collision_imminent(threshold_m=0.0)
    assert engine.collision_imminent(threshold_m=1.0e-6)

    away = MovingFrameUPDEEngine(
        2,
        omega=np.zeros(2),
        k_nm=np.zeros((2, 2), dtype=np.float64),
        dt=1.0e-3,
        positions_t0=np.array([2.0e-3, 3.0e-3], dtype=np.float64),
        velocities=np.array([1.0, 1.0], dtype=np.float64),
        spatial_modulator=SpatialCouplingModulator(K_base=1.0),
        solver="euler",
    )
    assert not away.collision_imminent(threshold_m=5.0e-4)
    with pytest.raises(ValueError, match="threshold_m"):
        away.collision_imminent(threshold_m=-1.0)


def test_two_body_merger_matches_manual_rk45_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_python(monkeypatch)
    phases = np.array([0.2, 0.4], dtype=np.float64)
    positions = np.array([-1.0e-6, 1.0e-6], dtype=np.float64)
    velocities = np.array([500.0, -500.0], dtype=np.float64)
    omega = np.array([0.1, -0.1], dtype=np.float64)
    knm = _two_body_knm(k=0.3)
    alpha = _zero_alpha()
    dt = 1.0e-9
    steps = 2
    modulator = SpatialCouplingModulator(K_base=0.8)
    engine = MovingFrameUPDEEngine(
        2,
        omega=omega,
        k_nm=knm,
        alpha=alpha,
        dt=dt,
        positions_t0=positions,
        velocities=velocities,
        spatial_modulator=modulator,
        doppler_strength=0.01,
        doppler_epsilon=1.0e-9,
        solver="rk45",
        phases=phases,
    )

    got = engine.run(n_steps=steps)

    manual_phases = phases.copy()
    manual_positions = positions.copy()
    for _ in range(steps):
        k_effective = modulator.modulate(knm, manual_positions.reshape(-1, 1))
        correction = doppler_term(
            velocities,
            k_effective,
            doppler_strength=0.01,
            doppler_epsilon=1.0e-9,
        )
        manual_phases = upde_run_omega_schedule_python(
            manual_phases,
            (omega + correction).reshape(1, -1),
            k_effective,
            alpha,
            0.0,
            0.0,
            dt,
            "rk45",
            1,
            1.0e-6,
            1.0e-3,
        )
        manual_positions = manual_positions + velocities * dt

    np.testing.assert_allclose(got, manual_phases, rtol=1.0e-6, atol=1.0e-12)
    np.testing.assert_allclose(engine.positions, manual_positions, atol=1.0e-15)
    np.testing.assert_allclose(engine.positions, np.zeros(2), atol=1.0e-15)


def test_stationary_positions_match_doppler_schedule_with_static_spatial_knm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_python(monkeypatch)
    phases = np.array([0.1, 0.3, 0.8], dtype=np.float64)
    positions = np.array([-1.0, 0.0, 2.0], dtype=np.float64)
    omega_schedule = np.array([[0.2, 0.1, -0.1], [0.3, 0.0, 0.2]], dtype=np.float64)
    velocity_schedule = np.zeros((2, 3), dtype=np.float64)
    knm = np.array(
        [[0.0, 0.2, 0.1], [0.2, 0.0, 0.4], [0.1, 0.4, 0.0]], dtype=np.float64
    )
    alpha = np.zeros((3, 3), dtype=np.float64)
    modulator = SpatialCouplingModulator(K_base=0.5)

    flat = moving_frame_run_python(
        phases,
        positions,
        omega_schedule,
        knm,
        alpha,
        velocity_schedule,
        modulator.K_base,
        modulator.decay_form,
        modulator.decay_exponent,
        modulator.decay_length_scale,
        modulator.epsilon,
        0.1,
        1.0e-9,
        0.0,
        0.0,
        0.01,
        "rk4",
        1,
        1.0e-6,
        1.0e-3,
    )

    static_knm = modulator.modulate(knm, positions.reshape(-1, 1))
    expected = upde_run_omega_schedule_python(
        phases,
        omega_schedule,
        static_knm,
        alpha,
        0.0,
        0.0,
        0.01,
        "rk4",
        1,
        1.0e-6,
        1.0e-3,
    )
    np.testing.assert_allclose(flat[:3], expected, atol=1.0e-12)
    np.testing.assert_allclose(flat[3:], positions, atol=1.0e-12)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"positions_t0": np.array([True, False])}, "positions"),
        ({"positions_t0": np.array([1.0, 2.0, 3.0])}, "positions"),
        ({"reference_point": float("nan")}, "reference_point"),
        ({"spatial_modulator": object()}, "spatial_modulator"),
    ],
)
def test_moving_frame_engine_invalid_boundaries_fail_closed(
    kwargs: dict[str, object],
    match: str,
) -> None:
    params: dict[str, object] = {
        "n": 2,
        "omega": np.zeros(2),
        "k_nm": _two_body_knm(),
        "alpha": 0.0,
        "dt": 0.01,
        "positions_t0": np.array([-1.0, 1.0], dtype=np.float64),
        "velocities": np.array([1.0, -1.0], dtype=np.float64),
        "spatial_modulator": SpatialCouplingModulator(K_base=1.0),
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match=match):
        MovingFrameUPDEEngine(**params)


def test_moving_frame_polyglot_benchmark_reports_available_language_slots() -> None:
    out = benchmark_upde_moving_frame_polyglot_gate(n=4, n_steps=3, calls=1, seed=11)

    assert out["suite"] == "upde_moving_frame_polyglot_gate"
    assert out["acceptance_passed"] == 1
    assert out["all_available_passed"] == 1
    assert out["parity_pass_count"] >= 1

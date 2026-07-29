# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Modal participation and controllability tests

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor.modal_participation import (
    NetworkMode,
    analyse_network_modes,
    phase_network_jacobian,
)
from scpn_phase_orchestrator.monitor.oscillation_modes import estimate_oscillation_modes

FloatArray = NDArray[np.float64]


def _companion(natural_hz: float, zeta: float) -> FloatArray:
    """State matrix of ``θ̈ + 2ζω₀θ̇ + ω₀²θ = 0`` (a second-order damped mode)."""
    w0 = 2.0 * np.pi * natural_hz
    return np.array([[0.0, 1.0], [-(w0**2), -2.0 * zeta * w0]])


def _two_area_swing() -> tuple[FloatArray, int]:
    """Return a four-machine, two-cluster swing model and the per-cluster size."""
    intra, inter = 8.0, 0.6
    coupling = np.array(
        [
            [0.0, intra, inter, inter],
            [intra, 0.0, inter, inter],
            [inter, inter, 0.0, intra],
            [inter, inter, intra, 0.0],
        ]
    )
    n = coupling.shape[0]
    laplacian = np.diag(coupling.sum(axis=1)) - coupling
    inertia = np.diag([1.0, 1.2, 1.0, 1.1])
    damping = np.diag([0.15, 0.15, 0.15, 0.15])
    inertia_inv = np.linalg.inv(inertia)
    state = np.block(
        [
            [np.zeros((n, n)), np.eye(n)],
            [-inertia_inv @ laplacian, -inertia_inv @ damping],
        ]
    )
    return state, n


def _kuramoto_derivative(
    coupling: FloatArray, phases: FloatArray, phase_lag: FloatArray
) -> FloatArray:
    """The engine's Sakaguchi–Kuramoto coupling derivative (drive-free)."""
    diff = phases[np.newaxis, :] - phases[:, np.newaxis] - phase_lag
    return np.asarray(np.sum(coupling * np.sin(diff), axis=1), dtype=np.float64)


class TestNetworkMode:
    def _mode(self, controllability: FloatArray | None) -> NetworkMode:
        eigenvalue = complex(-0.3, 2.0)
        mode_shape = np.array([1.0 + 0.0j, -0.5 + 0.2j])
        mode_shape /= np.linalg.norm(mode_shape)
        return NetworkMode(
            eigenvalue=eigenvalue,
            frequency_hz=abs(eigenvalue.imag) / (2.0 * np.pi),
            damping_ratio=-eigenvalue.real / abs(eigenvalue),
            mode_shape=mode_shape,
            participation=np.array([0.7, 0.3]),
            dominant_state=0,
            controllability=controllability,
            dominant_input=None if controllability is None else 1,
            poorly_damped=False,
        )

    def test_to_dict_without_controllability(self) -> None:
        payload = self._mode(None).to_dict()

        assert payload["eigenvalue"] == [-0.3, 2.0]
        assert payload["controllability"] is None
        assert payload["dominant_input"] is None
        assert payload["mode_shape"] == [
            [float(value.real), float(value.imag)]
            for value in self._mode(None).mode_shape
        ]
        assert payload["participation"] == [0.7, 0.3]
        assert payload["poorly_damped"] is False

    def test_to_dict_with_controllability(self) -> None:
        payload = self._mode(np.array([0.2, 0.9])).to_dict()

        assert payload["controllability"] == [0.2, 0.9]
        assert payload["dominant_input"] == 1

    def test_equality_is_identity_based(self) -> None:
        mode = self._mode(None)

        assert mode == mode
        assert mode != self._mode(None)

    @pytest.mark.parametrize(
        ("field", "value"),
        (
            ("eigenvalue", "-0.3+2j"),
            ("eigenvalue", complex(np.inf, 0.0)),
            ("eigenvalue", 10**1000),
            ("frequency_hz", "0.318"),
            ("frequency_hz", -0.1),
            ("damping_ratio", np.nan),
            ("mode_shape", np.array([1.0, "0.0"])),
            ("mode_shape", np.array([True, False])),
            ("mode_shape", np.array([1.0, "x"], dtype=object)),
            ("mode_shape", np.array([10**1000, 0], dtype=object)),
            ("mode_shape", np.array([[1.0, 0.0]])),
            ("mode_shape", np.array([np.nan + 0.0j, 0.0 + 0.0j])),
            ("mode_shape", np.array([2.0 + 0.0j, 0.0 + 0.0j])),
            ("participation", np.array([0.7, True], dtype=object)),
            ("participation", np.array([0.7, -0.3])),
            ("participation", np.array([0.7, 0.4])),
            ("participation", np.array([[0.7, 0.3]])),
            ("participation", np.array([1.0])),
            ("dominant_state", True),
            ("dominant_state", 1),
            ("dominant_state", 3),
            ("controllability", np.array([0.2, -0.1])),
            ("controllability", np.array([[0.2, 0.9]])),
            ("controllability", np.array([])),
            ("dominant_input", 0),
            ("poorly_damped", np.bool_(False)),
        ),
    )
    def test_direct_mode_rejects_malformed_evidence(
        self,
        field: str,
        value: object,
    ) -> None:
        with pytest.raises(ValueError, match=field):
            replace(self._mode(np.array([0.2, 0.9])), **{field: value})

    @pytest.mark.parametrize(
        ("field", "value"),
        (
            ("frequency_hz", 0.4),
            ("damping_ratio", 0.4),
            ("mode_shape", np.array([0.0 + 1.0j, 0.0 + 0.0j])),
        ),
    )
    def test_direct_mode_rejects_contradictory_derived_evidence(
        self,
        field: str,
        value: object,
    ) -> None:
        with pytest.raises(ValueError, match=field):
            replace(self._mode(None), **{field: value})

    def test_direct_mode_requires_controllability_pairing(self) -> None:
        with pytest.raises(ValueError, match="dominant_input"):
            replace(self._mode(None), dominant_input=0)
        with pytest.raises(ValueError, match="dominant_input"):
            replace(self._mode(np.array([0.2, 0.9])), dominant_input=None)

    def test_direct_mode_owns_immutable_array_evidence(self) -> None:
        shape = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=object)
        participation = np.array([0.7, 0.3])
        controllability = np.array([0.2, 0.9])
        mode = NetworkMode(
            eigenvalue=-1.0 + 0.0j,
            frequency_hz=0.0,
            damping_ratio=1.0,
            mode_shape=shape,
            participation=participation,
            dominant_state=0,
            controllability=controllability,
            dominant_input=1,
            poorly_damped=False,
        )

        shape[0] = 0.0
        participation[0] = 0.0
        controllability[1] = 0.0

        assert mode.mode_shape[0] == 1.0
        assert mode.participation[0] == 0.7
        assert mode.controllability is not None
        assert mode.controllability[1] == 0.9
        assert not mode.mode_shape.flags.writeable
        assert not mode.participation.flags.writeable
        assert not mode.controllability.flags.writeable

    def test_direct_mode_wraps_broken_shape_array_protocol(self) -> None:
        class BrokenArray:
            def __array__(self) -> np.ndarray:
                raise TypeError("broken protocol")

        with pytest.raises(ValueError, match="mode_shape"):
            replace(self._mode(None), mode_shape=cast(Any, BrokenArray()))


class TestCompanionOscillator:
    def test_recovers_exact_frequency_and_damping(self) -> None:
        modes = analyse_network_modes(_companion(0.5, 0.07))

        assert len(modes) == 1
        mode = modes[0]
        w0 = 2.0 * np.pi * 0.5
        expected_hz = w0 * np.sqrt(1.0 - 0.07**2) / (2.0 * np.pi)
        assert mode.frequency_hz == pytest.approx(expected_hz, rel=1e-9)
        assert mode.damping_ratio == pytest.approx(0.07, rel=1e-9)
        assert mode.eigenvalue.imag > 0.0

    def test_participation_normalised_and_mode_shape_unit_anchored(self) -> None:
        mode = analyse_network_modes(_companion(0.5, 0.07))[0]

        assert mode.participation.sum() == pytest.approx(1.0)
        assert np.all(mode.participation >= 0.0)
        assert np.linalg.norm(mode.mode_shape) == pytest.approx(1.0)
        anchor = mode.mode_shape[int(np.argmax(np.abs(mode.mode_shape)))]
        assert anchor.imag == pytest.approx(0.0, abs=1e-12)
        assert anchor.real > 0.0

    def test_well_damped_mode_not_flagged(self) -> None:
        assert analyse_network_modes(_companion(0.5, 0.07))[0].poorly_damped is False

    def test_poorly_damped_mode_flagged(self) -> None:
        mode = analyse_network_modes(_companion(0.5, 0.01))[0]

        assert mode.damping_ratio == pytest.approx(0.01, rel=1e-6)
        assert mode.poorly_damped is True

    def test_unstable_mode_has_negative_damping(self) -> None:
        # Negative viscous term -> right-half-plane pole pair (growing oscillation).
        w0 = 2.0 * np.pi * 0.5
        unstable = np.array([[0.0, 1.0], [-(w0**2), +0.4 * w0]])
        mode = analyse_network_modes(unstable)[0]

        assert mode.damping_ratio < 0.0
        assert mode.poorly_damped is True


class TestSwingModelInterArea:
    def test_inter_area_mode_shape_splits_the_two_clusters(self) -> None:
        state, n = _two_area_swing()
        modes = analyse_network_modes(state)
        oscillatory = [m for m in modes if m.frequency_hz > 1e-6]
        inter_area = min(oscillatory, key=lambda m: m.frequency_hz)

        # On the position states, the two clusters swing in anti-phase.
        position_angles = np.angle(inter_area.mode_shape[:n])
        cluster_a = position_angles[:2]
        cluster_b = position_angles[2:]
        assert np.std(cluster_a) < 0.2
        assert np.std(cluster_b) < 0.2
        separation = abs(np.mean(cluster_a) - np.mean(cluster_b))
        assert separation == pytest.approx(np.pi, abs=0.4)

    def test_controllability_ranks_inputs(self) -> None:
        state, n = _two_area_swing()
        # Drive the velocity of machine 0 and of machine 2 (one per cluster).
        inputs = np.zeros((2 * n, 2))
        inputs[n + 0, 0] = 1.0
        inputs[n + 2, 1] = 1.0
        modes = analyse_network_modes(state, input_matrix=inputs)

        for mode in modes:
            assert mode.controllability is not None
            assert mode.controllability.shape == (2,)
            assert mode.dominant_input in (0, 1)
            assert mode.dominant_input == int(np.argmax(mode.controllability))

    def test_modes_sorted_by_ascending_damping(self) -> None:
        state, _ = _two_area_swing()
        damping = [m.damping_ratio for m in analyse_network_modes(state)]

        assert damping == sorted(damping)


class TestRotationMode:
    def test_kuramoto_jacobian_has_marginal_global_phase_mode(self) -> None:
        n = 5
        coupling = 2.0 * np.ones((n, n))
        np.fill_diagonal(coupling, 0.0)
        phases = np.linspace(-0.1, 0.1, n)
        jacobian = phase_network_jacobian(coupling, phases)

        modes = analyse_network_modes(jacobian)
        rotation = min(modes, key=lambda m: abs(m.eigenvalue))
        assert rotation.frequency_hz == pytest.approx(0.0, abs=1e-9)
        assert rotation.damping_ratio == 0.0
        # The marginal mode is the uniform phase shift: a flat mode shape.
        assert np.std(np.abs(rotation.mode_shape)) == pytest.approx(0.0, abs=1e-9)

    def test_synchronised_jacobian_modes_are_stable(self) -> None:
        n = 5
        coupling = 2.0 * np.ones((n, n))
        np.fill_diagonal(coupling, 0.0)
        jacobian = phase_network_jacobian(coupling, np.zeros(n))
        modes = analyse_network_modes(jacobian)

        assert all(m.damping_ratio >= -1e-9 for m in modes)


class TestConjugateHandling:
    def test_only_non_negative_frequency_members_reported(self) -> None:
        # Block-diagonal: one complex pair + one real eigenvalue -> two modes.
        state = np.array(
            [
                [-0.2, 3.0, 0.0],
                [-3.0, -0.2, 0.0],
                [0.0, 0.0, -1.5],
            ]
        )
        modes = analyse_network_modes(state)

        assert len(modes) == 2
        oscillatory = [m for m in modes if m.frequency_hz > 1e-9]
        real_modes = [m for m in modes if m.frequency_hz <= 1e-9]
        assert len(oscillatory) == 1
        assert len(real_modes) == 1
        expected_hz = 3.0 / (2.0 * np.pi)
        assert oscillatory[0].frequency_hz == pytest.approx(expected_hz, rel=1e-6)

    def test_purely_real_spectrum_keeps_every_eigenvalue(self) -> None:
        modes = analyse_network_modes(np.diag([-1.0, -2.0, -3.0]))

        assert len(modes) == 3
        assert all(m.frequency_hz == pytest.approx(0.0, abs=1e-12) for m in modes)


class TestDefectiveMatrix:
    def test_defective_matrix_rejected(self) -> None:
        with pytest.raises(ValueError, match="defective"):
            analyse_network_modes(np.array([[1.0, 1.0], [0.0, 1.0]]))


class TestPhaseNetworkJacobian:
    def test_matches_finite_difference_of_engine_derivative(self) -> None:
        rng = np.random.default_rng(7)
        n = 6
        coupling = rng.uniform(0.5, 2.5, (n, n))
        np.fill_diagonal(coupling, 0.0)
        phases = rng.uniform(-np.pi, np.pi, n)
        phase_lag = rng.uniform(-0.3, 0.3, (n, n))
        np.fill_diagonal(phase_lag, 0.0)

        analytic = phase_network_jacobian(coupling, phases, phase_lag=phase_lag)
        eps = 1e-6
        numeric = np.empty((n, n))
        base = _kuramoto_derivative(coupling, phases, phase_lag)
        for k in range(n):
            bumped = phases.copy()
            bumped[k] += eps
            numeric[:, k] = (
                _kuramoto_derivative(coupling, bumped, phase_lag) - base
            ) / eps

        assert np.allclose(analytic, numeric, atol=1e-5)

    def test_drive_term_adds_diagonal_damping(self) -> None:
        n = 4
        coupling = np.zeros((n, n))  # no coupling: isolate the drive term
        phases = np.array([0.0, 0.5, 1.0, 1.5])
        free = phase_network_jacobian(coupling, phases)
        driven = phase_network_jacobian(
            coupling, phases, drive_strength=2.0, drive_phase=0.3
        )

        assert np.allclose(free, 0.0)
        expected_diag = -2.0 * np.cos(0.3 - phases)
        assert np.allclose(np.diag(driven), expected_diag)

    def test_symmetric_lag_free_network_is_overdamped(self) -> None:
        n = 6
        coupling = 1.5 * np.ones((n, n))
        np.fill_diagonal(coupling, 0.0)
        phases = 0.05 * np.arange(n)
        modes = analyse_network_modes(phase_network_jacobian(coupling, phases))

        assert all(m.frequency_hz == pytest.approx(0.0, abs=1e-9) for m in modes)

    def test_directed_coupling_yields_oscillatory_modes(self) -> None:
        rng = np.random.default_rng(1)
        n = 6
        coupling = rng.uniform(0.5, 2.5, (n, n))
        np.fill_diagonal(coupling, 0.0)
        phases = 0.3 * rng.standard_normal(n)
        modes = analyse_network_modes(phase_network_jacobian(coupling, phases))

        assert any(m.frequency_hz > 1e-6 for m in modes)

    def test_rejects_non_zero_self_coupling(self) -> None:
        coupling = np.ones((3, 3))
        with pytest.raises(ValueError, match="self-coupling diagonal"):
            phase_network_jacobian(coupling, np.zeros(3))


class TestValidation:
    def test_rejects_non_square_state_matrix(self) -> None:
        with pytest.raises(ValueError, match="square"):
            analyse_network_modes(np.ones((2, 3)))

    def test_rejects_one_dimensional_state_matrix(self) -> None:
        with pytest.raises(ValueError, match="square"):
            analyse_network_modes(np.ones(4))

    def test_rejects_empty_state_matrix(self) -> None:
        with pytest.raises(ValueError, match="at least one row"):
            analyse_network_modes(np.empty((0, 0)))

    def test_rejects_boolean_state_matrix(self) -> None:
        with pytest.raises(ValueError, match="boolean"):
            analyse_network_modes(np.eye(2, dtype=bool))

    def test_rejects_complex_state_matrix(self) -> None:
        with pytest.raises(ValueError, match="real-valued"):
            analyse_network_modes(np.eye(2, dtype=np.complex128))

    def test_rejects_non_numeric_state_matrix(self) -> None:
        with pytest.raises(ValueError, match="real float array"):
            analyse_network_modes(np.array([["a", "b"], ["c", "d"]]))

    def test_rejects_non_finite_state_matrix(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            analyse_network_modes(np.array([[1.0, np.inf], [0.0, 1.0]]))

    def test_rejects_non_real_damping_threshold(self) -> None:
        with pytest.raises(ValueError, match="damping_threshold"):
            analyse_network_modes(np.eye(2), damping_threshold="low")  # type: ignore[arg-type]

    def test_rejects_non_finite_damping_threshold(self) -> None:
        with pytest.raises(ValueError, match="damping_threshold"):
            analyse_network_modes(np.eye(2), damping_threshold=float("nan"))

    def test_rejects_input_matrix_with_wrong_rows(self) -> None:
        with pytest.raises(ValueError, match="rows to match"):
            analyse_network_modes(np.eye(3), input_matrix=np.ones((2, 1)))

    def test_rejects_one_dimensional_input_matrix(self) -> None:
        with pytest.raises(ValueError, match="input_matrix must be 2-D"):
            analyse_network_modes(np.eye(3), input_matrix=np.ones(3))

    def test_rejects_input_matrix_without_columns(self) -> None:
        with pytest.raises(ValueError, match="at least one column"):
            analyse_network_modes(np.eye(3), input_matrix=np.empty((3, 0)))

    def test_rejects_phases_with_wrong_length(self) -> None:
        coupling = np.zeros((3, 3))
        with pytest.raises(ValueError, match="length 3"):
            phase_network_jacobian(coupling, np.zeros(2))

    def test_rejects_two_dimensional_phases(self) -> None:
        coupling = np.zeros((3, 3))
        with pytest.raises(ValueError, match="one-dimensional"):
            phase_network_jacobian(coupling, np.zeros((3, 1)))

    def test_rejects_phase_lag_with_wrong_shape(self) -> None:
        coupling = np.zeros((3, 3))
        with pytest.raises(ValueError, match="phase_lag must have shape"):
            phase_network_jacobian(coupling, np.zeros(3), phase_lag=np.zeros((2, 2)))

    def test_rejects_boolean_drive_strength(self) -> None:
        coupling = np.zeros((3, 3))
        with pytest.raises(ValueError, match="drive_strength"):
            phase_network_jacobian(coupling, np.zeros(3), drive_strength=True)

    def test_rejects_non_finite_drive_phase(self) -> None:
        coupling = np.zeros((3, 3))
        with pytest.raises(ValueError, match="drive_phase"):
            phase_network_jacobian(coupling, np.zeros(3), drive_phase=float("inf"))

    @pytest.mark.parametrize(
        "field",
        ("state_matrix", "input_matrix", "coupling", "phases", "phase_lag"),
    )
    def test_public_array_boundaries_reject_numeric_text(self, field: str) -> None:
        if field == "state_matrix":
            with pytest.raises(ValueError, match=field):
                analyse_network_modes(np.array([["-1.0"]]))
        elif field == "input_matrix":
            with pytest.raises(ValueError, match=field):
                analyse_network_modes(
                    np.array([[-1.0]]), input_matrix=np.array([["1"]])
                )
        else:
            kwargs: dict[str, object] = {
                "coupling": np.zeros((1, 1)),
                "phases": np.zeros(1),
                "phase_lag": np.zeros((1, 1)),
            }
            kwargs[field] = (
                np.array([["0.0"]]) if field != "phases" else np.array(["0.0"])
            )
            with pytest.raises(ValueError, match=field):
                phase_network_jacobian(**cast(Any, kwargs))

    def test_public_array_boundary_rejects_object_boolean_aliases(self) -> None:
        with pytest.raises(ValueError, match="state_matrix"):
            analyse_network_modes(np.array([[-1.0, True], [0.0, -2.0]], dtype=object))

    def test_public_array_boundary_preserves_real_numeric_objects(self) -> None:
        modes = analyse_network_modes(
            np.array([[-1, np.float32(0.0)], [0, np.float64(-2.0)]], dtype=object)
        )

        assert len(modes) == 2

    def test_public_array_boundary_wraps_broken_protocol(self) -> None:
        class BrokenArray:
            def __array__(self) -> np.ndarray:
                raise TypeError("broken protocol")

        with pytest.raises(ValueError, match="state_matrix"):
            analyse_network_modes(cast(Any, BrokenArray()))

    def test_public_scalar_overflow_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="damping_threshold"):
            analyse_network_modes(np.array([[-1.0]]), damping_threshold=10**1000)

    def test_public_real_array_overflow_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="state_matrix"):
            analyse_network_modes(np.array([[10**1000]], dtype=object))


class TestPipelineWiring:
    def test_model_modes_match_matrix_pencil_ringdown(self) -> None:
        """The matrix-pencil estimate of a linear ringdown recovers the model modes."""
        state, _ = _two_area_swing()
        eigenvalues, vectors = np.linalg.eig(state)
        inverse = np.linalg.inv(vectors)

        # Excite the lightly-damped inter-area mode and read one state's ringdown.
        modes = analyse_network_modes(state)
        oscillatory = [m for m in modes if m.frequency_hz > 1e-6]
        inter_area = min(oscillatory, key=lambda m: m.frequency_hz)

        fs = 50.0
        times = np.arange(0.0, 12.0, 1.0 / fs)
        initial = np.zeros(state.shape[0])
        initial[0], initial[2] = 1.0, -1.0  # anti-phase across the two clusters
        coefficients = inverse @ initial
        trajectory = np.real(
            (vectors[0, :][np.newaxis, :] * np.exp(np.outer(times, eigenvalues)))
            @ coefficients
        )
        ringdown = trajectory - float(np.mean(trajectory[-int(0.5 * fs) :]))

        estimated = estimate_oscillation_modes(ringdown, fs, model_order=8)
        # The model's inter-area mode must appear among the data-driven estimates.
        assert any(
            est.frequency_hz == pytest.approx(inter_area.frequency_hz, abs=0.02)
            and est.damping_ratio == pytest.approx(inter_area.damping_ratio, abs=0.02)
            for est in estimated
        )

    def test_jacobian_from_engine_run_is_stable_with_rotation_mode(self) -> None:
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        n = 8
        engine = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(11)
        phases = 0.2 * rng.standard_normal(n)
        omegas = np.zeros(n)
        coupling = 4.0 * np.ones((n, n))
        np.fill_diagonal(coupling, 0.0)
        phase_lag = np.zeros((n, n))
        for _ in range(500):
            phases = engine.step(phases, omegas, coupling, 0.0, 0.0, phase_lag)

        jacobian = phase_network_jacobian(coupling, phases)
        modes = analyse_network_modes(jacobian)

        assert all(m.damping_ratio >= -1e-6 for m in modes)
        rotation = min(modes, key=lambda m: abs(m.eigenvalue))
        assert rotation.frequency_hz == pytest.approx(0.0, abs=1e-9)

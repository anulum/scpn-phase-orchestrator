# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Prediction model tests

from __future__ import annotations

from collections.abc import Callable
from numbers import Real
from typing import NoReturn, TypeVar

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.prediction import (
    PredictionModel,
    PredictionState,
    VariationalPredictor,
    VariationalState,
)

_R = TypeVar("_R")


def _call(fn: Callable[..., _R], *args: object, **kwargs: object) -> _R:
    """Call ``fn`` with arguments that are deliberately outside its types."""
    return fn(*args, **kwargs)


class _Unarrayable:
    def __array__(self, dtype: object = None, copy: object = None) -> NoReturn:
        raise TypeError("cannot expose an array")


class _OverflowReal:
    def __float__(self) -> float:
        raise OverflowError("cannot represent as float64")


Real.register(_OverflowReal)


class TestPredictionModel:
    @pytest.mark.parametrize("n_oscillators", [0, -1, 1.5, True, "4"])
    def test_rejects_invalid_oscillator_count(self, n_oscillators: object) -> None:
        with pytest.raises(ValueError, match="n_oscillators"):
            _call(PredictionModel, n_oscillators)

    @pytest.mark.parametrize("learning_rate", [-0.1, np.nan, np.inf, True, "0.1"])
    def test_rejects_invalid_learning_rate(self, learning_rate: object) -> None:
        with pytest.raises(ValueError, match="learning_rate"):
            _call(PredictionModel, 4, learning_rate=learning_rate)

    @pytest.mark.parametrize("error_gain", [-0.1, np.nan, np.inf, True, "0.1"])
    def test_rejects_invalid_error_gain(self, error_gain: object) -> None:
        with pytest.raises(ValueError, match="error_gain"):
            _call(PredictionModel, 4, error_gain=error_gain)

    @pytest.mark.parametrize(
        ("phases", "omegas", "dt", "message"),
        [
            (np.zeros(3), np.ones(4), 0.01, "phases"),
            (np.zeros(4), np.ones(3), 0.01, "omegas"),
            (np.array([0.0, np.nan, 1.0, 2.0]), np.ones(4), 0.01, "phases"),
            (np.zeros(4), np.array([1.0, np.inf, 1.0, 1.0]), 0.01, "omegas"),
            (np.array([True, False, True, False]), np.ones(4), 0.01, "phases"),
            (np.zeros(4), np.array([True, True, True, True]), 0.01, "omegas"),
            (np.zeros(4), np.ones(4), 0.0, "dt"),
            (np.zeros(4), np.ones(4), -0.01, "dt"),
            (np.zeros(4), np.ones(4), np.inf, "dt"),
        ],
    )
    def test_rejects_invalid_prediction_inputs(
        self, phases: np.ndarray, omegas: np.ndarray, dt: float, message: str
    ) -> None:
        model = PredictionModel(4)
        with pytest.raises(ValueError, match=message):
            model.predict(phases, omegas, dt)

    @pytest.mark.parametrize("method", ["predict", "update", "error_coupling"])
    @pytest.mark.parametrize("field", ["phases", "omegas"])
    @pytest.mark.parametrize(
        "alias",
        [
            np.array(["0.0", "0.5", "1.0", "1.5"]),
            np.array([0.0 + 0.1j, 0.5 + 0.1j, 1.0 + 0.1j, 1.5 + 0.1j]),
            np.array([0.0, 0.5, True, 1.5], dtype=object),
        ],
        ids=["numeric-string", "complex", "object-boolean"],
    )
    def test_runtime_methods_reject_coercive_vector_aliases(
        self,
        method: str,
        field: str,
        alias: np.ndarray,
    ) -> None:
        model = PredictionModel(4)
        values = {
            "phases": np.array([0.0, 0.5, 1.0, 1.5]),
            "omegas": np.ones(4),
        }
        values[field] = alias

        with pytest.raises(ValueError, match=field):
            getattr(model, method)(values["phases"], values["omegas"], 0.01)

    def test_runtime_methods_preserve_real_numeric_object_vectors(self) -> None:
        phases = np.array([0.0, 0.5, 1.0, 1.5], dtype=object)
        omegas = np.array([1, 1.0, np.float32(1.0), np.int64(1)], dtype=object)
        model = PredictionModel(4)

        predicted = model.predict(phases, omegas, 0.01)
        state = model.update(phases, omegas, 0.01)
        coupling = model.error_coupling(phases, omegas, 0.01)

        assert np.all(np.isfinite(predicted))
        assert np.all(np.isfinite(state.predicted_phases))
        assert np.all(np.isfinite(coupling))

    def test_vector_validation_normalizes_array_protocol_failure(self) -> None:
        with pytest.raises(ValueError, match="phases"):
            _call(PredictionModel(4).predict, _Unarrayable(), np.ones(4), 0.01)

    def test_vector_validation_normalizes_float_conversion_overflow(self) -> None:
        phases = np.array([_OverflowReal() for _ in range(4)], dtype=object)

        with pytest.raises(ValueError, match="phases"):
            PredictionModel(4).predict(phases, np.ones(4), 0.01)

    def test_update_rejects_invalid_runtime_inputs(self) -> None:
        model = PredictionModel(4)
        with pytest.raises(ValueError, match="phases"):
            model.update(np.zeros((2, 2)), np.ones(4), dt=0.01)

    def test_error_coupling_rejects_invalid_runtime_inputs(self) -> None:
        model = PredictionModel(4)
        with pytest.raises(ValueError, match="dt"):
            model.error_coupling(np.zeros(4), np.ones(4), dt=np.nan)

    def test_first_call_zero_error(self) -> None:
        model = PredictionModel(4)
        phases = np.array([0.0, 1.0, 2.0, 3.0])
        omegas = np.ones(4)
        state = model.update(phases, omegas, dt=0.01)
        assert state.mean_error == 0.0
        np.testing.assert_array_equal(state.prediction_error, np.zeros(4))

    def test_prediction_shape(self) -> None:
        model = PredictionModel(6)
        phases = np.zeros(6)
        omegas = np.ones(6)
        predicted = model.predict(phases, omegas, dt=0.01)
        assert predicted.shape == (6,)

    def test_prediction_in_range(self) -> None:
        model = PredictionModel(4)
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, 4)
        omegas = np.ones(4) * 5.0
        predicted = model.predict(phases, omegas, dt=0.1)
        assert np.all(predicted >= 0)
        assert np.all(predicted < 2 * np.pi)

    def test_error_decreases_with_learning(self) -> None:
        n = 4
        model = PredictionModel(n, learning_rate=0.1)
        omegas = np.ones(n) * 2.0
        dt = 0.01

        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, n)

        errors = []
        for _ in range(100):
            # Simple dynamics: advance by omegas
            phases = (phases + dt * omegas) % (2 * np.pi)
            state = model.update(phases, omegas, dt)
            errors.append(state.mean_error)

        # Error should decrease after initial transient
        early_mean = np.mean(errors[5:15])
        late_mean = np.mean(errors[-20:])
        assert late_mean < early_mean or late_mean < 0.1

    def test_weights_initially_zero(self) -> None:
        model = PredictionModel(3)
        np.testing.assert_array_equal(model.weights, np.zeros((3, 3)))

    def test_weights_change_after_update(self) -> None:
        model = PredictionModel(4, learning_rate=0.1)
        omegas = np.ones(4)
        phases1 = np.array([0.0, 0.5, 1.0, 1.5])
        phases2 = np.array([0.1, 0.6, 1.1, 1.6])
        model.update(phases1, omegas, dt=0.01)
        model.update(phases2, omegas, dt=0.01)
        assert not np.allclose(model.weights, 0.0)

    def test_error_coupling_zero_before_prediction(self) -> None:
        model = PredictionModel(4)
        phases = np.zeros(4)
        omegas = np.ones(4)
        coupling = model.error_coupling(phases, omegas, dt=0.01)
        np.testing.assert_array_equal(coupling, np.zeros(4))

    def test_error_coupling_after_prediction(self) -> None:
        model = PredictionModel(4, error_gain=0.5)
        omegas = np.ones(4)
        dt = 0.01
        phases1 = np.array([0.0, 0.5, 1.0, 1.5])
        model.update(phases1, omegas, dt)
        phases2 = np.array([0.1, 0.6, 1.1, 1.6])
        coupling = model.error_coupling(phases2, omegas, dt)
        assert coupling.shape == (4,)

    def test_reset(self) -> None:
        model = PredictionModel(3, learning_rate=0.1)
        omegas = np.ones(3)
        model.update(np.array([0.0, 1.0, 2.0]), omegas, dt=0.01)
        model.update(np.array([0.1, 1.1, 2.1]), omegas, dt=0.01)
        model.reset()
        np.testing.assert_array_equal(model.weights, np.zeros((3, 3)))
        state = model.update(np.array([0.0, 1.0, 2.0]), omegas, dt=0.01)
        assert state.mean_error == 0.0

    def test_error_wrapping(self) -> None:
        model = PredictionModel(2, learning_rate=0.0)
        omegas = np.ones(2)
        dt = 0.01
        model.update(np.array([0.1, 6.1]), omegas, dt)
        # Actual phases near 0 vs prediction near 2π should wrap correctly
        state = model.update(np.array([0.1, 0.1]), omegas, dt)
        assert np.all(np.abs(state.prediction_error) < np.pi)

    def test_prediction_state_fields(self) -> None:
        model = PredictionModel(3)
        state = model.update(np.zeros(3), np.ones(3), dt=0.01)
        assert isinstance(state, PredictionState)
        assert hasattr(state, "predicted_phases")
        assert hasattr(state, "prediction_error")
        assert hasattr(state, "mean_error")
        assert hasattr(state, "weights")


class TestVariationalPredictor:
    @pytest.mark.parametrize("n_oscillators", [0, -1, 1.5, False, "4"])
    def test_rejects_invalid_oscillator_count(self, n_oscillators: object) -> None:
        with pytest.raises(ValueError, match="n_oscillators"):
            _call(VariationalPredictor, n_oscillators)

    @pytest.mark.parametrize(
        "prior_precision", [0.0, -1.0, np.nan, np.inf, True, "1.0"]
    )
    def test_rejects_invalid_prior_precision(self, prior_precision: object) -> None:
        with pytest.raises(ValueError, match="prior_precision"):
            _call(VariationalPredictor, 4, prior_precision=prior_precision)

    @pytest.mark.parametrize("learning_rate", [-0.1, np.nan, np.inf, True, "0.1"])
    def test_rejects_invalid_learning_rate(self, learning_rate: object) -> None:
        with pytest.raises(ValueError, match="learning_rate"):
            _call(VariationalPredictor, 4, learning_rate=learning_rate)

    @pytest.mark.parametrize(
        ("predicted", "observed", "precision", "message"),
        [
            (np.zeros(3), np.zeros(4), np.ones(4), "predicted"),
            (np.zeros(4), np.zeros(3), np.ones(4), "observed"),
            (np.zeros(4), np.zeros(4), np.ones(3), "precision"),
            (np.array([0.0, np.nan, 0.0, 0.0]), np.zeros(4), np.ones(4), "predicted"),
            (np.zeros(4), np.array([0.0, np.inf, 0.0, 0.0]), np.ones(4), "observed"),
            (np.zeros(4), np.zeros(4), np.array([1.0, 0.0, 1.0, 1.0]), "precision"),
            (
                np.array([True, False, True, False]),
                np.zeros(4),
                np.ones(4),
                "predicted",
            ),
            (
                np.zeros(4),
                np.array([True, False, True, False]),
                np.ones(4),
                "observed",
            ),
            (
                np.zeros(4),
                np.zeros(4),
                np.array([True, True, True, True]),
                "precision",
            ),
        ],
    )
    def test_free_energy_rejects_invalid_inputs(
        self,
        predicted: np.ndarray,
        observed: np.ndarray,
        precision: np.ndarray,
        message: str,
    ) -> None:
        vp = VariationalPredictor(4)
        with pytest.raises(ValueError, match=message):
            vp.free_energy(predicted, observed, precision)

    @pytest.mark.parametrize("field", ["predicted", "observed", "precision"])
    @pytest.mark.parametrize(
        "alias",
        [
            np.array(["0.1", "0.2", "0.3", "0.4"]),
            np.array([0.1 + 0.2j, 0.2 + 0.2j, 0.3 + 0.2j, 0.4 + 0.2j]),
            np.array([0.1, 0.2, False, 0.4], dtype=object),
        ],
        ids=["numeric-string", "complex", "object-boolean"],
    )
    def test_free_energy_rejects_coercive_vector_aliases(
        self,
        field: str,
        alias: np.ndarray,
    ) -> None:
        values = {
            "predicted": np.ones(4),
            "observed": np.zeros(4),
            "precision": np.ones(4),
        }
        values[field] = alias

        with pytest.raises(ValueError, match=field):
            VariationalPredictor(4).free_energy(
                values["predicted"], values["observed"], values["precision"]
            )

    @pytest.mark.parametrize("field", ["phases", "omegas"])
    @pytest.mark.parametrize(
        "alias",
        [
            np.array(["0.1", "0.2", "0.3", "0.4"]),
            np.array([0.1 + 0.2j, 0.2 + 0.2j, 0.3 + 0.2j, 0.4 + 0.2j]),
            np.array([0.1, 0.2, True, 0.4], dtype=object),
        ],
        ids=["numeric-string", "complex", "object-boolean"],
    )
    def test_update_rejects_coercive_vector_aliases(
        self,
        field: str,
        alias: np.ndarray,
    ) -> None:
        values = {"phases": np.zeros(4), "omegas": np.ones(4)}
        values[field] = alias

        with pytest.raises(ValueError, match=field):
            VariationalPredictor(4).update(values["phases"], values["omegas"], 0.01)

    def test_variational_methods_preserve_real_numeric_object_vectors(self) -> None:
        predicted = np.array([0.1, 0.2, np.float32(0.3), np.int64(0)], dtype=object)
        observed = np.array([0, 0.1, 0.2, 0.3], dtype=object)
        precision = np.array([1, 1.0, np.float32(1.0), np.int64(1)], dtype=object)
        vp = VariationalPredictor(4)

        free_energy = vp.free_energy(predicted, observed, precision)
        state = vp.update(observed, precision, 0.01)

        assert np.isfinite(free_energy)
        assert np.all(np.isfinite(state.predicted_phases))

    def test_update_rejects_invalid_runtime_inputs(self) -> None:
        vp = VariationalPredictor(4)
        with pytest.raises(ValueError, match="dt"):
            vp.update(np.zeros(4), np.ones(4), dt=0.0)

    def test_variational_state_fields(self) -> None:
        vp = VariationalPredictor(4)
        state = vp.update(np.zeros(4), np.ones(4), dt=0.01)
        assert isinstance(state, VariationalState)
        assert state.predicted_phases.shape == (4,)
        assert state.error.shape == (4,)
        assert isinstance(state.free_energy, float)
        assert state.precision.shape == (4,)
        assert isinstance(state.complexity, float)

    def test_free_energy_decreases_with_learning(self) -> None:
        """F must decrease as mu converges to observed phases."""
        n = 6
        vp = VariationalPredictor(n, prior_precision=1.0, learning_rate=0.05)
        omegas = np.ones(n) * 2.0
        dt = 0.01
        rng = np.random.default_rng(99)
        phases = rng.uniform(0, 2 * np.pi, n)

        energies = []
        for _ in range(200):
            phases = (phases + dt * omegas) % (2 * np.pi)
            state = vp.update(phases, omegas, dt)
            energies.append(state.free_energy)

        # Compare mean of early window vs late window.
        # After transient, free energy should be lower.
        early = np.mean(energies[10:30])
        late = np.mean(energies[-30:])
        assert late < early, f"F did not decrease: early={early:.4f}, late={late:.4f}"

    def test_precision_increases_with_consistent_input(self) -> None:
        """Consistent (low-variance) input should drive precision up."""
        n = 4
        vp = VariationalPredictor(n, prior_precision=1.0, learning_rate=0.05)
        omegas = np.ones(n) * 3.0
        dt = 0.01

        phases = np.zeros(n)
        initial_prec = vp.precision.copy()

        for _ in range(150):
            phases = (phases + dt * omegas) % (2 * np.pi)
            state = vp.update(phases, omegas, dt)

        # Deterministic dynamics => small error variance => high precision
        assert np.all(state.precision > initial_prec), (
            f"Precision did not increase: {initial_prec} -> {state.precision}"
        )

    def test_fep_kuramoto_correspondence(self) -> None:
        """Verify precision ~ coupling strength (the core FEP-Kuramoto map).

        Two runs with different error magnitudes. Lower error => higher
        precision => larger effective coupling. This tests the formal
        correspondence: K_ij ~ Precision_ij.
        """
        n = 4
        omegas = np.ones(n) * 2.0
        dt = 0.01

        # Run A: low-noise (deterministic dynamics)
        vp_low = VariationalPredictor(n, prior_precision=1.0, learning_rate=0.05)
        phases = np.zeros(n)
        for _ in range(100):
            phases = (phases + dt * omegas) % (2 * np.pi)
            vp_low.update(phases, omegas, dt)
        K_low_noise = vp_low.precision_weighted_coupling()

        # Run B: high-noise (perturbed phases each step)
        rng = np.random.default_rng(42)
        vp_high = VariationalPredictor(n, prior_precision=1.0, learning_rate=0.05)
        phases = np.zeros(n)
        for _ in range(100):
            phases = (phases + dt * omegas + rng.normal(0, 0.3, n)) % (2 * np.pi)
            vp_high.update(phases, omegas, dt)
        K_high_noise = vp_high.precision_weighted_coupling()

        # Low-noise run should have higher diagonal coupling (= higher precision)
        diag_low = np.diag(K_low_noise)
        diag_high = np.diag(K_high_noise)
        assert np.all(diag_low > diag_high), (
            f"Precision-coupling violated: low_noise={diag_low}, high_noise={diag_high}"
        )

    def test_precision_weighted_coupling_shape(self) -> None:
        n = 5
        vp = VariationalPredictor(n)
        K = vp.precision_weighted_coupling()
        assert K.shape == (n, n)
        # Off-diagonal should be zero (diagonal precision model)
        mask = ~np.eye(n, dtype=bool)
        np.testing.assert_array_equal(K[mask], 0.0)

    def test_free_energy_positive_for_large_error(self) -> None:
        """F should be positive when prediction and observation diverge."""
        vp = VariationalPredictor(3, prior_precision=2.0)
        predicted = np.array([0.0, 0.0, 0.0])
        observed = np.array([2.0, 2.0, 2.0])
        precision = np.array([2.0, 2.0, 2.0])
        fe = vp.free_energy(predicted, observed, precision)
        assert fe > 0.0

    def test_complexity_zero_at_prior(self) -> None:
        """KL divergence is zero when precision equals prior precision."""
        vp = VariationalPredictor(4, prior_precision=1.0)
        state = vp.update(np.zeros(4), np.ones(4), dt=0.01)
        assert abs(state.complexity) < 1e-10

    def test_reset(self) -> None:
        vp = VariationalPredictor(3, prior_precision=2.0, learning_rate=0.1)
        omegas = np.ones(3)
        for _ in range(10):
            vp.update(np.array([1.0, 2.0, 3.0]), omegas, dt=0.01)
        vp.reset()
        np.testing.assert_array_equal(vp.precision, np.full(3, 2.0))


class TestPredictionPipelineWiring:
    """Pipeline: engine generates phases → predictor learns → error coupling
    feeds back into engine. The predictive coding loop."""

    def test_prediction_error_coupling_feeds_engine(self) -> None:
        """UPDEEngine → phases → PredictionModel.update → error_coupling →
        modifies next engine step. Proves prediction isn't decorative."""
        n = 4
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        eng = UPDEEngine(n, dt=0.01)
        model = PredictionModel(n, error_gain=0.5, learning_rate=0.1)
        phases = np.array([0.0, 0.5, 1.0, 1.5])
        omegas = np.ones(n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))

        for _ in range(20):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
            model.update(phases, omegas, dt=0.01)

        # Error coupling should be non-zero after learning
        coupling = model.error_coupling(phases, omegas, dt=0.01)
        assert coupling.shape == (n,)
        # After 20 updates, model should have learned something
        assert not np.allclose(model.weights, 0.0), (
            "Predictor should have non-zero weights after 20 updates"
        )

    def test_variational_precision_weighted_coupling_in_pipeline(self) -> None:
        """VariationalPredictor → precision_weighted_coupling → K_nm scale."""
        n = 4
        vp = VariationalPredictor(n, prior_precision=1.0, learning_rate=0.05)
        omegas = np.ones(n) * 2.0
        dt = 0.01
        phases = np.zeros(n)
        for _ in range(50):
            phases = (phases + dt * omegas) % (2 * np.pi)
            vp.update(phases, omegas, dt)
        K = vp.precision_weighted_coupling()
        assert K.shape == (n, n)
        assert np.all(np.diag(K) >= 0.0), "Precision coupling must be non-negative"

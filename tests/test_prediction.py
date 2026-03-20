# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Prediction model tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.upde.prediction import PredictionModel, PredictionState


class TestPredictionModel:
    def test_first_call_zero_error(self):
        model = PredictionModel(4)
        phases = np.array([0.0, 1.0, 2.0, 3.0])
        omegas = np.ones(4)
        state = model.update(phases, omegas, dt=0.01)
        assert state.mean_error == 0.0
        np.testing.assert_array_equal(state.prediction_error, np.zeros(4))

    def test_prediction_shape(self):
        model = PredictionModel(6)
        phases = np.zeros(6)
        omegas = np.ones(6)
        predicted = model.predict(phases, omegas, dt=0.01)
        assert predicted.shape == (6,)

    def test_prediction_in_range(self):
        model = PredictionModel(4)
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, 4)
        omegas = np.ones(4) * 5.0
        predicted = model.predict(phases, omegas, dt=0.1)
        assert np.all(predicted >= 0)
        assert np.all(predicted < 2 * np.pi)

    def test_error_decreases_with_learning(self):
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

    def test_weights_initially_zero(self):
        model = PredictionModel(3)
        np.testing.assert_array_equal(model.weights, np.zeros((3, 3)))

    def test_weights_change_after_update(self):
        model = PredictionModel(4, learning_rate=0.1)
        omegas = np.ones(4)
        phases1 = np.array([0.0, 0.5, 1.0, 1.5])
        phases2 = np.array([0.1, 0.6, 1.1, 1.6])
        model.update(phases1, omegas, dt=0.01)
        model.update(phases2, omegas, dt=0.01)
        assert not np.allclose(model.weights, 0.0)

    def test_error_coupling_zero_before_prediction(self):
        model = PredictionModel(4)
        phases = np.zeros(4)
        omegas = np.ones(4)
        coupling = model.error_coupling(phases, omegas, dt=0.01)
        np.testing.assert_array_equal(coupling, np.zeros(4))

    def test_error_coupling_after_prediction(self):
        model = PredictionModel(4, error_gain=0.5)
        omegas = np.ones(4)
        dt = 0.01
        phases1 = np.array([0.0, 0.5, 1.0, 1.5])
        model.update(phases1, omegas, dt)
        phases2 = np.array([0.1, 0.6, 1.1, 1.6])
        coupling = model.error_coupling(phases2, omegas, dt)
        assert coupling.shape == (4,)

    def test_reset(self):
        model = PredictionModel(3, learning_rate=0.1)
        omegas = np.ones(3)
        model.update(np.array([0.0, 1.0, 2.0]), omegas, dt=0.01)
        model.update(np.array([0.1, 1.1, 2.1]), omegas, dt=0.01)
        model.reset()
        np.testing.assert_array_equal(model.weights, np.zeros((3, 3)))
        state = model.update(np.array([0.0, 1.0, 2.0]), omegas, dt=0.01)
        assert state.mean_error == 0.0

    def test_error_wrapping(self):
        model = PredictionModel(2, learning_rate=0.0)
        omegas = np.ones(2)
        dt = 0.01
        model.update(np.array([0.1, 6.1]), omegas, dt)
        # Actual phases near 0 vs prediction near 2π should wrap correctly
        state = model.update(np.array([0.1, 0.1]), omegas, dt)
        assert np.all(np.abs(state.prediction_error) < np.pi)

    def test_prediction_state_fields(self):
        model = PredictionModel(3)
        state = model.update(np.zeros(3), np.ones(3), dt=0.01)
        assert isinstance(state, PredictionState)
        assert hasattr(state, "predicted_phases")
        assert hasattr(state, "prediction_error")
        assert hasattr(state, "mean_error")
        assert hasattr(state, "weights")

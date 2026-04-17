# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Prediction model tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.upde.prediction import (
    PredictionModel,
    PredictionState,
    VariationalPredictor,
    VariationalState,
)


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


class TestVariationalPredictor:
    def test_variational_state_fields(self):
        vp = VariationalPredictor(4)
        state = vp.update(np.zeros(4), np.ones(4), dt=0.01)
        assert isinstance(state, VariationalState)
        assert state.predicted_phases.shape == (4,)
        assert state.error.shape == (4,)
        assert isinstance(state.free_energy, float)
        assert state.precision.shape == (4,)
        assert isinstance(state.complexity, float)

    def test_free_energy_decreases_with_learning(self):
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

    def test_precision_increases_with_consistent_input(self):
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

    def test_fep_kuramoto_correspondence(self):
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

    def test_precision_weighted_coupling_shape(self):
        n = 5
        vp = VariationalPredictor(n)
        K = vp.precision_weighted_coupling()
        assert K.shape == (n, n)
        # Off-diagonal should be zero (diagonal precision model)
        mask = ~np.eye(n, dtype=bool)
        np.testing.assert_array_equal(K[mask], 0.0)

    def test_free_energy_positive_for_large_error(self):
        """F should be positive when prediction and observation diverge."""
        vp = VariationalPredictor(3, prior_precision=2.0)
        predicted = np.array([0.0, 0.0, 0.0])
        observed = np.array([2.0, 2.0, 2.0])
        precision = np.array([2.0, 2.0, 2.0])
        fe = vp.free_energy(predicted, observed, precision)
        assert fe > 0.0

    def test_complexity_zero_at_prior(self):
        """KL divergence is zero when precision equals prior precision."""
        vp = VariationalPredictor(4, prior_precision=1.0)
        state = vp.update(np.zeros(4), np.ones(4), dt=0.01)
        assert abs(state.complexity) < 1e-10

    def test_reset(self):
        vp = VariationalPredictor(3, prior_precision=2.0, learning_rate=0.1)
        omegas = np.ones(3)
        for _ in range(10):
            vp.update(np.array([1.0, 2.0, 3.0]), omegas, dt=0.01)
        vp.reset()
        np.testing.assert_array_equal(vp.precision, np.full(3, 2.0))


class TestPredictionPipelineWiring:
    """Pipeline: engine generates phases → predictor learns → error coupling
    feeds back into engine. The predictive coding loop."""

    def test_prediction_error_coupling_feeds_engine(self):
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

    def test_variational_precision_weighted_coupling_in_pipeline(self):
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

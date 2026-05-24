# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Prediction model contracts

"""Shape, finiteness, and parameter-preservation contracts for PredictionModel."""

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.upde.prediction import PredictionModel


class TestPredictionModelContract:
    """Verify PredictionModel stores parameters and produces valid output."""

    def test_error_gain_preserved(self):
        model = PredictionModel(4, error_gain=0.3)
        assert model.error_gain == 0.3

    def test_prediction_produces_valid_output(self):
        """PredictionModel.predict must return array of same shape as input."""
        model = PredictionModel(4, error_gain=0.5)
        phases = np.array([0.0, 0.5, 1.0, 1.5])
        omegas = np.ones(4)
        pred = model.predict(phases, omegas, dt=0.01)
        assert pred.shape == phases.shape
        assert np.all(np.isfinite(pred))

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Self-model error monitor input guards

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor.self_model import compute_self_model_error

_OBS = np.zeros((2, 4), dtype=np.float64)
_PRED = np.zeros((2, 4), dtype=np.float64)


def _compute(**overrides: Any) -> Any:
    kwargs: dict[str, Any] = {
        "observed_phases": _OBS,
        "predicted_phases": _PRED,
    }
    kwargs.update(overrides)
    observed = kwargs.pop("observed_phases")
    predicted = kwargs.pop("predicted_phases")
    return compute_self_model_error(observed, predicted, **kwargs)


class TestThresholdGuards:
    @pytest.mark.parametrize(
        ("tolerance", "match"),
        [
            ("loose", "tolerance must be a finite real value"),
            (float("inf"), "tolerance must be finite"),
            (-1.0, "tolerance must be non-negative"),
        ],
    )
    def test_rejects_invalid_tolerance(self, tolerance: Any, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            _compute(tolerance=tolerance)


class TestChannelMatrixGuards:
    def test_rejects_boolean_matrix(self) -> None:
        with pytest.raises(ValueError, match="must be numeric, got boolean"):
            _compute(observed_phases=np.array([[True, False]]))

    def test_rejects_non_coercible_matrix(self) -> None:
        bad = np.array([["a", "b"], ["c", "d"]], dtype=object)
        with pytest.raises(ValueError, match="convertible to a finite float array"):
            _compute(observed_phases=bad)

    def test_rejects_three_dimensional_matrix(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional or two-dimensional"):
            _compute(observed_phases=np.zeros((2, 2, 2)))


class TestOrderVectorGuards:
    def test_rejects_boolean_order(self) -> None:
        with pytest.raises(ValueError, match="must be numeric, got boolean"):
            _compute(
                observed_order=np.array([True, False]),
                predicted_order=np.zeros(2),
            )

    def test_rejects_non_coercible_order(self) -> None:
        bad = np.array(["a", "b"], dtype=object)
        with pytest.raises(ValueError, match="convertible to a finite float vector"):
            _compute(observed_order=bad, predicted_order=np.zeros(2))

    def test_rejects_two_dimensional_order(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional vector"):
            _compute(observed_order=np.zeros((2, 1)), predicted_order=np.zeros((2, 1)))

    def test_rejects_empty_order(self) -> None:
        with pytest.raises(ValueError, match="at least one value"):
            _compute(observed_order=np.array([]), predicted_order=np.array([]))

    def test_rejects_non_finite_order(self) -> None:
        with pytest.raises(ValueError, match="must contain finite values"):
            _compute(
                observed_order=np.array([np.inf, 0.0]),
                predicted_order=np.zeros(2),
            )

    def test_rejects_order_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="shapes must match"):
            _compute(observed_order=np.zeros(2), predicted_order=np.zeros(3))

    def test_rejects_order_channel_count_mismatch(self) -> None:
        with pytest.raises(ValueError, match="number of observed phases channels"):
            _compute(observed_order=np.zeros(3), predicted_order=np.zeros(3))


class TestChannelLabelGuards:
    def test_rejects_non_sequence_labels(self) -> None:
        with pytest.raises(ValueError, match="must be a sequence of strings"):
            _compute(channel_labels=5)

    def test_rejects_empty_label(self) -> None:
        with pytest.raises(ValueError, match="must not contain empty values"):
            _compute(channel_labels=["", "ch1"])


class TestChannelWeightGuards:
    def test_rejects_boolean_weights(self) -> None:
        with pytest.raises(ValueError, match="must be numeric, got boolean"):
            _compute(channel_weights=np.array([True, False]))

    def test_rejects_non_coercible_weights(self) -> None:
        bad = np.array(["a", "b"], dtype=object)
        with pytest.raises(ValueError, match="must be a numeric vector"):
            _compute(channel_weights=bad)

    def test_rejects_two_dimensional_weights(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional vector"):
            _compute(channel_weights=np.ones((2, 1)))

    def test_rejects_empty_weights(self) -> None:
        with pytest.raises(ValueError, match="must be non-empty"):
            _compute(channel_weights=np.array([]))

    def test_rejects_weight_count_mismatch(self) -> None:
        with pytest.raises(ValueError, match="length must match channel count"):
            _compute(channel_weights=np.array([1.0, 2.0, 3.0]))

    def test_rejects_non_finite_weights(self) -> None:
        with pytest.raises(ValueError, match="must contain finite values"):
            _compute(channel_weights=np.array([np.inf, 1.0]))

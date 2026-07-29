# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Self-model error monitor input guards

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor.self_model import (
    SelfModelErrorThresholdConfig,
    compute_self_model_error,
)

_OBS = np.zeros((2, 4), dtype=np.float64)
_PRED = np.zeros((2, 4), dtype=np.float64)


class _FloatLike:
    def __float__(self) -> float:
        return 0.25


class _ExplodingFloat(float):
    def __float__(self) -> float:
        raise OverflowError("adversarial float conversion")


class _BrokenArray:
    def __array__(self, dtype: object = None) -> np.ndarray:
        del dtype
        raise RuntimeError("broken array protocol")


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

    @pytest.mark.parametrize(
        "overrides",
        [
            {"tolerance": True},
            {"max_abs_tolerance": np.bool_(False)},
            {"order_tolerance": "0.1"},
            {"order_max_abs_tolerance": -0.1},
        ],
    )
    def test_rejects_coercive_or_invalid_threshold_configuration(
        self, overrides: dict[str, object]
    ) -> None:
        with pytest.raises(ValueError, match="tolerance"):
            _compute(**overrides)

    def test_direct_threshold_config_rejects_invalid_values(self) -> None:
        with pytest.raises(ValueError, match="order_tolerance"):
            SelfModelErrorThresholdConfig(
                tolerance=0.1,
                max_abs_tolerance=0.2,
                order_tolerance=True,
            )

    def test_rejects_threshold_whose_float_conversion_fails(self) -> None:
        with pytest.raises(ValueError, match="finite real value"):
            _compute(tolerance=_ExplodingFloat(0.1))


class TestChannelMatrixGuards:
    def test_rejects_boolean_matrix(self) -> None:
        with pytest.raises(ValueError, match="must be numeric, got boolean"):
            _compute(observed_phases=np.array([[True, False]]))

    def test_rejects_non_coercible_matrix(self) -> None:
        bad = np.array([["a", "b"], ["c", "d"]], dtype=object)
        with pytest.raises(ValueError, match="convertible to a finite float array"):
            _compute(observed_phases=bad)

    @pytest.mark.parametrize(
        "bad",
        [
            np.array([["0.1", "0.2"]]),
            np.array([[True, 0.2]], dtype=object),
            np.array([[1.0 + 0.0j, 0.2 + 0.0j]]),
            np.array([[_FloatLike(), 0.2]], dtype=object),
            _BrokenArray(),
        ],
    )
    def test_rejects_coercive_or_broken_phase_evidence(self, bad: object) -> None:
        with pytest.raises(ValueError, match="real float array"):
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

    @pytest.mark.parametrize(
        "bad",
        [
            np.array(["0.1", "0.2"]),
            np.array([True, 0.2], dtype=object),
            np.array([1.0 + 0.0j, 0.2 + 0.0j]),
            np.array([_FloatLike(), 0.2], dtype=object),
            _BrokenArray(),
        ],
    )
    def test_rejects_coercive_or_broken_order_evidence(self, bad: object) -> None:
        with pytest.raises(ValueError, match="real float vector"):
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

    @pytest.mark.parametrize(
        "labels",
        [
            [1, "ch1"],
            ["   ", "ch1"],
            ["ch0", "ch0"],
        ],
    )
    def test_rejects_noncanonical_or_duplicate_labels(
        self, labels: list[object]
    ) -> None:
        with pytest.raises(ValueError, match="channel_labels"):
            _compute(channel_labels=labels)

    def test_rejects_untrimmed_label(self) -> None:
        with pytest.raises(ValueError, match="canonical trimmed"):
            _compute(channel_labels=[" ch0", "ch1"])


class TestChannelWeightGuards:
    def test_rejects_boolean_weights(self) -> None:
        with pytest.raises(ValueError, match="must be numeric, got boolean"):
            _compute(channel_weights=np.array([True, False]))

    def test_rejects_non_coercible_weights(self) -> None:
        bad = np.array(["a", "b"], dtype=object)
        with pytest.raises(ValueError, match="must be a numeric vector"):
            _compute(channel_weights=bad)

    @pytest.mark.parametrize(
        "bad",
        [
            np.array(["1.0", "2.0"]),
            np.array([True, 2.0], dtype=object),
            np.array([1.0 + 0.0j, 2.0 + 0.0j]),
            np.array([_FloatLike(), 2.0], dtype=object),
            _BrokenArray(),
        ],
    )
    def test_rejects_coercive_or_broken_weights(self, bad: object) -> None:
        with pytest.raises(ValueError, match="real float vector"):
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

    def test_rejects_weight_sum_overflow(self) -> None:
        with pytest.raises(ValueError, match="sum must be finite and positive"):
            _compute(channel_weights=np.array([1.0e308, 1.0e308]))

    def test_rejects_object_array_whose_float_conversion_fails(self) -> None:
        bad = np.array([_ExplodingFloat(1.0), 2.0], dtype=object)
        with pytest.raises(ValueError, match="real float vector"):
            _compute(channel_weights=bad)


class TestEvidenceIdentityAndResultGuards:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"domain": "   "},
            {"domain": 7},
            {"scenario_id": ""},
            {"scenario_id": object()},
        ],
    )
    def test_rejects_noncanonical_evidence_identity(
        self, overrides: dict[str, object]
    ) -> None:
        with pytest.raises(ValueError, match="domain|scenario_id"):
            _compute(**overrides)

    @pytest.mark.parametrize(
        "changes",
        [
            {"channel_count": 3},
            {"overall_rmse": 99.0},
            {"channel_breaches": (False,)},
            {"breached": False},
            {"claim_boundary": "actuating"},
            {"non_actuating": False},
            {"execution_disabled": False},
            {"backend": "other"},
            {"record_hash": "0" * 64},
        ],
    )
    def test_direct_result_construction_replays_evidence(
        self, changes: dict[str, object]
    ) -> None:
        result = _compute(
            predicted_phases=np.full((2, 4), 0.2),
            tolerance=0.1,
            max_abs_tolerance=0.1,
        )
        with pytest.raises(ValueError):
            replace(result, **changes)

    @pytest.mark.parametrize(
        "changes",
        [
            {"sample_count": True},
            {"channel_labels": ["channel_0", "channel_1"]},
            {"channel_rmse": [0.2, 0.2]},
            {"channel_mae": (0.3, 0.2)},
            {"tolerance": 1},
            {"channel_breaches": (np.bool_(True), True)},
            {"weighted_rmse": 0.0},
            {"order_breached": False},
            {"breached": np.bool_(True)},
            {"record_hash": "NOT-A-SHA256"},
        ],
    )
    def test_direct_result_rejects_noncanonical_or_incoherent_fields(
        self, changes: dict[str, object]
    ) -> None:
        result = _compute(
            predicted_phases=np.full((2, 4), 0.2),
            tolerance=0.1,
            max_abs_tolerance=0.1,
        )
        with pytest.raises(ValueError):
            replace(result, **changes)

    @pytest.mark.parametrize(
        "changes",
        [
            {"channel_weights": [1.0, 2.0]},
            {"channel_weights": (0.0, 2.0)},
            {"weighted_mae": None},
        ],
    )
    def test_direct_weighted_result_rejects_incoherent_fields(
        self, changes: dict[str, object]
    ) -> None:
        result = _compute(
            predicted_phases=np.full((2, 4), 0.2),
            channel_weights=(1.0, 2.0),
            tolerance=0.1,
            max_abs_tolerance=0.1,
        )
        with pytest.raises(ValueError):
            replace(result, **changes)

    @pytest.mark.parametrize(
        "changes",
        [
            {"order_mae": None},
            {"order_mae": 0.3},
            {"order_breached": np.bool_(True)},
            {"order_breached": False},
        ],
    )
    def test_direct_order_result_rejects_incoherent_fields(
        self, changes: dict[str, object]
    ) -> None:
        result = _compute(
            observed_order=np.zeros(2),
            predicted_order=np.full(2, 0.2),
            tolerance=0.1,
            max_abs_tolerance=0.1,
        )
        with pytest.raises(ValueError):
            replace(result, **changes)

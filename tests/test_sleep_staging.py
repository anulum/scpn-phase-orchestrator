# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Sleep staging tests

from __future__ import annotations

from typing import get_type_hints

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor import sleep_staging as sleep_staging_module
from scpn_phase_orchestrator.monitor.sleep_staging import (
    classify_sleep_stage,
    ultradian_phase,
)
from tests.typing_contracts import assert_precise_ndarray_hint


def test_public_array_contracts_are_parameterised():
    hint = get_type_hints(ultradian_phase)["timestamps"]
    assert_precise_ndarray_hint(hint)
    assert "float64" in str(hint)


def test_n3_high_synchrony():
    assert classify_sleep_stage(0.85) == "N3"
    assert classify_sleep_stage(0.70) == "N3"


def test_n2_moderate_synchrony():
    assert classify_sleep_stage(0.55) == "N2"
    assert classify_sleep_stage(0.40) == "N2"


def test_n1_light_sleep():
    assert classify_sleep_stage(0.35) == "N1"
    assert classify_sleep_stage(0.30) == "N1"


def test_rem_with_functional_desync():
    assert classify_sleep_stage(0.25, functional_desync=True) == "REM"
    assert classify_sleep_stage(0.35, functional_desync=True) == "REM"


def test_wake_low_r_no_desync():
    assert classify_sleep_stage(0.15) == "Wake"
    assert classify_sleep_stage(0.0) == "Wake"


@pytest.mark.parametrize("value", [-0.01, 1.01, np.nan, np.inf, True])
def test_classify_sleep_stage_rejects_invalid_order_parameter(value):
    with pytest.raises((TypeError, ValueError), match="R"):
        classify_sleep_stage(value)


def test_classify_sleep_stage_rejects_object_complex_order_parameter() -> None:
    with pytest.raises((TypeError, ValueError), match="R.*real"):
        classify_sleep_stage(np.asarray(complex(0.7, 0.0), dtype=object))


@pytest.mark.parametrize("functional_desync", [0, 1, "yes"])
def test_classify_sleep_stage_requires_boolean_desync_flag(functional_desync):
    with pytest.raises(TypeError, match="functional_desync"):
        classify_sleep_stage(0.25, functional_desync=functional_desync)


def test_wake_very_low_r_even_with_desync():
    assert classify_sleep_stage(0.10, functional_desync=True) == "Wake"


def test_n1_without_desync_not_rem():
    assert classify_sleep_stage(0.32) == "N1"
    assert classify_sleep_stage(0.32, functional_desync=False) == "N1"


def test_boundary_at_n3_threshold():
    assert classify_sleep_stage(0.699) == "N2"
    assert classify_sleep_stage(0.700) == "N3"


def test_ultradian_phase_at_n3_onset():
    ts = np.array([0.0, 30.0, 60.0])
    stages = ["Wake", "N1", "N3"]
    phase = ultradian_phase(ts, stages)
    assert phase == 0.0


def test_ultradian_phase_halfway():
    # 45 minutes = half of 90-minute cycle
    ts = np.array([0.0, 45.0 * 60.0])
    stages = ["N3", "REM"]
    phase = ultradian_phase(ts, stages)
    np.testing.assert_allclose(phase, 0.5, atol=1e-6)


def test_ultradian_phase_wraps():
    # 90 minutes exactly → wraps to 0
    ts = np.array([0.0, 90.0 * 60.0])
    stages = ["N3", "N2"]
    phase = ultradian_phase(ts, stages)
    np.testing.assert_allclose(phase, 0.0, atol=1e-6)


def test_ultradian_no_n3_returns_zero():
    ts = np.array([0.0, 100.0, 200.0])
    stages = ["Wake", "N1", "N2"]
    assert ultradian_phase(ts, stages) == 0.0


def test_ultradian_empty_input():
    assert ultradian_phase(np.array([]), []) == 0.0


def test_ultradian_rejects_object_complex_timestamps_as_non_real() -> None:
    timestamps = np.asarray([0.0, complex(30.0, 0.0)], dtype=object)

    with pytest.raises(ValueError, match="timestamps must contain real-valued"):
        ultradian_phase(timestamps, ["N3", "REM"])


@pytest.mark.parametrize(
    ("timestamps", "stages", "match"),
    [
        (np.array([[0.0, 1.0]]), ["N3", "REM"], "timestamps"),
        (np.array([0.0, np.nan]), ["N3", "REM"], "timestamps"),
        (np.array([False, True]), ["N3", "REM"], "timestamps"),
        (np.array([60.0, 30.0]), ["N3", "REM"], "monotonic"),
        (np.array([0.0, 30.0]), ["N3"], "same length"),
        (np.array([0.0, 30.0]), ["N3", "Invalid"], "stage_history"),
    ],
)
def test_ultradian_rejects_invalid_history_contract(timestamps, stages, match):
    with pytest.raises(ValueError, match=match):
        ultradian_phase(timestamps, stages)


def test_optional_rust_classification_path_maps_stage_codes(monkeypatch):
    calls = []

    def fake_rust_classify(r_value, functional_desync):
        calls.append((r_value, functional_desync))
        return 4

    monkeypatch.setattr(sleep_staging_module, "_HAS_RUST", True)
    monkeypatch.setattr(
        sleep_staging_module,
        "_rust_classify",
        fake_rust_classify,
        raising=False,
    )

    assert classify_sleep_stage(0.21, functional_desync=True) == "REM"
    assert calls == [(0.21, True)]


def test_optional_rust_classification_rejects_invalid_stage_code(monkeypatch):
    monkeypatch.setattr(sleep_staging_module, "_HAS_RUST", True)
    monkeypatch.setattr(
        sleep_staging_module,
        "_rust_classify",
        lambda *_args: 9,
        raising=False,
    )

    with pytest.raises(ValueError, match="Rust sleep stage code"):
        classify_sleep_stage(0.21, functional_desync=True)


def test_optional_rust_ultradian_path_translates_stage_codes(monkeypatch):
    calls = []

    def fake_rust_ultradian(timestamps, codes):
        calls.append((timestamps.copy(), codes.copy()))
        return 0.625

    monkeypatch.setattr(sleep_staging_module, "_HAS_RUST", True)
    monkeypatch.setattr(
        sleep_staging_module,
        "_rust_ultradian",
        fake_rust_ultradian,
        raising=False,
    )

    timestamps = np.array([10.0, 70.0, 130.0], dtype=np.float64)
    phase = ultradian_phase(timestamps, ["Wake", "N3", "REM"])

    assert phase == 0.625
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], timestamps)
    assert calls[0][0].dtype == np.float64
    np.testing.assert_array_equal(calls[0][1], np.array([0, 3, 4], dtype=np.uint8))


@pytest.mark.parametrize("backend_value", [np.nan, np.inf, -0.1, 1.0])
def test_optional_rust_ultradian_rejects_nonphysical_phase(
    monkeypatch,
    backend_value: float,
) -> None:
    monkeypatch.setattr(sleep_staging_module, "_HAS_RUST", True)
    monkeypatch.setattr(
        sleep_staging_module,
        "_rust_ultradian",
        lambda *_args: backend_value,
        raising=False,
    )

    timestamps = np.array([10.0, 70.0, 130.0], dtype=np.float64)
    with pytest.raises(ValueError, match="Rust ultradian phase"):
        ultradian_phase(timestamps, ["Wake", "N3", "REM"])


class TestSleepStagingPipelineWiring:
    """Pipeline wiring: engine R → sleep stage classification."""

    def test_engine_r_to_sleep_stage(self):
        """UPDEEngine → R → classify_sleep_stage: high R → N3."""
        import numpy as np

        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import (
            compute_order_parameter,
        )

        n = 8
        eng = UPDEEngine(n, dt=0.01)
        # Synchronised initial conditions → high R
        phases = np.zeros(n)
        omegas = np.zeros(n)
        knm = 2.0 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        for _ in range(100):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        r, _ = compute_order_parameter(phases)
        stage = classify_sleep_stage(r)
        assert stage in ("N3", "N2", "N1", "REM", "Wake")
        if r > 0.65:
            assert stage == "N3"


@pytest.mark.parametrize(
    ("order_parameter", "desync", "expected"),
    [
        (0.85, False, "N3"),
        (0.50, False, "N2"),
        (0.35, False, "N1"),
        (0.35, True, "REM"),
        (0.25, True, "REM"),
        (0.25, False, "Wake"),
        (0.05, True, "Wake"),
    ],
)
def test_numpy_fallback_classifies_each_stage(
    monkeypatch, order_parameter, desync, expected
):
    """The pure-NumPy classifier resolves every stage band and REM split."""
    monkeypatch.setattr(sleep_staging_module, "_HAS_RUST", False)
    assert classify_sleep_stage(order_parameter, functional_desync=desync) == expected


def test_numpy_fallback_ultradian_phase_from_last_n3(monkeypatch):
    """The NumPy ultradian fallback measures elapsed fraction since the last N3."""
    monkeypatch.setattr(sleep_staging_module, "_HAS_RUST", False)
    timestamps = np.array([0.0, 60.0, 120.0, 180.0])
    stages = ["Wake", "N3", "N2", "REM"]
    phase = ultradian_phase(timestamps, stages)
    # Last N3 is at t=60 s; elapsed 120 s of the 5400 s ultradian period.
    assert phase == pytest.approx(120.0 / (90.0 * 60.0))


def test_numpy_fallback_ultradian_phase_without_n3_is_zero(monkeypatch):
    """With no N3 epoch the NumPy fallback returns a zero phase."""
    monkeypatch.setattr(sleep_staging_module, "_HAS_RUST", False)
    timestamps = np.array([0.0, 60.0, 120.0])
    assert ultradian_phase(timestamps, ["Wake", "N2", "REM"]) == 0.0


def test_timestamps_reject_non_castable_samples():
    """Timestamps that cannot be cast to float are rejected, not silently zeroed."""
    with pytest.raises(ValueError, match="timestamps must be a finite 1-D array"):
        ultradian_phase(np.array(["a", "b"], dtype=object), ["Wake", "N2"])


def test_timestamps_reject_complex_samples():
    """Complex timestamp samples are rejected before staging."""
    with pytest.raises(ValueError, match="timestamps must contain real-valued samples"):
        ultradian_phase(np.array([0.0 + 1.0j, 1.0 + 0.0j]), ["Wake", "N2"])


def test_rust_stage_code_rejects_non_real_output(monkeypatch):
    """A non-real Rust stage code is rejected rather than coerced."""
    monkeypatch.setattr(sleep_staging_module, "_HAS_RUST", True)
    monkeypatch.setattr(
        sleep_staging_module,
        "_rust_classify",
        lambda *_args: None,
        raising=False,
    )
    with pytest.raises(ValueError, match="Rust sleep stage code must be an integer"):
        classify_sleep_stage(0.85)


def test_rust_ultradian_rejects_non_real_output(monkeypatch):
    """A non-real Rust ultradian phase is rejected rather than coerced."""
    monkeypatch.setattr(sleep_staging_module, "_HAS_RUST", True)
    monkeypatch.setattr(
        sleep_staging_module,
        "_rust_ultradian",
        lambda *_args: None,
        raising=False,
    )
    with pytest.raises(ValueError, match="Rust ultradian phase must be a finite real"):
        ultradian_phase(np.array([0.0, 60.0]), ["N3", "REM"])

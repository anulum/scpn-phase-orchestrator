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


def _documented_stage(order_parameter: float, desync: bool) -> str:
    """Stage function as written in the API reference, kept independent."""
    if order_parameter >= 0.70:
        return "N3"
    if order_parameter >= 0.40:
        return "N2"
    if order_parameter >= 0.30:
        return "REM" if desync else "N1"
    if desync and order_parameter >= 0.20:
        return "REM"
    return "Wake"


def _band_probe_values() -> list[float]:
    """Every threshold, its float neighbours, the interval ends and a dense grid."""
    probes = {0.0, 1.0}
    for threshold in (0.20, 0.30, 0.40, 0.70):
        probes.update(
            {
                threshold,
                float(np.nextafter(threshold, 0.0)),
                float(np.nextafter(threshold, 1.0)),
            }
        )
    probes.update(float(value) for value in np.linspace(0.0, 1.0, 1001))
    return sorted(probes)


@pytest.mark.parametrize("desync", [False, True])
def test_classification_matches_documented_bands(desync: bool) -> None:
    """The active backend reproduces the documented piecewise stage function.

    Runs the Rust kernel when ``spo_kernel`` is installed and the NumPy path
    otherwise, so every environment checks the backend it actually uses.
    """
    for order_parameter in _band_probe_values():
        assert classify_sleep_stage(order_parameter, functional_desync=desync) == (
            _documented_stage(order_parameter, desync)
        ), order_parameter


def test_classification_accepts_numpy_real_and_boolean_inputs() -> None:
    """NumPy scalars from real pipelines are accepted and normalised."""
    spectral_ratio = np.array([0.2, 1.4])
    desync_flags = spectral_ratio > 1.0
    assert isinstance(desync_flags[1], np.bool_)

    assert classify_sleep_stage(np.float64(0.35), desync_flags[1]) == "REM"
    assert classify_sleep_stage(np.float32(0.35), desync_flags[0]) == "N1"
    assert classify_sleep_stage(np.float64(0.25), np.bool_(True)) == "REM"
    assert classify_sleep_stage(np.float64(0.25), np.bool_(False)) == "Wake"


# The four ``*_rejects_*`` tests that set ``_rust_classify`` or ``_rust_ultradian``
# substitute the native backend. A correct build cannot
# return an unknown stage code or a phase outside [0, 1) for validated input,
# so no real backend reaches these guards; they exist to catch a stale or
# mismatched spo_kernel build whose code table differs from this module. The
# nearest real coverage is test_classification_matches_documented_bands and
# test_ultradian_phase_matches_documented_formula, which run the real backend.
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


def _documented_ultradian_phase(timestamps: np.ndarray, stages: list[str]) -> float:
    """Ultradian phase as written in the API reference, kept independent."""
    n3_indices = [index for index, stage in enumerate(stages) if stage == "N3"]
    if not n3_indices:
        return 0.0
    elapsed = float(timestamps[-1] - timestamps[n3_indices[-1]])
    return (elapsed % 5400.0) / 5400.0


def test_ultradian_phase_matches_documented_formula() -> None:
    """The active backend matches the formula over random full-code histories.

    Every history carries all five stage labels, so the label-to-code
    translation is exercised for each code on the backend in use.
    """
    rng = np.random.default_rng(20260924)
    labels = ["Wake", "N1", "N2", "N3", "REM"]
    for _ in range(200):
        n_epochs = int(rng.integers(5, 400))
        steps = rng.choice([0.0, 30.0, 30.0, 30.0, 17.5], size=n_epochs - 1)
        timestamps = np.concatenate(([rng.uniform(0.0, 1.0e6)], steps)).cumsum()
        stages = [labels[i % 5] for i in range(5)] + [
            labels[int(code)] for code in rng.integers(0, 5, size=n_epochs - 5)
        ]
        order = rng.permutation(n_epochs)
        stages = [stages[int(i)] for i in order]
        phase = ultradian_phase(timestamps, stages)
        assert 0.0 <= phase < 1.0
        assert phase == pytest.approx(
            _documented_ultradian_phase(timestamps, stages), abs=1e-12
        )


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


# The tests below set ``_HAS_RUST`` to False to run the NumPy implementation in
# an environment where spo_kernel is installed. Nothing is substituted: the flag
# selects which real implementation executes, and without it the NumPy path is
# unreachable in a build that has the kernel.
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
    with pytest.raises(ValueError, match="timestamps must contain real-valued"):
        ultradian_phase(np.array(["a", "b"], dtype=object), ["Wake", "N2"])


@pytest.mark.parametrize(
    "timestamps",
    [
        np.array(["0", "30"]),
        np.array([b"0", b"30"]),
        np.array(["0.0", "30.0"], dtype=object),
        [0.0, "30"],
    ],
)
def test_timestamps_reject_numeric_text(timestamps) -> None:
    """Text that happens to parse as a number is not accepted as seconds."""
    with pytest.raises(ValueError, match="timestamps must contain real-valued"):
        ultradian_phase(timestamps, ["N3", "REM"])


@pytest.mark.parametrize(
    "timestamps",
    [
        np.array([0, 60_000], dtype="timedelta64[ms]"),
        np.array([0, 60], dtype="timedelta64[s]"),
        np.array(["2026-09-24T00:00", "2026-09-24T00:01"], dtype="datetime64[s]"),
    ],
)
def test_timestamps_reject_time_unit_dtypes(timestamps) -> None:
    """A unit-carrying time array would be read in its own unit, not seconds."""
    with pytest.raises(ValueError, match="plain seconds"):
        ultradian_phase(timestamps, ["N3", "REM"])


def test_timestamps_reject_boolean_items_in_object_arrays() -> None:
    """A boolean hidden in an object array is not read as 1.0 second."""
    with pytest.raises(ValueError, match="boolean"):
        ultradian_phase(np.array([0.0, True], dtype=object), ["N3", "REM"])


def test_timestamps_reject_ragged_input() -> None:
    """A ragged sequence is reported as a malformed timestamp array."""
    with pytest.raises(ValueError, match="finite 1-D array"):
        ultradian_phase([[0.0, 30.0], [60.0]], ["N3", "REM"])


def test_timestamps_accept_mixed_real_object_samples() -> None:
    """Python and NumPy reals in an object array are valid seconds."""
    timestamps = np.array([0, np.float32(30.0), 2700.0], dtype=object)
    phase = ultradian_phase(timestamps, ["N3", "N2", "REM"])
    assert phase == pytest.approx(0.5)


def test_stage_history_accepts_numpy_string_labels() -> None:
    """Labels held in a NumPy string array give the same phase as a list."""
    timestamps = np.array([0.0, 30.0, 2700.0])
    labels = ["N3", "N2", "REM"]
    assert ultradian_phase(timestamps, np.array(labels)) == ultradian_phase(
        timestamps, labels
    )


@pytest.mark.parametrize("bad_label", [["N3"], 3, np.int64(3), None])
def test_stage_history_rejects_non_string_labels(bad_label) -> None:
    """Unhashable or non-string labels fail with the documented ValueError."""
    with pytest.raises(ValueError, match="unknown sleep stage"):
        ultradian_phase(np.array([0.0, 30.0]), ["N3", bad_label])


def test_timestamps_reject_complex_samples():
    """Complex timestamp samples are rejected before staging."""
    with pytest.raises(ValueError, match="timestamps must contain real-valued samples"):
        ultradian_phase(np.array([0.0 + 1.0j, 1.0 + 0.0j]), ["Wake", "N2"])


# Substitutes the native backend; see the justification above
# test_optional_rust_classification_rejects_invalid_stage_code.
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


# Substitutes the native backend; see the justification above
# test_optional_rust_classification_rejects_invalid_stage_code.
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

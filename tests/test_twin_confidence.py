# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Online digital-twin confidence scoring tests

"""Algorithm, contract, dispatcher, calibration, and scoring tests.

Exercises the twin-confidence kernel through whatever backend is active plus
the calibration and scoring layer directly, including every validation branch
and the deterministic audit records.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor import twin_confidence as tc
from scpn_phase_orchestrator.monitor.twin_confidence import (
    TwinConfidenceBaseline,
    TwinConfidenceCalibrator,
    TwinDivergence,
    phase_order_divergence,
    score_twin_confidence,
)

TWO_PI = 2.0 * np.pi
LN2 = float(np.log(2.0))


# ---------------------------------------------------------------------
# Divergence kernel — algorithm invariants
# ---------------------------------------------------------------------


def test_identical_streams_have_zero_divergence() -> None:
    phases = np.array([0.1, 1.2, 2.3, 3.4, 4.5])
    order = np.array([0.4, 0.5, 0.6])
    div = phase_order_divergence(phases, phases.copy(), order, order.copy())
    assert div.phase_js_divergence == pytest.approx(0.0, abs=1e-12)
    assert div.order_wasserstein == pytest.approx(0.0, abs=1e-12)
    assert div.n_bins == 36
    assert div.backend in tc.AVAILABLE_BACKENDS


def test_disjoint_phase_support_approaches_ln2() -> None:
    model = np.full(16, 0.05)
    observed = np.full(16, np.pi + 0.05)
    order = np.full(4, 0.5)
    div = phase_order_divergence(model, observed, order, order, n_bins=36)
    assert div.phase_js_divergence == pytest.approx(LN2, abs=1e-9)


def test_order_shift_gives_exact_wasserstein() -> None:
    phases = np.array([0.1, 0.2, 0.3])
    model = np.array([0.2, 0.4, 0.6])
    observed = np.array([0.5, 0.7, 0.9])
    div = phase_order_divergence(phases, phases, model, observed)
    assert div.order_wasserstein == pytest.approx(0.3, abs=1e-9)


def test_js_divergence_is_symmetric() -> None:
    rng = np.random.default_rng(1)
    a = rng.uniform(0, TWO_PI, 128)
    b = rng.uniform(0, TWO_PI, 128)
    order = rng.uniform(0, 1, 16)
    forward = phase_order_divergence(a, b, order, order)
    backward = phase_order_divergence(b, a, order, order)
    assert forward.phase_js_divergence == pytest.approx(
        backward.phase_js_divergence, abs=1e-9
    )


def test_js_divergence_bounded_by_ln2() -> None:
    rng = np.random.default_rng(2)
    order = rng.uniform(0, 1, 8)
    for _ in range(50):
        a = rng.uniform(-20, 20, 64)
        b = rng.uniform(-20, 20, 64)
        div = phase_order_divergence(a, b, order, order, n_bins=24)
        assert -1e-12 <= div.phase_js_divergence <= LN2 + 1e-9
        assert 0.0 <= div.order_wasserstein <= 1.0 + 1e-12


def test_phase_wrapping_is_modulo_two_pi() -> None:
    base = np.array([0.3, 1.1, 2.2])
    shifted = base + TWO_PI * np.array([1.0, -2.0, 3.0])
    order = np.array([0.5, 0.5])
    div = phase_order_divergence(base, shifted, order, order)
    assert div.phase_js_divergence == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------
# Reference-kernel helpers (direct, for branch coverage)
# ---------------------------------------------------------------------


def test_empty_histogram_returns_uniform() -> None:
    hist = tc._phase_histogram(np.array([]), 4)
    assert hist == pytest.approx(np.full(4, 0.25))


def test_wasserstein_empty_window_is_zero() -> None:
    assert tc._wasserstein1(np.array([]), np.array([])) == 0.0


def test_python_kernel_matches_public_entry() -> None:
    rng = np.random.default_rng(3)
    a = rng.uniform(0, TWO_PI, 40)
    b = rng.uniform(0, TWO_PI, 40)
    c = rng.uniform(0, 1, 12)
    d = rng.uniform(0, 1, 12)
    raw = tc._python_kernel(a, b, c, d, 40, 12, 18)
    div = phase_order_divergence(a, b, c, d, n_bins=18)
    assert raw[0] == pytest.approx(div.phase_js_divergence, abs=1e-9)
    assert raw[1] == pytest.approx(div.order_wasserstein, abs=1e-9)


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------


def _ok_args() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array([0.1, 0.2, 0.3]),
        np.array([0.2, 0.3, 0.4]),
        np.array([0.4, 0.5]),
        np.array([0.5, 0.6]),
    )


@pytest.mark.parametrize("n_bins", [0, -3, 1.5, True])
def test_invalid_n_bins_rejected(n_bins: object) -> None:
    a, b, c, d = _ok_args()
    with pytest.raises(ValueError, match="n_bins"):
        phase_order_divergence(a, b, c, d, n_bins=n_bins)  # type: ignore[arg-type]


def test_boolean_alias_rejected() -> None:
    _, b, c, d = _ok_args()
    with pytest.raises(ValueError, match="boolean"):
        phase_order_divergence([True, False, True], b, c, d)  # type: ignore[arg-type]


def test_complex_alias_rejected() -> None:
    _, b, c, d = _ok_args()
    with pytest.raises(ValueError, match="real-valued"):
        phase_order_divergence([1 + 1j, 2, 3], b, c, d)  # type: ignore[arg-type]


def test_non_one_dimensional_phase_rejected() -> None:
    _, b, c, d = _ok_args()
    with pytest.raises(ValueError, match="one-dimensional"):
        phase_order_divergence(np.zeros((2, 2)), b, c, d)


def test_non_finite_phase_rejected() -> None:
    _, b, c, d = _ok_args()
    with pytest.raises(ValueError, match="finite"):
        phase_order_divergence([0.1, np.inf, 0.2], b, c, d)


def test_ragged_array_rejected() -> None:
    _, b, c, d = _ok_args()
    with pytest.raises(ValueError, match="real"):
        phase_order_divergence([[1.0], [2.0, 3.0]], b, c, d)  # type: ignore[list-item]


def test_order_out_of_unit_interval_rejected() -> None:
    a, b, _, d = _ok_args()
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        phase_order_divergence(a, b, np.array([0.5, 1.5]), d)


def test_empty_phase_rejected() -> None:
    with pytest.raises(ValueError, match="at least one phase"):
        phase_order_divergence(
            np.array([]), np.array([]), np.array([0.5]), np.array([0.5])
        )


def test_mismatched_phase_lengths_rejected() -> None:
    with pytest.raises(ValueError, match="observed_phases length"):
        phase_order_divergence(
            np.array([0.1, 0.2]),
            np.array([0.1, 0.2, 0.3]),
            np.array([0.5]),
            np.array([0.5]),
        )


def test_empty_order_rejected() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        phase_order_divergence(
            np.array([0.1]), np.array([0.2]), np.array([]), np.array([])
        )


def test_mismatched_order_lengths_rejected() -> None:
    with pytest.raises(ValueError, match="observed_order length"):
        phase_order_divergence(
            np.array([0.1]),
            np.array([0.2]),
            np.array([0.5, 0.6]),
            np.array([0.5]),
        )


# ---------------------------------------------------------------------
# Kernel output validation (defensive backend contract)
# ---------------------------------------------------------------------


def test_kernel_output_wrong_shape_rejected() -> None:
    with pytest.raises(ValueError, match=r"not \(2,\)"):
        tc._validate_kernel_output(np.array([0.1, 0.2, 0.3]), backend="python")


def test_kernel_output_non_finite_rejected() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        tc._validate_kernel_output(np.array([np.nan, 0.1]), backend="python")


def test_kernel_output_js_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match="Jensen"):
        tc._validate_kernel_output(np.array([1.0, 0.1]), backend="python")


def test_kernel_output_w1_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match="Wasserstein"):
        tc._validate_kernel_output(np.array([0.1, 2.0]), backend="python")


def test_kernel_output_clamps_tiny_negatives() -> None:
    js, w1 = tc._validate_kernel_output(np.array([-1e-13, -1e-13]), backend="python")
    assert js == 0.0
    assert w1 == 0.0


# ---------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------


def test_active_backend_in_available() -> None:
    assert tc.ACTIVE_BACKEND in tc.AVAILABLE_BACKENDS
    assert tc.AVAILABLE_BACKENDS[-1] == "python"


def test_dispatch_backend_returns_known_name() -> None:
    name, fn = tc._dispatch_backend()
    assert name in tc.AVAILABLE_BACKENDS
    assert (fn is None) == (name == "python")


def test_dispatch_falls_back_to_python(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tc, "ACTIVE_BACKEND", "python")
    monkeypatch.setattr(tc, "AVAILABLE_BACKENDS", ["python"])
    name, fn = tc._dispatch_backend()
    assert name == "python"
    assert fn is None


def test_dispatch_skips_failing_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom() -> tc._BackendFn:
        raise RuntimeError("backend down")

    monkeypatch.setattr(tc, "ACTIVE_BACKEND", "ghost")
    monkeypatch.setattr(tc, "AVAILABLE_BACKENDS", ["ghost", "python"])
    monkeypatch.setitem(tc._LOADERS, "ghost", _boom)
    tc._BACKEND_CACHE.pop("ghost", None)
    name, fn = tc._dispatch_backend()
    assert name == "python"
    assert fn is None


def test_dispatch_exhausts_without_python(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom() -> tc._BackendFn:
        raise OSError("backend down")

    monkeypatch.setattr(tc, "ACTIVE_BACKEND", "ghost")
    monkeypatch.setattr(tc, "AVAILABLE_BACKENDS", ["ghost"])
    monkeypatch.setitem(tc._LOADERS, "ghost", _boom)
    tc._BACKEND_CACHE.pop("ghost", None)
    name, fn = tc._dispatch_backend()
    assert name == "python"
    assert fn is None


def test_public_entry_uses_python_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tc, "_dispatch_backend", lambda: ("python", None))
    a, b, c, d = _ok_args()
    div = phase_order_divergence(a, b, c, d)
    assert div.backend == "python"
    assert div.phase_js_divergence >= 0.0


def test_load_backend_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def _loader() -> tc._BackendFn:
        calls["n"] += 1
        return tc._python_kernel

    monkeypatch.setitem(tc._LOADERS, "probe", _loader)
    tc._BACKEND_CACHE.pop("probe", None)
    first = tc._load_backend("probe")
    second = tc._load_backend("probe")
    assert first is second
    assert calls["n"] == 1


def test_resolve_backends_reports_python_floor() -> None:
    active, available = tc._resolve_backends()
    assert available[-1] == "python"
    assert active == available[0]


# ---------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------


def _nominal_baseline(seed: int = 10) -> TwinConfidenceBaseline:
    rng = np.random.default_rng(seed)
    cal = TwinConfidenceCalibrator()
    for _ in range(40):
        a = rng.uniform(0, TWO_PI, 128)
        b = a + rng.normal(0, 0.05, 128)
        ra = rng.uniform(0.45, 0.55, 32)
        rb = np.clip(ra + rng.normal(0, 0.01, 32), 0, 1)
        cal.observe(phase_order_divergence(a, b, ra, rb))
    return cal.baseline()


def test_calibrator_accumulates_samples() -> None:
    cal = TwinConfidenceCalibrator()
    assert cal.sample_count == 0
    div = TwinDivergence(0.01, 0.005, 36, "python")
    cal.observe(div)
    cal.observe_many([div, div])
    assert cal.sample_count == 3


def test_calibrator_requires_samples() -> None:
    with pytest.raises(ValueError, match="at least one nominal sample"):
        TwinConfidenceCalibrator().baseline()


@pytest.mark.parametrize("band_z", [-1.0, np.inf, True])
def test_calibrator_rejects_bad_band_z(band_z: object) -> None:
    with pytest.raises(ValueError, match="band_z"):
        TwinConfidenceCalibrator(band_z=band_z)  # type: ignore[arg-type]


def test_baseline_bands_and_audit_record() -> None:
    base = _nominal_baseline()
    assert base.sample_count == 40
    assert base.phase_js_upper_band == pytest.approx(
        base.phase_js_mean + base.band_z * base.phase_js_std
    )
    assert base.order_w1_upper_band == pytest.approx(
        base.order_w1_mean + base.band_z * base.order_w1_std
    )
    record = base.to_audit_record()
    assert record["sample_count"] == 40
    assert json.loads(json.dumps(record)) == record


# ---------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------


def test_nominal_tick_scores_healthy() -> None:
    base = _nominal_baseline()
    rng = np.random.default_rng(11)
    a = rng.uniform(0, TWO_PI, 128)
    b = a + rng.normal(0, 0.05, 128)
    ra = rng.uniform(0.45, 0.55, 32)
    rb = np.clip(ra + rng.normal(0, 0.01, 32), 0, 1)
    score = score_twin_confidence(phase_order_divergence(a, b, ra, rb), base)
    assert score.status == "healthy"
    assert score.confidence > 0.6
    assert score.phase_js_within_band
    assert score.order_w1_within_band


def test_divergent_tick_scores_critical() -> None:
    base = _nominal_baseline()
    model = np.random.default_rng(12).uniform(0, TWO_PI, 128)
    observed = np.full(128, 0.1)
    score = score_twin_confidence(
        phase_order_divergence(model, observed, np.full(32, 0.5), np.full(32, 0.05)),
        base,
    )
    assert score.status == "critical"
    assert score.confidence < 0.3
    assert not score.phase_js_within_band
    assert not score.order_w1_within_band


def test_confidence_is_one_below_baseline_mean() -> None:
    base = TwinConfidenceBaseline(0.05, 0.01, 0.05, 0.01, 100, 3.0)
    div = TwinDivergence(0.01, 0.01, 36, "python")
    score = score_twin_confidence(div, base)
    assert score.confidence == pytest.approx(1.0)
    assert score.composite_z == 0.0
    assert score.status == "healthy"


def test_warning_band_between_thresholds() -> None:
    base = TwinConfidenceBaseline(0.0, 0.1, 0.0, 0.1, 50, 3.0)
    # js=0.08 -> z=0.8 -> confidence=exp(-0.8)=0.449, inside [0.3, 0.9).
    div = TwinDivergence(0.08, 0.0, 36, "python")
    score = score_twin_confidence(
        div, base, sensitivity=1.0, warning_confidence=0.9, critical_confidence=0.3
    )
    assert score.status == "warning"
    assert 0.3 <= score.confidence < 0.9


@pytest.mark.parametrize("sensitivity", [0.0, -1.0, np.nan])
def test_invalid_sensitivity_rejected(sensitivity: float) -> None:
    base = _nominal_baseline()
    div = TwinDivergence(0.01, 0.01, 36, "python")
    with pytest.raises(ValueError, match="sensitivity"):
        score_twin_confidence(div, base, sensitivity=sensitivity)


def test_threshold_order_enforced() -> None:
    base = _nominal_baseline()
    div = TwinDivergence(0.01, 0.01, 36, "python")
    with pytest.raises(ValueError, match="critical_confidence"):
        score_twin_confidence(
            div, base, warning_confidence=0.3, critical_confidence=0.8
        )


def test_confidence_threshold_above_unit_rejected() -> None:
    base = _nominal_baseline()
    div = TwinDivergence(0.01, 0.01, 36, "python")
    with pytest.raises(ValueError, match="warning_confidence"):
        score_twin_confidence(div, base, warning_confidence=1.5)


def test_score_hash_is_deterministic_and_excludes_itself() -> None:
    base = _nominal_baseline()
    div = TwinDivergence(0.02, 0.01, 36, "python")
    first = score_twin_confidence(div, base)
    second = score_twin_confidence(div, base)
    assert first.score_hash == second.score_hash
    assert len(first.score_hash) == 64
    record = first.to_audit_record()
    assert record["score_hash"] == first.score_hash
    assert json.loads(json.dumps(record)) == record


def test_divergence_audit_record_round_trips() -> None:
    div = TwinDivergence(0.1, 0.2, 36, "rust")
    record = div.to_audit_record()
    assert record == {
        "phase_js_divergence": 0.1,
        "order_wasserstein": 0.2,
        "n_bins": 36,
        "backend": "rust",
    }
    assert json.loads(json.dumps(record)) == record

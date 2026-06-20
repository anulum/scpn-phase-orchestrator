# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Twin-confidence numerical stability tests

"""Numerical-stability and invariant tests for twin-confidence scoring.

Validates the mathematical contract of the divergence kernel and the confidence
map: divergence non-negativity and bounds, metric symmetry and identity of
indiscernibles, monotone confidence decay, calibration robustness, long-run
drift-freedom on identical streams, and overflow-free phase wrapping.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor.twin_confidence import (
    TwinConfidenceBaseline,
    TwinConfidenceCalibrator,
    TwinDivergence,
    phase_order_divergence,
    score_twin_confidence,
)

pytestmark = pytest.mark.slow

TWO_PI = 2.0 * np.pi
LN2 = float(np.log(2.0))


def test_divergence_bounds_hold_over_random_sweep() -> None:
    rng = np.random.default_rng(404)
    for _ in range(400):
        n = int(rng.integers(2, 256))
        w = int(rng.integers(2, 128))
        nb = int(rng.integers(2, 90))
        a = rng.uniform(-30.0, 30.0, n)
        b = rng.uniform(-30.0, 30.0, n)
        c = rng.uniform(0.0, 1.0, w)
        d = rng.uniform(0.0, 1.0, w)
        div = phase_order_divergence(a, b, c, d, n_bins=nb)
        assert -1e-12 <= div.phase_js_divergence <= LN2 + 1e-9
        assert -1e-12 <= div.order_wasserstein <= 1.0 + 1e-9


def test_js_symmetry_and_identity_of_indiscernibles() -> None:
    rng = np.random.default_rng(7)
    for _ in range(60):
        a = rng.uniform(0.0, TWO_PI, 96)
        b = rng.uniform(0.0, TWO_PI, 96)
        order = rng.uniform(0.0, 1.0, 24)
        forward = phase_order_divergence(a, b, order, order).phase_js_divergence
        backward = phase_order_divergence(b, a, order, order).phase_js_divergence
        assert forward == pytest.approx(backward, abs=1e-9)
        same = phase_order_divergence(a, a, order, order).phase_js_divergence
        assert same == pytest.approx(0.0, abs=1e-12)


def test_wasserstein_symmetry_and_identity() -> None:
    rng = np.random.default_rng(8)
    phases = rng.uniform(0.0, TWO_PI, 10)
    for _ in range(60):
        c = rng.uniform(0.0, 1.0, 32)
        d = rng.uniform(0.0, 1.0, 32)
        forward = phase_order_divergence(phases, phases, c, d).order_wasserstein
        backward = phase_order_divergence(phases, phases, d, c).order_wasserstein
        assert forward == pytest.approx(backward, abs=1e-12)
        same = phase_order_divergence(phases, phases, c, c).order_wasserstein
        assert same == pytest.approx(0.0, abs=1e-12)


def test_confidence_decays_monotonically_with_deviation() -> None:
    base = TwinConfidenceBaseline(0.0, 0.05, 0.0, 0.05, 100, 3.0)
    previous = 1.0 + 1e-9
    for step in range(0, 40):
        js = 0.01 * step
        score = score_twin_confidence(TwinDivergence(js, 0.0, 36, "python"), base)
        assert 0.0 <= score.confidence <= 1.0
        assert score.confidence <= previous + 1e-12
        previous = score.confidence


def test_confidence_is_unit_inside_band_and_drops_outside() -> None:
    base = TwinConfidenceBaseline(0.10, 0.02, 0.05, 0.01, 200, 3.0)
    inside = score_twin_confidence(TwinDivergence(0.09, 0.04, 36, "python"), base)
    assert inside.confidence == pytest.approx(1.0)
    assert inside.phase_js_within_band
    outside = score_twin_confidence(TwinDivergence(0.30, 0.30, 36, "python"), base)
    assert outside.confidence < inside.confidence
    assert not outside.phase_js_within_band


def test_calibration_is_robust_and_bounded() -> None:
    rng = np.random.default_rng(11)
    cal = TwinConfidenceCalibrator()
    for _ in range(120):
        a = rng.uniform(0.0, TWO_PI, 64)
        b = a + rng.normal(0.0, 0.04, 64)
        ra = rng.uniform(0.45, 0.55, 24)
        rb = np.clip(ra + rng.normal(0.0, 0.01, 24), 0.0, 1.0)
        cal.observe(phase_order_divergence(a, b, ra, rb))
    base = cal.baseline()
    assert base.phase_js_std >= 0.0
    assert base.order_w1_std >= 0.0
    assert base.sample_count == 120
    # Every nominal-like tick scores in [0, 1] with a valid status.
    for _ in range(50):
        a = rng.uniform(0.0, TWO_PI, 64)
        b = a + rng.normal(0.0, 0.04, 64)
        ra = rng.uniform(0.45, 0.55, 24)
        rb = np.clip(ra + rng.normal(0.0, 0.01, 24), 0.0, 1.0)
        score = score_twin_confidence(phase_order_divergence(a, b, ra, rb), base)
        assert 0.0 <= score.confidence <= 1.0
        assert score.status in {"healthy", "warning", "critical"}


def test_identical_streams_have_no_long_run_drift() -> None:
    rng = np.random.default_rng(12)
    base = TwinConfidenceBaseline(0.0, 1e-6, 0.0, 1e-6, 100, 3.0)
    for _ in range(200):
        phases = rng.uniform(-50.0, 50.0, 80)
        order = rng.uniform(0.0, 1.0, 30)
        div = phase_order_divergence(phases, phases.copy(), order, order.copy())
        assert div.phase_js_divergence == pytest.approx(0.0, abs=1e-12)
        assert div.order_wasserstein == pytest.approx(0.0, abs=1e-12)
        score = score_twin_confidence(div, base)
        assert score.confidence == pytest.approx(1.0)
        assert score.status == "healthy"


def test_large_phase_magnitudes_wrap_without_overflow() -> None:
    base = np.array([0.3, 1.1, 2.2, 4.0])
    order = np.array([0.5, 0.5])
    for k in (1.0, 100.0, 1e6, 1e9):
        shifted = base + TWO_PI * k
        div = phase_order_divergence(base, shifted, order, order, n_bins=36)
        assert np.isfinite(div.phase_js_divergence)
        assert div.phase_js_divergence == pytest.approx(0.0, abs=1e-6)


def test_score_hash_changes_with_inputs() -> None:
    base = TwinConfidenceBaseline(0.05, 0.01, 0.05, 0.01, 100, 3.0)
    a = score_twin_confidence(TwinDivergence(0.05, 0.05, 36, "python"), base)
    b = score_twin_confidence(TwinDivergence(0.20, 0.05, 36, "python"), base)
    assert a.score_hash != b.score_hash

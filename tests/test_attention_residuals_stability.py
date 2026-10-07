# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — AttnRes stability validation (Lyapunov criterion)

"""Measure bounded trajectory diagnostics for the phase attention adaptation.

The perturbation fixture retains its numerical slope ceiling of +0.05,
and the frozen-coupling fixture retains its +0.1 ceiling. Positive ceilings
and a frozen graph do not prove non-positive exponents of the time-dependent
law. These slow tests diagnose their sampled graphs; the research proposal's
global stability requirement remains unqualified.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from benchmarks.attnres_reference import FloatArray
from scpn_phase_orchestrator.coupling.attention_residuals import (
    attnres_modulate,
)
from scpn_phase_orchestrator.monitor.lyapunov import lyapunov_spectrum
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

TWO_PI = 2.0 * np.pi

pytestmark = pytest.mark.slow


def _symmetric_knm(n: int, strength: float, seed: int) -> FloatArray:
    """Build a seeded symmetric graph without self coupling."""
    rng = np.random.default_rng(seed)
    half = rng.uniform(0.0, 2.0 * strength, size=(n, n))
    knm = 0.5 * (half + half.T)
    np.fill_diagonal(knm, 0.0)
    return knm.astype(np.float64)


def _integrate_attnres(
    engine: UPDEEngine,
    phases: FloatArray,
    omegas: FloatArray,
    knm: FloatArray,
    alpha: FloatArray,
    n_steps: int,
    block_size: int = 4,
    lambda_: float = 0.5,
) -> FloatArray:
    """Run n_steps of AttnRes-modulated Kuramoto integration."""
    for _ in range(n_steps):
        knm_mod = attnres_modulate(knm, phases, block_size=block_size, lambda_=lambda_)
        phases = engine.step(phases, omegas, knm_mod, 0.0, 0.0, alpha)
    return phases


# ---------------------------------------------------------------------
# Perturbation-based max-Lyapunov estimator
# ---------------------------------------------------------------------


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(
    max_examples=3,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_attnres_perturbation_decay(seed: int) -> None:
    """Measure the nearby-trajectory slope against the existing +0.05 ceiling."""
    n = 16
    dt = 0.01
    n_warmup = 400
    n_measure = 300
    block_size = 4
    lambda_coupling = 0.5

    rng = np.random.default_rng(seed)
    omegas = (rng.standard_normal(n) * 0.3).astype(np.float64)
    knm = _symmetric_knm(n, strength=5.0 / n, seed=seed)  # supercritical
    alpha = np.zeros((n, n), dtype=np.float64)
    phases0 = rng.uniform(0.0, TWO_PI, size=n).astype(np.float64)

    engine = UPDEEngine(n_oscillators=n, dt=dt, method="euler")

    # Warm-up to reach the steady state.
    phases = _integrate_attnres(
        engine,
        phases0,
        omegas,
        knm,
        alpha,
        n_warmup,
        block_size=block_size,
        lambda_=lambda_coupling,
    )

    # Fork: clone the phase vector and add a tiny random perturbation.
    epsilon = 1e-8
    delta0 = rng.normal(size=n) * epsilon
    phases_a = phases.copy()
    phases_b = (phases + delta0) % TWO_PI

    # Track log |δθ| every step.
    log_norms: list[float] = []
    times: list[float] = []
    for step in range(n_measure):
        knm_mod = attnres_modulate(
            knm, phases_a, block_size=block_size, lambda_=lambda_coupling
        )
        phases_a = engine.step(phases_a, omegas, knm_mod, 0.0, 0.0, alpha)
        knm_mod_b = attnres_modulate(
            knm, phases_b, block_size=block_size, lambda_=lambda_coupling
        )
        phases_b = engine.step(phases_b, omegas, knm_mod_b, 0.0, 0.0, alpha)
        # Wrap-aware difference.
        raw = phases_b - phases_a
        diff = (raw + np.pi) % TWO_PI - np.pi
        norm = float(np.linalg.norm(diff))
        if norm > 0.0:
            log_norms.append(float(np.log(norm)))
            times.append(step * dt)

    # Regress log|δθ| vs time; slope is λ_max.
    # Use the last 60 % of samples so we avoid the initial transient.
    start = len(log_norms) * 2 // 5
    window_log = np.array(log_norms[start:])
    window_t = np.array(times[start:])
    assert window_t.size >= 10, "Insufficient perturbation samples for regression"

    slope, _intercept = np.polyfit(window_t, window_log, 1)

    # Accept up to +0.05 to absorb near-critical noise; a genuinely
    # unstable configuration would show λ_max ≫ 0.05.
    assert slope <= 0.05, (
        f"λ_max ≈ {slope:.4f} exceeds the 0.05 stability budget for seed={seed}"
    )


# ---------------------------------------------------------------------
# Frozen-K Lyapunov agreement
# ---------------------------------------------------------------------


def test_attnres_frozen_k_lyapunov_agrees() -> None:
    """Compare frozen-graph exponents against the existing +0.1 limits."""
    n = 12
    dt = 0.01
    seed = 2026

    rng = np.random.default_rng(seed)
    omegas = (rng.standard_normal(n) * 0.3).astype(np.float64)
    knm = _symmetric_knm(n, strength=5.0 / n, seed=seed)
    alpha = np.zeros((n, n), dtype=np.float64)
    phases0 = rng.uniform(0.0, TWO_PI, size=n).astype(np.float64)

    # Baseline spectrum (static K).
    baseline = lyapunov_spectrum(
        phases0,
        omegas,
        knm,
        alpha,
        dt=dt,
        n_steps=500,
        qr_interval=10,
    )

    # Integrate AttnRes to steady state, then freeze K.
    engine = UPDEEngine(n_oscillators=n, dt=dt, method="euler")
    phases_ss = _integrate_attnres(engine, phases0, omegas, knm, alpha, 400)
    knm_frozen = attnres_modulate(knm, phases_ss, lambda_=0.5)
    modulated = lyapunov_spectrum(
        phases_ss,
        omegas,
        knm_frozen,
        alpha,
        dt=dt,
        n_steps=500,
        qr_interval=10,
    )

    # R, the mean-field order parameter at the fixed point, determines
    # the magnitude of the leading exponent. Require that the AttnRes
    # fixture stays below its +0.1 ceiling and does not exceed the baseline
    # by more than 0.1; neither bound proves contraction.
    assert modulated[0] <= 0.1, (
        f"AttnRes frozen-K max Lyapunov {modulated[0]:.4f} exceeds the "
        f"0.1 stability ceiling"
    )
    assert modulated[0] - baseline[0] <= 0.1, (
        f"AttnRes max exponent {modulated[0]:.4f} exceeds baseline "
        f"{baseline[0]:.4f} by more than 0.1"
    )

    # Sanity — both spectra return N exponents.
    assert len(baseline) == n
    assert len(modulated) == n


# ---------------------------------------------------------------------
# Order parameter stability under long integration
# ---------------------------------------------------------------------


def test_attnres_long_run_r_stays_bounded() -> None:
    """Require finite long trajectories and the existing order-parameter bounds."""
    n = 16
    dt = 0.01
    n_steps = 2000

    rng = np.random.default_rng(17)
    omegas = (rng.standard_normal(n) * 0.3).astype(np.float64)
    knm = _symmetric_knm(n, strength=5.0 / n, seed=17)
    alpha = np.zeros((n, n), dtype=np.float64)
    phases = rng.uniform(0.0, TWO_PI, size=n).astype(np.float64)

    engine = UPDEEngine(n_oscillators=n, dt=dt, method="euler")

    r_trajectory: list[float] = []
    for _ in range(n_steps):
        knm_mod = attnres_modulate(knm, phases, lambda_=0.5)
        phases = engine.step(phases, omegas, knm_mod, 0.0, 0.0, alpha)
        assert np.all(np.isfinite(phases)), "AttnRes produced non-finite phases"
        r, _psi = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0 + 1e-12, f"R={r} out of [0, 1]"
        r_trajectory.append(float(r))

    # Last 200 steps should have a stable mean — variance proxy via
    # max minus min over the tail.
    tail = np.array(r_trajectory[-200:])
    assert tail.max() - tail.min() < 0.3, (
        f"R oscillates by {tail.max() - tail.min():.3f} at steady state, "
        f"suggesting a limit cycle rather than a fixed point"
    )

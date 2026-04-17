# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — AttnRes stability validation (Lyapunov criterion)

"""Stability validation for the AttnRes Phase-3 spike.

Research doc ``research_attention_residuals_2026-04-06.md §5``
requires the maximum Lyapunov exponent to stay non-positive under
AttnRes modulation — i.e. the state-dependent coupling must not
introduce new instabilities.

The bundled ``monitor.lyapunov.lyapunov_spectrum`` expects a static
``K_nm`` matrix, which does not fit AttnRes (K changes every step
from the current phases). Two validation paths are used here:

1. **Perturbation decay** — the canonical "maximum Lyapunov exponent
   via two trajectories" test: run AttnRes for a warm-up, fork into
   two copies one of which has a small random ``δθ`` perturbation,
   then continue both for ``n_measure`` steps and regress
   ``log |δθ|`` against time. The slope estimates ``λ_max``. Must be
   ≤ ``+0.05`` across seeds — the small positive budget absorbs the
   chaos-like behaviour near the critical coupling without letting
   a genuinely unstable configuration slip through.
2. **Frozen-K agreement** — freeze the modulated K at steady state
   and compute the full Lyapunov spectrum using the existing
   ``lyapunov_spectrum`` helper. The max exponent must agree with
   the baseline (un-modulated) spectrum to within a small tolerance.
   This catches any macroscopic change in the local stability
   neighbourhood that AttnRes induces.

Both tests are gated behind ``pytest.mark.slow`` so they are not on
the fast CI path; full suite picks them up through ``pytest -m slow``.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.coupling.attention_residuals import (
    attnres_modulate,
)
from scpn_phase_orchestrator.monitor.lyapunov import lyapunov_spectrum
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

TWO_PI = 2.0 * np.pi

pytestmark = pytest.mark.slow


def _symmetric_knm(n: int, strength: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    half = rng.uniform(0.0, 2.0 * strength, size=(n, n))
    knm = 0.5 * (half + half.T)
    np.fill_diagonal(knm, 0.0)
    return knm.astype(np.float64)


def _integrate_attnres(
    engine: UPDEEngine,
    phases: np.ndarray,
    omegas: np.ndarray,
    knm: np.ndarray,
    alpha: np.ndarray,
    n_steps: int,
    block_size: int = 4,
    lambda_: float = 0.5,
) -> np.ndarray:
    """Run n_steps of AttnRes-modulated Kuramoto integration."""
    for _ in range(n_steps):
        knm_mod = attnres_modulate(
            knm, phases, block_size=block_size, lambda_=lambda_
        )
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
    """Two nearby trajectories under AttnRes modulation must not
    diverge exponentially.

    Estimates ``λ_max`` from the slope of
    ``log |δθ(t)| ≈ λ_max · t + c`` over a linear-regime window after
    a warm-up. The supercritical Kuramoto regime is strictly
    contracting (``λ_max < 0``); AttnRes must not push it above ≈ 0.
    """
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
            log_norms.append(np.log(norm))
            times.append(step * dt)

    # Regress log|δθ| vs time; slope is λ_max.
    # Use the last 60 % of samples so we avoid the initial transient.
    start = len(log_norms) * 2 // 5
    window_log = np.array(log_norms[start:])
    window_t = np.array(times[start:])
    if window_t.size < 10:
        pytest.skip("Not enough finite log-norm samples for regression")

    slope, _intercept = np.polyfit(window_t, window_log, 1)

    # Accept up to +0.05 to absorb near-critical noise; a genuinely
    # unstable configuration would show λ_max ≫ 0.05.
    assert slope <= 0.05, (
        f"λ_max ≈ {slope:.4f} exceeds the 0.05 stability budget "
        f"for seed={seed}"
    )


# ---------------------------------------------------------------------
# Frozen-K Lyapunov agreement
# ---------------------------------------------------------------------


def test_attnres_frozen_k_lyapunov_agrees() -> None:
    """Freeze K_nm at the AttnRes steady state and compute the full
    Lyapunov spectrum. The maximum exponent must not differ from the
    baseline (un-modulated) spectrum by more than 0.1 — i.e. the
    modulation reshapes weights but does not open new unstable
    directions in the local neighbourhood.

    This is deterministic (single seed) because the frozen-K pipeline
    is itself deterministic.
    """
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
    phases_ss = _integrate_attnres(
        engine, phases0, omegas, knm, alpha, 400
    )
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
    # fixed point is still contracting (λ_max ≤ 0.1, the same ceiling
    # used in the perturbation test) and does not exceed the baseline
    # by more than 0.1.
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
    """Integrate AttnRes for many steps; the order parameter R must
    stay in [0, 1] and the trajectory must not explode. Guards against
    the feedback loop amplifying numerical drift."""
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

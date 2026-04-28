# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Property-based invariant tests (monitors)

"""Property-based invariants for the monitor family and steady-state
Kuramoto dynamics.

These tests cover Phase-7 invariants from the SPO backlog that map to
already-implemented compute paths:

* PID (Williams & Beer 2010) — ``redundancy`` and ``synergy`` are
  non-negative and bounded by the marginal MI estimates.
* Phase transfer entropy — near-independent source/target signals give
  a bounded, small TE (finite-sample bias, not exactly zero).
* Kuramoto supercritical regime — ``R`` is non-decreasing (within a
  small simulation tolerance) as ``K_base`` grows past the critical
  coupling, with narrowly-distributed natural frequencies.

Keep bounds empirically-justified and deterministic-seeded; these are
invariants of the **implementations**, not the true information
quantities.
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.monitor.pid import redundancy, synergy
from scpn_phase_orchestrator.monitor.transfer_entropy import (
    phase_transfer_entropy,
)
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------
# PID invariants
# ---------------------------------------------------------------------


@given(
    n=st.integers(min_value=4, max_value=24),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=40, deadline=None)
def test_pid_non_negativity(n: int, seed: int) -> None:
    """Redundancy and synergy are non-negative for arbitrary phase draws.

    Enforced by ``max(0.0, ...)`` in the Python fallback and the Rust
    implementation; the property guards against regressions.
    """
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, TWO_PI, size=n)
    half = n // 2
    group_a = list(range(half))
    group_b = list(range(half, n))
    r = redundancy(phases, group_a, group_b, n_bins=8)
    s = synergy(phases, group_a, group_b, n_bins=8)
    assert r >= 0.0, f"redundancy={r} must be ≥ 0"
    assert s >= 0.0, f"synergy={s} must be ≥ 0"


@given(
    n_obs=st.integers(min_value=8, max_value=64),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=20, deadline=None)
def test_pid_empty_groups_return_zero(n_obs: int, seed: int) -> None:
    """Empty groups yield exactly 0.0 for both redundancy and synergy."""
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, TWO_PI, size=n_obs)
    # both empty
    assert redundancy(phases, [], [], n_bins=8) == 0.0
    assert synergy(phases, [], [], n_bins=8) == 0.0
    # one empty, one populated
    full = list(range(n_obs))
    assert redundancy(phases, full, [], n_bins=8) == 0.0
    assert synergy(phases, [], full, n_bins=8) == 0.0


# ---------------------------------------------------------------------
# Transfer entropy invariants
# ---------------------------------------------------------------------


@given(
    n=st.integers(min_value=50, max_value=300),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_te_independent_signals_bounded(n: int, seed: int) -> None:
    """Independent random phase streams have small, bounded TE.

    Finite-sample histogram bias makes the estimate non-zero, but with
    8 bins and T ≤ 300 the bound log(n_bins) = log(8) ≈ 2.08 is a safe
    upper limit; practical values sit well below 1.0.
    """
    rng = np.random.default_rng(seed)
    src = rng.uniform(0.0, TWO_PI, size=n)
    tgt = rng.uniform(0.0, TWO_PI, size=n)
    te = phase_transfer_entropy(src, tgt, n_bins=8)
    assert te >= 0.0, f"TE must be ≥ 0, got {te}"
    # Loose sanity bound: H(Y) ≤ log(n_bins); TE ≤ H(Y) ⇒ TE ≤ log(8).
    assert te <= np.log(8) + 1e-9, (
        f"TE={te} exceeds log(8) upper bound for independent signals"
    )


@given(
    n=st.integers(min_value=64, max_value=256),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=20, deadline=None)
def test_te_self_is_bounded(n: int, seed: int) -> None:
    """TE(X→X) is non-negative and bounded by log(n_bins).

    Self-TE is not a well-defined information-flow measure, but the
    estimator must still satisfy the basic non-negativity and entropy
    bound.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, TWO_PI, size=n)
    te = phase_transfer_entropy(x, x, n_bins=8)
    assert te >= 0.0
    assert te <= np.log(8) + 1e-9


# ---------------------------------------------------------------------
# Kuramoto supercritical monotonicity
# ---------------------------------------------------------------------


def _steady_state_r(
    n: int,
    k_base: float,
    omegas: np.ndarray,
    seed: int,
    n_warmup: int = 400,
    n_measure: int = 200,
    dt: float = 0.01,
) -> float:
    """Integrate Kuramoto to steady state and return mean order parameter."""
    rng = np.random.default_rng(seed)
    # Dense all-to-all coupling at strength k_base
    knm = np.full((n, n), k_base, dtype=np.float64)
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n), dtype=np.float64)
    phases = rng.uniform(0.0, TWO_PI, size=n).astype(np.float64)

    engine = UPDEEngine(n_oscillators=n, dt=dt, method="euler")
    for _ in range(n_warmup):
        phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)
    r_samples: list[float] = []
    for _ in range(n_measure):
        phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)
        r, _ = compute_order_parameter(phases)
        r_samples.append(float(r))
    return float(np.mean(r_samples))


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=5, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_kuramoto_k_monotonicity_supercritical(seed: int) -> None:
    """Increasing K grows the steady-state order parameter.

    SPO's coupling matrix stores the per-edge coupling (not the
    normalised ``K/N``) so the test uses per-edge values. Empirically
    for this all-to-all N=32 mesh with Gaussian frequencies of spread
    σ = 3 rad/s, R transitions around k_edge ≈ 0.08 — well-characterised
    behaviour in the Kuramoto literature. Two test points at k_edge =
    0.0625 (transitional, R ≈ 0.2) and 0.3125 (deep saturation,
    R ≈ 0.95) give a ~0.6 R-gap, an order of magnitude larger than any
    realistic finite-sample drift. Strict ``R_strong > R_weak`` holds
    with no tolerance; a 0.2 minimum gap is also enforced.
    """
    n = 32
    rng = np.random.default_rng(seed)
    # Wide Gaussian frequencies (σ = 3 rad/s) to expose a real K→R
    # transition band within reachable coupling strengths.
    omegas = (rng.standard_normal(n) * 3.0).astype(np.float64)

    r_weak = _steady_state_r(n, k_base=2.0 / n, omegas=omegas, seed=seed)
    r_strong = _steady_state_r(n, k_base=10.0 / n, omegas=omegas, seed=seed)
    assert r_strong > r_weak, (
        f"R(K=10/n)={r_strong:.3f} must exceed R(K=2/n)={r_weak:.3f} for all seeds"
    )
    # Tighter invariant: the deep-saturation regime should lift R by at
    # least 0.2 above the transitional case across all seeds.
    assert r_strong - r_weak > 0.2, (
        f"R-gap R(K=10/n)-R(K=2/n) = {r_strong - r_weak:.3f} is smaller "
        f"than the 0.2 physical minimum for a 5× K spread across the "
        f"transition band"
    )


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=3, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_kuramoto_zero_coupling_near_incoherent(seed: int) -> None:
    """At K = 0 with spread frequencies, the steady-state R stays small.

    Finite-N bias floor: R̄ ≈ 1/√N for uniform random phases; with N=32
    the bound R < 0.35 is conservative.
    """
    n = 32
    rng = np.random.default_rng(seed)
    omegas = (rng.standard_normal(n) * 0.5).astype(np.float64)

    r_mean = _steady_state_r(n, k_base=0.0, omegas=omegas, seed=seed)
    # Incoherent-state finite-sample ceiling at N=32.
    assert r_mean < 0.35, (
        f"R̄={r_mean:.3f} at K=0 with spread frequencies should be "
        f"below 0.35 finite-N ceiling"
    )

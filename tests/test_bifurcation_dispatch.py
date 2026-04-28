# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Dispatch tests for upde.bifurcation

"""Verify that the Python-composite branch of
``upde.bifurcation`` routes through the 5-backend dispatcher in
:mod:`upde.basin_stability`.

The module carries two Rust fast paths (``_rust_trace``,
``_rust_find_kc``) that batch the whole K-sweep inside Rust.
When those are unavailable — or temporarily disabled for a test —
the Python composite must delegate to
``basin_stability.steady_state_r``, which itself dispatches
across Rust / Mojo / Julia / Go / Python.
"""

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.upde import basin_stability as bs
from scpn_phase_orchestrator.upde import bifurcation as bif


class TestPythonCompositeUsesDispatchedKernel:
    """Force ``_HAS_COMPOSITE_RUST = False`` to exercise the Python
    composite branch, and verify the per-K call goes through
    ``basin_stability.steady_state_r`` rather than a local kernel."""

    def _rewire(self, monkeypatch, calls: list[tuple]) -> None:
        monkeypatch.setattr(bif, "_HAS_COMPOSITE_RUST", False)
        orig = bs.steady_state_r

        def _spy(*args, **kwargs):
            calls.append((len(args), tuple(kwargs)))
            return orig(*args, **kwargs)

        monkeypatch.setattr(
            bif,
            "_dispatched_steady_state_r",
            _spy,
        )

    def test_trace_invokes_dispatched_kernel(self, monkeypatch):
        calls: list[tuple] = []
        self._rewire(monkeypatch, calls)
        omegas = np.array([1.0, 1.2, 0.9, 1.1])
        diagram = bif.trace_sync_transition(
            omegas,
            K_range=(0.0, 2.0),
            n_points=5,
            dt=0.01,
            n_transient=50,
            n_measure=30,
            seed=7,
        )
        assert len(diagram.points) == 5
        assert len(calls) == 5  # one per K sample
        for p in diagram.points:
            assert 0.0 <= p.R <= 1.0 + 1e-12

    def test_find_kc_invokes_dispatched_kernel(self, monkeypatch):
        calls: list[tuple] = []
        self._rewire(monkeypatch, calls)
        omegas = np.array([1.0, 1.2, 0.9, 1.1])
        kc = bif.find_critical_coupling(
            omegas,
            dt=0.01,
            n_transient=50,
            n_measure=30,
            tol=0.1,
            seed=7,
        )
        # At least the initial K=20 probe + one bisection step.
        assert len(calls) >= 2
        # Strong coupling guarantees a transition → finite K_c.
        assert np.isfinite(kc)

    def test_trace_finds_transition_in_fallback_mode(self, monkeypatch):
        """End-to-end: the Python composite path must produce a
        sensible R(K) curve — R grows with K across the sweep,
        reaching near-locked at the upper end of the range."""
        calls: list[tuple] = []
        self._rewire(monkeypatch, calls)
        omegas = np.zeros(5)  # identical oscillators
        diagram = bif.trace_sync_transition(
            omegas,
            K_range=(0.0, 3.0),
            n_points=4,
            dt=0.01,
            n_transient=100,
            n_measure=50,
            seed=1,
        )
        r_vals = diagram.R_values
        # Monotone growth from the K=0 baseline to the locked
        # regime at K=3 (R_final > 0.9 for identical oscillators).
        assert r_vals[-1] > r_vals[0]
        assert r_vals[-1] > 0.9

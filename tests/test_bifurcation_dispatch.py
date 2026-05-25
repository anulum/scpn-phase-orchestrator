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

import importlib.util
import sys
from types import ModuleType

import numpy as np
import pytest

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


class TestCompositeRustFastPathContracts:
    """Exercise the batched Rust fast-path boundary with deterministic fakes."""

    def _load_with_fake_spo_kernel(self, monkeypatch, fake_kernel):
        monkeypatch.setitem(sys.modules, "spo_kernel", fake_kernel)
        spec = importlib.util.spec_from_file_location(
            "test_bifurcation_with_fake_spo_kernel",
            bif.__file__,
        )
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_trace_uses_composite_rust_batch_and_preserves_kc(self, monkeypatch):
        captured: dict[str, object] = {}
        fake_kernel = ModuleType("spo_kernel")

        def _trace(
            omegas,
            knm_flat,
            alpha_flat,
            n,
            phases,
            K_start,
            K_stop,
            n_points,
            dt,
            n_transient,
            n_measure,
        ):
            captured["shape"] = (
                len(omegas),
                len(knm_flat),
                len(alpha_flat),
                len(phases),
            )
            captured["scan"] = (K_start, K_stop, n_points)
            captured["integration"] = (dt, n_transient, n_measure)
            return (
                np.array([0.0, 1.0, 2.0]),
                np.array([0.02, 0.11, 0.80]),
                1.25,
            )

        def _find(*_args):
            raise AssertionError("trace test must not call the Kc kernel")

        fake_kernel.trace_sync_transition_rust = _trace
        fake_kernel.find_critical_coupling_bif_rust = _find
        module = self._load_with_fake_spo_kernel(monkeypatch, fake_kernel)
        diagram = module.trace_sync_transition(
            np.array([-0.3, 0.0, 0.3]),
            K_range=(0.0, 2.0),
            n_points=3,
            dt=0.02,
            n_transient=7,
            n_measure=5,
            seed=11,
        )
        assert module._HAS_COMPOSITE_RUST is True
        assert captured["shape"] == (3, 9, 9, 3)
        assert captured["scan"] == (0.0, 2.0, 3)
        assert captured["integration"] == (0.02, 7, 5)
        assert diagram.K_critical == 1.25
        assert [point.stable for point in diagram.points] == [True, True, True]
        np.testing.assert_allclose(diagram.K_values, [0.0, 1.0, 2.0])
        np.testing.assert_allclose(diagram.R_values, [0.02, 0.11, 0.80])

    def test_trace_leaves_kc_unset_when_rust_reports_nan(self, monkeypatch):
        fake_kernel = ModuleType("spo_kernel")

        def _trace(*_args):
            return (
                np.array([0.0, 1.0]),
                np.array([0.01, 0.04]),
                float("nan"),
            )

        def _find(*_args):
            raise AssertionError("trace test must not call the Kc kernel")

        fake_kernel.trace_sync_transition_rust = _trace
        fake_kernel.find_critical_coupling_bif_rust = _find
        module = self._load_with_fake_spo_kernel(monkeypatch, fake_kernel)
        diagram = module.trace_sync_transition(
            np.array([-1.0, 1.0]),
            K_range=(0.0, 1.0),
            n_points=2,
            n_transient=3,
            n_measure=2,
        )
        assert diagram.K_critical is None
        np.testing.assert_allclose(diagram.R_values, [0.01, 0.04])

    @pytest.mark.parametrize(
        ("K_values", "R_values", "K_critical", "match"),
        [
            ([0.0], [0.1], 0.5, "unexpected shape"),
            ([0.0, 1.0], [0.1, 1.2], 0.5, "R outside"),
            ([1.0, 0.0], [0.1, 0.2], 0.5, "non-monotone"),
            ([0.0, 3.0], [0.1, 0.2], 0.5, "outside K_range"),
            ([0.0, 1.0], [0.1, 0.2], float("inf"), "K_critical"),
        ],
    )
    def test_trace_rejects_invalid_composite_rust_physics(
        self,
        monkeypatch,
        K_values,
        R_values,
        K_critical,
        match,
    ):
        fake_kernel = ModuleType("spo_kernel")

        def _trace(*_args):
            return (
                np.array(K_values, dtype=np.float64),
                np.array(R_values, dtype=np.float64),
                K_critical,
            )

        def _find(*_args):
            raise AssertionError("trace test must not call the Kc kernel")

        fake_kernel.trace_sync_transition_rust = _trace
        fake_kernel.find_critical_coupling_bif_rust = _find
        module = self._load_with_fake_spo_kernel(monkeypatch, fake_kernel)

        with pytest.raises(ValueError, match=match):
            module.trace_sync_transition(
                np.array([-1.0, 1.0]),
                K_range=(0.0, 1.0),
                n_points=2,
                n_transient=3,
                n_measure=2,
            )

    def test_find_kc_delegates_to_composite_rust_search(self, monkeypatch):
        captured: dict[str, object] = {}
        fake_kernel = ModuleType("spo_kernel")

        def _trace(*_args):
            raise AssertionError("find test must not call the trace kernel")

        def _find(
            omegas,
            knm_flat,
            alpha_flat,
            n,
            phases,
            dt,
            n_transient,
            n_measure,
            tol,
        ):
            captured["shape"] = (
                len(omegas),
                len(knm_flat),
                len(alpha_flat),
                len(phases),
                n,
            )
            captured["integration"] = (dt, n_transient, n_measure, tol)
            return 2.75

        fake_kernel.trace_sync_transition_rust = _trace
        fake_kernel.find_critical_coupling_bif_rust = _find
        module = self._load_with_fake_spo_kernel(monkeypatch, fake_kernel)
        kc = module.find_critical_coupling(
            np.array([-0.4, 0.0, 0.4]),
            dt=0.03,
            n_transient=9,
            n_measure=6,
            tol=0.2,
            seed=5,
        )
        assert kc == 2.75
        assert captured["shape"] == (3, 9, 9, 3, 3)
        assert captured["integration"] == (0.03, 9, 6, 0.2)

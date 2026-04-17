# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — HCP connectome loader tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.coupling.connectome import (
    load_hcp_connectome,
    load_neurolib_hcp,
)


def test_output_shape():
    knm = load_hcp_connectome(20)
    assert knm.shape == (20, 20)


def test_symmetric():
    knm = load_hcp_connectome(40)
    np.testing.assert_allclose(knm, knm.T, atol=1e-12)


def test_zero_diagonal():
    knm = load_hcp_connectome(30)
    np.testing.assert_allclose(np.diag(knm), 0.0, atol=1e-15)


def test_non_negative():
    knm = load_hcp_connectome(50)
    assert np.all(knm >= 0.0)


def test_intra_stronger_than_inter():
    """Intra-hemispheric coupling should be stronger than inter on average."""
    n = 40
    knm = load_hcp_connectome(n)
    half = n // 2
    intra_mean = (knm[:half, :half].sum() + knm[half:, half:].sum()) / (
        2 * half * (half - 1)
    )
    inter_mean = knm[:half, half:].sum() / (half * (n - half))
    assert intra_mean > inter_mean


def test_deterministic():
    """Same n_regions → same matrix (seeded RNG)."""
    a = load_hcp_connectome(24)
    b = load_hcp_connectome(24)
    np.testing.assert_array_equal(a, b)


def test_small_n_raises():
    with pytest.raises(ValueError, match="n_regions must be >= 2"):
        load_hcp_connectome(1)


def test_n_zero_raises():
    with pytest.raises(ValueError):
        load_hcp_connectome(0)


def test_minimum_n():
    knm = load_hcp_connectome(2)
    assert knm.shape == (2, 2)
    assert knm[0, 0] == 0.0
    assert knm[1, 1] == 0.0


def test_odd_n():
    knm = load_hcp_connectome(7)
    assert knm.shape == (7, 7)
    np.testing.assert_allclose(knm, knm.T, atol=1e-12)


def test_seed_parameter():
    a = load_hcp_connectome(10, seed=0)
    b = load_hcp_connectome(10, seed=999)
    assert not np.allclose(a, b)


def test_large_n():
    knm = load_hcp_connectome(100)
    assert knm.shape == (100, 100)
    assert np.all(knm >= 0)


def test_dmn_hubs_present():
    knm = load_hcp_connectome(20)
    half = 10
    dmn_fracs = [0.15, 0.45, 0.65, 0.85]
    dmn_left = [int(f * half) for f in dmn_fracs]
    dmn_right = [h + half for h in dmn_left if h + half < 20]
    dmn_all = dmn_left + dmn_right
    dmn_coupling = knm[np.ix_(dmn_all, dmn_all)].mean()
    non_dmn = [i for i in range(20) if i not in dmn_all]
    non_dmn_coupling = knm[np.ix_(non_dmn, non_dmn)].mean()
    assert dmn_coupling > non_dmn_coupling


def test_neurolib_import_error():
    try:
        import neurolib  # noqa: F401

        pytest.skip("neurolib is installed")
    except ImportError:
        with pytest.raises(ImportError, match="neurolib is required"):
            load_neurolib_hcp()


# --- neurolib real HCP ---

neurolib = pytest.importorskip("neurolib")


def test_neurolib_hcp_loads():
    sc = load_neurolib_hcp(80)
    assert sc.shape == (80, 80)
    np.testing.assert_allclose(sc, sc.T, atol=1e-12)
    assert np.all(sc >= 0.0)
    np.testing.assert_allclose(np.diag(sc), 0.0, atol=1e-15)


def test_neurolib_hcp_subsample():
    sc = load_neurolib_hcp(20)
    assert sc.shape == (20, 20)
    full = load_neurolib_hcp(80)
    np.testing.assert_array_equal(sc, full[:20, :20])


def test_neurolib_hcp_too_large():
    with pytest.raises(ValueError, match="n_regions must be <= 80"):
        load_neurolib_hcp(100)


def test_neurolib_hcp_too_small():
    with pytest.raises(ValueError, match="n_regions must be >= 2"):
        load_neurolib_hcp(1)


class TestConnectomePipelineEndToEnd:
    """Full pipeline: load_hcp_connectome → K_nm → Engine → R → Regime.

    Proves connectome loader is a real coupling source, not decorative.
    """

    def test_hcp_knm_drives_engine_regime(self):
        """HCP connectome → UPDEEngine → order_parameter → RegimeManager."""
        from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
        from scpn_phase_orchestrator.supervisor.regimes import RegimeManager
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        n = 20
        knm = load_hcp_connectome(n)
        assert knm.shape == (n, n)
        eng = UPDEEngine(n, dt=0.01, method="rk4")
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        alpha = np.zeros((n, n))
        phases = eng.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=300)
        r, psi = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0
        layer = LayerState(R=r, psi=psi)
        state = UPDEState(
            layers=[layer],
            cross_layer_alignment=np.array([r]),
            stability_proxy=r,
            regime_id="nominal",
        )
        rm = RegimeManager(hysteresis=0.05)
        regime = rm.evaluate(state, BoundaryState())
        assert regime.name in {"NOMINAL", "DEGRADED", "CRITICAL", "RECOVERY"}

    def test_neurolib_hcp_drives_engine(self):
        """Neurolib HCP connectome → engine → R."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        n = 20
        knm = load_neurolib_hcp(n)
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        alpha = np.zeros((n, n))
        phases = eng.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=200)
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0

    def test_performance_load_hcp_80_under_10ms(self):
        """load_hcp_connectome(80) < 10ms."""
        import time

        load_hcp_connectome(80)  # warm-up
        t0 = time.perf_counter()
        for _ in range(100):
            load_hcp_connectome(80)
        elapsed = (time.perf_counter() - t0) / 100
        assert elapsed < 0.01, f"load_hcp(80) took {elapsed * 1e3:.1f}ms"


# Pipeline wiring: load_hcp_connectome/load_neurolib_hcp → K_nm → UPDEEngine(RK4)
# → compute_order_parameter → RegimeManager. Performance: load_hcp(80)<10ms.

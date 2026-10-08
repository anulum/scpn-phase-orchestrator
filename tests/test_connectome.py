# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — HCP connectome loader tests

"""Exercise original synthetic, optional HCP and downstream phase consumers."""

from __future__ import annotations

from typing import cast, get_type_hints

import numpy as np
import pytest
from numpy.typing import NDArray

from benchmarks.connectome_reference import (
    reference_connectome,
    reference_hcp,
    reference_trajectory,
)
from scpn_phase_orchestrator.coupling.connectome import (
    load_hcp_connectome,
    load_neurolib_hcp,
)
from tests.test_connectome_real_runtime import inject_native_fault, installed_matrix
from tests.typing_contracts import assert_precise_ndarray_hint


def test_public_array_contracts_are_parameterised() -> None:
    """Public connectome loaders return typed float arrays."""
    for hint in [
        get_type_hints(load_hcp_connectome)["return"],
        get_type_hints(load_neurolib_hcp)["return"],
    ]:
        assert_precise_ndarray_hint(hint)
        assert "float64" in str(hint)


def test_output_shape() -> None:
    """The twenty-region public result preserves the declared square layout."""
    knm = load_hcp_connectome(20)
    assert knm.shape == (20, 20)


def test_symmetric() -> None:
    """Public structural weights remain symmetric across both hemispheres."""
    knm = load_hcp_connectome(40)
    np.testing.assert_allclose(knm, knm.T, atol=1e-12)


def test_zero_diagonal() -> None:
    """The public generator introduces no self-coupling at any region."""
    knm = load_hcp_connectome(30)
    np.testing.assert_allclose(np.diag(knm), 0.0, atol=1e-15)


def test_non_negative() -> None:
    """Noise clipping and hub assembly retain non-negative public weights."""
    knm = load_hcp_connectome(50)
    assert np.all(knm >= 0.0)


def test_intra_larger_than_inter() -> None:
    """Intra-hemispheric coupling should be larger than inter on average."""
    n = 40
    knm = load_hcp_connectome(n)
    half = n // 2
    intra_mean = (knm[:half, :half].sum() + knm[half:, half:].sum()) / (
        2 * half * (half - 1)
    )
    inter_mean = knm[:half, half:].sum() / (half * (n - half))
    assert intra_mean > inter_mean


def test_deterministic() -> None:
    """Same n_regions → same matrix (seeded RNG)."""
    a = load_hcp_connectome(24)
    b = load_hcp_connectome(24)
    np.testing.assert_array_equal(a, b)


def test_small_n_raises() -> None:
    """A one-region request is refused before any generator runs."""
    with pytest.raises(ValueError, match="n_regions must be >= 2"):
        load_hcp_connectome(1)


def test_n_zero_raises() -> None:
    """An empty region request cannot enter dense generation."""
    with pytest.raises(ValueError):
        load_hcp_connectome(0)


def test_n_regions_rejects_bool_and_non_integer() -> None:
    """Counts retain their integer meaning before optional native dispatch."""
    with pytest.raises(TypeError, match="n_regions must be an integer"):
        load_hcp_connectome(True)
    with pytest.raises(TypeError, match="n_regions must be an integer"):
        load_hcp_connectome(cast(int, 2.5))


def test_seed_rejects_bool_and_out_of_u64_range() -> None:
    """Seed aliases and negative values are refused before RNG execution."""
    with pytest.raises(TypeError, match="seed must be an integer"):
        load_hcp_connectome(2, seed=False)
    with pytest.raises(ValueError, match="seed must be an integer in the u64 range"):
        load_hcp_connectome(2, seed=-1)


def test_minimum_n() -> None:
    """The minimum two-region graph preserves its exact zero diagonal."""
    knm = load_hcp_connectome(2)
    assert knm.shape == (2, 2)
    assert knm[0, 0] == 0.0
    assert knm[1, 1] == 0.0


def test_odd_n() -> None:
    """An extra right-hemisphere region preserves symmetric public weights."""
    knm = load_hcp_connectome(7)
    assert knm.shape == (7, 7)
    np.testing.assert_allclose(knm, knm.T, atol=1e-12)


def test_seed_parameter() -> None:
    """Changing a valid seed changes the actual generated noise."""
    a = load_hcp_connectome(10, seed=0)
    b = load_hcp_connectome(10, seed=999)
    assert not np.allclose(a, b)


def test_large_n() -> None:
    """The hundred-region public matrix remains finite and non-negative."""
    knm = load_hcp_connectome(100)
    assert knm.shape == (100, 100)
    assert np.all(knm >= 0)


def test_dmn_hubs_present() -> None:
    """Exercise the public loader contract for dmn hubs present."""
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


@pytest.mark.native_runtime
def test_optional_rust_loader_returns_validated_matrix_contract() -> None:
    """The original compiled installed owner obeys the full independent edge law."""
    actual = installed_matrix("rust", 3, 17)
    np.testing.assert_allclose(actual, reference_connectome(3, 17, "rust"), atol=3e-14)


@pytest.mark.native_runtime
def test_optional_rust_loader_rejects_contract_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A self-edge fault after original generation is refused by the public loader."""

    def corrupt(matrix: NDArray[np.float64]) -> object:
        """Inject the invalid self-edge after the original native producer."""
        matrix[0, 0] = 0.25
        return matrix

    inject_native_fault(monkeypatch, corrupt)
    with pytest.raises(ValueError, match="diagonal must be zero"):
        load_hcp_connectome(3, 17)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            np.asarray([[0.0, np.bool_(True)], [np.bool_(True), 0.0]], dtype=object),
            "must not contain boolean values",
        ),
        (
            np.asarray(
                [[0.0, complex(0.2, 0.0)], [complex(0.2, 0.0), 0.0]],
                dtype=object,
            ),
            "must contain real-valued weights",
        ),
        (
            np.asarray([["0.0", "0.2"], ["0.2", "0.0"]]),
            "must not contain numeric-string aliases",
        ),
        (
            np.asarray([[b"0.0", b"0.2"], [b"0.2", b"0.0"]]),
            "must not contain numeric-string aliases",
        ),
    ],
)
@pytest.mark.native_runtime
def test_optional_rust_loader_rejects_coercive_source_aliases(
    monkeypatch: pytest.MonkeyPatch,
    payload: object,
    message: str,
) -> None:
    """Declared source-alias faults are refused after original native execution."""

    def corrupt(matrix: NDArray[np.float64]) -> object:
        """Inject one scalar alias while retaining original generated weights."""
        damaged = matrix.astype(object)
        bad = np.asarray(payload, dtype=object)[0, 1]
        damaged[0, 1] = damaged[1, 0] = bad
        return damaged

    inject_native_fault(monkeypatch, corrupt)
    with pytest.raises(ValueError, match=message):
        load_hcp_connectome(2, 17)


@pytest.mark.native_runtime
def test_neurolib_import_error() -> None:
    """The actually neurolib-absent installed profile refuses real-data loading."""
    with pytest.raises(ImportError, match="neurolib is required"):
        installed_matrix("python", 80, kind="hcp")


# --- neurolib real HCP ---


@pytest.mark.native_runtime
def test_neurolib_hcp_loads() -> None:
    """The real installed HCP reader matches independent subject-file averaging."""
    actual = installed_matrix("rust", 80, kind="hcp")
    np.testing.assert_allclose(actual, reference_hcp(), atol=2e-15, rtol=2e-15)
    np.testing.assert_array_equal(np.diag(actual), np.zeros(80))


@pytest.mark.native_runtime
def test_neurolib_hcp_subsample() -> None:
    """Real installed cortical slicing preserves original subject-average weights."""
    actual = installed_matrix("rust", 20, kind="hcp")
    np.testing.assert_allclose(actual, reference_hcp(20), atol=2e-15, rtol=2e-15)


@pytest.mark.native_runtime
def test_neurolib_hcp_too_large() -> None:
    """The real public dataset path refuses counts exceeding its cortical atlas."""
    with pytest.raises(ValueError, match="n_regions must be <= 80"):
        installed_matrix("rust", 100, kind="hcp")


@pytest.mark.native_runtime
def test_neurolib_hcp_too_small() -> None:
    """The real public dataset path refuses a one-region request before data I/O."""
    with pytest.raises(ValueError, match="n_regions must be >= 2"):
        installed_matrix("rust", 1, kind="hcp")


class TestConnectomePipelineEndToEnd:
    """Full pipeline: load_hcp_connectome → K_nm → Engine → R → Regime.

    Proves connectome loader is a real coupling source, not decorative.
    """

    def test_hcp_knm_drives_engine_regime(self) -> None:
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
        initial = phases.copy()
        expected = reference_trajectory(initial, knm, 0.01, 300, "rk4")
        omegas = np.ones(n)
        alpha = np.zeros((n, n))
        phases = eng.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=300)
        np.testing.assert_allclose(
            np.angle(np.exp(1j * (phases - expected))), 0.0, atol=3e-12
        )
        assert np.max(np.abs(np.angle(np.exp(1j * (phases - initial - 3.0))))) > 0.01
        r, psi = compute_order_parameter(phases)
        independent_order = np.mean(np.exp(1j * expected))
        assert r == pytest.approx(abs(independent_order), abs=3e-12)
        assert np.angle(
            np.exp(1j * (psi - np.angle(independent_order)))
        ) == pytest.approx(0.0, abs=3e-12)
        layer = LayerState(R=r, psi=psi)
        state = UPDEState(
            layers=[layer],
            cross_layer_alignment=np.array([r]),
            stability_proxy=r,
            regime_id="nominal",
        )
        rm = RegimeManager(hysteresis=0.05)
        regime = rm.evaluate(state, BoundaryState())
        expected_r = abs(independent_order)
        expected_regime = (
            "CRITICAL"
            if expected_r < 0.3
            else "DEGRADED"
            if expected_r < 0.6
            else "NOMINAL"
        )
        assert regime.name == expected_regime

    @pytest.mark.native_runtime
    def test_neurolib_hcp_drives_engine(self) -> None:
        """Neurolib HCP connectome → engine → R."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        n = 20
        knm = load_neurolib_hcp(n)
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        expected = reference_trajectory(phases, knm, 0.01, 200, "euler")
        omegas = np.ones(n)
        alpha = np.zeros((n, n))
        phases = eng.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=200)
        np.testing.assert_allclose(
            np.angle(np.exp(1j * (phases - expected))), 0.0, atol=3e-12
        )
        r, _ = compute_order_parameter(phases)
        assert r == pytest.approx(abs(np.mean(np.exp(1j * expected))), abs=3e-12)

    def test_performance_load_hcp_80_under_10ms(self) -> None:
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

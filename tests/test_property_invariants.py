# SCPN Phase Orchestrator — Property-based invariant tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from scpn_phase_orchestrator.binding.types import BoundaryDef
from scpn_phase_orchestrator.coupling import CouplingBuilder
from scpn_phase_orchestrator.monitor.boundaries import (
    BoundaryObserver,
    BoundaryState,
)
from scpn_phase_orchestrator.supervisor.regimes import RegimeManager
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState
from scpn_phase_orchestrator.upde.order_params import (
    compute_order_parameter,
    compute_plv,
)

TWO_PI = 2.0 * np.pi

_FINITE_FLOATS = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
)

phase_arrays = arrays(
    dtype=np.float64,
    shape=st.integers(min_value=2, max_value=200),
    elements=_FINITE_FLOATS,
)


@given(phases=phase_arrays)
@settings(max_examples=200)
def test_order_parameter_bounds(phases: np.ndarray) -> None:
    """R in [0, 1], psi in [0, 2pi) for any finite phase array."""
    r, psi = compute_order_parameter(phases)
    assert -1e-12 <= r <= 1.0 + 1e-12, f"R={r}"
    assert -1e-12 <= psi < TWO_PI + 1e-12, f"psi={psi}"


@given(
    n=st.integers(min_value=2, max_value=16),
    base=st.floats(min_value=0.01, max_value=10.0, allow_nan=False),
    alpha=st.floats(min_value=0.01, max_value=5.0, allow_nan=False),
)
@settings(max_examples=100)
def test_coupling_symmetry(n: int, base: float, alpha: float) -> None:
    """CouplingBuilder: Knm == Knm^T, non-negative, zero-diagonal."""
    cs = CouplingBuilder().build(n, base, alpha)
    knm = cs.knm
    np.testing.assert_allclose(knm, knm.T, atol=1e-12, err_msg="not symmetric")
    assert np.all(knm >= 0.0), "negative entries"
    np.testing.assert_allclose(np.diag(knm), 0.0, atol=1e-15, err_msg="diagonal")


@given(
    a=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=2, max_value=500),
        elements=_FINITE_FLOATS,
    ),
)
@settings(max_examples=200)
def test_plv_bounds(a: np.ndarray) -> None:
    """PLV in [0, 1] for any two equal-length phase arrays."""
    b = np.roll(a, 1)
    plv = compute_plv(a, b)
    assert -1e-12 <= plv <= 1.0 + 1e-12, f"PLV={plv}"


@given(
    r_val=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    n_layers=st.integers(min_value=1, max_value=16),
)
@settings(max_examples=200)
def test_regime_determinism(r_val: float, n_layers: int) -> None:
    """Same input always produces the same regime."""
    layers = [LayerState(R=r_val, psi=0.0) for _ in range(n_layers)]
    state = UPDEState(
        layers=layers,
        cross_layer_alignment=np.zeros((n_layers, n_layers)),
        stability_proxy=0.0,
        regime_id="nominal",
    )
    boundary = BoundaryState()
    mgr1 = RegimeManager()
    mgr2 = RegimeManager()
    r1 = mgr1.evaluate(state, boundary)
    r2 = mgr2.evaluate(state, boundary)
    assert r1 == r2


@given(
    n_defs=st.integers(min_value=1, max_value=20),
    vals=st.dictionaries(
        keys=st.text(min_size=1, max_size=8, alphabet="abcdefgh"),
        values=st.floats(min_value=-100, max_value=100, allow_nan=False),
        min_size=1,
        max_size=10,
    ),
)
@settings(max_examples=100)
def test_boundary_monotonicity(n_defs: int, vals: dict[str, float]) -> None:
    """Violations count <= definitions count."""
    keys = list(vals.keys())
    defs = [
        BoundaryDef(
            name=f"b{i}",
            variable=keys[i % len(keys)],
            lower=-50.0,
            upper=50.0,
            severity="soft" if i % 2 == 0 else "hard",
        )
        for i in range(n_defs)
    ]
    observer = BoundaryObserver(defs)
    state = observer.observe(vals)
    assert len(state.violations) <= n_defs


@given(
    n=st.integers(min_value=2, max_value=16),
    dt=st.floats(min_value=1e-4, max_value=0.1, allow_nan=False),
)
@settings(max_examples=100)
def test_upde_phase_bounding(n: int, dt: float) -> None:
    """UPDEEngine.step() returns all phases in [0, 2*pi)."""
    rng = np.random.default_rng(0)
    phases = rng.uniform(0, TWO_PI, n)
    omegas = rng.uniform(-10.0, 10.0, n)
    knm = rng.uniform(0, 1.0, (n, n))
    knm = (knm + knm.T) / 2.0
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))

    engine = UPDEEngine(n, dt)
    result = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)

    assert result.shape == (n,)
    assert np.all(result >= 0.0), f"negative phase: {result.min()}"
    assert np.all(result < TWO_PI + 1e-12), f"phase >= 2pi: {result.max()}"
    assert np.all(np.isfinite(result)), "non-finite phase"

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for full multi-head AttnRes

"""Exercise the public phase attention coupling law and its error boundaries.

Signed graph/topology, periodicity, identity, projection shape and measurement
contracts use original production calls. The order-parameter comparison is a
bounded supercritical fixture with identical initial conditions, not a proof
of global stability or calibrated-anchor preservation. Invalid output/factory
substitutions are negative controls only and supply no successful runtime proof.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy.typing import DTypeLike, NDArray

from benchmarks.attnres_reference import (
    AttnResOptions,
    FloatArray,
    phase_attention_oracle,
)
from scpn_phase_orchestrator.coupling import attention_residuals as attnres_mod
from scpn_phase_orchestrator.coupling.attention_residuals import (
    ACTIVE_BACKEND,
    AVAILABLE_BACKENDS,
    PHASE_EMBED_DIM,
    attnres_modulate,
    default_projections,
)
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

TWO_PI = 2.0 * np.pi


class _FloatRejectingArray:
    """Reject floating coercion while permitting the array protocol object probe."""

    def __array__(self, dtype: DTypeLike | None = None) -> NDArray[np.object_]:
        """Expose object values only, so public ingress must reject float conversion."""
        if dtype is not None and np.dtype(dtype) == np.dtype(object):
            return np.array([0.0], dtype=object)
        raise TypeError("float conversion rejected")


def _symmetric_knm(n: int, strength: float = 0.3, seed: int = 0) -> FloatArray:
    """Draw a reproducible undirected graph with no self-coupling."""
    rng = np.random.default_rng(seed)
    half = rng.uniform(0.0, 2.0 * strength, size=(n, n))
    knm = 0.5 * (half + half.T)
    np.fill_diagonal(knm, 0.0)
    return knm.astype(np.float64)


# ---------------------------------------------------------------------
# default_projections
# ---------------------------------------------------------------------


class TestDefaultProjections:
    """Check seeded Fourier projection shapes, variability and integer domains."""

    def test_shapes(self) -> None:
        """Preserve all per-head dimensions and the square output projection."""
        w_q, w_k, w_v, w_o = default_projections(n_heads=4, seed=0)
        d_model = PHASE_EMBED_DIM
        d_head = d_model // 4
        assert w_q.shape == (4, d_model, d_head)
        assert w_k.shape == (4, d_model, d_head)
        assert w_v.shape == (4, d_model, d_head)
        assert w_o.shape == (4 * d_head, d_model)

    def test_seed_reproducible(self) -> None:
        """Reproduce all four projection buffers from the same integer seed."""
        a = default_projections(n_heads=4, seed=42)
        b = default_projections(n_heads=4, seed=42)
        for x, y in zip(a, b, strict=True):
            np.testing.assert_array_equal(x, y)

    def test_different_seeds_differ(self) -> None:
        """Produce different parameters when the seed changes."""
        a = default_projections(n_heads=4, seed=0)
        b = default_projections(n_heads=4, seed=1)
        # At least one of the four should differ — they're independently seeded.
        assert any(not np.array_equal(x, y) for x, y in zip(a, b, strict=True))

    def test_non_divisible_rejected(self) -> None:
        """Refuse a head count that cannot partition the feature width."""
        with pytest.raises(ValueError, match="not divisible"):
            default_projections(n_heads=3, d_model=8)

    @pytest.mark.parametrize("n_heads", [0, True, 1.5])
    def test_invalid_head_count_rejected(self, n_heads: object) -> None:
        """Reject zero, boolean and fractional head-count controls."""
        with pytest.raises(ValueError, match="n_heads"):
            default_projections(n_heads=cast("int", n_heads))

    @pytest.mark.parametrize("d_model", [0, True, 7, 8.5])
    def test_invalid_model_width_rejected(self, d_model: object) -> None:
        """Reject zero, boolean, odd and fractional Fourier widths."""
        with pytest.raises(ValueError, match="d_model"):
            default_projections(n_heads=1, d_model=cast("int", d_model))


# ---------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------


def test_symmetry_preserved() -> None:
    """Preserve undirected coupling under arbitrary oscillator phases."""
    knm = _symmetric_knm(8, seed=1)
    theta = np.linspace(0.0, TWO_PI, 8, endpoint=False)
    k_mod = attnres_modulate(knm, theta, lambda_=0.5)
    np.testing.assert_allclose(k_mod, k_mod.T, atol=1e-12)


def test_zero_diagonal_preserved() -> None:
    """Keep every self-coupling exactly zero after modulation."""
    knm = _symmetric_knm(12, seed=2)
    theta = np.random.default_rng(0).uniform(0.0, TWO_PI, size=12)
    k_mod = attnres_modulate(knm, theta, lambda_=0.5)
    np.testing.assert_array_equal(np.diag(k_mod), np.zeros(12))


def test_lambda_zero_is_identity() -> None:
    """Retain the input matrix when modulation is explicitly disabled."""
    knm = _symmetric_knm(16, seed=3)
    theta = np.random.default_rng(5).uniform(0.0, TWO_PI, size=16)
    k_mod = attnres_modulate(knm, theta, lambda_=0.0)
    np.testing.assert_array_equal(k_mod, knm)


def test_existing_zeros_stay_zero() -> None:
    """Preserve deliberately absent graph edges across phase-dependent attention."""
    knm = _symmetric_knm(8, seed=4)
    # Knock out symmetric pairs
    knm[0, 3] = knm[3, 0] = 0.0
    knm[2, 5] = knm[5, 2] = 0.0
    theta = np.random.default_rng(1).uniform(0.0, TWO_PI, size=8)
    k_mod = attnres_modulate(knm, theta, lambda_=0.5)
    for i, j in [(0, 3), (3, 0), (2, 5), (5, 2)]:
        assert k_mod[i, j] == 0.0


def test_periodicity_in_phase_angles_is_preserved() -> None:
    """Preserve coupling when phases advance by integral complete turns."""
    knm = _symmetric_knm(7, seed=9)
    theta = np.linspace(0.0, TWO_PI, 7, endpoint=False)
    shifted = theta + 4.0 * TWO_PI

    out = attnres_modulate(knm, theta, lambda_=0.5)
    shifted_out = attnres_modulate(knm, shifted, lambda_=0.5)

    np.testing.assert_allclose(out, shifted_out, atol=1e-12)


def test_block_size_restricts_attention() -> None:
    """With block_size = 2, pairs with |i - j| > 2 keep original K."""
    n = 12
    knm = _symmetric_knm(n, seed=7)
    theta = np.random.default_rng(4).uniform(0.0, TWO_PI, size=n)
    k_mod = attnres_modulate(knm, theta, block_size=2, lambda_=0.5)
    for i in range(n):
        for j in range(n):
            if abs(i - j) > 2:
                assert k_mod[i, j] == pytest.approx(knm[i, j], abs=1e-12), (
                    f"out-of-block ({i}, {j}) was modulated"
                )


def test_full_attention_default() -> None:
    """Modulate distant graph edges with the default unbounded attention mask."""
    n = 16
    knm = _symmetric_knm(n, strength=0.1, seed=99)
    theta = np.random.default_rng(13).uniform(0.0, TWO_PI, size=n)
    k_full = attnres_modulate(knm, theta, lambda_=0.5)
    k_block = attnres_modulate(knm, theta, block_size=2, lambda_=0.5)
    # The two results must differ for at least one distant pair.
    far_mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if abs(i - j) > 2:
                far_mask[i, j] = True
    assert np.any(np.abs(k_full - k_block)[far_mask] > 1e-8)


def test_small_temperature_remains_finite_and_modulates() -> None:
    """Keep concentrated, representable attention finite and nontrivial."""
    knm = _symmetric_knm(10, strength=0.9, seed=2026)
    theta = np.linspace(0.0, TWO_PI, knm.shape[0], endpoint=False)

    out = attnres_modulate(
        knm,
        theta,
        temperature=1e-6,
        lambda_=0.5,
        block_size=None,
    )

    assert np.all(np.isfinite(out))
    assert np.allclose(np.diag(out), np.zeros(out.shape[0]))
    assert np.any(np.abs(out - knm) > 1e-10)


# ---------------------------------------------------------------------
# Validation criterion — R within 5 % of baseline
# ---------------------------------------------------------------------


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(
    max_examples=3,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_paired_supercritical_order_parameter_stays_within_five_percent(
    seed: int,
) -> None:
    """Keep paired supercritical fixture coherence within five percent."""
    n = 16
    dt = 0.01
    n_warmup = 300
    n_measure = 200
    lambda_ = 0.1  # small modulation so the physics stays dominated
    # by the base coupling — doc §5 explicitly uses small λ here.

    rng = np.random.default_rng(seed)
    omegas = (rng.standard_normal(n) * 0.5).astype(np.float64)
    knm = _symmetric_knm(n, strength=5.0 / n, seed=seed)
    alpha = np.zeros((n, n), dtype=np.float64)

    phases0 = rng.uniform(0.0, TWO_PI, size=n).astype(np.float64)

    def _r(knm_fn: Callable[[FloatArray], FloatArray]) -> float:
        """Integrate identical phases and average measured coherence."""
        phases = phases0.copy()
        engine = UPDEEngine(n_oscillators=n, dt=dt, method="euler")
        for _ in range(n_warmup):
            phases = engine.step(phases, omegas, knm_fn(phases), 0.0, 0.0, alpha)
        rs: list[float] = []
        for _ in range(n_measure):
            phases = engine.step(phases, omegas, knm_fn(phases), 0.0, 0.0, alpha)
            r, _ = compute_order_parameter(phases)
            rs.append(float(r))
        return float(np.mean(rs))

    r_base = _r(lambda _theta: knm)
    r_attn = _r(lambda theta: attnres_modulate(knm, theta, lambda_=lambda_))

    rel = abs(r_attn - r_base) / max(r_base, 1e-6)
    assert rel <= 0.05, (
        f"R(attnres)={r_attn:.4f} vs R(baseline)={r_base:.4f}, "
        f"relative change {rel * 100:.2f}% exceeds 5 % budget"
    )


# ---------------------------------------------------------------------
# Constructor contracts
# ---------------------------------------------------------------------


class TestContractFailures:
    """Reject malformed measurements and projection topology before computation."""

    def test_non_square_knm_rejected(self) -> None:
        """Refuse a coupling buffer that is not a square graph."""
        with pytest.raises(ValueError, match="square"):
            attnres_modulate(np.zeros((4, 5)), np.zeros(4))

    def test_theta_shape_mismatch_rejected(self) -> None:
        """Refuse phases whose cardinality differs from the oscillator count."""
        with pytest.raises(ValueError, match="does not match"):
            attnres_modulate(_symmetric_knm(4), np.zeros(6))

    def test_block_size_zero_rejected(self) -> None:
        """Reject a zero index-band radius before optional dispatch."""
        with pytest.raises(ValueError, match="block_size"):
            attnres_modulate(_symmetric_knm(4), np.zeros(4), block_size=0)

    def test_temperature_zero_rejected(self) -> None:
        """Refuse an undefined zero-temperature attention calculation."""
        with pytest.raises(ValueError, match="temperature"):
            attnres_modulate(_symmetric_knm(4), np.zeros(4), temperature=0.0)

    def test_negative_lambda_rejected(self) -> None:
        """Reject modulation strength outside its non-negative domain."""
        with pytest.raises(ValueError, match="lambda_"):
            attnres_modulate(_symmetric_knm(4), np.zeros(4), lambda_=-0.1)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("knm", np.inf),
            ("theta", np.nan),
            ("temperature", np.inf),
            ("lambda_", np.nan),
        ],
    )
    def test_non_finite_numeric_inputs_rejected(
        self,
        field: str,
        value: float,
    ) -> None:
        """Reject non-finite coupling, phases and scalar controls at ingress."""
        knm = _symmetric_knm(4)
        theta = np.zeros(4)
        kwargs: dict[str, float] = {}
        if field == "knm":
            knm[0, 1] = knm[1, 0] = value
        elif field == "theta":
            theta[0] = value
        else:
            kwargs[field] = value

        with pytest.raises(ValueError, match=field):
            attnres_modulate(knm, theta, **cast("AttnResOptions", kwargs))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"temperature": True},
            {"lambda_": True},
            {"projection_seed": -1},
        ],
    )
    def test_scalar_constructor_aliases_rejected(
        self,
        kwargs: dict[str, object],
    ) -> None:
        """Reject boolean strengths/temperatures and negative projection seeds."""
        with pytest.raises(ValueError):
            attnres_modulate(
                _symmetric_knm(4), np.zeros(4), **cast("AttnResOptions", kwargs)
            )

    def test_default_projection_factory_must_fill_all_missing_slots(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Refuse an intentionally incomplete default factory result."""

        def broken_defaults(
            n_heads: int, seed: int
        ) -> tuple[FloatArray, None, FloatArray, FloatArray]:
            """Return an invalid missing-slot fixture solely to exercise refusal."""
            w_q, w_k, w_v, w_o = default_projections(n_heads=n_heads, seed=seed)
            return w_q, None, w_v, w_o

        monkeypatch.setattr(attnres_mod, "default_projections", broken_defaults)
        with pytest.raises(ValueError, match="attention projections"):
            attnres_modulate(_symmetric_knm(4), np.zeros(4), w_q=None)

    def test_query_key_value_shape_mismatch_rejected(self) -> None:
        """Reject incompatible query, key and value head tensors."""
        w_q = np.zeros((2, 4, 2), dtype=np.float64)
        w_k = np.zeros((2, 4, 1), dtype=np.float64)
        w_v = np.zeros((2, 4, 2), dtype=np.float64)
        w_o = np.zeros((4, 4), dtype=np.float64)
        with pytest.raises(ValueError, match="shape mismatch"):
            attnres_modulate(
                _symmetric_knm(4),
                np.zeros(4),
                w_q=w_q,
                w_k=w_k,
                w_v=w_v,
                w_o=w_o,
                n_heads=2,
            )

    def test_projection_head_count_mismatch_rejected(self) -> None:
        """Reject projection heads that disagree with the requested count."""
        w_q = np.zeros((1, 4, 4), dtype=np.float64)
        w_k = np.zeros((1, 4, 4), dtype=np.float64)
        w_v = np.zeros((1, 4, 4), dtype=np.float64)
        w_o = np.zeros((4, 4), dtype=np.float64)
        with pytest.raises(ValueError, match="leading dim"):
            attnres_modulate(
                _symmetric_knm(4),
                np.zeros(4),
                w_q=w_q,
                w_k=w_k,
                w_v=w_v,
                w_o=w_o,
                n_heads=2,
            )

    def test_projection_model_width_mismatch_rejected(self) -> None:
        """Reject a projection width incompatible with the Fourier topology."""
        w_q = np.zeros((2, 5, 2), dtype=np.float64)
        w_k = np.zeros((2, 5, 2), dtype=np.float64)
        w_v = np.zeros((2, 5, 2), dtype=np.float64)
        w_o = np.zeros((4, 5), dtype=np.float64)
        with pytest.raises(ValueError, match="d_model"):
            attnres_modulate(
                _symmetric_knm(4),
                np.zeros(4),
                w_q=w_q,
                w_k=w_k,
                w_v=w_v,
                w_o=w_o,
                n_heads=2,
            )

    def test_odd_projection_model_width_rejected(self) -> None:
        """Refuse an odd feature width that cannot form complete harmonic pairs."""
        w_q = np.zeros((1, 7, 7), dtype=np.float64)
        w_k = np.zeros((1, 7, 7), dtype=np.float64)
        w_v = np.zeros((1, 7, 7), dtype=np.float64)
        w_o = np.zeros((7, 7), dtype=np.float64)
        with pytest.raises(ValueError, match="d_model"):
            attnres_modulate(
                _symmetric_knm(4),
                np.zeros(4),
                w_q=w_q,
                w_k=w_k,
                w_v=w_v,
                w_o=w_o,
                n_heads=1,
            )

    def test_non_finite_projection_rejected(self) -> None:
        """Reject an infinite projection entry before native execution."""
        w_q, w_k, w_v, w_o = default_projections(n_heads=1)
        w_q = w_q.copy()
        w_q[0, 0, 0] = np.inf
        with pytest.raises(ValueError, match="w_q"):
            attnres_modulate(
                _symmetric_knm(4),
                np.zeros(4),
                w_q=w_q,
                w_k=w_k,
                w_v=w_v,
                w_o=w_o,
                n_heads=1,
            )

    def test_output_projection_shape_mismatch_rejected(self) -> None:
        """Refuse an output matrix incompatible with concatenated heads."""
        w_q = np.zeros((2, 4, 2), dtype=np.float64)
        w_k = np.zeros((2, 4, 2), dtype=np.float64)
        w_v = np.zeros((2, 4, 2), dtype=np.float64)
        w_o = np.zeros((3, 4), dtype=np.float64)
        with pytest.raises(ValueError, match="w_o shape"):
            attnres_modulate(
                _symmetric_knm(4),
                np.zeros(4),
                w_q=w_q,
                w_k=w_k,
                w_v=w_v,
                w_o=w_o,
                n_heads=2,
            )

    def test_projection_tensors_must_be_rank_three(self) -> None:
        """Reject rank-two query, key and value tensors."""
        w_q = np.zeros((2, 4), dtype=np.float64)
        w_k = np.zeros((2, 4), dtype=np.float64)
        w_v = np.zeros((2, 4), dtype=np.float64)
        w_o = np.zeros((4, 4), dtype=np.float64)

        with pytest.raises(ValueError, match="3-D"):
            attnres_modulate(
                _symmetric_knm(4),
                np.zeros(4),
                w_q=w_q,
                w_k=w_k,
                w_v=w_v,
                w_o=w_o,
                n_heads=2,
            )

    def test_projection_head_width_must_match_model_width(self) -> None:
        """Reject a head split that does not reconstruct the feature width."""
        w_q = np.zeros((2, 8, 3), dtype=np.float64)
        w_k = np.zeros((2, 8, 3), dtype=np.float64)
        w_v = np.zeros((2, 8, 3), dtype=np.float64)
        w_o = np.zeros((6, 8), dtype=np.float64)

        with pytest.raises(ValueError, match="d_model"):
            attnres_modulate(
                _symmetric_knm(4),
                np.zeros(4),
                w_q=w_q,
                w_k=w_k,
                w_v=w_v,
                w_o=w_o,
                n_heads=2,
            )


# ---------------------------------------------------------------------
# Idempotence
# ---------------------------------------------------------------------


def test_deterministic_same_inputs() -> None:
    """Reproduce the modulated matrix from identical inputs and parameters."""
    knm = _symmetric_knm(8, seed=11)
    theta = np.linspace(0.1, 0.1 + TWO_PI, 8, endpoint=False)
    a = attnres_modulate(knm, theta, lambda_=0.3)
    b = attnres_modulate(knm, theta, lambda_=0.3)
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------
# Multi-backend dispatcher
# ---------------------------------------------------------------------


class TestDispatcher:
    """Exercise original public owners and deterministic optional runtime selection."""

    def test_python_is_always_available(self) -> None:
        """Retain the mandatory NumPy owner after optional loader discovery."""
        assert "python" in AVAILABLE_BACKENDS
        assert AVAILABLE_BACKENDS[-1] == "python"

    def test_active_backend_is_first_available(self) -> None:
        """Report the first discovered owner as the automatic initial choice."""
        assert AVAILABLE_BACKENDS[0] == ACTIVE_BACKEND

    def test_available_owners_follow_fixed_preference_order(self) -> None:
        """Preserve the documented fixed loader order without asserting speed."""
        canonical = ["rust", "mojo", "julia", "go", "python"]
        indices = [canonical.index(b) for b in AVAILABLE_BACKENDS]
        assert indices == sorted(indices)

    def test_named_rust_executes_original_coupling_or_refuses_absence(self) -> None:
        """Compute through the installed Rust owner or prove its real absence."""
        knm = _symmetric_knm(3)
        theta = np.linspace(0.0, TWO_PI, 3, endpoint=False)
        if "rust" not in AVAILABLE_BACKENDS:
            with pytest.raises(ImportError):
                attnres_modulate(knm, theta, backend="rust")
            return
        weights = default_projections(n_heads=1, d_model=4, seed=42)
        expected = phase_attention_oracle(knm, theta, weights, strength=0.25)
        output = attnres_modulate(
            knm,
            theta,
            w_q=weights[0],
            w_k=weights[1],
            w_v=weights[2],
            w_o=weights[3],
            n_heads=1,
            lambda_=0.25,
            backend="rust",
        )
        np.testing.assert_allclose(output, expected, rtol=0.0, atol=1e-12)
        assert np.max(np.abs(output - knm)) > 1e-4

    def test_automatic_and_numpy_coupling_match_independent_law(self) -> None:
        """Default public dispatch and an explicit real NumPy call agree."""
        knm = _symmetric_knm(5)
        theta = np.linspace(0.0, TWO_PI, 5, endpoint=False)
        weights = default_projections(seed=12)
        expected = phase_attention_oracle(knm, theta, weights)
        actual = attnres_modulate(knm, theta, projection_seed=12)
        numpy = attnres_modulate(knm, theta, projection_seed=12, backend="python")
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(numpy, expected, rtol=0.0, atol=1e-12)

    def test_repeated_automatic_calls_match_independent_law(self) -> None:
        """Repeated public calls retain numerical behavior without changing loaders."""
        knm = _symmetric_knm(4)
        theta = np.array([0.1, 0.7, 1.8, 2.4])
        weights = default_projections(seed=6)
        expected = phase_attention_oracle(knm, theta, weights)
        for _ in range(2):
            output = attnres_modulate(knm, theta, projection_seed=6)
            np.testing.assert_allclose(output, expected, rtol=0.0, atol=1e-12)
        assert np.max(np.abs(output - knm)) > 1e-4

    def test_explicit_numpy_works_alongside_optional_owners(self) -> None:
        """Explicit NumPy computation is available independent of optional owners."""
        knm = _symmetric_knm(4)
        theta = np.linspace(0.0, TWO_PI, 4, endpoint=False)
        weights = default_projections()
        expected = phase_attention_oracle(knm, theta, weights)
        actual = attnres_modulate(knm, theta, backend="python")
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_python_reference_lambda_zero_returns_independent_identity_copy() -> None:
    """The public NumPy identity preserves values without aliasing the input."""
    knm = _symmetric_knm(5, seed=17)
    theta = np.linspace(0.0, TWO_PI, 5, endpoint=False)
    out = attnres_modulate(knm, theta, lambda_=0.0, backend="python")
    np.testing.assert_array_equal(out, knm)
    assert not np.shares_memory(out, knm)


def test_python_reference_block_mask_preserves_out_of_band_edges() -> None:
    """The public NumPy mask preserves distant edges and modulates neighbours."""
    knm = _symmetric_knm(7, strength=0.2, seed=23)
    theta = np.random.default_rng(23).uniform(0.0, TWO_PI, size=7)
    modulated = attnres_modulate(
        knm, theta, block_size=1, lambda_=0.4, backend="python"
    )
    far = np.abs(np.arange(7)[:, None] - np.arange(7)[None, :]) > 1
    np.testing.assert_array_equal(modulated[far], knm[far])
    assert np.max(np.abs(modulated - knm)) > 1e-4


def test_public_entry_rejects_boolean_and_complex_payload_aliases() -> None:
    """Reject boolean, complex and uncoercible measurement payloads."""
    knm = _symmetric_knm(3)
    theta = np.zeros(3, dtype=np.float64)

    with pytest.raises(ValueError, match="knm.*boolean"):
        attnres_modulate(np.array([[False, True], [True, False]]), np.zeros(2))

    with pytest.raises(ValueError, match="theta.*real-valued"):
        attnres_modulate(knm, np.array([0.0, 1.0 + 0.0j, 2.0], dtype=object))

    with pytest.raises(ValueError, match="theta.*finite real array"):
        attnres_modulate(knm, cast("FloatArray", _FloatRejectingArray()))

    w_q, w_k, w_v, w_o = default_projections(n_heads=1)
    w_q_obj = w_q.astype(object)
    w_q_obj[0, 0, 0] = 1.0 + 0.0j
    with pytest.raises(ValueError, match="w_q.*real-valued"):
        attnres_modulate(
            knm,
            theta,
            w_q=w_q_obj,
            w_k=w_k,
            w_v=w_v,
            w_o=w_o,
            n_heads=1,
        )


@pytest.mark.parametrize(
    "field",
    ["knm", "theta", "w_q", "w_k", "w_v", "w_o"],
)
def test_public_entry_rejects_numeric_string_array_aliases(field: str) -> None:
    """Reject numeric text in every coupling, phase and projection buffer."""
    knm = _symmetric_knm(3)
    theta = np.zeros(3, dtype=np.float64)
    w_q, w_k, w_v, w_o = default_projections(n_heads=1)
    payload: dict[str, object] = {
        "knm": knm,
        "theta": theta,
        "w_q": w_q,
        "w_k": w_k,
        "w_v": w_v,
        "w_o": w_o,
    }
    payload[field] = np.asarray(payload[field]).astype(str)

    with pytest.raises(ValueError, match=rf"{field}.*numeric-string"):
        attnres_modulate(
            cast("FloatArray", payload["knm"]),
            cast("FloatArray", payload["theta"]),
            w_q=cast("FloatArray", payload["w_q"]),
            w_k=cast("FloatArray", payload["w_k"]),
            w_v=cast("FloatArray", payload["w_v"]),
            w_o=cast("FloatArray", payload["w_o"]),
            n_heads=1,
        )


def test_public_entry_rejects_non_physical_coupling_topology() -> None:
    """Reject directed input and nonzero self-coupling for this undirected law."""
    theta = np.zeros(3, dtype=np.float64)

    asymmetric = _symmetric_knm(3)
    asymmetric[0, 1] += 0.25
    with pytest.raises(ValueError, match="knm must be symmetric"):
        attnres_modulate(asymmetric, theta)

    self_coupled = _symmetric_knm(3)
    self_coupled[1, 1] = 0.1
    with pytest.raises(ValueError, match="knm diagonal"):
        attnres_modulate(self_coupled, theta)


def test_public_entry_preserves_empty_system_contract() -> None:
    """Return a typed empty matrix for an empty oscillator graph."""
    out = attnres_modulate(
        np.zeros((0, 0), dtype=np.float64),
        np.zeros(0, dtype=np.float64),
    )

    assert out.shape == (0, 0)
    assert out.dtype == np.float64


def test_projection_seed_rejects_boolean_alias() -> None:
    """Reject boolean seeds in the factory and public modulation API."""
    with pytest.raises(ValueError, match="seed"):
        default_projections(seed=True)

    with pytest.raises(ValueError, match="projection_seed"):
        attnres_modulate(_symmetric_knm(3), np.zeros(3), projection_seed=True)


def test_backend_output_must_preserve_attnres_physics_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse an injected self-coupling result instead of publishing it."""

    def non_physical_backend(*args: object) -> FloatArray:
        """Supply deliberately invalid output for a refusal-only boundary control."""
        n = cast("int", args[6])
        out = np.zeros((n, n), dtype=np.float64)
        out[0, 0] = 1.0
        return out.ravel()

    monkeypatch.setattr(attnres_mod, "_dispatch_backend", lambda: non_physical_backend)

    with pytest.raises(ValueError, match="backend output.*diagonal"):
        attnres_modulate(_symmetric_knm(3), np.zeros(3), n_heads=1, lambda_=0.25)


def test_backend_output_rejects_numeric_string_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse injected numeric text before output coercion."""

    def string_backend(*args: object) -> NDArray[np.str_]:
        """Supply numeric text for a refusal-only output coercion control."""
        return np.asarray(args[0]).astype(str)

    monkeypatch.setattr(attnres_mod, "_dispatch_backend", lambda: string_backend)

    with pytest.raises(ValueError, match="backend output.*numeric-string"):
        attnres_modulate(_symmetric_knm(3), np.zeros(3), n_heads=1, lambda_=0.25)


def test_backend_output_must_not_create_zero_input_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse an injected result that creates a deliberately absent edge."""

    def non_physical_backend(*args: object) -> FloatArray:
        """Supply deliberately invalid output for a refusal-only boundary control."""
        n = cast("int", args[6])
        out = np.zeros((n, n), dtype=np.float64)
        out[0, 1] = out[1, 0] = 0.5
        return out.ravel()

    knm = np.zeros((3, 3), dtype=np.float64)
    knm[1, 2] = knm[2, 1] = 0.25

    monkeypatch.setattr(attnres_mod, "_dispatch_backend", lambda: non_physical_backend)

    with pytest.raises(ValueError, match="preserve zero"):
        attnres_modulate(knm, np.zeros(3), n_heads=1, lambda_=0.25)

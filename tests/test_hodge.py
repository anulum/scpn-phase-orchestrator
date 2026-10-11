# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for the combinatorial Hodge decomposition

"""Public Hodge projections, original runtime execution and input refusals."""

from __future__ import annotations

import cProfile
from collections.abc import Callable
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.coupling.hodge import HodgeResult, hodge_decomposition

FloatArray = NDArray[np.float64]
ObjectArray = NDArray[np.object_]
BoolArray = NDArray[np.bool_]
# Malformed source values deliberately cross the public FloatArray boundary.
invalid_input_call = cast(Callable[[object, object], HodgeResult], hodge_decomposition)
invalid_triangle_call = cast(
    Callable[[FloatArray, FloatArray, object], HodgeResult], hodge_decomposition
)


def _symmetric_coupling(n: int, k: float = 1.0) -> FloatArray:
    """Construct a uniform undirected graph without self edges."""
    knm = np.full((n, n), k, dtype=np.float64)
    np.fill_diagonal(knm, 0.0)
    return knm


class _ArrayProtocolSwitch:
    """Expose different arrays for object-probe and numeric coercion paths."""

    def __init__(self, object_array: ObjectArray, numeric_array: BoolArray) -> None:
        self._object_array = object_array
        self._numeric_array = numeric_array

    def __array__(
        self,
        dtype: np.dtype[np.float64]
        | np.dtype[np.object_]
        | type[np.float64]
        | type[object]
        | None = None,
    ) -> ObjectArray | BoolArray:
        """Return the probe array for object dtype and the numeric array otherwise."""
        if dtype is not None and np.dtype(dtype) == np.dtype(object):
            return self._object_array
        return self._numeric_array


class TestHodgeDecomposition:
    """Public flow, source boundary and runtime-dispatch contracts."""

    def test_returns_hodge_result_dataclass(self) -> None:
        """Zero phases produce correctly sized public decomposition fields."""
        res = hodge_decomposition(_symmetric_coupling(3), np.zeros(3))
        assert isinstance(res, HodgeResult)
        assert res.gradient.shape == (3, 3)
        assert res.curl.shape == (3, 3)
        assert res.harmonic.shape == (3, 3)
        assert res.flow.shape == (3, 3)
        assert res.potential.shape == (3,)
        assert isinstance(res.betti_one, int)

    def test_reconstruction(self) -> None:
        """Gradient, curl and harmonic components reconstruct the input current."""
        rng = np.random.default_rng(42)
        knm = rng.standard_normal((5, 5))
        phases = rng.uniform(0, 2 * np.pi, 5)
        res = hodge_decomposition(knm, phases)
        np.testing.assert_allclose(
            res.gradient + res.curl + res.harmonic, res.flow, atol=1e-10
        )

    def test_zero_coupling_is_all_zero(self) -> None:
        """An uncoupled network has no gradient, curl or harmonic current."""
        res = hodge_decomposition(np.zeros((4, 4)), np.linspace(0, 1, 4))
        np.testing.assert_allclose(res.flow, 0.0, atol=1e-15)
        np.testing.assert_allclose(res.gradient, 0.0, atol=1e-15)
        np.testing.assert_allclose(res.curl, 0.0, atol=1e-15)
        np.testing.assert_allclose(res.harmonic, 0.0, atol=1e-15)
        assert res.betti_one == 0

    def test_complete_graph_curl_free_and_harmonic_free(self) -> None:
        """A triangle-filled complete graph carries no harmonic flow."""
        knm = _symmetric_coupling(4)
        phases = np.array([0.0, 0.5, 1.0, 1.5])
        res = hodge_decomposition(knm, phases)
        np.testing.assert_allclose(res.harmonic, 0.0, atol=1e-10)
        assert res.betti_one == 0

    def test_two_node_flow_closed_form(self) -> None:
        """Two coupled oscillators: a single edge, pure gradient flow."""
        a = 1.7
        phases = np.array([0.25, 1.1])
        knm = np.array([[0.0, a], [a, 0.0]])
        res = hodge_decomposition(knm, phases)
        expected = a * np.sin(phases[1] - phases[0])
        assert res.flow[0, 1] == pytest.approx(expected)
        np.testing.assert_allclose(res.gradient, res.flow, atol=1e-12)
        np.testing.assert_allclose(res.curl, 0.0, atol=1e-12)
        np.testing.assert_allclose(res.harmonic, 0.0, atol=1e-12)
        assert res.betti_one == 0

    def test_global_phase_shift_invariance(self) -> None:
        """A common phase offset leaves each physical flow component unchanged."""
        rng = np.random.default_rng(7)
        knm = rng.standard_normal((5, 5))
        knm = 0.5 * (knm + knm.T)
        phases = rng.uniform(-np.pi, np.pi, 5)
        base = hodge_decomposition(knm, phases)
        shifted = hodge_decomposition(knm, phases + 17.25)
        np.testing.assert_allclose(shifted.gradient, base.gradient, atol=1e-12)
        np.testing.assert_allclose(shifted.curl, base.curl, atol=1e-12)
        np.testing.assert_allclose(shifted.harmonic, base.harmonic, atol=1e-12)

    def test_empty_phases(self) -> None:
        """An empty public problem has empty components and zero topology."""
        res = hodge_decomposition(np.zeros((0, 0)), np.array([]))
        assert res.gradient.size == 0
        assert res.curl.size == 0
        assert res.harmonic.size == 0
        assert res.betti_one == 0

    @pytest.mark.parametrize("backend", [None, "python", "rust", "go", "julia", "mojo"])
    def test_empty_network_checks_coupling_and_triangles(
        self, backend: str | None
    ) -> None:
        """Every selector validates empty-network dimensions and simplex support."""
        phases = np.empty(0, dtype=np.float64)
        coupling = np.empty((0, 0), dtype=np.float64)
        result = hodge_decomposition(coupling, phases, triangles=[], backend=backend)
        for component in (result.gradient, result.curl, result.harmonic, result.flow):
            assert component.shape == (0, 0)
            assert component.dtype == np.float64
        assert result.potential.shape == (0,)
        assert result.betti_one == 0
        with pytest.raises(ValueError, match=r"does not match \(0, 0\)"):
            hodge_decomposition(np.ones((1, 1)), phases, backend=backend)
        with pytest.raises(ValueError, match=r"triangle node outside \[0, 0\)"):
            hodge_decomposition(
                coupling, phases, triangles=[(0, 1, 2)], backend=backend
            )

    def test_single_oscillator(self) -> None:
        """A lone oscillator cannot carry an edge current."""
        res = hodge_decomposition(np.array([[0.0]]), np.array([1.0]))
        np.testing.assert_allclose(res.flow, 0.0)
        np.testing.assert_allclose(res.gradient, 0.0)
        np.testing.assert_allclose(res.curl, 0.0)

    def test_uniform_scale_scales_all_components(self) -> None:
        """Scaling K_nm by a scalar scales every flow component linearly."""
        rng = np.random.default_rng(123)
        knm = rng.standard_normal((5, 5))
        phases = rng.uniform(0.0, 2 * np.pi, 5)
        base = hodge_decomposition(knm, phases, backend="python")
        doubled = hodge_decomposition(2.5 * knm, phases, backend="python")
        np.testing.assert_allclose(doubled.gradient, 2.5 * base.gradient, rtol=1e-10)
        np.testing.assert_allclose(doubled.curl, 2.5 * base.curl, rtol=1e-10)
        np.testing.assert_allclose(doubled.harmonic, 2.5 * base.harmonic, atol=1e-10)

    def test_boolean_phase_alias_is_rejected(self) -> None:
        """Boolean phase aliases refuse at the public source boundary."""
        knm = np.zeros((2, 2), dtype=np.float64)
        with pytest.raises(ValueError, match="phases must not contain boolean"):
            invalid_input_call(knm, [True, 0.5])

    def test_boolean_coupling_alias_is_rejected(self) -> None:
        """Boolean coupling aliases refuse before flow construction."""
        phases = np.array([0.0, 0.5], dtype=np.float64)
        knm = [[0.0, True], [0.0, 0.0]]
        with pytest.raises(ValueError, match="knm must not contain boolean"):
            invalid_input_call(knm, phases)

    def test_numeric_string_phase_alias_is_rejected(self) -> None:
        """Text phase aliases cannot masquerade as measured angles."""
        knm = np.zeros((2, 2), dtype=np.float64)
        with pytest.raises(ValueError, match="phases.*numeric-string"):
            hodge_decomposition(knm, np.array(["0.0", "0.5"], dtype=object))

    def test_numeric_string_coupling_alias_is_rejected(self) -> None:
        """Text weights refuse before numeric coupling conversion."""
        phases = np.array([0.0, 0.5], dtype=np.float64)
        knm = np.array([[0.0, "1.0"], ["1.0", 0.0]], dtype=object)
        with pytest.raises(ValueError, match="knm.*numeric-string"):
            invalid_input_call(knm, phases)

    def test_numeric_string_triangle_node_alias_is_rejected(self) -> None:
        """Topology indices require numeric source values rather than text."""
        phases = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        knm = _symmetric_coupling(3)
        with pytest.raises(ValueError, match="triangle node.*numeric-string"):
            invalid_triangle_call(knm, phases, [["0", 1, 2]])

    def test_phase_object_with_bool_numeric_dtype_is_rejected(self) -> None:
        """A coercion object cannot smuggle boolean angles past the source check."""
        knm = np.zeros((2, 2), dtype=np.float64)
        phases = _ArrayProtocolSwitch(
            object_array=np.array([0.0, 1.0], dtype=object),
            numeric_array=np.array([True, False], dtype=bool),
        )
        with pytest.raises(ValueError, match="phases must not contain boolean"):
            invalid_input_call(knm, phases)

    def test_phase_cast_failure_is_rejected(self) -> None:
        """Nonnumeric phase objects refuse through the public decomposition."""
        knm = np.zeros((2, 2), dtype=np.float64)
        phases = np.array(["left", object()], dtype=object)
        with pytest.raises(ValueError, match="phases must be a finite 1-D"):
            invalid_input_call(knm, phases)

    def test_phase_matrix_is_rejected(self) -> None:
        """The public phase contract rejects a two-dimensional phase array."""
        knm = np.zeros((2, 2), dtype=np.float64)
        phases = np.zeros((1, 2), dtype=np.float64)
        with pytest.raises(ValueError, match="phases must be a finite 1-D"):
            invalid_input_call(knm, phases)

    def test_coupling_object_with_bool_numeric_dtype_is_rejected(self) -> None:
        """A coercion object cannot smuggle boolean coupling weights."""
        phases = np.array([0.0, 0.5], dtype=np.float64)
        knm = _ArrayProtocolSwitch(
            object_array=np.zeros((2, 2), dtype=object),
            numeric_array=np.array([[True, False], [False, True]], dtype=bool),
        )
        with pytest.raises(ValueError, match="knm must not contain boolean"):
            invalid_input_call(knm, phases)

    def test_coupling_cast_failure_is_rejected(self) -> None:
        """Nonnumeric weight objects refuse before decomposition."""
        phases = np.array([0.0, 0.5], dtype=np.float64)
        knm = np.array([["left", object()], [object(), "right"]], dtype=object)
        with pytest.raises(ValueError, match="knm must be a finite square matrix"):
            invalid_input_call(knm, phases)

    def test_coupling_nonfinite_is_rejected(self) -> None:
        """Infinite coupling cannot enter the public flow calculation."""
        phases = np.array([0.0, 0.5], dtype=np.float64)
        knm = np.array([[0.0, np.inf], [np.inf, 0.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="knm must contain only finite"):
            invalid_input_call(knm, phases)

    def test_dispatch_skips_duplicate_and_failed_backends(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Injected duplicate unavailability still reaches the real NumPy caller."""
        from scpn_phase_orchestrator.coupling import hodge as hodge_mod

        attempts: list[str] = []

        def unavailable(owner: str) -> hodge_mod.HodgeBackend:
            attempts.append(owner)
            raise RuntimeError("injected backend unavailability")

        monkeypatch.setattr(hodge_mod, "ACTIVE_BACKEND", "julia")
        monkeypatch.setattr(hodge_mod, "AVAILABLE_BACKENDS", ["julia", "python"])
        monkeypatch.setattr(hodge_mod, "_load_backend", unavailable)
        coupling = _symmetric_coupling(2)
        phases = np.array([0.0, 0.5])
        result = hodge_decomposition(coupling, phases)
        assert attempts == ["julia"]
        assert result.gradient[0, 1] == pytest.approx(np.sin(0.5))
        monkeypatch.setattr(hodge_mod, "AVAILABLE_BACKENDS", [])
        attempts.clear()
        result = hodge_decomposition(coupling, phases)
        assert attempts == ["julia"]
        assert result.gradient[0, 1] == pytest.approx(np.sin(0.5))

    def test_python_tuple_api_matches_public_result(self) -> None:
        """Public NumPy components match the analytic complete-triangle projection."""
        result = hodge_decomposition(
            _symmetric_coupling(3), np.array([0.0, 0.5, 1.0]), backend="python"
        )
        potential_gap = (np.sin(0.5) + np.sin(1.0)) / 3.0
        circulation = (2.0 * np.sin(0.5) - np.sin(1.0)) / 3.0
        gradient = potential_gap * np.array(
            [[0.0, 1.0, 2.0], [-1.0, 0.0, 1.0], [-2.0, -1.0, 0.0]]
        )
        curl = circulation * np.array(
            [[0.0, 1.0, -1.0], [-1.0, 0.0, 1.0], [1.0, -1.0, 0.0]]
        )
        np.testing.assert_allclose(result.gradient, gradient, atol=1e-12)
        np.testing.assert_allclose(result.curl, curl, atol=1e-12)
        np.testing.assert_allclose(result.harmonic, 0.0, atol=1e-12)
        np.testing.assert_allclose(
            result.potential, [-potential_gap, 0.0, potential_gap], atol=1e-12
        )

    def test_empty_pseudoinverse_returns_empty_vector(self) -> None:
        """The internal zero-dimensional solver boundary returns an empty vector."""
        import scpn_phase_orchestrator.coupling.hodge as hodge_mod

        result = hodge_mod._psd_pinv_apply(
            np.zeros((0, 0), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )
        assert result.shape == (0,)

    @pytest.mark.native_runtime
    @pytest.mark.parametrize("backend", ["rust", "go", "julia", "mojo"])
    def test_matching_backend_output_is_used(self, backend: str) -> None:
        """A required original owner returns the nonzero triangle projections."""
        coupling = _symmetric_coupling(3)
        phases = np.array([0.0, 0.5, 1.0])
        measured = cProfile.Profile()
        measured.enable()
        try:
            result = hodge_decomposition(coupling, phases, backend=backend)
        finally:
            measured.disable()
        calls = sum(
            entry.callcount
            for entry in measured.getstats()
            if (
                isinstance(entry.code, str)
                and f"hodge_decomposition_{backend}>" in entry.code
            )
            or (
                not isinstance(entry.code, str)
                and entry.code.co_name == f"hodge_decomposition_{backend}"
            )
        )
        assert calls == 1
        potential_gap = (np.sin(0.5) + np.sin(1.0)) / 3.0
        circulation = (2.0 * np.sin(0.5) - np.sin(1.0)) / 3.0
        gradient = potential_gap * np.array(
            [[0.0, 1.0, 2.0], [-1.0, 0.0, 1.0], [-2.0, -1.0, 0.0]]
        )
        curl = circulation * np.array(
            [[0.0, 1.0, -1.0], [-1.0, 0.0, 1.0], [1.0, -1.0, 0.0]]
        )
        np.testing.assert_allclose(result.gradient, gradient, atol=1e-12)
        np.testing.assert_allclose(result.curl, curl, atol=1e-12)
        np.testing.assert_allclose(result.harmonic, 0.0, atol=1e-12)
        assert result.betti_one == 0


class TestHodgePipelineWiring:
    """Pipeline: engine phases → Hodge decomposition → coupling analysis."""

    @pytest.mark.parametrize(
        "backend",
        [
            "python",
            pytest.param("rust", marks=pytest.mark.native_runtime),
            pytest.param("go", marks=pytest.mark.native_runtime),
            pytest.param("julia", marks=pytest.mark.native_runtime),
            pytest.param("mojo", marks=pytest.mark.native_runtime),
        ],
    )
    def test_engine_phases_to_hodge(self, backend: str) -> None:
        """The real engine output drives each selected public Hodge owner.

        Reconstructed flow matches the independently calculated current of
        the evolved phases; a complete filled graph has no harmonic part.
        """
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        n = 6
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = rng.uniform(0.1, 0.5, (n, n))
        knm = (knm + knm.T) / 2
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        for _ in range(100):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)

        res = hodge_decomposition(knm, phases, backend=backend)
        expected_flow = knm * np.sin(phases[None, :] - phases[:, None])
        np.testing.assert_allclose(res.flow, expected_flow, atol=1e-10)
        assert np.all(np.isfinite(res.gradient))
        assert np.all(np.isfinite(res.curl))
        np.testing.assert_allclose(
            res.gradient + res.curl + res.harmonic, res.flow, atol=1e-10
        )
        # Complete graph (all triangles present) → no harmonic content.
        np.testing.assert_allclose(res.harmonic, 0.0, atol=1e-10)
        assert res.betti_one == 0

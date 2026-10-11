# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hodge decomposition dispatch and validation tests

"""Public Hodge projections, runtime selection and malformed-input refusal."""

from __future__ import annotations

import cProfile
import sys
import types
from collections.abc import Callable
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_phase_orchestrator.coupling.hodge as hodge
from scpn_phase_orchestrator.coupling import hodge_decomposition
from scpn_phase_orchestrator.coupling.hodge import HodgeResult

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
invalid_input_call = cast(Callable[[object, object], HodgeResult], hodge_decomposition)


@pytest.fixture(name="python_backend")
def _python_backend() -> str:
    """Select the public NumPy implementation without replacing dispatch."""
    return "python"


def _matrix_reference(
    knm_flat: FloatArray,
    phases: FloatArray,
    n: int,
    edges_flat: IntArray,
    n_edges: int,
    tris_flat: IntArray,
    n_tris: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Supply the real public NumPy baseline only to an explicit output fault."""
    result = hodge_decomposition(knm_flat.reshape(n, n), phases, backend="python")
    return result.gradient, result.curl, result.harmonic


def _assert_public_output_refused(
    monkeypatch: pytest.MonkeyPatch, output: object, match: str
) -> None:
    """Submit an explicitly malformed response through the public dispatcher."""

    def malformed_backend(*args: object) -> object:
        """Return the specified output fault rather than a numerical solver result."""
        return output

    monkeypatch.setattr(hodge, "_dispatch", lambda: malformed_backend)
    with pytest.raises(ValueError, match=match):
        hodge_decomposition(np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([0.0, 0.5]))


class TestHodgeDecompositionSemantics:
    """Physical projections and topology through the public NumPy owner."""

    def test_components_reconstruct_flow(self, python_backend: str) -> None:
        """The three projections reconstruct a directed-weight Kuramoto current."""
        phases = np.array([0.1, 0.9, -0.6], dtype=np.float64)
        knm = np.array(
            [[0.0, 0.3, -0.8], [0.1, 0.0, 0.5], [-0.4, -0.2, 0.0]],
            dtype=np.float64,
        )
        res = hodge_decomposition(knm, phases, backend=python_backend)
        np.testing.assert_allclose(
            res.gradient + res.curl + res.harmonic, res.flow, atol=1e-12
        )

    def test_filled_complete_graph_has_zero_harmonic(self, python_backend: str) -> None:
        """Filling every triangle removes the complete graph harmonic subspace."""
        phases = np.array([0.4, -0.1, 1.0], dtype=np.float64)
        knm = np.array(
            [[0.0, 0.5, 1.0], [0.5, 0.0, -0.4], [1.0, -0.4, 0.0]],
            dtype=np.float64,
        )
        res = hodge_decomposition(knm, phases, backend=python_backend)
        np.testing.assert_allclose(res.harmonic, 0.0, atol=1e-10)
        assert res.betti_one == 0

    def test_components_orthogonal_and_antisymmetric(self, python_backend: str) -> None:
        """Projections alternate with orientation and have zero pairwise products."""
        rng = np.random.default_rng(4)
        knm = rng.standard_normal((6, 6))
        phases = rng.uniform(0, 2 * np.pi, 6)
        res = hodge_decomposition(knm, phases, backend=python_backend)
        upper = np.triu_indices(6, k=1)
        for a, b in (
            (res.gradient, res.curl),
            (res.gradient, res.harmonic),
            (res.curl, res.harmonic),
        ):
            assert abs(float(np.sum(a[upper] * b[upper]))) < 1e-10
        for comp in (res.gradient, res.curl, res.harmonic):
            np.testing.assert_allclose(comp, -comp.T, atol=1e-12)

    def test_hodge_result_is_dataclass(self, python_backend: str) -> None:
        """A lone oscillator produces the public result with its node potential."""
        res = hodge_decomposition(
            np.array([[0.0]]), np.array([0.5]), backend=python_backend
        )
        assert isinstance(res, HodgeResult)
        assert res.gradient.shape == (1, 1)
        assert res.flow.shape == (1, 1)
        assert res.potential.shape == (1,)


class TestHodgeBackendDispatch:
    """Actual runtime execution and explicit malformed-response fault refusal."""

    def test_python_backend_is_selected_when_dispatch_returns_none(
        self,
        python_backend: str,
    ) -> None:
        """Explicit NumPy selection computes finite two-node flow components."""
        phases = np.array([0.0, np.pi / 2], dtype=np.float64)
        knm = np.array([[0.0, 2.0], [1.0, 0.0]], dtype=np.float64)
        res = hodge_decomposition(knm, phases, backend=python_backend)
        assert np.isfinite(res.gradient).all()
        assert np.isfinite(res.curl).all()
        assert np.isfinite(res.harmonic).all()

    def test_active_backend_result_is_forwarded(self) -> None:
        """The declared runtime must execute the actual public edge decomposition."""
        phases = np.array([0.3, -0.2])
        knm = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        backend = hodge.ACTIVE_BACKEND
        measurement = cProfile.Profile()
        measurement.enable()
        try:
            res = hodge_decomposition(knm, phases, backend=backend)
        finally:
            measurement.disable()
        if backend != "python":
            observed = sum(
                entry.callcount
                for entry in measurement.getstats()
                if (
                    isinstance(entry.code, str)
                    and f"hodge_decomposition_{backend}>" in entry.code
                )
                or (
                    not isinstance(entry.code, str)
                    and entry.code.co_name == f"hodge_decomposition_{backend}"
                )
            )
            assert observed == 1
        expected = np.sin(-0.5)
        assert res.flow[0, 1] == pytest.approx(expected)
        np.testing.assert_allclose(res.gradient, res.flow, atol=1e-12)
        np.testing.assert_allclose(res.curl, 0.0, atol=1e-12)
        np.testing.assert_allclose(res.harmonic, 0.0, atol=1e-12)

    def test_invalid_backend_result_falls_back_to_python_reference(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A deliberately doubled native gradient falls back to the physical current."""

        def fake_backend(
            knm_flat: FloatArray,
            phases: FloatArray,
            n: int,
            edges_flat: IntArray,
            n_edges: int,
            tris_flat: IntArray,
            n_tris: int,
        ) -> tuple[FloatArray, FloatArray, FloatArray]:
            gradient, curl, harmonic = _matrix_reference(
                knm_flat, phases, n, edges_flat, n_edges, tris_flat, n_tris
            )
            return gradient * 2.0, curl.copy(), harmonic.copy()

        monkeypatch.setattr(hodge, "_dispatch", lambda: fake_backend)

        phases = np.array([0.3, -0.2])
        knm = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        res = hodge_decomposition(knm, phases)

        np.testing.assert_allclose(
            res.gradient + res.curl + res.harmonic, res.flow, atol=1e-12
        )
        np.testing.assert_allclose(res.curl, 0.0, atol=1e-12)

    def test_backend_boolean_output_alias_is_rejected_before_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A boolean response fault refuses before any fallback can publish it."""

        def fake_backend(*args: object) -> tuple[object, object, object]:
            n = args[2]
            assert isinstance(n, int)
            bad = np.zeros((n, n), dtype=object)
            bad[0, 1] = np.bool_(True)
            bad[1, 0] = -1.0
            zeros = np.zeros((n, n), dtype=np.float64)
            return bad, zeros, zeros.copy()

        monkeypatch.setattr(hodge, "_dispatch", lambda: fake_backend)

        phases = np.array([0.3, -0.2])
        knm = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="finite real-valued"):
            hodge_decomposition(knm, phases)

    def test_backend_non_antisymmetric_output_is_rejected_before_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A symmetric response fault cannot be published as an alternating flow."""

        def fake_backend(*args: object) -> tuple[object, object, object]:
            n = args[2]
            assert isinstance(n, int)
            bad = np.full((n, n), 1.0, dtype=np.float64)
            zeros = np.zeros((n, n), dtype=np.float64)
            return bad, zeros, zeros.copy()

        monkeypatch.setattr(hodge, "_dispatch", lambda: fake_backend)

        phases = np.array([0.3, -0.2])
        knm = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="antisymmetric"):
            hodge_decomposition(knm, phases)


class TestHodgeValidation:
    """Measured-array cardinality, numeric source types and published output guards."""

    def test_empty_phases_produces_empty_results(
        self,
        python_backend: str,
    ) -> None:
        """A valid empty public network returns empty projections."""
        res = hodge_decomposition(
            np.zeros((0, 0), dtype=np.float64),
            np.array([], dtype=np.float64),
            backend=python_backend,
        )
        assert res.gradient.size == 0
        assert res.curl.size == 0
        assert res.harmonic.size == 0
        assert isinstance(res.gradient, np.ndarray)

    def test_non_finite_phase_values_raise(self, python_backend: str) -> None:
        """A NaN phase is refused before current construction."""
        phases = np.array([0.0, np.nan], dtype=np.float64)
        knm = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="phases must contain only finite values"):
            hodge_decomposition(knm, phases, backend=python_backend)

    def test_non_finite_coupling_values_raise(self, python_backend: str) -> None:
        """An infinite weight is refused before projection."""
        phases = np.array([0.0, 1.0], dtype=np.float64)
        knm = np.array([[0.0, np.inf], [1.0, 0.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="knm must contain only finite values"):
            hodge_decomposition(knm, phases, backend=python_backend)

    def test_invalid_phase_shape_is_rejected(self, python_backend: str) -> None:
        """A phase matrix is refused where a vector is required."""
        phases = np.array([[0.0, 1.0]], dtype=np.float64)
        knm = np.eye(2, dtype=np.float64)
        with pytest.raises(
            ValueError, match="phases must be a finite 1-D phase vector"
        ):
            hodge_decomposition(knm, phases, backend=python_backend)

    def test_shape_mismatch_between_knm_and_phases_is_rejected(
        self,
        python_backend: str,
    ) -> None:
        """The coupling size must equal the oscillator count."""
        phases = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        knm = np.eye(2, dtype=np.float64)
        with pytest.raises(ValueError, match=r"does not match \(3, 3\)"):
            hodge_decomposition(knm, phases, backend=python_backend)

    def test_boolean_phases_rejected(self, python_backend: str) -> None:
        """Boolean angles are refused before numeric coercion."""
        phases = np.array([True, False], dtype=np.bool_)
        knm = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="phases must not contain boolean values"):
            hodge_decomposition(knm, phases, backend=python_backend)

    def test_empty_boolean_phase_dtype_rejected(self, python_backend: str) -> None:
        """An empty boolean vector retains its invalid source dtype."""
        phases = np.array([], dtype=np.bool_)
        knm = np.zeros((0, 0), dtype=np.float64)
        with pytest.raises(ValueError, match="phases must not contain boolean values"):
            hodge_decomposition(knm, phases, backend=python_backend)

    def test_numpy_boolean_phase_alias_rejected(self, python_backend: str) -> None:
        """A NumPy boolean inside a numeric object vector is refused."""
        phases = np.array([0.0, np.bool_(True)], dtype=object)
        knm = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="phases must not contain boolean values"):
            hodge_decomposition(knm, phases, backend=python_backend)

    def test_non_numeric_phase_object_array_rejected(self, python_backend: str) -> None:
        """Text phase objects cannot enter the numerical solver."""
        phases = np.array(["bad"], dtype=object)
        knm = np.zeros((1, 1), dtype=np.float64)
        with pytest.raises(ValueError, match="phases must be a finite 1-D"):
            hodge_decomposition(knm, phases, backend=python_backend)

    def test_boolean_coupling_matrix_rejected(self, python_backend: str) -> None:
        """Boolean edge weights are refused at the public input boundary."""
        phases = np.array([0.0, 1.0], dtype=np.float64)
        knm = np.array([[False, True], [True, False]], dtype=np.bool_)
        with pytest.raises(ValueError, match="knm must not contain boolean values"):
            hodge_decomposition(knm, phases, backend=python_backend)

    def test_empty_boolean_coupling_dtype_rejected(self) -> None:
        """The public empty-network boundary still refuses boolean coupling."""
        knm = np.zeros((0, 0), dtype=np.bool_)
        with pytest.raises(ValueError, match="knm must not contain boolean values"):
            invalid_input_call(knm, np.empty(0, dtype=np.float64))

    def test_numpy_boolean_coupling_alias_rejected(self, python_backend: str) -> None:
        """An object matrix cannot conceal a NumPy boolean weight."""
        phases = np.array([0.0, 1.0], dtype=np.float64)
        knm = np.array([[0.0, np.bool_(True)], [1.0, 0.0]], dtype=object)
        with pytest.raises(ValueError, match="knm must not contain boolean values"):
            hodge_decomposition(knm, phases, backend=python_backend)

    def test_non_numeric_coupling_object_array_rejected(
        self, python_backend: str
    ) -> None:
        """Nonnumeric weight objects are refused before projection."""
        phases = np.array([0.0], dtype=np.float64)
        knm = np.array([["bad"]], dtype=object)
        with pytest.raises(ValueError, match="knm must be a finite square matrix"):
            hodge_decomposition(knm, phases, backend=python_backend)

    def test_empty_pseudoinverse_application_returns_empty_vector(self) -> None:
        """The internal zero-dimensional solver defence returns an empty vector."""
        result = hodge._psd_pinv_apply(
            np.zeros((0, 0), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )
        np.testing.assert_array_equal(result, np.zeros(0, dtype=np.float64))

    def test_backend_output_shape_is_validated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An injected wrong-size response is refused through the public caller."""
        _assert_public_output_refused(
            monkeypatch,
            (
                np.zeros((1, 1), dtype=np.float64),
                np.zeros((2, 2), dtype=np.float64),
                np.zeros((2, 2), dtype=np.float64),
            ),
            "invalid shape",
        )

    def test_backend_output_finiteness_is_validated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An injected NaN response is refused through the public caller."""
        _assert_public_output_refused(
            monkeypatch,
            (
                np.full((2, 2), np.nan, dtype=np.float64),
                np.zeros((2, 2), dtype=np.float64),
                np.zeros((2, 2), dtype=np.float64),
            ),
            "non-finite",
        )

    def test_backend_output_boolean_alias_is_validated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An injected boolean response is refused through the public caller."""
        _assert_public_output_refused(
            monkeypatch,
            (
                np.array([[0.0, np.bool_(True)], [-1.0, 0.0]], dtype=object),
                np.zeros((2, 2), dtype=np.float64),
                np.zeros((2, 2), dtype=np.float64),
            ),
            "finite real-valued",
        )

    def test_backend_output_complex_alias_is_validated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An injected complex response is refused through the public caller."""
        _assert_public_output_refused(
            monkeypatch,
            (
                np.array([[0.0, 1.0 + 0.0j], [-1.0, 0.0]], dtype=object),
                np.zeros((2, 2), dtype=np.float64),
                np.zeros((2, 2), dtype=np.float64),
            ),
            "finite real-valued",
        )

    def test_backend_output_antisymmetry_is_validated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An injected symmetric response is refused through the public caller."""
        _assert_public_output_refused(
            monkeypatch,
            (
                np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
                np.zeros((2, 2), dtype=np.float64),
                np.zeros((2, 2), dtype=np.float64),
            ),
            "antisymmetric",
        )


def test_large_phase_values_preserve_phase_shift_invariance(
    python_backend: str,
) -> None:
    """Large finite phases preserve finite projections and flow reconstruction."""
    knm = np.array(
        [[0.0, 0.3, -0.2], [0.3, 0.0, 0.8], [-0.2, 0.8, 0.0]],
        dtype=np.float64,
    )
    base = hodge_decomposition(knm, np.array([0.1, -0.2, 0.3]), backend=python_backend)
    large = hodge_decomposition(
        knm, np.array([1e8, -2e8, 3e8]) + 0.1, backend=python_backend
    )
    # sin(θ_j − θ_i) is bounded; components stay finite under huge phases.
    assert np.all(np.isfinite(large.gradient))
    assert np.all(np.isfinite(large.curl))
    assert np.all(np.isfinite(large.harmonic))
    np.testing.assert_allclose(
        large.gradient + large.curl + large.harmonic, large.flow, atol=1e-10
    )
    assert base.betti_one == large.betti_one


class TestDispatchFallbackChain:
    """Real cached owners and explicit optional-loader failures."""

    def test_dispatch_returns_python_when_every_backend_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit loader faults still reach the real NumPy computation."""

        def _fail() -> None:
            raise ImportError("backend unavailable")

        monkeypatch.setattr(hodge, "_BACKEND_CACHE", {})
        monkeypatch.setattr(hodge, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(hodge, "AVAILABLE_BACKENDS", ["go"])
        monkeypatch.setattr(hodge, "_LOADERS", {"rust": _fail, "go": _fail})

        result = hodge_decomposition(
            np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([0.0, 0.4])
        )
        assert result.gradient[0, 1] == pytest.approx(np.sin(0.4), abs=1e-12)

    @pytest.mark.native_runtime
    @pytest.mark.parametrize("backend", ["go"])
    def test_dispatch_falls_back_to_next_backend_when_active_fails(
        self, monkeypatch: pytest.MonkeyPatch, backend: str
    ) -> None:
        """An explicit loader fault must fall through to the real Go operation."""
        calls = {"rust": 0, "go": 0}
        original_go_loader = hodge._LOADERS[backend]

        def unavailable_rust() -> hodge.HodgeBackend:
            calls["rust"] += 1
            raise ImportError("injected Rust loader unavailability")

        def observed_go_loader() -> hodge.HodgeBackend:
            calls["go"] += 1
            return original_go_loader()

        monkeypatch.setattr(hodge, "_BACKEND_CACHE", {})
        monkeypatch.setattr(hodge, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(hodge, "AVAILABLE_BACKENDS", ["rust", "go", "python"])
        monkeypatch.setattr(
            hodge, "_LOADERS", {"rust": unavailable_rust, "go": observed_go_loader}
        )
        phases = np.array([0.3, -0.2])
        knm = np.array([[0.0, 1.0], [1.0, 0.0]])
        result = hodge_decomposition(knm, phases)
        assert calls == {"rust": 1, "go": 1}
        assert result.gradient[0, 1] == pytest.approx(np.sin(-0.5))
        np.testing.assert_allclose(result.gradient, result.flow, atol=1e-12)
        np.testing.assert_allclose(result.curl, 0.0, atol=1e-12)
        np.testing.assert_allclose(result.harmonic, 0.0, atol=1e-12)

    @pytest.mark.native_runtime
    @pytest.mark.parametrize("backend", ["rust", "go", "julia", "mojo"])
    def test_dispatch_uses_cached_loader_once(self, backend: str) -> None:
        """Repeated public calls reuse the real loader admitted at module import."""
        phases = np.array([0.3, -0.2])
        knm = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        hodge_decomposition(knm, phases, backend=backend)
        measurement = cProfile.Profile()
        measurement.enable()
        try:
            first = hodge_decomposition(knm, phases, backend=backend)
            second = hodge_decomposition(knm, phases, backend=backend)
        finally:
            measurement.disable()
        loads = sum(
            entry.callcount
            for entry in measurement.getstats()
            if not isinstance(entry.code, str)
            and entry.code.co_name == f"_load_{backend}_fn"
        )
        assert loads == 0
        for result in (first, second):
            assert result.gradient[0, 1] == pytest.approx(np.sin(-0.5))
            np.testing.assert_allclose(result.curl, 0.0, atol=1e-12)
            np.testing.assert_allclose(result.harmonic, 0.0, atol=1e-12)


@pytest.mark.parametrize("error_type", [ImportError, RuntimeError, OSError, KeyError])
def test_explicit_backend_refuses_loader_fault(
    monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]
) -> None:
    """An injected loader fault cannot silently satisfy a native request."""

    def failed_loader() -> hodge.HodgeBackend:
        raise error_type("injected native loader failure")

    monkeypatch.setattr(hodge, "_BACKEND_CACHE", {})
    monkeypatch.setitem(hodge._LOADERS, "rust", failed_loader)
    with pytest.raises(ImportError, match="Hodge backend 'rust' is unavailable"):
        hodge_decomposition(
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([0.0, 0.4]),
            backend="rust",
        )


def test_explicit_backend_refuses_numerical_fault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A numerically wrong native response is refused under explicit selection."""

    def wrong_result(
        knm: FloatArray,
        phases: FloatArray,
        n: int,
        edges: IntArray,
        n_edges: int,
        tris: IntArray,
        n_tris: int,
    ) -> hodge.HodgeTuple:
        baseline = hodge_decomposition(knm.reshape(n, n), phases, backend="python")
        return baseline.gradient * 2.0, baseline.curl, baseline.harmonic

    monkeypatch.setattr(hodge, "_BACKEND_CACHE", {})
    monkeypatch.setitem(hodge._LOADERS, "rust", lambda: wrong_result)
    with pytest.raises(ValueError, match="disagrees with the reference"):
        hodge_decomposition(
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([0.0, 0.4]),
            backend="rust",
        )


def test_explicit_rust_refuses_missing_export(monkeypatch: pytest.MonkeyPatch) -> None:
    """An installed-module export fault cannot hide behind the NumPy fallback."""
    monkeypatch.setitem(sys.modules, "spo_kernel", types.ModuleType("spo_kernel"))
    monkeypatch.setattr(hodge, "_BACKEND_CACHE", {})
    with pytest.raises(ImportError, match="Hodge backend 'rust' is unavailable"):
        hodge_decomposition(
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([0.0, 0.4]),
            backend="rust",
        )

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Algorithm tests for order parameters

"""Algorithm-level tests for ``upde/order_params.py``.

Covers the three compute kernels — ``compute_order_parameter``,
``compute_plv``, ``compute_layer_coherence`` — plus dispatcher
invariants. Per-backend parity lives in
``test_order_params_backends.py``; stability tests live in
``test_order_params_stability.py``.
"""

from __future__ import annotations

import cmath
import cProfile
from collections.abc import Callable
from typing import cast

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from scpn_phase_orchestrator.upde.order_params import (
    ACTIVE_BACKEND,
    AVAILABLE_BACKENDS,
    compute_layer_coherence,
    compute_order_parameter,
    compute_plv,
    debiased_squared_order_parameter,
)
from tests._runtime_owners import NEEDS_KERNEL, owner_params

TWO_PI = 2.0 * np.pi

phase_arrays = arrays(
    dtype=np.float64,
    shape=st.integers(min_value=2, max_value=200),
    elements=st.floats(
        min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False
    ),
)


# ---------------------------------------------------------------------
# compute_order_parameter
# ---------------------------------------------------------------------


class TestOrderParameter:
    @given(phases=phase_arrays)
    @settings(max_examples=30, deadline=None)
    def test_r_bounded_unit_interval(self, phases: np.ndarray) -> None:
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0 + 1e-12

    @given(phases=phase_arrays)
    @settings(max_examples=30, deadline=None)
    def test_psi_in_zero_2pi(self, phases: np.ndarray) -> None:
        _, psi = compute_order_parameter(phases)
        assert 0.0 <= psi < TWO_PI + 1e-12

    def test_empty_returns_zero_zero(self) -> None:
        r, psi = compute_order_parameter(np.array([], dtype=np.float64))
        assert r == 0.0 and psi == 0.0

    def test_full_synchrony_r_one(self) -> None:
        phases = np.full(32, 1.234)
        r, _ = compute_order_parameter(phases)
        assert r == pytest.approx(1.0, abs=1e-12)

    def test_antiphase_pairs_r_zero(self) -> None:
        phases = np.concatenate([np.zeros(8), np.full(8, np.pi)])
        r, _ = compute_order_parameter(phases)
        assert r == pytest.approx(0.0, abs=1e-12)

    def test_uniform_distribution_r_small(self) -> None:
        phases = np.linspace(0.0, TWO_PI, 100, endpoint=False)
        r, _ = compute_order_parameter(phases)
        assert r < 1e-10

    def test_single_oscillator_r_one(self) -> None:
        r, psi = compute_order_parameter(np.array([0.7]))
        assert r == pytest.approx(1.0, abs=1e-12)
        assert psi == pytest.approx(0.7, abs=1e-12)

    @pytest.mark.parametrize(
        "phases",
        [
            np.array([0.0, np.nan]),
            np.array([0.0, np.inf]),
            np.array([True, False]),
            np.array([0.0, True], dtype=object),
            np.array([0.0 + 0.0j, 1.0 + 0.0j]),
            np.array(["0.0", "1.0"]),
        ],
    )
    def test_rejects_invalid_phase_values(self, phases: np.ndarray) -> None:
        with pytest.raises(ValueError, match="phases"):
            compute_order_parameter(phases)

    def test_debiased_order_parameter_rejects_numeric_string_phases(self) -> None:
        """The bias-corrected public boundary must not launder text phases."""
        with pytest.raises(ValueError, match="phases must be numeric"):
            debiased_squared_order_parameter(np.array(["0.0", "1.0"]))

    @pytest.mark.parametrize("backend_r", [-0.1, 1.1])
    def test_backend_material_order_parameter_bound_violation_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch, backend_r: float
    ) -> None:
        import scpn_phase_orchestrator.upde.order_params as op_mod

        def fake_order(phases: np.ndarray) -> tuple[float, float]:
            return backend_r, float(phases[0])

        monkeypatch.setattr(op_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(
            op_mod,
            "_load_backend",
            lambda name: {"order_parameter": fake_order},
        )

        with pytest.raises(ValueError, match="coherence magnitude"):
            op_mod.compute_order_parameter(np.array([0.25], dtype=np.float64))

    def test_backend_order_parameter_roundoff_overrun_is_clamped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scpn_phase_orchestrator.upde.order_params as op_mod

        def fake_order(phases: np.ndarray) -> tuple[float, float]:
            return 1.0 + np.finfo(np.float64).eps, float(phases[0])

        monkeypatch.setattr(op_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(op_mod, "_HAS_RUST", True)
        monkeypatch.setattr(
            op_mod,
            "_load_backend",
            lambda name: {"order_parameter": fake_order},
        )

        r, psi = op_mod.compute_order_parameter(np.array([0.25], dtype=np.float64))

        assert r == 1.0
        assert psi == 0.25

    def test_backend_non_finite_mean_phase_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scpn_phase_orchestrator.upde.order_params as op_mod

        def fake_order(phases: np.ndarray) -> tuple[float, float]:
            return 0.5, float("nan")

        monkeypatch.setattr(op_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(op_mod, "_HAS_RUST", True)
        monkeypatch.setattr(
            op_mod,
            "_load_backend",
            lambda name: {"order_parameter": fake_order},
        )

        with pytest.raises(ValueError, match="mean phase"):
            op_mod.compute_order_parameter(np.array([0.25], dtype=np.float64))


# ---------------------------------------------------------------------
# compute_plv
# ---------------------------------------------------------------------


class TestPLV:
    @given(a=phase_arrays, b=phase_arrays)
    @settings(max_examples=20, deadline=None)
    def test_plv_bounded_unit_interval(self, a: np.ndarray, b: np.ndarray) -> None:
        n = min(a.size, b.size)
        val = compute_plv(a[:n], b[:n])
        assert 0.0 <= val <= 1.0 + 1e-12

    def test_empty_returns_zero(self) -> None:
        assert compute_plv(np.array([]), np.array([])) == 0.0

    def test_identical_series_plv_one(self) -> None:
        rng = np.random.default_rng(0)
        phases = rng.uniform(0.0, TWO_PI, size=50)
        assert compute_plv(phases, phases) == pytest.approx(1.0, abs=1e-12)

    def test_constant_offset_plv_one(self) -> None:
        rng = np.random.default_rng(1)
        phases = rng.uniform(0.0, TWO_PI, size=50)
        offset = phases + 0.7
        assert compute_plv(phases, offset) == pytest.approx(1.0, abs=1e-12)

    def test_uncorrelated_plv_small(self) -> None:
        rng = np.random.default_rng(42)
        a = rng.uniform(0.0, TWO_PI, size=2000)
        b = rng.uniform(0.0, TWO_PI, size=2000)
        assert compute_plv(a, b) < 0.08

    def test_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="equal-length"):
            compute_plv(np.zeros(5), np.zeros(6))

    def test_empty_non_empty_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="equal-length"):
            compute_plv(np.array([], dtype=np.float64), np.zeros(1, dtype=np.float64))

    @pytest.mark.parametrize(
        ("a", "b", "match"),
        [
            (np.array([0.0, np.nan]), np.zeros(2), "phases_a"),
            (np.zeros(2), np.array([0.0, np.inf]), "phases_b"),
            (np.array([True, False]), np.zeros(2), "phases_a"),
            (np.zeros(2), np.array([True, False]), "phases_b"),
            (np.array([0.0, True], dtype=object), np.zeros(2), "phases_a"),
            (np.zeros(2), np.array([0.0, False], dtype=object), "phases_b"),
            (np.array([0.0 + 0.0j, 1.0 + 0.0j]), np.zeros(2), "phases_a"),
            (np.zeros(2), np.array([0.0 + 0.0j, 1.0 + 0.0j]), "phases_b"),
            (np.array(["0.0", "1.0"]), np.zeros(2), "phases_a"),
            (np.zeros(2), np.array(["0.0", "1.0"]), "phases_b"),
        ],
    )
    def test_rejects_invalid_plv_values(
        self, a: np.ndarray, b: np.ndarray, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            compute_plv(a, b)

    @pytest.mark.parametrize("backend_plv", [-0.1, 1.1])
    def test_backend_material_plv_bound_violation_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch, backend_plv: float
    ) -> None:
        import scpn_phase_orchestrator.upde.order_params as op_mod

        def fake_plv(a: np.ndarray, b: np.ndarray) -> float:
            return backend_plv

        monkeypatch.setattr(op_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(op_mod, "_load_backend", lambda name: {"plv": fake_plv})

        with pytest.raises(ValueError, match="coherence magnitude"):
            op_mod.compute_plv(
                np.array([0.0, 0.1], dtype=np.float64),
                np.array([0.0, 0.1], dtype=np.float64),
            )

    def test_backend_plv_roundoff_overrun_is_clamped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scpn_phase_orchestrator.upde.order_params as op_mod

        def fake_plv(a: np.ndarray, b: np.ndarray) -> float:
            assert a.shape == b.shape
            return 1.0 + np.finfo(np.float64).eps

        monkeypatch.setattr(op_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(op_mod, "_HAS_RUST", True)
        monkeypatch.setattr(op_mod, "_load_backend", lambda name: {"plv": fake_plv})

        val = op_mod.compute_plv(
            np.array([0.0, 0.1], dtype=np.float64),
            np.array([0.0, 0.1], dtype=np.float64),
        )

        assert val == 1.0

    def test_backend_non_finite_plv_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scpn_phase_orchestrator.upde.order_params as op_mod

        def fake_plv(a: np.ndarray, b: np.ndarray) -> float:
            return float("nan")

        monkeypatch.setattr(op_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(op_mod, "_HAS_RUST", True)
        monkeypatch.setattr(op_mod, "_load_backend", lambda name: {"plv": fake_plv})

        with pytest.raises(ValueError, match="coherence magnitude"):
            op_mod.compute_plv(
                np.array([0.0, 0.1], dtype=np.float64),
                np.array([0.0, 0.1], dtype=np.float64),
            )


# ---------------------------------------------------------------------
# compute_layer_coherence
# ---------------------------------------------------------------------


class TestLayerCoherence:
    def test_full_synchrony_of_subset(self) -> None:
        phases = np.array([0.0, 0.1, np.pi, np.pi + 0.1, 2.0, 3.0], dtype=np.float64)
        r = compute_layer_coherence(phases, np.array([0, 1], dtype=np.int64))
        assert r > 0.99

    def test_empty_mask_returns_zero(self) -> None:
        phases = np.array([0.0, 1.0, 2.0])
        assert compute_layer_coherence(phases, np.array([], dtype=np.int64)) == 0.0

    def test_bool_mask_supported(self) -> None:
        phases = np.array([0.0, np.pi, 0.05, np.pi + 0.05])
        bool_mask = np.array([True, False, True, False])
        idx_mask = np.array([0, 2], dtype=np.int64)
        r_bool = compute_layer_coherence(phases, bool_mask)
        r_idx = compute_layer_coherence(phases, idx_mask)
        assert r_bool == pytest.approx(r_idx, abs=1e-12)

    def test_single_index_r_one(self) -> None:
        phases = np.array([0.7, 1.0, 2.0])
        r = compute_layer_coherence(phases, np.array([0], dtype=np.int64))
        assert r == pytest.approx(1.0, abs=1e-12)

    def test_bool_mask_length_must_match_phase_vector(self) -> None:
        phases = np.array([0.0, 0.5, 1.0])

        with pytest.raises(ValueError, match="layer_mask"):
            compute_layer_coherence(phases, np.array([True, False]))

    def test_integer_indices_must_reference_existing_oscillators(self) -> None:
        phases = np.array([0.0, 0.5, 1.0])

        with pytest.raises(ValueError, match="layer_mask"):
            compute_layer_coherence(phases, np.array([0, 3], dtype=np.int64))

    def test_integer_indices_must_not_repeat_oscillators(self) -> None:
        phases = np.array([0.0, np.pi, np.pi])

        with pytest.raises(ValueError, match="layer_mask"):
            compute_layer_coherence(phases, np.array([0, 0, 1], dtype=np.int64))

    def test_layer_indices_must_not_truncate_float_values(self) -> None:
        phases = np.array([0.0, 0.5, 1.0])

        with pytest.raises(ValueError, match="layer_mask indices must be integers"):
            compute_layer_coherence(phases, np.array([0.0, 1.5]))

    def test_layer_indices_reject_mixed_boolean_aliases(self) -> None:
        phases = np.array([0.0, 0.5, 1.0])

        with pytest.raises(
            ValueError, match="layer_mask indices must not contain boolean values"
        ):
            compute_layer_coherence(phases, np.array([0, True], dtype=object))

    @pytest.mark.parametrize(
        "phases",
        [
            np.array([0.0, np.nan]),
            np.array([0.0, np.inf]),
            np.array([True, False]),
            np.array([0.0, True], dtype=object),
            np.array([0.0 + 0.0j, 1.0 + 0.0j]),
            np.array(["0.0", "1.0"]),
        ],
    )
    def test_rejects_invalid_layer_phase_values(self, phases: np.ndarray) -> None:
        with pytest.raises(ValueError, match="phases"):
            compute_layer_coherence(phases, np.array([0], dtype=np.int64))

    @pytest.mark.parametrize("backend_r", [-0.1, 1.1])
    def test_backend_material_layer_coherence_bound_violation_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch, backend_r: float
    ) -> None:
        import scpn_phase_orchestrator.upde.order_params as op_mod

        def fake_layer(phases: np.ndarray, indices: np.ndarray) -> float:
            return backend_r

        monkeypatch.setattr(op_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(
            op_mod,
            "_load_backend",
            lambda name: {"layer_coherence": fake_layer},
        )

        with pytest.raises(ValueError, match="coherence magnitude"):
            op_mod.compute_layer_coherence(
                np.array([0.0, 0.1], dtype=np.float64),
                np.array([0, 1], dtype=np.int64),
            )

    def test_backend_layer_coherence_roundoff_overrun_is_clamped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scpn_phase_orchestrator.upde.order_params as op_mod

        def fake_layer(phases: np.ndarray, indices: np.ndarray) -> float:
            assert phases.size >= indices.size
            return 1.0 + np.finfo(np.float64).eps

        monkeypatch.setattr(op_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(op_mod, "_HAS_RUST", True)
        monkeypatch.setattr(
            op_mod,
            "_load_backend",
            lambda name: {"layer_coherence": fake_layer},
        )

        r = op_mod.compute_layer_coherence(
            np.array([0.0, 0.1], dtype=np.float64),
            np.array([0, 1], dtype=np.int64),
        )

        assert r == 1.0

    def test_backend_non_finite_layer_coherence_is_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import scpn_phase_orchestrator.upde.order_params as op_mod

        def fake_layer(phases: np.ndarray, indices: np.ndarray) -> float:
            return float("nan")

        monkeypatch.setattr(op_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(op_mod, "_HAS_RUST", True)
        monkeypatch.setattr(
            op_mod,
            "_load_backend",
            lambda name: {"layer_coherence": fake_layer},
        )

        with pytest.raises(ValueError, match="coherence magnitude"):
            op_mod.compute_layer_coherence(
                np.array([0.0, 0.1], dtype=np.float64),
                np.array([0, 1], dtype=np.int64),
            )


# ---------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------


class TestDispatcher:
    @NEEDS_KERNEL
    def test_rust_loader_maps_spo_kernel_symbols(self) -> None:
        """The compiled Rust loader maps kernels that follow the three laws."""
        import scpn_phase_orchestrator.upde.order_params as op_mod

        if "rust" not in op_mod.AVAILABLE_BACKENDS:
            pytest.skip("rust unavailable")
        loaded = op_mod._load_rust_fns()
        rng = np.random.default_rng(61)
        phases = rng.uniform(-np.pi, np.pi, 11)
        other = rng.uniform(-np.pi, np.pi, 11)
        indices = np.array([0, 3, 4, 9], dtype=np.int64)

        mean = sum(cmath.exp(1j * v) for v in phases.tolist()) / 11
        r, psi = cast(
            "Callable[[np.ndarray], tuple[float, float]]", loaded["order_parameter"]
        )(phases)
        assert r == pytest.approx(abs(mean), abs=1e-12)
        gap = (psi - cmath.phase(mean) + np.pi) % (2 * np.pi) - np.pi
        assert abs(gap) < 1e-12
        diffs = (a - b for a, b in zip(phases.tolist(), other.tolist(), strict=True))
        plv = abs(sum(cmath.exp(1j * d) for d in diffs) / 11)
        assert cast("Callable[..., float]", loaded["plv"])(
            phases, other
        ) == pytest.approx(plv, abs=1e-12)
        layer = abs(sum(cmath.exp(1j * phases[k]) for k in indices.tolist()) / 4)
        got = cast("Callable[..., float]", loaded["layer_coherence"])(phases, indices)
        assert got == pytest.approx(layer, abs=1e-12)

    def test_python_is_always_available(self) -> None:
        assert "python" in AVAILABLE_BACKENDS
        assert AVAILABLE_BACKENDS[-1] == "python"

    def test_active_backend_is_first_available(self) -> None:
        assert ACTIVE_BACKEND in AVAILABLE_BACKENDS

    def test_available_backends_keep_canonical_fallback_order(self) -> None:
        canonical = ["rust", "mojo", "julia", "go", "python"]
        indices = [canonical.index(b) for b in AVAILABLE_BACKENDS]
        assert indices == sorted(indices)

    def test_order_parameter_probe_uses_python_path(self) -> None:
        """The Python probe runs the NumPy order parameter once and is finite."""
        import scpn_phase_orchestrator.upde.order_params as op_mod

        profile = cProfile.Profile()
        profile.enable()
        try:
            elapsed = op_mod._order_parameter_probe_seconds("python")
        finally:
            profile.disable()
        calls = sum(
            entry.callcount
            for entry in profile.getstats()
            if not isinstance(entry.code, str)
            and entry.code.co_name == "_python_order_parameter"
        )
        assert calls == 1
        assert 0.0 <= elapsed < float("inf")

    def test_resolve_backends_falls_back_to_python_if_optional_backends_fail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scpn_phase_orchestrator.upde.order_params as op_mod

        def broken_loader(name: str) -> dict[str, object]:
            raise ImportError(f"{name} unavailable")

        monkeypatch.setattr(op_mod, "_load_backend", broken_loader)
        monkeypatch.setattr(op_mod, "_BACKEND_CACHE", {})

    def test_dispatch_falls_back_to_next_backend_when_active_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scpn_phase_orchestrator.upde.order_params as op_mod

        calls: dict[str, int] = {"rust": 0, "go": 0}

        def _fail_rust() -> dict[str, object]:
            calls["rust"] += 1
            raise ImportError("rust unavailable")

        def _ok_go() -> dict[str, object]:
            calls["go"] += 1
            return {
                "order_parameter": lambda phases: (0.75, 0.0),
                "plv": lambda a, b: 0.5,
                "layer_coherence": lambda phases, idx: 0.25,
            }

        monkeypatch.setattr(op_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(op_mod, "AVAILABLE_BACKENDS", ["rust", "go", "python"])
        monkeypatch.setattr(op_mod, "_LOADERS", {"rust": _fail_rust, "go": _ok_go})
        monkeypatch.setattr(op_mod, "_BACKEND_CACHE", {})

        owner, fn = op_mod._dispatch("order_parameter")
        assert owner == "go"
        assert callable(fn)
        assert fn(np.array([0.0, 1.0], dtype=np.float64)) == (0.75, 0.0)
        assert calls == {"rust": 1, "go": 1}

    @pytest.mark.parametrize("owner", owner_params(["rust", "mojo", "julia", "go"]))
    def test_dispatch_uses_cached_loader_once(
        self, owner: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On a cold cache two loads of an installed owner run its loader once."""
        import scpn_phase_orchestrator.upde.order_params as op_mod

        if owner not in op_mod.AVAILABLE_BACKENDS:
            pytest.skip(f"{owner} unavailable")
        monkeypatch.setattr(op_mod, "_BACKEND_CACHE", {})
        profile = cProfile.Profile()
        profile.enable()
        try:
            first = op_mod._load_backend(owner)
            second = op_mod._load_backend(owner)
        finally:
            profile.disable()
        loads = sum(
            entry.callcount
            for entry in profile.getstats()
            if not isinstance(entry.code, str)
            and entry.code.co_name == f"_load_{owner}_fns"
        )
        assert loads == 1
        assert first is second
        assert set(first) == {"order_parameter", "plv", "layer_coherence"}

    def test_dispatch_returns_none_when_active_backend_symbol_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scpn_phase_orchestrator.upde.order_params as op_mod

        monkeypatch.setattr(op_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(op_mod, "_HAS_RUST", True)
        monkeypatch.setattr(op_mod, "_load_backend", lambda name: {})

        assert op_mod._dispatch("plv") == ("python", None)

    def test_probe_marks_faulty_backend_unusable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scpn_phase_orchestrator.upde.order_params as op_mod

        def broken_backend(name: str) -> dict[str, object]:
            raise RuntimeError(f"{name} unavailable during probe")

        monkeypatch.setattr(op_mod, "_load_backend", broken_backend)

        elapsed = op_mod._order_parameter_probe_seconds("rust")

        assert elapsed == float("inf")

    def test_resolve_backends_keeps_working_backend_before_python(self) -> None:
        """Resolution lists the working real loaders in order, then Python."""
        import scpn_phase_orchestrator.upde.order_params as op_mod

        loadable = []
        for name in ("rust", "mojo", "julia", "go"):
            try:
                op_mod._LOADERS[name]()
            except (ImportError, RuntimeError, OSError, KeyError):
                continue
            loadable.append(name)

        active, available = op_mod._resolve_backends()

        assert available == [*loadable, "python"]
        assert active in available
        assert op_mod._order_parameter_probe_seconds(active) < float("inf")

    def test_dispatch_falls_through_when_active_backend_has_no_symbol(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Automatic dispatch skips an owner that lacks the kernel.

        Every shipped owner exports all three kernels, so an incomplete owner
        can only be a stand-in; this is the justified negative contract for
        the fall-through branch.
        """
        import scpn_phase_orchestrator.upde.order_params as op_mod

        monkeypatch.setattr(op_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(op_mod, "AVAILABLE_BACKENDS", ["rust", "go", "python"])
        monkeypatch.setattr(op_mod, "_BACKEND_CACHE", {})
        monkeypatch.setattr(
            op_mod,
            "_load_backend",
            lambda name: (
                {} if name == "rust" else {"order_parameter": lambda phases: (0.3, 0.0)}
            ),
        )

        owner, fn = op_mod._dispatch("order_parameter")
        assert owner == "go"
        assert callable(fn)
        assert fn(np.array([0.0], dtype=np.float64)) == (0.3, 0.0)

    def test_dispatch_falls_back_when_active_backend_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scpn_phase_orchestrator.upde.order_params as op_mod

        def failing_loader(name: str) -> dict[str, object]:
            raise OSError(f"{name} backend failed after selection")

        monkeypatch.setattr(op_mod, "ACTIVE_BACKEND", "mojo")
        monkeypatch.setattr(op_mod, "_HAS_RUST", False)
        monkeypatch.setattr(op_mod, "_load_backend", failing_loader)

        assert op_mod._dispatch("plv") == ("python", None)

    @pytest.mark.parametrize(
        "owner", owner_params(["python", "rust", "mojo", "julia", "go"])
    )
    def test_public_kernels_use_active_backend_with_flat_contiguous_payloads(
        self, owner: str
    ) -> None:
        """Each named owner serves the public kernels on strided input."""
        import scpn_phase_orchestrator.upde.order_params as op_mod

        if owner not in op_mod.AVAILABLE_BACKENDS:
            pytest.skip(f"{owner} unavailable")
        for slot in ("order_parameter", "plv", "layer_coherence"):
            assert op_mod.order_params_backend(slot, owner) == owner
        rng = np.random.default_rng(67)
        grid = rng.uniform(-np.pi, np.pi, (2, 13))
        phases, other = grid[0, ::1], grid[1, ::-1]
        assert not other.flags.c_contiguous
        mask = np.zeros(13, dtype=bool)
        mask[[1, 2, 7, 12]] = True
        atol = 1e-9 if owner == "mojo" else 1e-12

        def mean(values: list[float]) -> complex:
            """Return the mean unit phasor."""
            return sum(cmath.exp(1j * v) for v in values) / len(values)

        r, psi = compute_order_parameter(phases, backend=owner)
        z = mean(phases.tolist())
        assert r == pytest.approx(abs(z), abs=atol)
        assert abs((psi - cmath.phase(z) + np.pi) % (2 * np.pi) - np.pi) < atol
        diffs = [a - b for a, b in zip(phases.tolist(), other.tolist(), strict=True)]
        assert compute_plv(phases, other, backend=owner) == pytest.approx(
            abs(mean(diffs)), abs=atol
        )
        layer = abs(mean(phases[mask].tolist()))
        assert compute_layer_coherence(phases, mask, backend=owner) == pytest.approx(
            layer, abs=atol
        )

    def test_named_owner_refusals(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An absent or incomplete named owner and unknown names refuse."""
        import scpn_phase_orchestrator.upde.order_params as op_mod

        def broken(name: str) -> dict[str, object]:
            """Fail like a missing shared library."""
            raise OSError(f"{name} missing")

        monkeypatch.setattr(op_mod, "_load_backend", broken)
        with pytest.raises(RuntimeError, match="'go' unavailable"):
            compute_order_parameter(np.zeros(3), backend="go")
        monkeypatch.setattr(op_mod, "_load_backend", lambda name: {"plv": object()})
        with pytest.raises(RuntimeError, match="'go' lacks layer_coherence"):
            compute_layer_coherence(np.zeros(3), np.array([0]), backend="go")
        with pytest.raises(ValueError, match="unknown order-parameter backend"):
            compute_plv(np.zeros(2), np.zeros(2), backend="cuda")
        with pytest.raises(ValueError, match="unknown order-parameter operation"):
            op_mod.order_params_backend("chimera")

    def test_layer_coherence_empty_selected_subarray_returns_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scpn_phase_orchestrator.upde.order_params as op_mod

        monkeypatch.setattr(op_mod, "ACTIVE_BACKEND", "python")
        phases = np.empty((1, 0), dtype=np.float64)
        indices = np.array([0], dtype=np.int64)

        assert op_mod.compute_layer_coherence(phases, indices) == 0.0

    def test_python_fallback_kernels_preserve_physical_invariants(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scpn_phase_orchestrator.upde.order_params as op_mod

        monkeypatch.setattr(op_mod, "ACTIVE_BACKEND", "python")
        phases = np.array([0.0, np.pi], dtype=np.float64)
        shifted = phases + 0.5

        r, psi = op_mod.compute_order_parameter(phases)
        plv = op_mod.compute_plv(phases, shifted)
        layer = op_mod.compute_layer_coherence(
            phases,
            np.array([0, 1], dtype=np.int64),
        )

        assert r == pytest.approx(0.0, abs=1e-12)
        assert 0.0 <= psi < TWO_PI
        assert plv == pytest.approx(1.0, abs=1e-12)
        assert layer == pytest.approx(0.0, abs=1e-12)


# Salvaged module-specific behavioural contracts from deleted broad tests.
class TestOrderParameterEdgeCases:
    """Verify order parameter functions handle edge cases with defined behaviour,
    not just "doesn't crash"."""

    def test_empty_phases_returns_zero_r_and_psi(self) -> None:
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        r, psi = compute_order_parameter(np.array([]))
        assert r == 0.0 and psi == 0.0, "Empty phases must give R=0, Ψ=0"

    def test_single_phase_returns_r_one(self) -> None:
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        r, psi = compute_order_parameter(np.array([1.23]))
        assert abs(r - 1.0) < 1e-10, "Single oscillator must have R=1"
        assert abs(psi - 1.23) < 1e-10, "Single oscillator Ψ must equal its phase"

    def test_plv_two_empty_series_returns_zero(self) -> None:
        from scpn_phase_orchestrator.upde.order_params import compute_plv

        assert compute_plv(np.array([]), np.array([])) == 0.0

    def test_plv_one_sided_empty_series_is_invalid(self) -> None:
        from scpn_phase_orchestrator.upde.order_params import compute_plv

        with pytest.raises(ValueError, match="equal-length"):
            compute_plv(np.array([]), np.array([1.0]))
        with pytest.raises(ValueError, match="equal-length"):
            compute_plv(np.array([1.0]), np.array([]))

    def test_plv_identical_phases_returns_one(self) -> None:
        from scpn_phase_orchestrator.upde.order_params import compute_plv

        phases = np.array([0.5, 0.5, 0.5, 0.5])
        plv = compute_plv(phases, phases)
        assert abs(plv - 1.0) < 1e-10, "PLV of identical phases must be 1"

    def test_plv_anti_phase_returns_one(self) -> None:
        """PLV measures consistency of phase difference, not alignment.
        Anti-phase (π offset) has a consistent difference → PLV = 1."""
        from scpn_phase_orchestrator.upde.order_params import compute_plv

        a = np.array([0.0, 0.5, 1.0, 1.5])
        b = a + np.pi
        plv = compute_plv(a, b)
        assert abs(plv - 1.0) < 1e-10, f"Anti-phase PLV should be 1, got {plv:.4f}"


# Salvaged module-specific behavioural contracts from deleted mixed tests.
class TestOrderParamsPythonPath:
    def test_compute_order_parameter_python(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scpn_phase_orchestrator.upde.order_params as op_mod

        monkeypatch.setattr(op_mod, "_HAS_RUST", False)
        phases = np.zeros(8)
        r, psi = op_mod.compute_order_parameter(phases)
        np.testing.assert_allclose(r, 1.0, atol=1e-10)

    def test_compute_plv_python(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import scpn_phase_orchestrator.upde.order_params as op_mod

        monkeypatch.setattr(op_mod, "_HAS_RUST", False)
        a = np.zeros(10)
        b = np.zeros(10)
        plv = op_mod.compute_plv(a, b)
        np.testing.assert_allclose(plv, 1.0, atol=1e-10)

    def test_compute_plv_size_mismatch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import scpn_phase_orchestrator.upde.order_params as op_mod

        monkeypatch.setattr(op_mod, "_HAS_RUST", False)
        with pytest.raises(ValueError, match="equal-length"):
            op_mod.compute_plv(np.zeros(5), np.zeros(3))


# ──────────────────────────────────────────────────────────────────────
# knm.py: force Python fallback for build()
# ──────────────────────────────────────────────────────────────────────


class TestDebiasedSquaredOrderParameter:
    def test_perfect_synchrony_is_one(self) -> None:
        phases = np.full(5, 1.234)
        assert debiased_squared_order_parameter(phases) == pytest.approx(1.0)

    def test_two_antiphase_oscillators_give_minus_one(self) -> None:
        # R = 0 for opposed phases; (2·0 − 1) / (2 − 1) = −1.
        phases = np.array([0.0, np.pi])
        assert debiased_squared_order_parameter(phases) == pytest.approx(-1.0)

    def test_equally_spaced_phases_are_below_zero(self) -> None:
        # Perfectly uniform (R = 0) → (N·0 − 1)/(N − 1) = −1/(N − 1), honestly < 0.
        n = 8
        phases = np.arange(n, dtype=np.float64) * (TWO_PI / n)
        assert debiased_squared_order_parameter(phases) == pytest.approx(-1.0 / (n - 1))

    def test_never_exceeds_one(self) -> None:
        rng = np.random.default_rng(3)
        for _ in range(50):
            phases = rng.uniform(0.0, TWO_PI, 16)
            assert debiased_squared_order_parameter(phases) <= 1.0 + 1e-12

    def test_removes_the_small_n_bias(self) -> None:
        # Raw R² carries the 1/N floor; the debiased estimator averages to ~0.
        rng = np.random.default_rng(0)
        n = 8
        raw_squared = []
        debiased = []
        for _ in range(2000):
            phases = rng.uniform(0.0, TWO_PI, n)
            coherence, _ = compute_order_parameter(phases)
            raw_squared.append(coherence * coherence)
            debiased.append(debiased_squared_order_parameter(phases))
        assert np.mean(raw_squared) == pytest.approx(1.0 / n, abs=0.03)
        assert abs(float(np.mean(debiased))) < 0.02

    def test_single_oscillator_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least two phases"):
            debiased_squared_order_parameter(np.array([0.5]))

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least two phases"):
            debiased_squared_order_parameter(np.array([]))

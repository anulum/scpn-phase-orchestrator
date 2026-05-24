# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Psychedelic simulation protocol tests

from __future__ import annotations

from typing import Any, get_type_hints

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor import psychedelic as psychedelic_mod
from scpn_phase_orchestrator.monitor.psychedelic import (
    entropy_from_phases,
    reduce_coupling,
    simulate_psychedelic_trajectory,
)
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from tests.typing_contracts import assert_precise_ndarray_hint


def test_public_array_contracts_are_parameterised():
    hints = (
        get_type_hints(reduce_coupling)["knm"],
        get_type_hints(reduce_coupling)["return"],
        get_type_hints(entropy_from_phases)["phases"],
        get_type_hints(simulate_psychedelic_trajectory)["phases"],
        get_type_hints(simulate_psychedelic_trajectory)["omegas"],
        get_type_hints(simulate_psychedelic_trajectory)["knm"],
        get_type_hints(simulate_psychedelic_trajectory)["alpha"],
    )

    for hint in hints:
        assert_precise_ndarray_hint(hint)
        assert "float64" in str(hint)


def test_reduce_coupling_zero_factor_keeps_original():
    knm = np.ones((5, 5))
    result = reduce_coupling(knm, 0.0)
    np.testing.assert_array_equal(result, knm)


def test_reduce_coupling_full_reduction_gives_zero():
    knm = np.ones((5, 5)) * 3.0
    result = reduce_coupling(knm, 1.0)
    np.testing.assert_allclose(result, 0.0, atol=1e-15)


def test_reduce_coupling_half():
    knm = np.eye(4) * 2.0
    result = reduce_coupling(knm, 0.5)
    np.testing.assert_allclose(result, np.eye(4), atol=1e-15)


@pytest.mark.parametrize(
    ("knm", "match"),
    [
        (np.ones(4), "knm must be a finite 2-D matrix"),
        (np.array([[0.0, np.nan], [1.0, 0.0]]), "knm"),
        (np.array([[0.0, True], [1.0, 0.0]], dtype=object), "knm"),
    ],
)
def test_reduce_coupling_rejects_invalid_coupling_matrix(knm, match):
    with pytest.raises(ValueError, match=match):
        reduce_coupling(knm, 0.5)


@pytest.mark.parametrize("reduction_factor", [-0.1, 1.1, np.nan, True])
def test_reduce_coupling_rejects_invalid_reduction_factor(reduction_factor):
    with pytest.raises((TypeError, ValueError), match="reduction_factor"):
        reduce_coupling(np.eye(3), reduction_factor)


@pytest.mark.parametrize(
    "backend_output",
    [
        np.array([0.0], dtype=np.float64),
        np.array([[0.0, 1.0]], dtype=np.float64),
        np.array([[0.0, np.nan], [1.0, 0.0]], dtype=np.float64),
        np.array([[False, True], [True, False]], dtype=np.bool_),
        np.array([[0.0, np.bool_(True)], [1.0, 0.0]], dtype=object),
    ],
)
def test_reduce_coupling_rejects_invalid_rust_reduce_output(
    monkeypatch: pytest.MonkeyPatch,
    backend_output: np.ndarray,
) -> None:
    monkeypatch.setattr(psychedelic_mod, "_HAS_RUST_REDUCE", True)
    monkeypatch.setattr(
        psychedelic_mod, "_rust_reduce", lambda *_args: backend_output, raising=False
    )

    with pytest.raises(ValueError, match="reduced coupling"):
        reduce_coupling(np.eye(2), 0.5)


def test_reduce_coupling_accepts_flat_backend_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend flat n*n payloads are restored to the public matrix contract."""
    monkeypatch.setattr(psychedelic_mod, "_HAS_RUST_REDUCE", True)
    monkeypatch.setattr(
        psychedelic_mod,
        "_rust_reduce",
        lambda flat, factor: flat * (1.0 - factor),
        raising=False,
    )

    result = reduce_coupling(np.eye(3), 0.25)

    np.testing.assert_allclose(result, np.eye(3) * 0.75)


def test_entropy_uniform_phases_high():
    """Uniformly distributed phases should have near-maximal entropy."""
    phases = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    ent = entropy_from_phases(phases)
    max_entropy = np.log(36)  # 36 bins
    assert ent > 0.9 * max_entropy


def test_entropy_concentrated_phases_low():
    """All phases at same value → only 1 bin occupied → entropy = 0."""
    phases = np.full(100, 1.0)
    ent = entropy_from_phases(phases)
    assert ent == 0.0


def test_entropy_empty_phases():
    assert entropy_from_phases(np.array([])) == 0.0


@pytest.mark.parametrize(
    "phases",
    [
        np.array([[0.0, 1.0]]),
        np.array([0.0, np.inf]),
        np.array([0.0, True], dtype=object),
    ],
)
def test_entropy_rejects_non_vector_or_non_finite_phases(phases):
    with pytest.raises(ValueError, match="phases"):
        entropy_from_phases(phases)


@pytest.mark.parametrize("n_bins", [0, 1, False, 18.5])
def test_entropy_rejects_invalid_bin_counts(n_bins):
    with pytest.raises((TypeError, ValueError), match="n_bins"):
        entropy_from_phases(np.linspace(0.0, 1.0, 8), n_bins=n_bins)


def test_entropy_accepts_numpy_integer_bin_count() -> None:
    ent = entropy_from_phases(np.linspace(0.0, 1.0, 8), n_bins=np.int64(4))

    assert 0.0 <= ent <= np.log(4)


@pytest.mark.parametrize(
    "backend_value",
    [-0.1, np.nan, np.inf, [0.5], np.log(4) + 1.0, True, np.bool_(True)],
)
def test_entropy_invalid_backend_payload_falls_back(
    monkeypatch: pytest.MonkeyPatch,
    backend_value: Any,
) -> None:
    phases = np.linspace(0.0, 1.0, 16)
    monkeypatch.setattr(
        psychedelic_mod, "_dispatch", lambda: lambda *_args: backend_value
    )

    ent = entropy_from_phases(phases, n_bins=4)

    assert 0.0 <= ent <= np.log(4)


def test_simulate_trajectory_returns_correct_length():
    n = 10
    engine = UPDEEngine(n, dt=0.01)
    rng = np.random.default_rng(7)
    phases = rng.uniform(0, 2 * np.pi, n)
    omegas = rng.normal(1.0, 0.1, n)
    knm = np.ones((n, n)) * 0.5
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))
    schedule = [0.0, 0.3, 0.6, 0.9]

    results = simulate_psychedelic_trajectory(
        engine,
        phases,
        omegas,
        knm,
        alpha,
        schedule,
        n_steps_per_level=50,
    )
    assert len(results) == 4
    for rec in results:
        assert "R" in rec
        assert "entropy" in rec
        assert "chimera_index" in rec
        assert rec["phases"].shape == (n,)


def test_trajectory_entropy_increases_with_coupling_reduction():
    """Entropic brain hypothesis: lower coupling → higher entropy (on average)."""
    n = 30
    engine = UPDEEngine(n, dt=0.01)
    # Start synchronised
    phases = np.zeros(n)
    omegas = np.ones(n)
    knm = np.ones((n, n)) * 2.0
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))
    schedule = [0.0, 0.5, 0.9]

    results = simulate_psychedelic_trajectory(
        engine,
        phases,
        omegas,
        knm,
        alpha,
        schedule,
        n_steps_per_level=200,
    )
    # With strong coupling reduction, entropy should not decrease overall
    assert results[-1]["entropy"] >= results[0]["entropy"] - 0.5


def test_simulate_trajectory_rejects_mismatched_runtime_shapes():
    n = 4
    engine = UPDEEngine(n, dt=0.01)
    phases = np.linspace(0.0, 1.0, n)
    omegas = np.ones(n - 1)
    knm = np.eye(n)
    alpha = np.zeros((n, n))
    with pytest.raises(ValueError, match="omegas"):
        simulate_psychedelic_trajectory(
            engine,
            phases,
            omegas,
            knm,
            alpha,
            [0.0],
            n_steps_per_level=1,
        )


def test_simulate_trajectory_rejects_empty_phase_vector():
    engine = UPDEEngine(1, dt=0.01)

    with pytest.raises(ValueError, match="at least one oscillator"):
        simulate_psychedelic_trajectory(
            engine,
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            [0.0],
            n_steps_per_level=1,
        )


def test_simulate_trajectory_with_empty_schedule_returns_empty_series():
    n = 6
    engine = UPDEEngine(n, dt=0.01)
    phases = np.arange(n, dtype=float)
    omegas = np.ones(n)
    knm = np.ones((n, n))
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))

    results = simulate_psychedelic_trajectory(
        engine,
        phases,
        omegas,
        knm,
        alpha,
        reduction_schedule=[],
        n_steps_per_level=5,
    )

    assert results == []


def test_simulate_trajectory_rejects_invalid_schedule_and_step_count():
    n = 4
    engine = UPDEEngine(n, dt=0.01)
    phases = np.linspace(0.0, 1.0, n)
    omegas = np.ones(n)
    knm = np.eye(n)
    alpha = np.zeros((n, n))
    with pytest.raises(ValueError, match="reduction_schedule"):
        simulate_psychedelic_trajectory(
            engine,
            phases,
            omegas,
            knm,
            alpha,
            [0.0, 1.2],
            n_steps_per_level=1,
        )
    with pytest.raises((TypeError, ValueError), match="reduction_schedule"):
        simulate_psychedelic_trajectory(
            engine,
            phases,
            omegas,
            knm,
            alpha,
            0.5,  # type: ignore[arg-type]
            n_steps_per_level=1,
        )
    with pytest.raises((TypeError, ValueError), match="reduction_schedule"):
        simulate_psychedelic_trajectory(
            engine,
            phases,
            omegas,
            knm,
            alpha,
            [[0.0, 0.5]],  # type: ignore[list-item]
            n_steps_per_level=1,
        )
    with pytest.raises((TypeError, ValueError), match="n_steps_per_level"):
        simulate_psychedelic_trajectory(
            engine,
            phases,
            omegas,
            knm,
            alpha,
            [0.0],
            n_steps_per_level=True,
        )


class TestPsychedelicPipelineWiring:
    """Pipeline: UPDEEngine → psychedelic simulation → R + entropy."""

    def test_psychedelic_sim_produces_valid_metrics(self):
        """simulate_psychedelic_trajectory uses engine internally and
        produces R∈[0,1] and entropy≥0 at each coupling reduction level."""
        n = 8
        engine = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = np.ones((n, n)) * 0.5
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))

        results = simulate_psychedelic_trajectory(
            engine,
            phases,
            omegas,
            knm,
            alpha,
            reduction_schedule=[0.0, 0.5],
            n_steps_per_level=30,
        )
        for rec in results:
            assert 0.0 <= rec["R"] <= 1.0
            assert rec["entropy"] >= 0.0


class TestPsychedelicBackendDispatch:
    def test_dispatch_falls_back_to_python_when_backend_loading_fails(
        self, monkeypatch
    ):
        previous_backend = psychedelic_mod.ACTIVE_BACKEND
        previous_available = list(psychedelic_mod.AVAILABLE_BACKENDS)
        previous_loader = psychedelic_mod._LOADERS["go"]
        psychedelic_mod.ACTIVE_BACKEND = "go"
        psychedelic_mod.AVAILABLE_BACKENDS = ["go", "python"]
        psychedelic_mod._BACKEND_FN_CACHE.clear()
        monkeypatch.setitem(
            psychedelic_mod._LOADERS,
            "go",
            lambda: (_ for _ in ()).throw(ImportError("go backend unavailable")),
        )
        try:
            fn = psychedelic_mod._dispatch()
        finally:
            psychedelic_mod.ACTIVE_BACKEND = previous_backend
            psychedelic_mod.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(psychedelic_mod._LOADERS, "go", previous_loader)
            psychedelic_mod._BACKEND_FN_CACHE.clear()

        assert fn is None

    def test_dispatch_caches_loaded_backend_function(self, monkeypatch):
        previous_backend = psychedelic_mod.ACTIVE_BACKEND
        previous_available = list(psychedelic_mod.AVAILABLE_BACKENDS)
        previous_loader = psychedelic_mod._LOADERS["go"]
        psychedelic_mod.ACTIVE_BACKEND = "go"
        psychedelic_mod.AVAILABLE_BACKENDS = ["go", "python"]
        psychedelic_mod._BACKEND_FN_CACHE.clear()
        call_count = 0

        def fake_fn(_phases, _n_bins):
            return 0.0

        def fake_loader():
            nonlocal call_count
            call_count += 1
            return fake_fn

        monkeypatch.setitem(psychedelic_mod._LOADERS, "go", fake_loader)
        try:
            fn1 = psychedelic_mod._dispatch()
            fn2 = psychedelic_mod._dispatch()
        finally:
            psychedelic_mod.ACTIVE_BACKEND = previous_backend
            psychedelic_mod.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(psychedelic_mod._LOADERS, "go", previous_loader)
            psychedelic_mod._BACKEND_FN_CACHE.clear()

        assert fn1 is fake_fn
        assert fn2 is fake_fn
        assert call_count == 1

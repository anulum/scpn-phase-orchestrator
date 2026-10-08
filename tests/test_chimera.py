# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Chimera state detection tests

"""Exercise original public chimera measurement and classification contracts."""

from __future__ import annotations

import cProfile
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from benchmarks.chimera_local_order_reference import scalar_local_order
from scpn_phase_orchestrator.monitor import chimera as chimera_module
from scpn_phase_orchestrator.monitor.chimera import (
    ChimeraState,
    detect_chimera,
    local_order_parameter,
)
from tests.test_chimera_real_runtime import installed_absent_probe

FloatArray = NDArray[np.float64]


def _uniform_knm(n: int, strength: float = 1.0) -> FloatArray:
    """Construct finite positive all-to-all coupling with a zero diagonal."""
    knm = np.full((n, n), strength, dtype=np.float64)
    np.fill_diagonal(knm, 0.0)
    return knm


def test_fully_synchronised_all_coherent() -> None:
    """Classify every coupled synchronized oscillator as coherent."""
    phases = np.zeros(20)
    knm = _uniform_knm(20)
    state = detect_chimera(phases, knm)
    assert len(state.coherent_indices) == 20
    assert len(state.incoherent_indices) == 0
    assert state.chimera_index == 0.0


def test_uniform_random_phases_mostly_incoherent() -> None:
    """Resolve low neighbourhood coherence on a seeded random population."""
    rng = np.random.default_rng(7)
    phases = rng.uniform(0, 2 * np.pi, 100)
    knm = _uniform_knm(100)
    state = detect_chimera(phases, knm)
    assert len(state.incoherent_indices) > 50


def test_chimera_state_has_mixed_groups() -> None:
    """Half synchronised, half scattered — the hallmark chimera.

    Uses block-diagonal coupling so each group's R_i is computed only
    from its own neighbors, preventing dilution.
    """
    n = 40
    half = n // 2
    phases = np.zeros(n)
    rng = np.random.default_rng(42)
    phases[half:] = rng.uniform(0, 2 * np.pi, half)
    # Block coupling: each half only couples within itself
    knm = np.zeros((n, n))
    knm[:half, :half] = 1.0
    knm[half:, half:] = 1.0
    np.fill_diagonal(knm, 0.0)
    state = detect_chimera(phases, knm)
    assert len(state.coherent_indices) > 0
    assert len(state.incoherent_indices) > 0


def test_chimera_index_between_zero_and_one() -> None:
    """Keep the public boundary fraction in its declared interval."""
    rng = np.random.default_rng(99)
    phases = rng.uniform(0, 2 * np.pi, 50)
    knm = _uniform_knm(50)
    state = detect_chimera(phases, knm)
    assert 0.0 <= state.chimera_index <= 1.0


def test_empty_phases_returns_empty_state() -> None:
    """Preserve the valid empty classification identity."""
    state = detect_chimera(np.array([]), np.zeros((0, 0)))
    assert state == ChimeraState()


def test_two_oscillators_in_phase() -> None:
    """Classify two coupled synchronized oscillators as coherent."""
    phases = np.array([0.0, 0.0])
    knm = np.array([[0.0, 1.0], [1.0, 0.0]])
    state = detect_chimera(phases, knm)
    assert len(state.coherent_indices) == 2


def test_no_coupling_gives_zero_r_local() -> None:
    """With zero coupling, no oscillator has neighbors — all R_i = 0."""
    phases = np.zeros(10)
    knm = np.zeros((10, 10))
    state = detect_chimera(phases, knm)
    # R_i = 0 < 0.3 → all incoherent
    assert len(state.incoherent_indices) == 10
    assert len(state.coherent_indices) == 0


def test_dataclass_fields() -> None:
    """Retain the public result fields and original values."""
    state = ChimeraState(
        coherent_indices=[0, 1],
        incoherent_indices=[3],
        chimera_index=0.25,
    )
    assert state.coherent_indices == [0, 1]
    assert state.incoherent_indices == [3]
    assert state.chimera_index == 0.25


@pytest.mark.parametrize(
    "payload",
    [
        {"coherent_indices": [0, 0], "incoherent_indices": [], "chimera_index": 0.0},
        {"coherent_indices": [True], "incoherent_indices": [], "chimera_index": 0.0},
        {"coherent_indices": [-1], "incoherent_indices": [], "chimera_index": 0.0},
        {"coherent_indices": [1], "incoherent_indices": [1], "chimera_index": 0.0},
        {"coherent_indices": [], "incoherent_indices": [], "chimera_index": -0.1},
        {"coherent_indices": [], "incoherent_indices": [], "chimera_index": np.nan},
        {"coherent_indices": [], "incoherent_indices": [], "chimera_index": True},
        {
            "coherent_indices": [],
            "incoherent_indices": [],
            "chimera_index": np.bool_(True),
        },
        {"coherent_indices": [], "incoherent_indices": [], "chimera_index": 0.5 + 0.0j},
    ],
)
def test_chimera_state_rejects_invalid_public_record(
    payload: dict[str, object],
) -> None:
    """Refuse genuine invalid record fields without coercing their original types."""
    with pytest.raises(ValueError):
        ChimeraState(
            coherent_indices=cast(list[int], payload["coherent_indices"]),
            incoherent_indices=cast(list[int], payload["incoherent_indices"]),
            chimera_index=cast(float, payload["chimera_index"]),
        )


@pytest.mark.parametrize(
    ("phases", "knm", "match"),
    [
        (np.array([0.0, True], dtype=object), np.zeros((2, 2)), "phases"),
        (
            np.array(["0.0", "0.5"], dtype=object),
            np.zeros((2, 2)),
            "numeric-string",
        ),
        (np.array([0.0 + 0.0j, 0.5 + 0.25j]), np.zeros((2, 2)), "real-valued"),
        (
            np.array([0.0 + 0.25j, 0.5], dtype=object),
            np.zeros((2, 2)),
            "real-valued",
        ),
        (np.zeros((1, 2)), np.zeros((2, 2)), "phases"),
        (np.array([0.0, np.nan]), np.zeros((2, 2)), "phases"),
        (np.zeros(2), np.array([[0.0, True], [0.0, 0.0]], dtype=object), "knm"),
        (
            np.zeros(2),
            np.array([["0.0", "1.0"], ["1.0", "0.0"]], dtype=object),
            "numeric-string",
        ),
        (
            np.zeros(2),
            np.array([[0.0 + 0.0j, 1.0 + 0.25j], [1.0, 0.0 + 0.0j]]),
            "real-valued",
        ),
        (
            np.zeros(2),
            np.array([[0.0, 1.0 + 0.25j], [1.0, 0.0]], dtype=object),
            "real-valued",
        ),
        (np.zeros(2), np.zeros((2, 3)), "knm"),
        (np.zeros(2), np.array([[0.0, np.inf], [0.0, 0.0]]), "knm"),
        (np.zeros(2), np.array([[1.0, 0.0], [0.0, 0.0]]), "self-coupling"),
    ],
)
def test_rejects_invalid_chimera_inputs(
    phases: object,
    knm: object,
    match: str,
) -> None:
    """Refuse source aliases, nonfinite values and invalid graph dimensions."""
    with pytest.raises(ValueError, match=match):
        local_order_parameter(cast(FloatArray, phases), cast(FloatArray, knm))


@pytest.mark.native_runtime
def test_local_order_parameter_uses_backend_when_available() -> None:
    """Observe the original Rust builtin through the actual preferred public API."""
    phases = np.array([0.0, 0.2, 0.4])
    knm = _uniform_knm(3)
    assert chimera_module.ACTIVE_BACKEND == "rust"
    profiler = cProfile.Profile()
    with profiler:
        local = local_order_parameter(phases, knm)
    np.testing.assert_allclose(
        local, scalar_local_order(phases, knm), atol=1e-12, rtol=0
    )
    calls = sum(
        entry.callcount
        for entry in profiler.getstats()
        if isinstance(entry.code, str) and "detect_chimera_rust" in entry.code
    )
    assert calls == 1


def test_local_order_parameter_falls_back_when_backend_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Computation faults propagate in the retained original fallback test slot."""

    def failing(_p: FloatArray, _k: FloatArray, _n: int) -> FloatArray:
        """Raise solely as an explicit negative execution-fault control."""
        raise RuntimeError("boom")

    monkeypatch.setattr(chimera_module, "_dispatch", lambda backend=None: failing)
    with pytest.raises(RuntimeError, match="boom"):
        local_order_parameter(np.array([0.0, 0.2, 0.4]), _uniform_knm(3))


@pytest.mark.parametrize(
    "backend_output",
    [
        np.array([0.5], dtype=np.float64),
        np.array([0.5, np.nan], dtype=np.float64),
        np.array([0.5, 1.1], dtype=np.float64),
        np.array([-0.1, 0.5], dtype=np.float64),
        np.array([0.5 + 0.0j, 0.5 + 0.25j]),
        np.array([0.5 + 0.25j, 0.5], dtype=object),
        np.array([True, False], dtype=np.bool_),
        np.array([0.5, np.bool_(True)], dtype=object),
        np.array(["0.5", "0.25"], dtype=object),
    ],
)
def test_local_order_parameter_invalid_backend_payload_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    backend_output: FloatArray,
) -> None:
    """Reject deliberately invalid execution returns as a negative boundary control."""

    def _fake_backend(
        _phases: FloatArray,
        _knm_flat: FloatArray,
        _n: int,
    ) -> FloatArray:
        """Return the original invalid payload solely for negative validation."""
        return backend_output

    monkeypatch.setattr(chimera_module, "_dispatch", lambda backend=None: _fake_backend)
    phases = np.array([0.0, 0.0], dtype=np.float64)
    knm = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)

    with pytest.raises(ValueError):
        local_order_parameter(phases, knm)


@pytest.mark.native_runtime
def test_dispatch_falls_back_to_python_when_loader_fails() -> None:
    """A current installed profile genuinely lacks Go and preserves Python output."""
    record = installed_absent_probe()
    assert "go" in cast(list[str], record["missing"])
    assert record["active"] == "python"
    assert record["local"] == [0.0, 0.0]


@pytest.mark.native_runtime
def test_dispatch_uses_cached_loader_once() -> None:
    """Real Go public calls reuse the resolved loader and execute its bridge."""
    phases = np.array([0.0, 0.2, 0.4])
    knm = _uniform_knm(3)
    expected = scalar_local_order(phases, knm)
    np.testing.assert_allclose(
        local_order_parameter(phases, knm, backend="go"), expected, atol=1e-12, rtol=0
    )
    profiler = cProfile.Profile()
    with profiler:
        first = local_order_parameter(phases, knm, backend="go")
        second = local_order_parameter(phases, knm, backend="go")
    np.testing.assert_allclose(first, expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(second, expected, atol=1e-12, rtol=0)
    names = [
        entry.code.co_name
        for entry in profiler.getstats()
        if not isinstance(entry.code, str)
    ]
    assert "_load_go_fn" not in names
    calls = sum(
        entry.callcount
        for entry in profiler.getstats()
        if not isinstance(entry.code, str)
        and entry.code.co_name == "local_order_parameter_go"
    )
    assert calls == 2


class TestChimeraPipelineWiring:
    """Pipeline: engine phases → detect_chimera → chimera_index."""

    def test_engine_phases_to_chimera_detection(self) -> None:
        """Classify actual UPDE engine phases through the public chimera monitor."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        n = 16
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = rng.normal(1.0, 0.5, n)
        knm = _uniform_knm(n, 0.3)
        alpha = np.zeros((n, n))
        for _ in range(200):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)

        state = detect_chimera(phases, _uniform_knm(n, 0.3))
        assert isinstance(state, ChimeraState)
        assert 0.0 <= state.chimera_index <= 1.0
        total = len(state.coherent_indices) + len(state.incoherent_indices)
        assert total == n

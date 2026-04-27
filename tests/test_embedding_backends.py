# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-backend parity for embedding primitives

"""Cross-backend parity for the three embedding primitives.

Tolerances:

* ``delay_embed`` — exact (pure indexing).
* ``mutual_information`` — 1e-9; the NumPy reference uses
  ``np.histogram2d`` which chooses bin edges through a slightly
  different interior path from our manual ``min/max`` binning
  (difference ≤ 3e-11 on random signals).
* ``nearest_neighbor_distances`` — 1e-9 float + array-exact
  indices.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.monitor import embedding as em_mod
from scpn_phase_orchestrator.monitor.embedding import (
    AVAILABLE_BACKENDS,
    delay_embed,
    mutual_information,
    nearest_neighbor_distances,
)


def _force(backend: str) -> str:
    prev = em_mod.ACTIVE_BACKEND
    em_mod.ACTIVE_BACKEND = backend
    return prev


def _reset(prev: str) -> None:
    em_mod.ACTIVE_BACKEND = prev


def _reference_mi(sig: np.ndarray, lag: int, n_bins: int) -> float:
    prev = _force("python")
    try:
        return mutual_information(sig, lag, n_bins)
    finally:
        _reset(prev)


def _reference_nn(emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prev = _force("python")
    try:
        return nearest_neighbor_distances(emb)
    finally:
        _reset(prev)


def _reference_de(sig, delay, dim) -> np.ndarray:
    prev = _force("python")
    try:
        return delay_embed(sig, delay, dim)
    finally:
        _reset(prev)


def _signal(seed: int, t: int = 200) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sin(np.linspace(0, 10 * np.pi, t)) + 0.1 * rng.normal(0, 1, t)


class TestDelayEmbedParity:
    @given(
        t=st.integers(min_value=10, max_value=100),
        delay=st.integers(min_value=1, max_value=5),
        dim=st.integers(min_value=1, max_value=4),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_all_backends_exact(
        self,
        t: int,
        delay: int,
        dim: int,
        seed: int,
    ):
        if t - (dim - 1) * delay <= 0:
            return
        sig = _signal(seed, t)
        ref = _reference_de(sig, delay, dim)
        for backend in AVAILABLE_BACKENDS:
            prev = _force(backend)
            try:
                got = delay_embed(sig, delay, dim)
            finally:
                _reset(prev)
            np.testing.assert_array_equal(
                got,
                ref,
                err_msg=f"{backend} delay_embed diverged",
            )


class TestMutualInformationParity:
    @pytest.mark.parametrize("seed", [0, 42])
    @pytest.mark.parametrize("lag", [1, 5, 20])
    def test_non_python_backends_match(
        self,
        seed: int,
        lag: int,
    ) -> None:
        sig = _signal(seed)
        ref = _reference_mi(sig, lag, 16)
        for backend in AVAILABLE_BACKENDS:
            if backend == "python":
                continue
            prev = _force(backend)
            try:
                got = mutual_information(sig, lag, 16)
            finally:
                _reset(prev)
            assert abs(got - ref) < 1e-9, f"{backend} MI diverged: {got} vs {ref}"


class TestNearestNeighborParity:
    @pytest.mark.parametrize("seed", [0, 77])
    def test_dist_and_idx(self, seed: int) -> None:
        sig = _signal(seed)
        emb = delay_embed(sig, 5, 3)
        ref_dist, ref_idx = _reference_nn(emb)
        for backend in AVAILABLE_BACKENDS:
            if backend == "python":
                continue
            prev = _force(backend)
            try:
                dist, idx = nearest_neighbor_distances(emb)
            finally:
                _reset(prev)
            np.testing.assert_allclose(
                dist,
                ref_dist,
                atol=1e-9,
                err_msg=f"{backend} NN dist diverged",
            )
            np.testing.assert_array_equal(
                idx,
                ref_idx,
                err_msg=f"{backend} NN idx diverged",
            )


class TestDispatcherFallthroughForRust:
    def test_rust_active_still_resolves_mi_nn(self) -> None:
        """Rust has no standalone MI / NN FFI; with Rust active the
        dispatcher must still produce a value via the next backend."""
        if "rust" not in AVAILABLE_BACKENDS:
            pytest.skip("Rust backend not built")
        sig = _signal(0)
        prev = _force("rust")
        try:
            mi = mutual_information(sig, 5, 16)
            emb = delay_embed(sig, 5, 3)
            dist, idx = nearest_neighbor_distances(emb)
        finally:
            _reset(prev)
        assert np.isfinite(mi)
        assert dist.size == emb.shape[0]
        assert idx.size == emb.shape[0]

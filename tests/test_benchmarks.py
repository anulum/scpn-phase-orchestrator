# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — pytest-benchmark performance regression tests

"""Performance benchmarks for hot-path functions.

Run: py -3.12 -m pytest tests/test_benchmarks.py --benchmark-only
Compare: py -3.12 -m pytest tests/test_benchmarks.py --benchmark-compare

These benchmarks track performance regressions across commits.
They are NOT run during normal CI (--benchmark-disable is the default).
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.coupling.spectral import (
    fiedler_value,
    graph_laplacian,
)
from scpn_phase_orchestrator.monitor.chimera import detect_chimera
from scpn_phase_orchestrator.monitor.npe import compute_npe
from scpn_phase_orchestrator.monitor.recurrence import rqa
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

TWO_PI = 2.0 * np.pi


def _setup(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, TWO_PI, n)
    omegas = rng.uniform(-1, 1, n)
    raw = rng.uniform(0.3, 1.0, (n, n))
    knm = 0.5 * (raw + raw.T)
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))
    return phases, omegas, knm, alpha


# ── Engine step benchmarks ───────────────────────────────────────────────


class TestEngineStepBenchmark:
    @pytest.mark.benchmark(group="engine-step")
    @pytest.mark.parametrize("n", [8, 32, 128])
    def test_upde_euler_step(self, benchmark, n: int) -> None:
        phases, omegas, knm, alpha = _setup(n)
        eng = UPDEEngine(n, dt=0.01, method="euler")
        benchmark(eng.step, phases, omegas, knm, 0.0, 0.0, alpha)

    @pytest.mark.benchmark(group="engine-step")
    @pytest.mark.parametrize("n", [8, 32, 128])
    def test_upde_rk4_step(self, benchmark, n: int) -> None:
        phases, omegas, knm, alpha = _setup(n)
        eng = UPDEEngine(n, dt=0.01, method="rk4")
        benchmark(eng.step, phases, omegas, knm, 0.0, 0.0, alpha)


# ── Order parameter benchmarks ───────────────────────────────────────────


class TestOrderParameterBenchmark:
    @pytest.mark.benchmark(group="order-param")
    @pytest.mark.parametrize("n", [16, 64, 256])
    def test_compute_R(self, benchmark, n: int) -> None:
        phases = np.random.default_rng(0).uniform(0, TWO_PI, n)
        benchmark(compute_order_parameter, phases)


# ── Spectral benchmarks ─────────────────────────────────────────────────


class TestSpectralBenchmark:
    @pytest.mark.benchmark(group="spectral")
    @pytest.mark.parametrize("n", [16, 64, 256])
    def test_graph_laplacian(self, benchmark, n: int) -> None:
        _, _, knm, _ = _setup(n)
        benchmark(graph_laplacian, knm)

    @pytest.mark.benchmark(group="spectral")
    @pytest.mark.parametrize("n", [16, 64, 256])
    def test_fiedler_value(self, benchmark, n: int) -> None:
        _, _, knm, _ = _setup(n)
        benchmark(fiedler_value, knm)


# ── Monitor benchmarks ──────────────────────────────────────────────────


class TestMonitorBenchmark:
    @pytest.mark.benchmark(group="monitor")
    @pytest.mark.parametrize("n", [16, 64, 256])
    def test_compute_npe(self, benchmark, n: int) -> None:
        phases = np.random.default_rng(0).uniform(0, TWO_PI, n)
        benchmark(compute_npe, phases)

    @pytest.mark.benchmark(group="monitor")
    @pytest.mark.parametrize("n", [8, 32])
    def test_detect_chimera(self, benchmark, n: int) -> None:
        phases, _, knm, _ = _setup(n)
        benchmark(detect_chimera, phases, knm)

    @pytest.mark.benchmark(group="monitor")
    def test_rqa_t50(self, benchmark) -> None:
        traj = np.random.default_rng(0).standard_normal((50, 2))
        benchmark(rqa, traj, 1.0)

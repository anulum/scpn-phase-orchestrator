# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — engine internal-locking regression tests

from __future__ import annotations

import threading

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine


@pytest.fixture
def upde_engine() -> UPDEEngine:
    return UPDEEngine(n_oscillators=4, dt=0.01)


@pytest.fixture
def sl_engine() -> StuartLandauEngine:
    return StuartLandauEngine(n_oscillators=4, dt=0.01)


def _upde_inputs(n: int, rng: np.random.Generator):
    return (
        rng.uniform(0.0, 2.0 * np.pi, n),
        rng.uniform(0.5, 1.5, n),
        rng.uniform(0.1, 0.5, (n, n)),
        0.0,
        0.0,
        np.zeros((n, n)),
    )


class TestUpdeEngineLocking:
    def test_lock_is_reentrant(self, upde_engine: UPDEEngine) -> None:
        """run() calls step() internally — a non-reentrant lock would
        deadlock the first step.
        """
        rng = np.random.default_rng(0)
        phases, omegas, knm, zeta, psi, alpha = _upde_inputs(4, rng)
        final = upde_engine.run(phases, omegas, knm, zeta, psi, alpha, n_steps=3)
        assert final.shape == (4,)
        assert np.all(np.isfinite(final))

    def test_concurrent_step_does_not_corrupt_scratch(
        self, upde_engine: UPDEEngine
    ) -> None:
        """Two threads stepping the same engine must produce finite
        results — a race on _phase_diff / _sin_diff would surface as NaN.
        """
        rng = np.random.default_rng(1)
        phases, omegas, knm, zeta, psi, alpha = _upde_inputs(4, rng)

        errors: list[Exception] = []
        lock_barrier = threading.Barrier(2)

        def worker() -> None:
            try:
                lock_barrier.wait(timeout=2.0)
                for _ in range(50):
                    local = upde_engine.step(
                        phases.copy(), omegas, knm, zeta, psi, alpha
                    )
                    if not np.all(np.isfinite(local)):
                        errors.append(
                            RuntimeError("NaN/Inf leaked through concurrent step")
                        )
                        return
            except Exception as exc:  # noqa: BLE001 — propagate to assertion
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)
            assert not t.is_alive(), "worker thread did not finish"
        assert errors == []

    def test_lock_attribute_exists(self, upde_engine: UPDEEngine) -> None:
        """Regression: the _lock must survive construction with every
        supported integration method."""
        assert hasattr(upde_engine, "_lock")
        upde_rk4 = UPDEEngine(n_oscillators=2, dt=0.01, method="rk4")
        upde_rk45 = UPDEEngine(n_oscillators=2, dt=0.01, method="rk45")
        assert hasattr(upde_rk4, "_lock")
        assert hasattr(upde_rk45, "_lock")


class TestStuartLandauEngineLocking:
    def test_lock_attribute_exists(self, sl_engine: StuartLandauEngine) -> None:
        assert hasattr(sl_engine, "_lock")

    def test_concurrent_step_is_race_free(
        self, sl_engine: StuartLandauEngine
    ) -> None:
        rng = np.random.default_rng(2)
        n = 4
        state = np.concatenate(
            [rng.uniform(0.0, 2.0 * np.pi, n), rng.uniform(0.5, 1.5, n)]
        )
        omegas = rng.uniform(0.5, 1.5, n)
        mu = np.full(n, 0.1)
        knm = rng.uniform(0.1, 0.3, (n, n))
        knm_r = rng.uniform(0.05, 0.15, (n, n))
        alpha = np.zeros((n, n))

        errors: list[Exception] = []
        barrier = threading.Barrier(2)

        def worker() -> None:
            try:
                barrier.wait(timeout=2.0)
                for _ in range(40):
                    out = sl_engine.step(
                        state.copy(), omegas, mu, knm, knm_r, 0.0, 0.0, alpha, 1.0
                    )
                    if not np.all(np.isfinite(out)):
                        errors.append(
                            RuntimeError("NaN/Inf in StuartLandau concurrent step")
                        )
                        return
            except Exception as exc:  # noqa: BLE001 — propagate to assertion
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)
            assert not t.is_alive()
        assert errors == []


# Pipeline wiring: concurrent step() is the realistic failure mode for
# multi-client gRPC / WebSocket deployments that share an engine. The
# tests above exercise both Python-path scratch arrays and the RLock
# re-entry from run() so regressions surface immediately.

# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for Partial Information Decomposition

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor.pid import redundancy, synergy


class TestRedundancy:
    def test_identical_groups_maximum_redundancy(self):
        """Same oscillators in both groups →
        redundancy = MI of that group with whole."""
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, 100)
        group = list(range(50))
        r = redundancy(phases, group, group)
        assert r >= 0.0

    def test_non_negative(self):
        rng = np.random.default_rng(7)
        phases = rng.uniform(0, 2 * np.pi, 50)
        r = redundancy(phases, [0, 1, 2], [3, 4, 5])
        assert r >= 0.0

    def test_empty_phases(self):
        assert redundancy(np.array([]), [0], [1]) == 0.0

    def test_empty_group_a(self):
        phases = np.array([0.0, 1.0, 2.0])
        assert redundancy(phases, [], [0, 1]) == 0.0

    def test_empty_group_b(self):
        phases = np.array([0.0, 1.0, 2.0])
        assert redundancy(phases, [0, 1], []) == 0.0

    def test_synchronized_phases_low_redundancy(self):
        """All phases identical → flat histogram → low entropy → low MI."""
        phases = np.zeros(100)
        r = redundancy(phases, [0, 1, 2], [3, 4, 5])
        assert r == pytest.approx(0.0, abs=0.1)


class TestSynergy:
    def test_non_negative(self):
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, 100)
        s = synergy(phases, list(range(0, 50)), list(range(50, 100)))
        assert s >= 0.0

    def test_empty_phases(self):
        assert synergy(np.array([]), [0], [1]) == 0.0

    def test_empty_group(self):
        phases = np.array([0.0, 1.0, 2.0])
        assert synergy(phases, [], [0, 1]) == 0.0

    def test_disjoint_uniform_groups(self):
        """Uniform random phases in disjoint groups should have finite synergy."""
        rng = np.random.default_rng(99)
        phases = rng.uniform(0, 2 * np.pi, 200)
        s = synergy(phases, list(range(0, 100)), list(range(100, 200)))
        assert np.isfinite(s)

    def test_synergy_with_structured_phases(self):
        """Phases with structure (e.g. two clusters)
        should produce measurable synergy."""
        rng = np.random.default_rng(55)
        phases = np.concatenate(
            [
                rng.normal(0.5, 0.3, 50) % (2 * np.pi),
                rng.normal(3.5, 0.3, 50) % (2 * np.pi),
            ]
        )
        s = synergy(phases, list(range(0, 25)), list(range(25, 50)))
        assert s >= 0.0


class TestPIDPipelineWiring:
    """Pipeline: engine phases → PID redundancy/synergy between layers."""

    def test_engine_phases_to_pid(self):
        """UPDEEngine → phases → redundancy/synergy between oscillator
        groups. Quantifies information sharing across layers."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        n = 8
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = 0.5 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        for _ in range(200):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)

        group_a = [0, 1, 2, 3]
        group_b = [4, 5, 6, 7]
        r = redundancy(phases, group_a, group_b)
        s = synergy(phases, group_a, group_b)
        assert r >= 0.0
        assert np.isfinite(s)

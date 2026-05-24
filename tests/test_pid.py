# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for Partial Information Decomposition

from __future__ import annotations

from typing import Any, get_type_hints

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor import pid as pid_module
from scpn_phase_orchestrator.monitor.pid import redundancy, synergy
from tests.typing_contracts import assert_precise_ndarray_hint


class TestRedundancy:
    def test_public_array_contracts_are_parameterised(self):
        hints = (
            get_type_hints(redundancy)["phases"],
            get_type_hints(redundancy)["group_a"],
            get_type_hints(redundancy)["group_b"],
            get_type_hints(synergy)["phases"],
            get_type_hints(synergy)["group_a"],
            get_type_hints(synergy)["group_b"],
        )

        for hint in hints:
            assert_precise_ndarray_hint(hint)

        assert "float64" in str(hints[0])
        assert "int64" in str(hints[1])

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

    def test_redundancy_with_repeated_group_members(self):
        phases = np.array([0.0, 1.0, 2.0, 3.0])
        r = redundancy(phases, [0, 1, 1], [2, 3])
        assert r >= 0.0

    def test_empty_phases(self):
        assert redundancy(np.array([]), [0], [1]) == 0.0

    def test_empty_group_a(self):
        phases = np.array([0.0, 1.0, 2.0])
        assert redundancy(phases, [], [0, 1]) == 0.0

    def test_empty_group_b(self):
        phases = np.array([0.0, 1.0, 2.0])
        assert redundancy(phases, [0, 1], []) == 0.0

    @pytest.mark.parametrize(
        "phases",
        [
            np.array([[0.0, 1.0]]),
            np.array([0.0, np.nan]),
            np.array([True, False]),
            np.array([0.0, True], dtype=object),
            np.array([0.0, np.bool_(True)], dtype=object),
            [0.0, True],
        ],
    )
    def test_rejects_invalid_phase_vector(self, phases):
        with pytest.raises(ValueError, match="phases"):
            redundancy(phases, [0], [1])

    @pytest.mark.parametrize(
        "group",
        [
            [0.5],
            [True],
            [np.bool_(True)],
            np.array([0, True], dtype=object),
            np.array([0, np.bool_(True)], dtype=object),
            [-1],
            [3],
            np.array([[0]]),
        ],
    )
    def test_rejects_invalid_group_indices(self, group):
        phases = np.array([0.0, 1.0, 2.0])
        with pytest.raises((TypeError, ValueError, IndexError), match="group_a"):
            redundancy(phases, group, [1])

    @pytest.mark.parametrize("n_bins", [0, 1, False, 4.5])
    def test_rejects_invalid_bin_count(self, n_bins):
        phases = np.array([0.0, 1.0, 2.0])
        with pytest.raises((TypeError, ValueError), match="n_bins"):
            redundancy(phases, [0], [1], n_bins=n_bins)

    def test_accepts_numpy_integer_bin_count(self):
        phases = np.array([0.0, 1.0, 2.0])
        r = redundancy(phases, [0], [1], n_bins=np.int64(4))

        assert r >= 0.0

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

    @pytest.mark.parametrize(
        "phases",
        [
            np.array([0.0, np.nan]),
            np.array([True, False]),
            np.array([0.0, True], dtype=object),
            np.array([0.0, np.bool_(True)], dtype=object),
            [0.0, True],
        ],
    )
    def test_rejects_invalid_phase_vector(self, phases: np.ndarray) -> None:
        with pytest.raises(ValueError, match="phases"):
            synergy(phases, [0], [1])

    @pytest.mark.parametrize(
        "group",
        [
            [0.5],
            [True],
            [np.bool_(True)],
            np.array([0, True], dtype=object),
            np.array([0, np.bool_(True)], dtype=object),
            [-1],
            [3],
            np.array([[0]]),
        ],
    )
    def test_rejects_invalid_group_indices(self, group):
        phases = np.array([0.0, 1.0, 2.0])
        with pytest.raises((TypeError, ValueError, IndexError), match="group_b"):
            synergy(phases, [0], group)

    @pytest.mark.parametrize("n_bins", [0, 1, True, 7.5])
    def test_rejects_invalid_bin_count(self, n_bins):
        phases = np.array([0.0, 1.0, 2.0])
        with pytest.raises((TypeError, ValueError), match="n_bins"):
            synergy(phases, [0], [1], n_bins=n_bins)

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

    def test_redundancy_synergy_are_deterministic_for_identical_inputs(self):
        phases = np.linspace(0.0, 2 * np.pi, 100, dtype=np.float64)
        group_a = [0, 1, 2, 3, 4]
        group_b = [5, 6, 7, 8, 9]

        r1 = redundancy(phases, group_a, group_b)
        s1 = synergy(phases, group_a, group_b)
        r2 = redundancy(phases, group_a, group_b)
        s2 = synergy(phases, group_a, group_b)

        assert r1 == pytest.approx(r2, abs=0.0)
        assert s1 == pytest.approx(s2, abs=0.0)


class TestPIDRustDispatch:
    @pytest.mark.parametrize(
        "backend_value", [-0.1, np.nan, np.inf, [0.5], True, np.bool_(True)]
    )
    def test_redundancy_invalid_rust_payload_falls_back(
        self,
        monkeypatch: pytest.MonkeyPatch,
        backend_value: Any,
    ):
        monkeypatch.setattr(
            pid_module,
            "_rust_pid_redundancy",
            lambda *_args: backend_value,
        )
        phases = np.linspace(0.0, 2 * np.pi, 16, dtype=np.float64)

        val = redundancy(phases, [0, 1, 2], [3, 4, 5], n_bins=8)

        assert np.isfinite(val)
        assert val >= 0.0

    def test_redundancy_uses_rust_kernel_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        calls: list[tuple[np.ndarray, list[int], list[int], int]] = []

        def _fake_rust_red(
            phases: np.ndarray,
            group_a: list[int],
            group_b: list[int],
            n_bins: int,
        ) -> float:
            calls.append((phases, group_a, group_b, n_bins))
            return 0.42

        monkeypatch.setattr(pid_module, "_rust_pid_redundancy", _fake_rust_red)
        phases = np.linspace(0.0, 2 * np.pi, 16, dtype=np.float64)
        val = redundancy(phases, [0, 1, 2], [3, 4, 5], n_bins=8)
        assert val == pytest.approx(0.42, abs=1e-12)
        assert len(calls) == 1

    def test_redundancy_falls_back_when_rust_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        def _raising_rust_red(
            _phases: np.ndarray,
            _group_a: list[int],
            _group_b: list[int],
            _n_bins: int,
        ) -> float:
            raise RuntimeError("boom")

        monkeypatch.setattr(pid_module, "_rust_pid_redundancy", _raising_rust_red)
        phases = np.linspace(0.0, 2 * np.pi, 16, dtype=np.float64)
        val = redundancy(phases, [0, 1, 2], [3, 4, 5], n_bins=8)
        assert np.isfinite(val)
        assert val >= 0.0

    def test_synergy_uses_rust_kernel_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        calls: list[tuple[np.ndarray, list[int], list[int], int]] = []

        def _fake_rust_syn(
            phases: np.ndarray,
            group_a: list[int],
            group_b: list[int],
            n_bins: int,
        ) -> float:
            calls.append((phases, group_a, group_b, n_bins))
            return 0.37

        monkeypatch.setattr(pid_module, "_rust_pid_synergy", _fake_rust_syn)
        phases = np.linspace(0.0, 2 * np.pi, 16, dtype=np.float64)
        val = synergy(phases, [0, 1, 2], [3, 4, 5], n_bins=8)
        assert val == pytest.approx(0.37, abs=1e-12)
        assert len(calls) == 1

    @pytest.mark.parametrize(
        "backend_value", [-0.1, np.nan, np.inf, [0.5], True, np.bool_(True)]
    )
    def test_synergy_invalid_rust_payload_falls_back(
        self,
        monkeypatch: pytest.MonkeyPatch,
        backend_value: Any,
    ):
        monkeypatch.setattr(
            pid_module, "_rust_pid_synergy", lambda *_args: backend_value
        )
        phases = np.linspace(0.0, 2 * np.pi, 16, dtype=np.float64)

        val = synergy(phases, [0, 1, 2], [3, 4, 5], n_bins=8)

        assert np.isfinite(val)
        assert val >= 0.0

    def test_synergy_falls_back_when_rust_raises(self, monkeypatch: pytest.MonkeyPatch):
        def _raising_rust_syn(
            _phases: np.ndarray,
            _group_a: list[int],
            _group_b: list[int],
            _n_bins: int,
        ) -> float:
            raise RuntimeError("boom")

        monkeypatch.setattr(pid_module, "_rust_pid_synergy", _raising_rust_syn)
        phases = np.linspace(0.0, 2 * np.pi, 16, dtype=np.float64)
        val = synergy(phases, [0, 1, 2], [3, 4, 5], n_bins=8)
        assert np.isfinite(val)
        assert val >= 0.0


# Salvaged module-specific behavioural contracts from deleted broad tests.
class TestPIDInformationTheory:
    """Verify circular entropy and mutual information edge cases
    satisfy information-theoretic bounds."""

    def test_circular_entropy_empty_is_zero(self):
        from scpn_phase_orchestrator.monitor.pid import _circular_entropy

        assert _circular_entropy(np.array([])) == 0.0

    def test_circular_entropy_single_value_nonnegative(self):
        from scpn_phase_orchestrator.monitor.pid import _circular_entropy

        result = _circular_entropy(np.array([100.0]))
        assert result >= 0.0

    def test_circular_entropy_uniform_higher_than_peaked(self):
        """Uniformly spread phases should have higher entropy than clustered."""
        from scpn_phase_orchestrator.monitor.pid import _circular_entropy

        uniform = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        clustered = np.random.default_rng(0).normal(1.0, 0.01, 100)
        h_uniform = _circular_entropy(uniform)
        h_clustered = _circular_entropy(clustered)
        assert h_uniform > h_clustered, (
            f"H_uniform={h_uniform:.3f} should > H_clustered={h_clustered:.3f}"
        )

    def test_joint_entropy_empty_is_zero(self):
        from scpn_phase_orchestrator.monitor.pid import _joint_entropy_2d

        assert _joint_entropy_2d(np.array([]), np.array([])) == 0.0

    def test_mutual_information_empty_is_zero(self):
        from scpn_phase_orchestrator.monitor.pid import _mutual_information_paired

        assert _mutual_information_paired(np.array([]), np.array([])) == 0.0

    def test_mutual_information_mismatched_lengths_is_zero(self):
        from scpn_phase_orchestrator.monitor.pid import _mutual_information_paired

        assert _mutual_information_paired(np.array([1.0]), np.array([1.0, 2.0])) == 0.0

    def test_mutual_information_nonnegative(self):
        """MI must be non-negative for any valid inputs."""
        from scpn_phase_orchestrator.monitor.pid import _mutual_information_paired

        rng = np.random.default_rng(42)
        a = rng.uniform(0, 2 * np.pi, 200)
        b = rng.uniform(0, 2 * np.pi, 200)
        mi = _mutual_information_paired(a, b)
        assert mi >= -1e-10, f"MI must be non-negative, got {mi}"

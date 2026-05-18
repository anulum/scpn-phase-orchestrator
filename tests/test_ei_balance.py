# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — EI balance tests

from __future__ import annotations

from typing import get_type_hints

import numpy as np
import pytest

from scpn_phase_orchestrator.coupling import ei_balance as ei_balance_module
from scpn_phase_orchestrator.coupling.ei_balance import (
    EIBalance,
    adjust_ei_ratio,
    compute_ei_balance,
)


def test_public_array_contracts_are_parameterised() -> None:
    """Public E/I balance array contracts stay element-typed."""
    for hint in [
        get_type_hints(compute_ei_balance)["knm"],
        get_type_hints(adjust_ei_ratio)["knm"],
        get_type_hints(adjust_ei_ratio)["return"],
    ]:
        assert "numpy.ndarray" in str(hint)
        assert "float64" in str(hint)


def _uniform_knm(n: int, k: float = 1.0) -> np.ndarray:
    knm = np.full((n, n), k)
    np.fill_diagonal(knm, 0.0)
    return knm


class TestComputeEIBalance:
    def test_equal_groups_balanced(self):
        knm = _uniform_knm(6)
        bal = compute_ei_balance(knm, [0, 1, 2], [3, 4, 5])
        assert abs(bal.ratio - 1.0) < 1e-10
        assert bal.is_balanced

    def test_stronger_excitatory(self):
        knm = _uniform_knm(4)
        knm[0, :] *= 2.0
        knm[1, :] *= 2.0
        bal = compute_ei_balance(knm, [0, 1], [2, 3])
        assert bal.ratio > 1.0
        assert not bal.is_balanced

    def test_no_inhibitory(self):
        knm = _uniform_knm(4)
        bal = compute_ei_balance(knm, [0, 1, 2, 3], [])
        assert bal.inhibitory_strength == 0.0

    def test_no_excitatory(self):
        knm = _uniform_knm(4)
        bal = compute_ei_balance(knm, [], [0, 1, 2, 3])
        assert bal.excitatory_strength == 0.0

    def test_out_of_range_indices(self):
        knm = _uniform_knm(4)
        bal = compute_ei_balance(knm, [0, 1, 99], [2, 3])
        assert bal.excitatory_strength > 0
        expected = compute_ei_balance(knm, [0, 1], [2, 3])
        assert bal == expected

    def test_negative_indices_are_rejected(self):
        knm = _uniform_knm(4)
        with pytest.raises(ValueError, match="indices"):
            compute_ei_balance(knm, [-1], [2, 3])

    def test_returns_dataclass(self):
        knm = _uniform_knm(4)
        bal = compute_ei_balance(knm, [0, 1], [2, 3])
        assert isinstance(bal, EIBalance)
        assert hasattr(bal, "ratio")
        assert hasattr(bal, "is_balanced")


class TestAdjustEIRatio:
    def test_already_balanced(self):
        knm = _uniform_knm(4)
        result = adjust_ei_ratio(knm, [0, 1], [2, 3], target_ratio=1.0)
        np.testing.assert_array_almost_equal(result, knm)

    def test_scales_inhibitory(self):
        knm = _uniform_knm(4)
        knm[0, :] *= 2.0
        knm[1, :] *= 2.0
        result = adjust_ei_ratio(knm, [0, 1], [2, 3], target_ratio=1.0)
        bal = compute_ei_balance(result, [0, 1], [2, 3])
        assert abs(bal.ratio - 1.0) < 0.1

    def test_no_inhibitory_returns_copy(self):
        knm = _uniform_knm(4)
        result = adjust_ei_ratio(knm, [0, 1, 2, 3], [], target_ratio=1.0)
        np.testing.assert_array_equal(result, knm)
        assert result is not knm

    def test_preserves_diagonal_zero(self):
        knm = _uniform_knm(4)
        knm[0, :] *= 3.0
        result = adjust_ei_ratio(knm, [0, 1], [2, 3], target_ratio=1.0)
        np.testing.assert_array_equal(np.diag(result), np.zeros(4))

    def test_target_ratio_above_one(self):
        knm = _uniform_knm(6)
        result = adjust_ei_ratio(knm, [0, 1, 2], [3, 4, 5], target_ratio=2.0)
        bal = compute_ei_balance(result, [0, 1, 2], [3, 4, 5])
        assert abs(bal.ratio - 2.0) < 0.1

    def test_negative_indices_are_rejected(self):
        knm = _uniform_knm(4)
        with pytest.raises(ValueError, match="indices"):
            adjust_ei_ratio(knm, [0, 1], [-1], target_ratio=1.0)

    def test_optional_rust_paths_preserve_contract(self, monkeypatch):
        calls = []

        def fake_rust_ei(flat_knm, n, excitatory, inhibitory):
            calls.append(
                ("ei", flat_knm.copy(), n, excitatory.copy(), inhibitory.copy())
            )
            return 2.0, 4.0, 2.0, False

        def fake_rust_adjust(flat_knm, n, excitatory, inhibitory, target_ratio):
            calls.append(
                (
                    "adjust",
                    flat_knm.copy(),
                    n,
                    excitatory.copy(),
                    inhibitory.copy(),
                    target_ratio,
                )
            )
            return flat_knm * 3.0

        monkeypatch.setattr(ei_balance_module, "_HAS_RUST", True)
        monkeypatch.setattr(
            ei_balance_module, "_rust_ei", fake_rust_ei, raising=False
        )
        monkeypatch.setattr(
            ei_balance_module, "_rust_adjust", fake_rust_adjust, raising=False
        )

        knm = np.array([[0.0, 0.5], [1.5, 0.0]], dtype=np.float64)
        balance = compute_ei_balance(knm, [0], [1])
        adjusted = adjust_ei_ratio(knm, [0], [1], target_ratio=1.25)

        assert balance == EIBalance(
            ratio=2.0,
            excitatory_strength=4.0,
            inhibitory_strength=2.0,
            is_balanced=False,
        )
        np.testing.assert_array_equal(adjusted, knm * 3.0)
        assert calls[0][0] == "ei"
        np.testing.assert_array_equal(calls[0][1], np.array([0.0, 0.5, 1.5, 0.0]))
        assert calls[0][2] == 2
        np.testing.assert_array_equal(calls[0][3], np.array([0]))
        np.testing.assert_array_equal(calls[0][4], np.array([1]))
        assert calls[1][0] == "adjust"
        np.testing.assert_array_equal(calls[1][1], np.array([0.0, 0.5, 1.5, 0.0]))
        assert calls[1][2] == 2
        np.testing.assert_array_equal(calls[1][3], np.array([0]))
        np.testing.assert_array_equal(calls[1][4], np.array([1]))
        assert calls[1][5] == 1.25


class TestEIBalancePipelineWiring:
    """Pipeline: adjust_ei_ratio → balanced K_nm → engine."""

    def test_ei_balanced_knm_drives_engine(self):
        """adjust_ei_ratio → balanced K_nm → UPDEEngine → R∈[0,1].
        Proves EI balance feeds valid coupling into simulation."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import (
            compute_order_parameter,
        )

        n = 6
        knm = _uniform_knm(n)
        knm[:3, :] *= 2.0  # excitatory stronger
        balanced = adjust_ei_ratio(
            knm,
            [0, 1, 2],
            [3, 4, 5],
            target_ratio=1.0,
        )
        bal = compute_ei_balance(balanced, [0, 1, 2], [3, 4, 5])
        assert abs(bal.ratio - 1.0) < 0.2

        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        alpha = np.zeros((n, n))
        for _ in range(100):
            phases = eng.step(
                phases,
                omegas,
                balanced,
                0.0,
                0.0,
                alpha,
            )
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0

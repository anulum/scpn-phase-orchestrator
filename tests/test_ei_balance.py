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
from tests.typing_contracts import assert_precise_ndarray_hint


def test_public_array_contracts_are_parameterised() -> None:
    """Public E/I balance array contracts stay element-typed."""
    for hint in [
        get_type_hints(compute_ei_balance)["knm"],
        get_type_hints(adjust_ei_ratio)["knm"],
        get_type_hints(adjust_ei_ratio)["return"],
    ]:
        assert_precise_ndarray_hint(hint)
        assert "float64" in str(hint)


def _uniform_knm(n: int, k: float = 1.0) -> np.ndarray:
    knm = np.full((n, n), k)
    np.fill_diagonal(knm, 0.0)
    return knm


class TestComputeEIBalance:
    def test_exact_row_mean_ratio_contract(self):
        """E/I balance is the ratio of mean outgoing typed-row strengths."""
        knm = np.array(
            [
                [0.0, 2.0, 4.0, 6.0],
                [8.0, 0.0, 10.0, 12.0],
                [1.0, 3.0, 0.0, 5.0],
                [7.0, 9.0, 11.0, 0.0],
            ],
            dtype=np.float64,
        )
        bal = compute_ei_balance(knm, [0, 1], [2, 3])
        expected_e = float(np.mean(knm[[0, 1], :]))
        expected_i = float(np.mean(knm[[2, 3], :]))

        assert bal.excitatory_strength == pytest.approx(expected_e)
        assert bal.inhibitory_strength == pytest.approx(expected_i)
        assert bal.ratio == pytest.approx(expected_e / expected_i)
        assert bal.is_balanced is True

    def test_interaction_type_breakdown_matches_blocks(self):
        """The four directed interaction-type means (Kuroki & Mizuseki
        2025) equal the corresponding source→target sub-block means."""
        knm = np.array(
            [
                [0.0, 2.0, 4.0, 6.0],
                [8.0, 0.0, 10.0, 12.0],
                [1.0, 3.0, 0.0, 5.0],
                [7.0, 9.0, 11.0, 0.0],
            ],
            dtype=np.float64,
        )
        e, i = [0, 1], [2, 3]
        bal = compute_ei_balance(knm, e, i)
        assert bal.e_to_e == pytest.approx(float(np.mean(knm[np.ix_(e, e)])))
        assert bal.e_to_i == pytest.approx(float(np.mean(knm[np.ix_(e, i)])))
        assert bal.i_to_e == pytest.approx(float(np.mean(knm[np.ix_(i, e)])))
        assert bal.i_to_i == pytest.approx(float(np.mean(knm[np.ix_(i, i)])))

    def test_aggregate_strength_is_block_blend(self):
        """Each aggregate strength is the count-weighted blend of its two
        outgoing interaction-type blocks (equal group sizes → mean)."""
        knm = _uniform_knm(6)
        bal = compute_ei_balance(knm, [0, 1, 2], [3, 4, 5])
        assert bal.excitatory_strength == pytest.approx(
            0.5 * (bal.e_to_e + bal.e_to_i)
        )
        assert bal.inhibitory_strength == pytest.approx(
            0.5 * (bal.i_to_e + bal.i_to_i)
        )

    def test_empty_groups_zero_interaction_types(self):
        knm = _uniform_knm(4)
        bal = compute_ei_balance(knm, [0, 1], [])
        assert bal.e_to_i == 0.0
        assert bal.i_to_e == 0.0
        assert bal.i_to_i == 0.0
        assert bal.e_to_e == pytest.approx(float(np.mean(knm[np.ix_([0, 1], [0, 1])])))

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

    def test_boolean_coupling_alias_is_rejected(self):
        with pytest.raises(ValueError, match="knm must not contain boolean"):
            compute_ei_balance([[0.0, True], [1.0, 0.0]], [0], [1])

    @pytest.mark.parametrize("indices", [[True], [np.bool_(True)]])
    def test_boolean_indices_are_rejected(self, indices):
        knm = _uniform_knm(2)
        with pytest.raises(ValueError, match="excitatory indices"):
            compute_ei_balance(knm, indices, [1])

    def test_non_finite_coupling_is_rejected(self):
        with pytest.raises(ValueError, match="knm must contain only finite"):
            compute_ei_balance([[0.0, np.nan], [1.0, 0.0]], [0], [1])

    def test_non_square_coupling_is_rejected(self):
        with pytest.raises(ValueError, match="finite square matrix"):
            compute_ei_balance([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0]], [0], [1])

    def test_returns_dataclass(self):
        knm = _uniform_knm(4)
        bal = compute_ei_balance(knm, [0, 1], [2, 3])
        assert isinstance(bal, EIBalance)
        assert bal == EIBalance(
            ratio=1.0,
            excitatory_strength=0.75,
            inhibitory_strength=0.75,
            is_balanced=True,
            e_to_e=0.5,
            e_to_i=1.0,
            i_to_e=1.0,
            i_to_i=0.5,
        )


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
        assert bal.ratio == pytest.approx(1.0)

    def test_adjustment_scales_only_inhibitory_rows_to_target(self):
        knm = np.array(
            [
                [0.0, 2.0, 4.0, 6.0],
                [8.0, 0.0, 10.0, 12.0],
                [1.0, 3.0, 0.0, 5.0],
                [7.0, 9.0, 11.0, 0.0],
            ],
            dtype=np.float64,
        )
        before = compute_ei_balance(knm, [0, 1], [2, 3])
        target_ratio = 1.5
        expected_scale = before.ratio / target_ratio

        result = adjust_ei_ratio(knm, [0, 1], [2, 3], target_ratio=target_ratio)
        after = compute_ei_balance(result, [0, 1], [2, 3])

        np.testing.assert_allclose(result[[0, 1], :], knm[[0, 1], :])
        np.testing.assert_allclose(result[[2, 3], :], knm[[2, 3], :] * expected_scale)
        np.testing.assert_allclose(knm[2, :], np.array([1.0, 3.0, 0.0, 5.0]))
        assert after.ratio == pytest.approx(target_ratio)

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

    def test_boolean_indices_are_rejected(self):
        knm = _uniform_knm(2)
        with pytest.raises(ValueError, match="inhibitory indices"):
            adjust_ei_ratio(knm, [0], [True], target_ratio=1.0)

    @pytest.mark.parametrize("target_ratio", [0.0, -1.0, np.nan, True])
    def test_invalid_target_ratio_is_rejected(self, target_ratio):
        knm = _uniform_knm(4)
        with pytest.raises((TypeError, ValueError), match="target_ratio"):
            adjust_ei_ratio(knm, [0, 1], [2, 3], target_ratio=target_ratio)

    def test_optional_rust_paths_preserve_contract(self, monkeypatch):
        calls = []

        def fake_rust_ei(flat_knm, n, excitatory, inhibitory):
            calls.append(
                ("ei", flat_knm.copy(), n, excitatory.copy(), inhibitory.copy())
            )
            return 2.0, 4.0, 2.0, False, 5.0, 6.0, 7.0, 8.0

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
        monkeypatch.setattr(ei_balance_module, "_rust_ei", fake_rust_ei, raising=False)
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
            e_to_e=5.0,
            e_to_i=6.0,
            i_to_e=7.0,
            i_to_i=8.0,
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
        phases = np.linspace(-0.7, 0.7, n) % (2 * np.pi)
        initial_r, _ = compute_order_parameter(phases)
        omegas = np.zeros(n)
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
        assert r > initial_r
        assert 0.0 <= r <= 1.0

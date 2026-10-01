# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — EI balance tests

"""Exercise E/I numerical admission, real runtimes and UPDE consumption."""

from __future__ import annotations

from typing import cast, get_type_hints

import numpy as np
import pytest
from numpy.typing import NDArray

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


def _uniform_knm(n: int, k: float = 1.0) -> NDArray[np.float64]:
    """Construct uniform finite coupling with a zero diagonal."""
    knm = np.full((n, n), k)
    np.fill_diagonal(knm, 0.0)
    return knm


class TestComputeEIBalance:
    """Verify signed source summaries, directed blocks and input admission."""

    def test_exact_row_mean_ratio_contract(self) -> None:
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

    def test_interaction_type_breakdown_matches_blocks(self) -> None:
        """Match the four directed means to the actual source-to-target blocks."""
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

    def test_aggregate_strength_is_block_blend(self) -> None:
        """Blend directed block means when equal groups partition all targets."""
        knm = _uniform_knm(6)
        bal = compute_ei_balance(knm, [0, 1, 2], [3, 4, 5])
        assert bal.excitatory_strength == pytest.approx(0.5 * (bal.e_to_e + bal.e_to_i))
        assert bal.inhibitory_strength == pytest.approx(0.5 * (bal.i_to_e + bal.i_to_i))

    def test_empty_groups_zero_interaction_types(self) -> None:
        """Empty groups zero interaction types."""
        knm = _uniform_knm(4)
        bal = compute_ei_balance(knm, [0, 1], [])
        assert bal.e_to_i == 0.0
        assert bal.i_to_e == 0.0
        assert bal.i_to_i == 0.0
        assert bal.e_to_e == pytest.approx(float(np.mean(knm[np.ix_([0, 1], [0, 1])])))

    def test_equal_groups_balanced(self) -> None:
        """Equal groups balanced."""
        knm = _uniform_knm(6)
        bal = compute_ei_balance(knm, [0, 1, 2], [3, 4, 5])
        assert abs(bal.ratio - 1.0) < 1e-10
        assert bal.is_balanced

    def test_stronger_excitatory(self) -> None:
        """Stronger excitatory."""
        knm = _uniform_knm(4)
        knm[0, :] *= 2.0
        knm[1, :] *= 2.0
        bal = compute_ei_balance(knm, [0, 1], [2, 3])
        assert bal.ratio > 1.0
        assert not bal.is_balanced

    def test_no_inhibitory(self) -> None:
        """No inhibitory."""
        knm = _uniform_knm(4)
        bal = compute_ei_balance(knm, [0, 1, 2, 3], [])
        assert bal.inhibitory_strength == 0.0

    def test_no_excitatory(self) -> None:
        """No excitatory."""
        knm = _uniform_knm(4)
        bal = compute_ei_balance(knm, [], [0, 1, 2, 3])
        assert bal.excitatory_strength == 0.0

    def test_out_of_range_indices(self) -> None:
        """Out of range indices."""
        knm = _uniform_knm(4)
        bal = compute_ei_balance(knm, [0, 1, 99], [2, 3])
        assert bal.excitatory_strength > 0
        expected = compute_ei_balance(knm, [0, 1], [2, 3])
        assert bal == expected

    def test_negative_indices_are_rejected(self) -> None:
        """Negative indices are rejected."""
        knm = _uniform_knm(4)
        with pytest.raises(ValueError, match="indices"):
            compute_ei_balance(knm, [-1], [2, 3])

    def test_boolean_coupling_alias_is_rejected(self) -> None:
        """Boolean coupling alias is rejected."""
        with pytest.raises(ValueError, match="knm must not contain boolean"):
            compute_ei_balance(
                cast(NDArray[np.float64], [[0.0, True], [1.0, 0.0]]), [0], [1]
            )

    def test_numpy_boolean_coupling_alias_is_rejected(self) -> None:
        """Numpy boolean coupling alias is rejected."""
        knm = np.array([[0.0, np.bool_(True)], [1.0, 0.0]], dtype=object)
        with pytest.raises(ValueError, match="knm must not contain boolean"):
            compute_ei_balance(knm, [0], [1])

    @pytest.mark.parametrize("indices", [[True], [np.bool_(True)]])
    def test_boolean_indices_are_rejected(self, indices: list[int]) -> None:
        """Boolean indices are rejected."""
        knm = _uniform_knm(2)
        with pytest.raises(ValueError, match="excitatory indices"):
            compute_ei_balance(knm, indices, [1])

    def test_non_finite_coupling_is_rejected(self) -> None:
        """Non finite coupling is rejected."""
        with pytest.raises(ValueError, match="knm must contain only finite"):
            compute_ei_balance(np.array([[0.0, np.nan], [1.0, 0.0]]), [0], [1])

    def test_non_square_coupling_is_rejected(self) -> None:
        """Non square coupling is rejected."""
        with pytest.raises(ValueError, match="finite square matrix"):
            compute_ei_balance(np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0]]), [0], [1])

    def test_returns_dataclass(self) -> None:
        """Returns dataclass."""
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
    """Verify inhibitory rescaling, preserved source bytes and invalid targets."""

    def test_already_balanced(self) -> None:
        """Already balanced."""
        knm = _uniform_knm(4)
        result = adjust_ei_ratio(knm, [0, 1], [2, 3], target_ratio=1.0)
        np.testing.assert_array_almost_equal(result, knm)

    def test_scales_inhibitory(self) -> None:
        """Scales inhibitory."""
        knm = _uniform_knm(4)
        knm[0, :] *= 2.0
        knm[1, :] *= 2.0
        result = adjust_ei_ratio(knm, [0, 1], [2, 3], target_ratio=1.0)
        bal = compute_ei_balance(result, [0, 1], [2, 3])
        assert bal.ratio == pytest.approx(1.0)

    def test_adjustment_scales_only_inhibitory_rows_to_target(self) -> None:
        """Adjustment scales only inhibitory rows to target."""
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

    def test_no_inhibitory_returns_copy(self) -> None:
        """No inhibitory returns copy."""
        knm = _uniform_knm(4)
        result = adjust_ei_ratio(knm, [0, 1, 2, 3], [], target_ratio=1.0)
        np.testing.assert_array_equal(result, knm)
        assert result is not knm

    def test_preserves_diagonal_zero(self) -> None:
        """Preserves diagonal zero."""
        knm = _uniform_knm(4)
        knm[0, :] *= 3.0
        result = adjust_ei_ratio(knm, [0, 1], [2, 3], target_ratio=1.0)
        np.testing.assert_array_equal(np.diag(result), np.zeros(4))

    def test_target_ratio_above_one(self) -> None:
        """Target ratio above one."""
        knm = _uniform_knm(6)
        result = adjust_ei_ratio(knm, [0, 1, 2], [3, 4, 5], target_ratio=2.0)
        bal = compute_ei_balance(result, [0, 1, 2], [3, 4, 5])
        assert abs(bal.ratio - 2.0) < 0.1

    def test_negative_indices_are_rejected(self) -> None:
        """Negative indices are rejected."""
        knm = _uniform_knm(4)
        with pytest.raises(ValueError, match="indices"):
            adjust_ei_ratio(knm, [0, 1], [-1], target_ratio=1.0)

    def test_boolean_indices_are_rejected(self) -> None:
        """Boolean indices are rejected."""
        knm = _uniform_knm(2)
        with pytest.raises(ValueError, match="inhibitory indices"):
            adjust_ei_ratio(knm, [0], [True], target_ratio=1.0)

    def test_numpy_boolean_coupling_alias_is_rejected(self) -> None:
        """Numpy boolean coupling alias is rejected."""
        knm = np.array([[0.0, np.bool_(True)], [1.0, 0.0]], dtype=object)
        with pytest.raises(ValueError, match="knm must not contain boolean"):
            adjust_ei_ratio(knm, [0], [1], target_ratio=1.0)

    @pytest.mark.parametrize("target_ratio", [0.0, -1.0, np.nan, True])
    def test_invalid_target_ratio_is_rejected(self, target_ratio: float) -> None:
        """Invalid target ratio is rejected."""
        knm = _uniform_knm(4)
        with pytest.raises((TypeError, ValueError), match="target_ratio"):
            adjust_ei_ratio(knm, [0, 1], [2, 3], target_ratio=target_ratio)

    def test_optional_rust_paths_preserve_contract(self) -> None:
        """Keep all directed means and scaled entries tied to actual input."""
        knm = np.array([[0.0, 0.5], [1.5, 0.0]], dtype=np.float64)
        balance = compute_ei_balance(knm, [0], [1])
        adjusted = adjust_ei_ratio(knm, [0], [1], target_ratio=1.25)
        assert balance == EIBalance(1 / 3, 0.25, 0.75, False, 0.0, 0.5, 1.5, 0.0)
        np.testing.assert_allclose(adjusted, [[0.0, 0.5], [0.4, 0.0]])
        np.testing.assert_array_equal(knm, [[0.0, 0.5], [1.5, 0.0]])
        assert compute_ei_balance(adjusted, [0], [1]).ratio == pytest.approx(1.25)


class TestEIBalancePipelineWiring:
    """Pipeline: adjust_ei_ratio → balanced K_nm → engine."""

    def test_ei_balanced_knm_drives_engine(self) -> None:
        """Increase coherence using adjusted coupling in the real UPDE engine."""
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


def test_validate_knm_rejects_non_coercible_matrix() -> None:
    """A matrix whose entries cannot become float64 fails closed."""
    knm = np.array([["a", "b"], ["c", "d"]], dtype=object)
    with pytest.raises(ValueError, match="finite square matrix"):
        compute_ei_balance(knm, [0], [1])


class TestRuntimeEdgeContracts:
    """Exercise the same edge contracts in native and kernel-absent installs."""

    def test_compute_matches_block_mean_contract(self) -> None:
        """Compute matches block mean contract."""
        knm = _uniform_knm(4, k=2.0)

        bal = compute_ei_balance(knm, [0, 1], [2, 3])

        assert bal.ratio == pytest.approx(1.0)
        assert bal.is_balanced is True
        # within-group blocks include the zero diagonal -> mean 1.0
        assert bal.e_to_e == pytest.approx(1.0)
        assert bal.i_to_i == pytest.approx(1.0)
        # cross-group blocks have no diagonal -> mean 2.0
        assert bal.e_to_i == pytest.approx(2.0)
        assert bal.i_to_e == pytest.approx(2.0)

    def test_compute_reports_infinite_ratio_when_only_inhibition_is_silent(
        self,
    ) -> None:
        """Compute reports infinite ratio when only inhibition is silent."""
        knm = np.zeros((4, 4), dtype=np.float64)
        knm[0, :] = 1.0
        knm[1, :] = 1.0

        bal = compute_ei_balance(knm, [0, 1], [2, 3])

        assert bal.ratio == float("inf")
        assert bal.is_balanced is False

    def test_compute_with_empty_groups_is_neutral(self) -> None:
        """Compute with empty groups is neutral."""
        knm = _uniform_knm(3, k=1.0)

        bal = compute_ei_balance(knm, [], [])

        assert bal.ratio == pytest.approx(1.0)
        assert bal.excitatory_strength == pytest.approx(0.0)
        assert bal.inhibitory_strength == pytest.approx(0.0)
        assert bal.e_to_e == pytest.approx(0.0)
        assert bal.i_to_i == pytest.approx(0.0)

    def test_adjust_scales_inhibitory_rows_toward_target(self) -> None:
        """Adjust scales inhibitory rows toward target."""
        knm = _uniform_knm(4, k=2.0)

        adjusted = adjust_ei_ratio(knm, [0, 1], [2, 3], target_ratio=2.0)

        # current ratio 1.0, target 2.0 -> inhibitory rows scaled by 0.5
        np.testing.assert_allclose(adjusted[2], knm[2] * 0.5)
        np.testing.assert_allclose(adjusted[3], knm[3] * 0.5)
        np.testing.assert_allclose(adjusted[0], knm[0])

    def test_adjust_returns_copy_when_inhibition_is_silent(self) -> None:
        """Adjust returns copy when inhibition is silent."""
        knm = np.zeros((4, 4), dtype=np.float64)
        knm[0, :] = 1.0
        knm[1, :] = 1.0

        adjusted = adjust_ei_ratio(knm, [0, 1], [2, 3], target_ratio=1.0)

        np.testing.assert_array_equal(adjusted, knm)
        assert adjusted is not knm

    def test_adjust_returns_copy_when_already_at_target(self) -> None:
        """Adjust returns copy when already at target."""
        knm = _uniform_knm(4, k=2.0)

        adjusted = adjust_ei_ratio(knm, [0, 1], [2, 3], target_ratio=1.0)

        np.testing.assert_array_equal(adjusted, knm)
        assert adjusted is not knm


def test_ei_balance_public_facade_runs_in_actual_installation() -> None:
    """Resolve lazy public exports and exercise summary plus adjustment."""
    from scpn_phase_orchestrator.coupling import (
        adjust_ei_ratio as public_adjust,
    )
    from scpn_phase_orchestrator.coupling import (
        compute_ei_balance as public_compute,
    )

    knm = np.array([[0.0, 2.0], [1.0, 0.0]])
    assert public_compute(knm, [0], [1]).ratio == 2.0
    np.testing.assert_array_equal(public_adjust(knm, [0], [1]), [[0, 2], [2, 0]])


@pytest.mark.parametrize("exc,inh", [([0, 0, 1], [2, 2]), ([1, 0, 1, 99], [2, 99])])
def test_index_groups_are_sets(exc: list[int], inh: list[int]) -> None:
    """Count each source once and apply each inhibitory adjustment once."""
    knm = np.array([[0.0, 2.0, 4.0], [6.0, 0.0, 8.0], [10.0, 12.0, 0.0]])
    before = knm.copy()
    balance = compute_ei_balance(knm, exc, inh)
    assert balance.excitatory_strength == pytest.approx(10 / 3)
    assert balance.inhibitory_strength == pytest.approx(22 / 3)
    assert balance.e_to_e == 2.0
    assert balance.e_to_i == 6.0
    assert balance.i_to_e == 11.0
    assert balance.i_to_i == 0.0
    adjusted = adjust_ei_ratio(knm, exc, inh)
    np.testing.assert_allclose(adjusted[2], [50 / 11, 60 / 11, 0])
    np.testing.assert_array_equal(adjusted[:2], knm[:2])
    assert compute_ei_balance(adjusted, exc, inh).ratio == pytest.approx(1.0)
    np.testing.assert_array_equal(knm, before)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_signed_source_means_preserve_quotient_and_adjustment(sign: float) -> None:
    """Treat inhibitory silence by magnitude and retain the signed quotient."""
    knm = np.array([[0.0, 2.0 * sign], [-1.0, 0.0]])
    before = knm.copy()
    assert compute_ei_balance(knm, [0], [1]).ratio == -2.0 * sign
    adjusted = adjust_ei_ratio(knm, [0], [1])
    np.testing.assert_array_equal(adjusted, [[0.0, 2.0 * sign], [2.0 * sign, 0.0]])
    assert compute_ei_balance(adjusted, [0], [1]).ratio == 1.0
    np.testing.assert_array_equal(knm, before)


@pytest.mark.parametrize("value", [0.0, 1e308, -1e308])
def test_finite_group_means_do_not_overflow(value: float) -> None:
    """Keep representable means finite even when their unscaled sum overflows."""
    knm = np.full((2, 2), value)
    b = compute_ei_balance(knm, [0], [1])
    assert b == EIBalance(1.0, value, value, True, value, value, value, value)
    adjusted = adjust_ei_ratio(knm, [0], [1])
    np.testing.assert_array_equal(adjusted, knm)
    assert not np.shares_memory(adjusted, knm)


def test_cancelling_large_group_has_finite_mean() -> None:
    """Retain zero means from signed, individually finite large couplings."""
    knm = np.array([[1e308, 1e308, -1e308, -1e308]] * 4)
    b = compute_ei_balance(knm, [0, 1], [2, 3])
    assert b.excitatory_strength == 0.0
    assert b.inhibitory_strength == 0.0
    assert b.ratio == 1.0
    assert b.e_to_e == 1e308
    assert b.e_to_i == -1e308


@pytest.mark.parametrize(
    "knm,target",
    [
        (np.array([[0.0, 2.0], [1.0, 0.0]]), 1e-310),
        (np.array([[0.0, 1e308], [1e308, -1e308 + 1e294]]), 1.0),
    ],
)
def test_adjustment_overflow_refuses_without_source_mutation(
    knm: NDArray[np.float64],
    target: float,
) -> None:
    """Refuse non-finite scale or elements while leaving the source intact."""
    before = knm.copy()
    with pytest.raises(ValueError, match="adjustment must remain finite"):
        adjust_ei_ratio(knm, [0], [1], target)
    np.testing.assert_array_equal(knm, before)
    recovered = adjust_ei_ratio(np.array([[0.0, 2.0], [1.0, 0.0]]), [0], [1])
    np.testing.assert_array_equal(recovered, [[0.0, 2.0], [2.0, 0.0]])


def test_empty_coupling_and_ignored_indices_are_neutral() -> None:
    """Keep the zero-oscillator summary and independent adjustment valid."""
    knm = np.empty((0, 0))
    b = compute_ei_balance(knm, [99], [99])
    assert b == EIBalance(1.0, 0.0, 0.0, True, 0.0, 0.0, 0.0, 0.0)
    adjusted = adjust_ei_ratio(knm, [], [])
    assert adjusted.shape == (0, 0)
    assert not np.shares_memory(adjusted, knm)


def test_overlapping_groups_use_declared_source_means() -> None:
    """Allow overlapping observation sets without asserting target attainment."""
    knm = np.array([[0.0, 2.0], [1.0, 0.0]])
    b = compute_ei_balance(knm, [0, 1], [1])
    assert b.excitatory_strength == 0.75
    assert b.inhibitory_strength == 0.5
    assert b.ratio == 1.5
    np.testing.assert_allclose(
        adjust_ei_ratio(knm, [0, 1], [1]), [[0.0, 2.0], [1.5, 0.0]]
    )


def test_strided_readonly_coupling_returns_independent_adjustment() -> None:
    """Accept real non-contiguous matrices and preserve readonly source bytes."""
    storage = np.arange(16.0, dtype=np.float64).reshape(4, 4)
    knm = storage[::2, ::2]
    knm.flags.writeable = False
    before = storage.copy()
    adjusted = adjust_ei_ratio(knm, [0], [1])
    np.testing.assert_allclose(adjusted, [[0.0, 2.0], [8 / 9, 10 / 9]])
    np.testing.assert_array_equal(storage, before)
    assert not np.shares_memory(adjusted, storage)


def test_silent_excitation_returns_independent_copy() -> None:
    """Leave positive inhibitory coupling unchanged when excitation is silent."""
    knm = np.array([[0.0, 0.0], [1.0, 0.0]])
    adjusted = adjust_ei_ratio(knm, [0], [1])
    np.testing.assert_array_equal(adjusted, knm)
    assert not np.shares_memory(adjusted, knm)
    assert compute_ei_balance(adjusted, [0], [1]).ratio == 0.0


@pytest.mark.parametrize("excitation", [-2.0, 2.0])
@pytest.mark.parametrize("inhibition", [-1e308, 1e308])
def test_underflowing_scale_refuses_preserves_source_and_recovers(
    excitation: float, inhibition: float
) -> None:
    """Refuse either signed zero scale through the real public runtime."""
    knm = np.array([[0.0, excitation], [inhibition, 0.0]])
    before = knm.copy()
    with pytest.raises(ValueError, match="non-zero scale"):
        adjust_ei_ratio(knm, [0, 0], [1, 1], target_ratio=1e308)
    np.testing.assert_array_equal(knm, before)
    recovered = adjust_ei_ratio(knm, [0, 0], [1, 1])
    assert recovered[1, 0] == pytest.approx(excitation)
    assert compute_ei_balance(recovered, [0], [1]).ratio == pytest.approx(1.0)
    assert not np.shares_memory(recovered, knm)
    np.testing.assert_array_equal(knm, before)


def test_representable_scale_can_produce_silent_inhibition() -> None:
    """Retain the summary's threshold after valid finite row rescaling."""
    knm = np.array([[0.0, 2.0], [1.0, 0.0]])
    adjusted = adjust_ei_ratio(knm, [0], [1], target_ratio=1e20)
    assert adjusted[1, 0] == pytest.approx(2e-20, rel=1e-15, abs=0.0)
    assert compute_ei_balance(adjusted, [0], [1]).ratio == float("inf")
    assert not np.shares_memory(adjusted, knm)
    np.testing.assert_array_equal(knm, [[0.0, 2.0], [1.0, 0.0]])

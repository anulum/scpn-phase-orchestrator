# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Spatial modulator validation and jacobian guards

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest

from scpn_phase_orchestrator.coupling.spatial_modulator import (
    SpatialCouplingModulator,
    _load_julia_fn,
    _validate_backend_output,
)

_POSITIONS = np.array([[0.0], [1.0]], dtype=np.float64)
_KNM = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)


def _const_distance_fn(matrix: np.ndarray) -> Callable[..., np.ndarray]:
    """Return a distance_fn that ignores its inputs and yields ``matrix``."""

    def _fn(_left: object, _right: object) -> np.ndarray:
        return matrix

    return _fn


class TestScalarAndFormGuards:
    """Constructor guards over scalars, decay form and distance_fn."""

    def test_rejects_boolean_k_base(self) -> None:
        with pytest.raises(ValueError, match="K_base must be a finite real scalar"):
            SpatialCouplingModulator(K_base=True)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"K_base": "1.0"}, "K_base.*numeric-string"),
            ({"decay_exponent": "1.0"}, "decay_exponent.*numeric-string"),
            ({"decay_length_scale": "1.0"}, "decay_length_scale.*numeric-string"),
            ({"epsilon": "1e-12"}, "epsilon.*numeric-string"),
        ],
    )
    def test_rejects_numeric_string_scalar_controls(
        self, kwargs: dict[str, object], match: str
    ) -> None:
        controls = cast("dict[str, Any]", {"K_base": 1.0} | kwargs)
        with pytest.raises(ValueError, match=match):
            SpatialCouplingModulator(**controls)

    def test_rejects_non_finite_length_scale(self) -> None:
        with pytest.raises(ValueError, match="decay_length_scale must be finite"):
            SpatialCouplingModulator(K_base=1.0, decay_length_scale=float("inf"))

    def test_rejects_non_positive_decay_exponent(self) -> None:
        with pytest.raises(ValueError, match="decay_exponent must be positive"):
            SpatialCouplingModulator(K_base=1.0, decay_exponent=0.0)

    def test_rejects_non_callable_distance_fn(self) -> None:
        with pytest.raises(ValueError, match="distance_fn must be callable"):
            SpatialCouplingModulator(K_base=1.0, distance_fn=5)  # type: ignore[arg-type]  # invalid on purpose


class TestPositionGuards:
    """``_validate_positions`` reached through ``modulation_matrix``."""

    def test_rejects_numeric_string_positions(self) -> None:
        modulator = SpatialCouplingModulator(K_base=1.0)
        with pytest.raises(ValueError, match="positions.*numeric-string"):
            modulator.modulation_matrix(np.array(["0.0", "1.0"], dtype=object))

    def test_rejects_boolean_positions(self) -> None:
        modulator = SpatialCouplingModulator(K_base=1.0)
        with pytest.raises(ValueError, match="positions must not contain boolean"):
            modulator.modulation_matrix(np.array([True, False]))

    def test_rejects_complex_positions(self) -> None:
        modulator = SpatialCouplingModulator(K_base=1.0)
        with pytest.raises(ValueError, match="finite real coordinates"):
            modulator.modulation_matrix(np.array([0.0 + 1.0j, 1.0]))

    def test_rejects_non_coercible_positions(self) -> None:
        modulator = SpatialCouplingModulator(K_base=1.0)
        with pytest.raises(ValueError, match="finite real array"):
            modulator.modulation_matrix(np.array([["a"], ["b"]], dtype=object))

    def test_rejects_three_dimensional_positions(self) -> None:
        modulator = SpatialCouplingModulator(K_base=1.0)
        with pytest.raises(ValueError, match=r"shape \(n,\) or \(n, dim\)"):
            modulator.modulation_matrix(np.zeros((2, 2, 2)))

    def test_rejects_empty_positions(self) -> None:
        modulator = SpatialCouplingModulator(K_base=1.0)
        with pytest.raises(ValueError, match="at least one oscillator"):
            modulator.modulation_matrix(np.zeros((0, 1)))


class TestKnmBaseGuards:
    """``_validate_knm_base`` reached through ``modulate``."""

    def test_rejects_numeric_string_base(self) -> None:
        modulator = SpatialCouplingModulator(K_base=1.0)
        base = np.array([["0.0", "1.0"], ["1.0", "0.0"]], dtype=object)
        with pytest.raises(ValueError, match="k_nm_base.*numeric-string"):
            modulator.modulate(base, _POSITIONS)

    def test_rejects_boolean_base(self) -> None:
        modulator = SpatialCouplingModulator(K_base=1.0)
        with pytest.raises(ValueError, match="k_nm_base must not contain boolean"):
            modulator.modulate(np.array([[False, True], [True, False]]), _POSITIONS)

    def test_rejects_non_coercible_base(self) -> None:
        modulator = SpatialCouplingModulator(K_base=1.0)
        base = np.array([["a", "b"], ["c", "d"]], dtype=object)
        with pytest.raises(ValueError, match="finite real square matrix"):
            modulator.modulate(base, _POSITIONS)

    def test_rejects_non_square_base(self) -> None:
        modulator = SpatialCouplingModulator(K_base=1.0)
        with pytest.raises(ValueError, match="finite real square matrix"):
            modulator.modulate(np.zeros((2, 3)), _POSITIONS)

    def test_rejects_base_shape_mismatch(self) -> None:
        modulator = SpatialCouplingModulator(K_base=1.0)
        with pytest.raises(ValueError, match="does not match"):
            modulator.modulate(np.zeros((3, 3)), _POSITIONS)

    def test_rejects_non_finite_base(self) -> None:
        modulator = SpatialCouplingModulator(K_base=1.0)
        base = np.array([[0.0, np.inf], [np.inf, 0.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="only finite values"):
            modulator.modulate(base, _POSITIONS)


class TestDistanceMatrixGuards:
    """``_validate_distance_matrix`` reached through a custom distance kernel."""

    def test_rejects_numeric_string_distance_matrix(self) -> None:
        modulator = SpatialCouplingModulator(
            K_base=1.0,
            distance_fn=_const_distance_fn(
                np.array([["0.0", "1.0"], ["1.0", "0.0"]], dtype=object)
            ),
        )
        with pytest.raises(ValueError, match="distance matrix.*numeric-string"):
            modulator.distance_matrix(_POSITIONS)

    def test_rejects_complex_distance_matrix(self) -> None:
        modulator = SpatialCouplingModulator(
            K_base=1.0,
            distance_fn=_const_distance_fn(np.array([[0.0, 1.0j], [1.0j, 0.0]])),
        )
        with pytest.raises(ValueError, match="finite real-valued"):
            modulator.distance_matrix(_POSITIONS)

    def test_rejects_non_coercible_distance_matrix(self) -> None:
        modulator = SpatialCouplingModulator(
            K_base=1.0,
            distance_fn=_const_distance_fn(
                np.array([["a", "b"], ["c", "d"]], dtype=object)
            ),
        )
        with pytest.raises(ValueError, match="finite real-valued"):
            modulator.distance_matrix(_POSITIONS)

    def test_rejects_distance_matrix_shape_mismatch(self) -> None:
        modulator = SpatialCouplingModulator(
            K_base=1.0, distance_fn=_const_distance_fn(np.zeros((3, 3)))
        )
        with pytest.raises(ValueError, match="does not match"):
            modulator.distance_matrix(_POSITIONS)

    def test_rejects_non_finite_distance_matrix(self) -> None:
        modulator = SpatialCouplingModulator(
            K_base=1.0,
            distance_fn=_const_distance_fn(
                np.array([[0.0, np.inf], [np.inf, 0.0]], dtype=np.float64)
            ),
        )
        with pytest.raises(ValueError, match="only finite values"):
            modulator.distance_matrix(_POSITIONS)

    def test_rejects_negative_distance_matrix(self) -> None:
        modulator = SpatialCouplingModulator(
            K_base=1.0,
            distance_fn=_const_distance_fn(
                np.array([[0.0, -1.0], [-1.0, 0.0]], dtype=np.float64)
            ),
        )
        with pytest.raises(ValueError, match="must be non-negative"):
            modulator.distance_matrix(_POSITIONS)

    def test_rejects_non_zero_diagonal_distance_matrix(self) -> None:
        modulator = SpatialCouplingModulator(
            K_base=1.0,
            distance_fn=_const_distance_fn(
                np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
            ),
        )
        with pytest.raises(ValueError, match="diagonal must be zero"):
            modulator.distance_matrix(_POSITIONS)

    def test_rejects_asymmetric_distance_matrix(self) -> None:
        modulator = SpatialCouplingModulator(
            K_base=1.0,
            distance_fn=_const_distance_fn(
                np.array([[0.0, 2.0], [1.0, 0.0]], dtype=np.float64)
            ),
        )
        with pytest.raises(ValueError, match="must be symmetric"):
            modulator.distance_matrix(_POSITIONS)

    def test_accepts_valid_custom_distance_matrix(self) -> None:
        modulator = SpatialCouplingModulator(
            K_base=1.0,
            distance_fn=_const_distance_fn(
                np.array([[0.0, 2.0], [2.0, 0.0]], dtype=np.float64)
            ),
        )
        distances = modulator.distance_matrix(_POSITIONS)
        np.testing.assert_array_equal(distances, np.array([[0.0, 2.0], [2.0, 0.0]]))


class TestPythonDecayWeights:
    """``modulation_matrix`` exercises the NumPy decay-weight reference path."""

    @pytest.mark.parametrize("decay_form", ["exponential", "power_law"])
    def test_decay_form_produces_zero_diagonal_field(self, decay_form: str) -> None:
        modulator = SpatialCouplingModulator(K_base=1.0, decay_form=decay_form)

        field = modulator.modulation_matrix(_POSITIONS)

        assert field.shape == (2, 2)
        np.testing.assert_array_equal(np.diag(field), np.zeros(2))
        assert np.all(np.isfinite(field))


class TestBackendOutputContract:
    """``_validate_backend_output`` guards untrusted compute-backend output."""

    def test_rejects_numeric_string_output(self) -> None:
        with pytest.raises(ValueError, match="numeric-string"):
            _validate_backend_output(
                np.array(["0.0", "0.5", "0.5", "0.0"], dtype=object), n=2
            )

    def test_rejects_complex_output(self) -> None:
        with pytest.raises(ValueError, match="finite real-valued"):
            _validate_backend_output(np.array([0.0j, 1.0j, 1.0j, 0.0j]), n=2)

    def test_rejects_non_coercible_output(self) -> None:
        bad = np.array(["a", "b", "c", "d"], dtype=object)
        with pytest.raises(ValueError, match="finite real-valued"):
            _validate_backend_output(bad, n=2)

    def test_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="must be"):
            _validate_backend_output(np.zeros(5), n=2)

    def test_rejects_non_finite_output(self) -> None:
        with pytest.raises(ValueError, match="only finite values"):
            _validate_backend_output(np.array([0.0, np.inf, 1.0, 0.0]), n=2)

    def test_rejects_non_zero_diagonal_output(self) -> None:
        with pytest.raises(ValueError, match="diagonal must be zero"):
            _validate_backend_output(np.array([1.0, 1.0, 1.0, 1.0]), n=2)


class TestJacobianPositions:
    """Closed-form position jacobian over each supported decay form."""

    def test_rejects_custom_distance_fn(self) -> None:
        modulator = SpatialCouplingModulator(
            K_base=1.0, distance_fn=_const_distance_fn(np.zeros((2, 2)))
        )
        with pytest.raises(ValueError, match="default Euclidean distance"):
            modulator.jacobian_positions(_POSITIONS)

    @pytest.mark.parametrize(
        "decay_form",
        ["inverse_plus_one", "exponential", "power_law", "inverse_distance"],
    )
    def test_jacobian_is_finite_with_antisymmetric_pair(self, decay_form: str) -> None:
        modulator = SpatialCouplingModulator(K_base=0.7, decay_form=decay_form)

        jac = modulator.jacobian_positions(_POSITIONS)

        assert jac.shape == (2, 2, 2, 1)
        assert np.all(np.isfinite(jac))
        # entry i,j is the negative of its position-swapped counterpart
        np.testing.assert_allclose(jac[0, 1, 0], -jac[0, 1, 1])

    def test_jacobian_is_zero_for_coincident_oscillators(self) -> None:
        modulator = SpatialCouplingModulator(K_base=0.7, decay_form="inverse_plus_one")

        jac = modulator.jacobian_positions(np.array([[0.0], [0.0]], dtype=np.float64))

        np.testing.assert_array_equal(jac, np.zeros((2, 2, 2, 1)))


def test_load_julia_fn_rejects_juliacall_without_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A present-but-uninitialised juliacall must be reported unavailable.

    When the Julia runtime cannot inject ``juliacall.Main`` (for example under
    coverage's thread tracer) the package still imports. The backend probe
    must reject it here so dispatch falls through, instead of selecting Julia
    and crashing with ImportError at call time.
    """
    import sys
    import types

    stub = types.ModuleType("juliacall")  # deliberately lacks ``Main``
    monkeypatch.setitem(sys.modules, "juliacall", stub)

    with pytest.raises(ImportError, match="Main unavailable"):
        _load_julia_fn()

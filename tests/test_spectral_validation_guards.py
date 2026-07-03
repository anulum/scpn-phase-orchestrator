# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Spectral input and FFI-output validation guards

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.coupling import (
    _spectral_validation as spectral_validation,
)
from scpn_phase_orchestrator.coupling.spectral import (
    _validate_rust_fiedler_vector,
    _validate_spectral_output,
    fiedler_value,
    spectral_eig,
    sync_convergence_rate,
)


def _valid_knm() -> np.ndarray:
    return np.array([[0.0, 0.4], [0.4, 0.0]], dtype=np.float64)


class _FailedArrayProtocol:
    def __array__(
        self,
        dtype: object | None = None,
        copy: object | None = None,
    ) -> np.ndarray:
        raise TypeError("array protocol failed")


class _FailedObjectArrayProtocol:
    def __array__(
        self,
        dtype: object | None = None,
        copy: object | None = None,
    ) -> np.ndarray:
        if dtype is not None:
            raise TypeError("object array protocol failed")
        return np.array([1.0, 2.0], dtype=np.float64)


class TestCouplingMatrixObjectArrayGuards:
    """Object-dtype matrices that smuggle non-real entries must fail closed."""

    def test_rejects_object_matrix_with_complex_entry(self) -> None:
        knm = np.array(
            [[0.0, complex(0.4, 0.0)], [0.4, 0.0]],
            dtype=object,
        )
        with pytest.raises(ValueError, match="real-valued"):
            spectral_eig(knm)

    def test_rejects_object_matrix_with_string_entry(self) -> None:
        knm = np.array([[0.0, "x"], [0.4, 0.0]], dtype=object)
        with pytest.raises(ValueError, match="finite square matrix"):
            fiedler_value(knm)


class TestOmegaVectorGuards:
    """Frequency vectors must coerce to a finite 1-D float array."""

    def test_rejects_two_dimensional_omegas(self) -> None:
        with pytest.raises(ValueError, match="1-D frequency vector"):
            sync_convergence_rate(_valid_knm(), np.zeros((2, 1)))

    def test_rejects_non_numeric_omegas(self) -> None:
        omegas = np.array(["a", "b"], dtype=object)
        with pytest.raises(ValueError, match="1-D frequency vector"):
            sync_convergence_rate(_valid_knm(), omegas)


class TestSpectralOutputContract:
    """``_validate_spectral_output`` guards untrusted backend primitive output."""

    def test_rejects_non_pair_output(self) -> None:
        with pytest.raises(ValueError, match=r"\(eigvals, fiedler\)"):
            _validate_spectral_output([1.0, 2.0, 3.0], n=2)

    def test_rejects_complex_eigenvalues(self) -> None:
        value = (np.array([0.0j, 1.0j]), np.array([0.0, 1.0]))
        with pytest.raises(ValueError, match="real-valued numeric arrays"):
            _validate_spectral_output(value, n=2)

    def test_rejects_non_numeric_arrays(self) -> None:
        value = (np.array(["a", "b"], dtype=object), np.array([0.0, 1.0]))
        with pytest.raises(ValueError, match="must be numeric"):
            _validate_spectral_output(value, n=2)

    def test_rejects_eigenvalue_shape_mismatch(self) -> None:
        value = (np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]))
        with pytest.raises(ValueError, match="eigenvalue shape"):
            _validate_spectral_output(value, n=2)

    def test_rejects_fiedler_shape_mismatch(self) -> None:
        value = (np.array([0.0, 1.0]), np.array([0.0, 1.0, 2.0]))
        with pytest.raises(ValueError, match="fiedler shape"):
            _validate_spectral_output(value, n=2)

    def test_rejects_non_finite_output(self) -> None:
        value = (np.array([0.0, np.inf]), np.array([0.0, 1.0]))
        with pytest.raises(ValueError, match="only finite values"):
            _validate_spectral_output(value, n=2)

    def test_rejects_unsorted_eigenvalues(self) -> None:
        value = (np.array([1.0, 0.0]), np.array([0.0, 1.0]))
        with pytest.raises(ValueError, match="sorted ascending"):
            _validate_spectral_output(value, n=2)

    def test_rejects_zero_fiedler_vector(self) -> None:
        value = (np.array([0.0, 1.0]), np.array([0.0, 0.0]))
        with pytest.raises(ValueError, match="fiedler vector must be non-zero"):
            _validate_spectral_output(value, n=2)


class TestSharedSpectralBackendInputContract:
    """Direct optional backends share one fail-closed input contract."""

    def test_accepts_valid_flattened_matrix_and_integer_n(self) -> None:
        k, n = spectral_validation.validate_spectral_backend_inputs(
            [0.0, 0.4, 0.4, 0.0],
            2,
        )

        assert n == 2
        assert k.dtype == np.float64
        assert k.shape == (4,)
        assert k.flags.c_contiguous

    def test_rejects_boolean_array(self) -> None:
        with pytest.raises(ValueError, match="boolean"):
            spectral_validation.validate_spectral_backend_inputs(
                np.array([True, False, False, True]),
                2,
            )

    def test_rejects_complex_array(self) -> None:
        with pytest.raises(ValueError, match="real-valued"):
            spectral_validation.validate_spectral_backend_inputs(
                np.array([0.0, 1.0 + 0.0j, 0.0, 0.0]),
                2,
            )

    def test_rejects_non_numeric_array(self) -> None:
        with pytest.raises(ValueError, match="finite one-dimensional"):
            spectral_validation.validate_spectral_backend_inputs(
                np.array(["x", "y"], dtype=object),
                1,
            )

    def test_rejects_two_dimensional_array(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional"):
            spectral_validation.validate_spectral_backend_inputs(np.zeros((2, 2)), 2)

    def test_rejects_non_finite_array(self) -> None:
        with pytest.raises(ValueError, match="finite values"):
            spectral_validation.validate_spectral_backend_inputs(
                np.array([0.0, np.inf, 0.0, 0.0]),
                2,
            )

    def test_rejects_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match=r"n\*n"):
            spectral_validation.validate_spectral_backend_inputs(np.zeros(3), 2)

    def test_rejects_negative_n(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            spectral_validation.validate_spectral_backend_inputs(
                np.array([], dtype=np.float64),
                -1,
            )


class TestSharedSpectralBackendOutputContract:
    """Direct optional backends share the public eigensystem output contract."""

    def test_accepts_valid_output_as_contiguous_float64_arrays(self) -> None:
        value = (
            np.array([-1e-12, 1.0], dtype=np.float32),
            np.array([1.0, -1.0], dtype=np.float32),
        )

        eigvals, fiedler = spectral_validation.validate_spectral_backend_output(
            value,
            n=2,
        )

        assert eigvals.dtype == np.float64
        assert fiedler.dtype == np.float64
        assert eigvals.flags.c_contiguous
        assert fiedler.flags.c_contiguous
        np.testing.assert_array_equal(eigvals, np.array([0.0, 1.0]))
        np.testing.assert_array_equal(fiedler, np.array([1.0, -1.0]))

    def test_rejects_non_integer_oscillator_count(self) -> None:
        value = (np.array([], dtype=np.float64), np.array([], dtype=np.float64))
        with pytest.raises(ValueError, match="n must be a non-negative integer"):
            spectral_validation.validate_spectral_backend_output(value, n=True)

    def test_rejects_non_pair_output(self) -> None:
        with pytest.raises(ValueError, match=r"\(eigvals, fiedler\)"):
            spectral_validation.validate_spectral_backend_output(
                [1.0, 2.0, 3.0],
                n=2,
            )

    def test_rejects_boolean_alias_output(self) -> None:
        value = (
            np.array([0.0, np.bool_(True)], dtype=object),
            np.array([1.0, -1.0]),
        )
        with pytest.raises(ValueError, match="real-valued numeric arrays"):
            spectral_validation.validate_spectral_backend_output(value, n=2)

    def test_rejects_complex_alias_output(self) -> None:
        value = (
            np.array([0.0, np.complex128(1.0 + 0.0j)], dtype=object),
            np.array([1.0, -1.0]),
        )
        with pytest.raises(ValueError, match="real-valued numeric arrays"):
            spectral_validation.validate_spectral_backend_output(value, n=2)

    def test_rejects_non_numeric_arrays(self) -> None:
        value = (np.array(["a", "b"], dtype=object), np.array([1.0, -1.0]))
        with pytest.raises(ValueError, match="must be numeric"):
            spectral_validation.validate_spectral_backend_output(value, n=2)

    def test_rejects_failed_array_protocol_as_non_numeric(self) -> None:
        value = (_FailedArrayProtocol(), np.array([1.0, -1.0]))
        with pytest.raises(ValueError, match="must be numeric"):
            spectral_validation.validate_spectral_backend_output(value, n=2)

    def test_rejects_failed_object_array_protocol_as_non_numeric(self) -> None:
        value = (_FailedObjectArrayProtocol(), np.array([1.0, -1.0]))
        with pytest.raises(ValueError, match="must be numeric"):
            spectral_validation.validate_spectral_backend_output(value, n=2)

    def test_rejects_eigenvalue_shape_mismatch(self) -> None:
        value = (np.array([0.0, 1.0, 2.0]), np.array([1.0, -1.0]))
        with pytest.raises(ValueError, match="eigenvalue shape"):
            spectral_validation.validate_spectral_backend_output(value, n=2)

    def test_rejects_fiedler_shape_mismatch(self) -> None:
        value = (np.array([0.0, 1.0]), np.array([1.0, -1.0, 0.0]))
        with pytest.raises(ValueError, match="fiedler shape"):
            spectral_validation.validate_spectral_backend_output(value, n=2)

    def test_rejects_non_finite_output(self) -> None:
        value = (np.array([0.0, np.inf]), np.array([1.0, -1.0]))
        with pytest.raises(ValueError, match="only finite values"):
            spectral_validation.validate_spectral_backend_output(value, n=2)

    def test_rejects_negative_eigenvalues(self) -> None:
        value = (np.array([-1e-4, 1.0]), np.array([1.0, -1.0]))
        with pytest.raises(ValueError, match="non-negative"):
            spectral_validation.validate_spectral_backend_output(value, n=2)

    def test_rejects_unsorted_eigenvalues(self) -> None:
        value = (np.array([1.0, 0.0]), np.array([1.0, -1.0]))
        with pytest.raises(ValueError, match="sorted ascending"):
            spectral_validation.validate_spectral_backend_output(value, n=2)

    def test_rejects_zero_fiedler_vector(self) -> None:
        value = (np.array([0.0, 1.0]), np.array([0.0, 0.0]))
        with pytest.raises(ValueError, match="fiedler vector must be non-zero"):
            spectral_validation.validate_spectral_backend_output(value, n=2)


class TestRustFiedlerVectorContract:
    """``_validate_rust_fiedler_vector`` guards untrusted Rust FFI vectors."""

    def test_rejects_non_numeric_vector(self) -> None:
        with pytest.raises(ValueError, match="must be numeric"):
            _validate_rust_fiedler_vector(np.array(["a", "b"], dtype=object), n=2)

    def test_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="must be"):
            _validate_rust_fiedler_vector(np.array([0.0, 1.0, 2.0]), n=2)

    def test_rejects_non_finite_vector(self) -> None:
        with pytest.raises(ValueError, match="only finite values"):
            _validate_rust_fiedler_vector(np.array([0.0, np.inf]), n=2)

    def test_rejects_zero_vector(self) -> None:
        with pytest.raises(ValueError, match="must be non-zero"):
            _validate_rust_fiedler_vector(np.array([0.0, 0.0]), n=2)

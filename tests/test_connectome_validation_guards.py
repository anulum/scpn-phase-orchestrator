# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Connectome matrix-validation and FFI guard contracts

"""Exercise structural admission through the original public loaders.

Owned admission module: scpn_phase_orchestrator.coupling._connectome_validation.
Every fault modifies the output of an original provider after its real execution.
Such negative/representation contracts are separate from unmodified owner proof.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from benchmarks.connectome_reference import reference_connectome, reference_hcp
from scpn_phase_orchestrator.coupling import connectome
from tests.test_connectome_real_runtime import (
    inject_hcp_fault,
    inject_native_fault,
    installed_matrix,
)

pytestmark = pytest.mark.native_runtime


def test_coerce_rejects_non_float_convertible_object_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nonnumeric data injected after genuine dataset I/O cannot enter coupling."""

    def corrupt(matrix: NDArray[np.float64]) -> object:
        """Replace one real provider entry by invalid text."""
        damaged = matrix.astype(object)
        damaged[0, 1] = damaged[1, 0] = "notnumeric"
        return damaged

    inject_hcp_fault(monkeypatch, corrupt)
    with pytest.raises(ValueError, match="must be a float matrix"):
        connectome.load_neurolib_hcp(2)


def test_coerce_rejects_wrong_shape_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    """A truncated original provider matrix fails its eighty-region contract."""
    inject_hcp_fault(monkeypatch, lambda matrix: matrix[:2, :2])
    with pytest.raises(ValueError, match=r"shape \(80, 80\), got \(2, 2\)"):
        connectome.load_neurolib_hcp(3)


def test_coerce_rejects_asymmetric_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    """An asymmetric provider-data fault is refused before cortical slicing."""

    def corrupt(matrix: NDArray[np.float64]) -> object:
        """Change one directed entry of the real provider output."""
        matrix[0, 1] += 0.25
        return matrix

    inject_hcp_fault(monkeypatch, corrupt)
    with pytest.raises(ValueError, match="must be symmetric"):
        connectome.load_neurolib_hcp(2)


def test_coerce_accepts_clean_symmetric_matrix() -> None:
    """Unmodified installed HCP ingress preserves independently averaged weights."""
    actual = installed_matrix("rust", 2, kind="hcp")
    np.testing.assert_allclose(actual, reference_hcp(2), atol=2e-15, rtol=2e-15)


def test_coerce_accepts_finite_real_numeric_object_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real provider values retain documented object-storage compatibility.

    This representation-boundary contract changes storage after actual I/O,
    without qualifying that modified provider as a successful original backend.
    """
    inject_hcp_fault(monkeypatch, lambda matrix: matrix.astype(object))
    actual = connectome.load_neurolib_hcp(2)
    np.testing.assert_allclose(actual, reference_hcp(2), atol=2e-15, rtol=2e-15)
    assert actual.dtype == np.float64 and actual.flags.c_contiguous


@pytest.mark.parametrize(
    ("dtype", "message"),
    [
        (bool, "must not contain boolean values"),
        (complex, "must contain real-valued weights"),
    ],
)
def test_rust_loader_rejects_native_bool_or_complex_dtype(
    monkeypatch: pytest.MonkeyPatch, dtype: type[bool] | type[complex], message: str
) -> None:
    """Native output dtype faults are refused after original generated values."""
    inject_native_fault(monkeypatch, lambda matrix: matrix.astype(dtype))
    with pytest.raises(ValueError, match=message):
        connectome.load_hcp_connectome(2, 4242)


def test_rust_loader_rejects_object_array_without_float_coercion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nonnumeric native-output data cannot be silently float-coerced."""

    def corrupt(matrix: NDArray[np.float64]) -> object:
        """Inject nonnumeric text after original native generation."""
        damaged = matrix.astype(object)
        damaged[0, 1] = "notnumeric"
        return damaged

    inject_native_fault(monkeypatch, corrupt)
    with pytest.raises(ValueError, match="must contain real-valued weights"):
        connectome.load_hcp_connectome(2, 4343)


def test_module_marks_rust_absent_when_kernel_cannot_import() -> None:
    """An actually uninstalled kernel executes the original Python public law."""
    actual = installed_matrix("python", 3, 42)
    np.testing.assert_allclose(
        actual, reference_connectome(3, 42, "python"), atol=3e-14
    )

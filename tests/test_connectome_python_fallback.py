# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Connectome Python fallback contracts

"""Real optional HCP and kernel-absent public profile contracts."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from benchmarks.connectome_reference import reference_connectome, reference_hcp
from scpn_phase_orchestrator.coupling import connectome
from tests.test_connectome_real_runtime import inject_hcp_fault, installed_matrix

pytestmark = pytest.mark.native_runtime


def test_connectome_python_fallback_preserves_brain_network_structure() -> None:
    """The actually kernel-absent installed owner preserves every declared edge."""
    actual = installed_matrix("python", 16, 7)
    np.testing.assert_allclose(
        actual, reference_connectome(16, 7, "python"), atol=3e-14
    )
    np.testing.assert_array_equal(np.diag(actual), np.zeros(16))


def test_neurolib_hcp_loader_validates_and_slices_dataset() -> None:
    """Original installed HCP data agrees with independent actual subject assets."""
    actual = installed_matrix("rust", 5, kind="hcp")
    np.testing.assert_allclose(actual, reference_hcp(5), atol=2e-15, rtol=2e-15)
    with pytest.raises(ValueError, match=">= 2"):
        connectome.load_neurolib_hcp(1)
    with pytest.raises(TypeError, match="n_regions must be an integer"):
        connectome.load_neurolib_hcp(True)
    with pytest.raises(ValueError, match="<= 80"):
        connectome.load_neurolib_hcp(81)


@pytest.mark.parametrize(
    ("matrix_update", "message"),
    [
        (lambda matrix: matrix.__setitem__((0, 1), np.nan), "finite"),
        (lambda matrix: matrix.__setitem__((0, 1), -0.25), "non-negative"),
        (lambda matrix: matrix.__setitem__((0, 1), np.bool_(True)), "boolean"),
        (lambda matrix: matrix.__setitem__((0, 1), complex(0.2, 0.0)), "real-valued"),
        (
            lambda matrix: matrix.__setitem__((0, 1), np.str_("0.2")),
            "numeric-string aliases",
        ),
        (
            lambda matrix: matrix.__setitem__((0, 1), np.bytes_(b"0.2")),
            "numeric-string aliases",
        ),
    ],
)
def test_neurolib_hcp_loader_rejects_invalid_dataset_contract(
    monkeypatch: pytest.MonkeyPatch,
    matrix_update: Callable[[NDArray[np.object_]], None],
    message: str,
) -> None:
    """Faults after original HCP file loading fail at the public ingress boundary.

    This is an injected negative provider-data contract, not backend-success or
    anatomical-data qualification. The original Dataset and assets execute first.
    """

    def corrupt(matrix: NDArray[np.float64]) -> object:
        """Inject one declared invalid entry into original subject-average weights."""
        damaged = matrix.astype(object)
        matrix_update(damaged)
        damaged[1, 0] = damaged[0, 1]
        return damaged

    inject_hcp_fault(monkeypatch, corrupt)
    with pytest.raises(ValueError, match=message):
        connectome.load_neurolib_hcp(5)


def test_neurolib_hcp_loader_reports_missing_optional_dependency() -> None:
    """An actually uninstalled optional provider produces the public ImportError."""
    with pytest.raises(ImportError, match="neurolib is required"):
        installed_matrix("python", 2, kind="hcp")

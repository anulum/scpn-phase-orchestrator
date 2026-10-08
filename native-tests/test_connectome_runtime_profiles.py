# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Original native connectome ABI regressions

"""Observe the installed builtin and its real admission, recovery and edge laws."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from importlib import import_module
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from benchmarks.connectome_reference import reference_connectome

pytestmark = pytest.mark.native_runtime


@pytest.mark.parametrize("n_regions", [0, 1])
def test_original_native_refuses_small_counts_with_numerical_recovery(
    n_regions: int,
) -> None:
    """Admitted integer counts below two fail before original native generation."""
    generator = cast(
        "Callable[[object, object], NDArray[np.float64]]",
        import_module("spo_kernel").load_hcp_connectome_rust,
    )
    assert inspect.isbuiltin(generator)
    with pytest.raises(ValueError, match="n_regions must be >= 2"):
        generator(n_regions, 42)
    np.testing.assert_allclose(
        generator(3, 42).reshape(3, 3), reference_connectome(3, 42, "rust")
    )


@pytest.mark.parametrize("n_regions", [2, 3, 16])
@pytest.mark.parametrize("seed", [0, 42, 2**64 - 1])
def test_original_native_generator_matches_independent_edge_equations(
    n_regions: int, seed: int
) -> None:
    """The unchanged two-argument builtin returns its full row-major edge law."""
    generator = cast(
        "Callable[[object, object], NDArray[np.float64]]",
        import_module("spo_kernel").load_hcp_connectome_rust,
    )
    assert inspect.isbuiltin(generator)
    actual = generator(n_regions, seed)
    assert actual.shape == (n_regions * n_regions,)
    assert actual.dtype == np.float64 and actual.flags.c_contiguous
    np.testing.assert_allclose(
        actual.reshape(n_regions, n_regions),
        reference_connectome(n_regions, seed, "rust"),
        rtol=3e-14,
        atol=3e-14,
    )


@pytest.mark.parametrize("argument", ["n_regions", "seed"])
@pytest.mark.parametrize(
    "value",
    [True, np.bool_(True), "4", 4.0, np.timedelta64(4, "ms"), None],
)
def test_original_native_metadata_refusal_preserves_following_generation(
    argument: str, value: object
) -> None:
    """Scalar aliases fail at the actual boundary without damaging later calls."""
    generator = cast(
        "Callable[[object, object], NDArray[np.float64]]",
        import_module("spo_kernel").load_hcp_connectome_rust,
    )
    with pytest.raises(ValueError):
        generator(
            value if argument == "n_regions" else 3, value if argument == "seed" else 42
        )
    np.testing.assert_allclose(
        generator(3, 42).reshape(3, 3), reference_connectome(3, 42, "rust")
    )


@pytest.mark.parametrize("n_regions", [2**30, 2**32, np.iinfo(np.intp).max])
def test_original_native_refuses_unaddressable_matrix_without_panic(
    n_regions: int,
) -> None:
    """Both byte-count and square-count overflow return the native ValueError."""
    generator = cast(
        "Callable[[object, object], NDArray[np.float64]]",
        import_module("spo_kernel").load_hcp_connectome_rust,
    )
    with pytest.raises(ValueError, match="exceeds addressable float64 storage"):
        generator(n_regions, 42)
    np.testing.assert_allclose(
        generator(2, 42).reshape(2, 2), [[0.0, 4.95], [4.95, 0.0]]
    )

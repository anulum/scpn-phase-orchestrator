# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Native embedding measurement controls

"""Exercise original metadata types at installed embedding entry points."""

from __future__ import annotations

import numpy as np
import pytest
import spo_kernel


@pytest.mark.parametrize(
    "surface,field",
    [
        ("delay_embed_rust", "delay"),
        ("delay_embed_rust", "dimension"),
        ("optimal_delay_rust", "max_lag"),
        ("optimal_delay_rust", "n_bins"),
        ("optimal_dimension_rust", "delay"),
        ("optimal_dimension_rust", "max_dim"),
        ("optimal_dimension_rust", "rtol"),
        ("optimal_dimension_rust", "atol"),
    ],
)
@pytest.mark.parametrize("value", [True, np.bool_(True), "2", np.timedelta64(2, "ns")])
def test_native_embedding_rejects_metadata_aliases(
    surface: str, field: str, value: object
) -> None:
    """Counts and tolerances reject aliases before native parameter extraction."""
    controls: dict[str, object] = (
        {"delay": 2, "dimension": 3}
        if surface == "delay_embed_rust"
        else {"max_lag": 4, "n_bins": 4}
        if surface == "optimal_delay_rust"
        else {"delay": 1, "max_dim": 3, "rtol": 15.0, "atol": 2.0}
    )
    controls[field] = value
    with pytest.raises(ValueError):
        getattr(spo_kernel, surface)(np.arange(24, dtype=np.float64), **controls)


def test_native_embedding_checks_window_product_before_allocation() -> None:
    """An overflowing embedding window is refused rather than wrapped."""
    with pytest.raises(ValueError, match="overflows"):
        spo_kernel.delay_embed_rust(
            np.arange(24, dtype=np.float64), int(np.iinfo(np.uintp).max), 3
        )


def test_native_dimension_large_delay_has_no_candidates() -> None:
    """A valid huge delay cannot become negative after native signed conversion."""
    assert (
        spo_kernel.optimal_dimension_rust(
            np.arange(24, dtype=np.float64), int(np.iinfo(np.uintp).max)
        )
        == 1
    )


@pytest.mark.parametrize(
    "surface", ["delay_embed_rust", "optimal_delay_rust", "optimal_dimension_rust"]
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_native_embedding_rejects_nonfinite_samples(surface: str, value: float) -> None:
    """Native ordering and distance computations require finite signal samples."""
    controls = (
        {"delay": 1, "dimension": 2}
        if surface == "delay_embed_rust"
        else {"delay": 1}
        if surface == "optimal_dimension_rust"
        else {}
    )
    with pytest.raises(ValueError):
        getattr(spo_kernel, surface)(np.array([0.0, value, 1.0, 2.0]), **controls)


def test_native_delay_checks_histogram_product_before_allocation() -> None:
    """Overflowing histogram metadata is refused before allocating a table."""
    bins = 1 << (np.dtype(np.uintp).itemsize * 4)
    with pytest.raises(ValueError, match="overflows"):
        spo_kernel.optimal_delay_rust(np.arange(24, dtype=np.float64), 4, bins)


@pytest.mark.parametrize("field", ["rtol", "atol"])
@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_native_dimension_rejects_invalid_tolerances(field: str, value: float) -> None:
    """Distance thresholds remain finite and nonnegative after source checks."""
    controls = {"delay": 1, "max_dim": 3, "rtol": 15.0, "atol": 2.0}
    controls[field] = value
    with pytest.raises(ValueError):
        spo_kernel.optimal_dimension_rust(np.arange(24, dtype=np.float64), **controls)

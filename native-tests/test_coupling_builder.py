# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Native coupling builder source types

"""Test original measurement types at the installed coupling builder boundary."""

from __future__ import annotations

import numpy as np
import pytest
import spo_kernel


@pytest.mark.parametrize("field", ["n", "base_strength", "decay_alpha"])
@pytest.mark.parametrize("value", [True, np.bool_(True), "1", np.timedelta64(1, "ms")])
def test_native_builder_rejects_scalar_aliases(field: str, value: object) -> None:
    """Counts and coefficients are checked before native parameter extraction."""
    controls: dict[str, object] = {"n": 3, "base_strength": 0.5, "decay_alpha": 0.2}
    controls[field] = value
    with pytest.raises(ValueError):
        spo_kernel.PyCouplingBuilder().build(**controls)


@pytest.mark.parametrize(
    "value",
    [True, np.bool_(True), "1", np.timedelta64(1, "ms"), np.datetime64("2026-01-01")],
)
def test_native_projection_rejects_source_aliases(value: object) -> None:
    """Projection cannot erase the type of the original distance-free weights."""
    with pytest.raises(ValueError):
        spo_kernel.PyCouplingBuilder.project([0.0, value, value, 0.0], 2)


def test_native_projection_preserves_real_numeric_objects() -> None:
    """Real object coefficients retain symmetry projection and zero diagonal."""
    actual = spo_kernel.PyCouplingBuilder.project(
        np.array([0, 2.0, 1.0, 0], dtype=object), np.int64(2)
    )
    np.testing.assert_array_equal(actual, [0.0, 1.5, 1.5, 0.0])


def test_native_builder_rejects_dimension_product_overflow() -> None:
    """An unrepresentable matrix size is refused before native allocation."""
    with pytest.raises(ValueError, match="n\\*n overflows"):
        spo_kernel.PyCouplingBuilder().build(
            1 << (np.dtype(np.uintp).itemsize * 4), 0.5, 0.2
        )

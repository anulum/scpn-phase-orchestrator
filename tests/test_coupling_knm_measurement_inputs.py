# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Coupling builder measurement ingress

"""Exercise source types and immutable state through public coupling construction."""

from __future__ import annotations

from typing import TypeAlias, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.coupling.knm import CouplingBuilder

FloatArray: TypeAlias = NDArray[np.float64]


@pytest.mark.parametrize(
    "dtype",
    ["U8", "bool", "complex128", "timedelta64[ms]", "datetime64[ns]", "object-time"],
)
def test_template_switch_rejects_source_aliases(dtype: str) -> None:
    """Reject coercible templates before replacing a real coupling snapshot."""
    builder = CouplingBuilder()
    state = builder.build(3, 0.5, 0.2)
    original = state.knm.copy()
    k = np.ones((3, 3)) - np.eye(3)
    value = cast(
        FloatArray,
        np.array([np.timedelta64(1, "ns")] * 9, dtype=object).reshape(3, 3)
        if dtype == "object-time"
        else k.astype(dtype),
    )
    with pytest.raises(ValueError):
        builder.switch_template(state, "invalid", {"invalid": value})
    np.testing.assert_array_equal(state.knm, original)


@pytest.mark.parametrize("field", ["n_layers", "base_strength", "decay_alpha"])
def test_builder_rejects_temporal_controls(field: str) -> None:
    """A temporal scalar cannot become a layer count or coupling coefficient."""
    value = np.timedelta64(2, "ms")
    with pytest.raises(ValueError):
        CouplingBuilder().build(
            cast(int, value) if field == "n_layers" else 3,
            cast(float, value) if field == "base_strength" else 0.5,
            cast(float, value) if field == "decay_alpha" else 0.2,
        )


def test_template_switch_preserves_real_numeric_objects() -> None:
    """Valid object coefficients retain exact values without mutating old state."""
    builder = CouplingBuilder()
    state = builder.build(3, 0.5, 0.2)
    original = state.knm.copy()
    k = (np.ones((3, 3)) - np.eye(3)) * 0.7
    actual = builder.switch_template(
        state, "objects", {"objects": cast(FloatArray, k.astype(object))}
    )
    np.testing.assert_array_equal(actual.knm, k)
    np.testing.assert_array_equal(actual.alpha, state.alpha)
    np.testing.assert_array_equal(state.knm, original)

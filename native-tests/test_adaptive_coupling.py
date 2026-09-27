# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Native adaptive coupling boundaries

"""Exercise original metadata and matrix domains at the installed Rust boundary."""

from __future__ import annotations

import numpy as np
import pytest
import spo_kernel


@pytest.mark.parametrize("field", ["n", "lr", "decay"])
@pytest.mark.parametrize("value", [True, np.bool_(True), "1", np.timedelta64(1, "ms")])
def test_native_adaptation_rejects_control_aliases(field: str, value: object) -> None:
    """Metadata extraction preserves alias rejection before numeric coercion."""
    controls: dict[str, object] = {"n": 2, "lr": 0.01, "decay": 0.0}
    controls[field] = value
    with pytest.raises(ValueError):
        spo_kernel.te_adapt_coupling_rust(
            np.array([0.0, 1.0, 1.0, 0.0]), np.array([0.0, 0.2, 0.3, 0.0]), **controls
        )


@pytest.mark.parametrize("field", ["knm", "te"])
@pytest.mark.parametrize(
    "values",
    [
        [0.0, 1.0, 0.0],
        [0.0, np.nan, 1.0, 0.0],
        [0.0, -1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 0.0],
    ],
)
def test_native_adaptation_rejects_matrix_faults(
    field: str, values: list[float]
) -> None:
    """Reject malformed native matrices before indexing or publishing an update."""
    valid = np.array([0.0, 1.0, 1.0, 0.0])
    invalid = np.asarray(values)
    with pytest.raises(ValueError):
        spo_kernel.te_adapt_coupling_rust(
            invalid if field == "knm" else valid,
            invalid if field == "te" else valid,
            2,
            0.01,
            0.0,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("lr", -0.1),
        ("lr", np.nan),
        ("lr", np.inf),
        ("decay", -0.1),
        ("decay", 1.1),
        ("decay", np.nan),
        ("decay", np.inf),
    ],
)
def test_native_adaptation_rejects_out_of_domain_controls(
    field: str, value: float
) -> None:
    """The native update shares finite learning-rate and unit-interval decay rules."""
    valid = np.array([0.0, 1.0, 1.0, 0.0])
    with pytest.raises(ValueError):
        spo_kernel.te_adapt_coupling_rust(
            valid,
            valid,
            2,
            value if field == "lr" else 0.01,
            value if field == "decay" else 0.0,
        )


def test_native_adaptation_rejects_overflowing_output() -> None:
    """Finite inputs cannot publish an infinite adapted coupling matrix."""
    valid = np.array([0.0, 1e308, 1e308, 0.0])
    with pytest.raises(ValueError, match="adapted coupling"):
        spo_kernel.te_adapt_coupling_rust(valid, valid, 2, 2.0, 0.0)


def test_native_adaptation_normalises_negligible_negative_scores() -> None:
    """Match the public score tolerance before forming the coupling update."""
    k = np.array([0.0, 1.0, 1.0, 0.0])
    t = np.array([0.0, -1e-13, 0.2, 0.0])
    result = spo_kernel.te_adapt_coupling_rust(k, t, 2, 0.5, 0.1)
    np.testing.assert_allclose(
        result, np.array([0.0, 0.9, 1.0, 0.0]), rtol=1e-12, atol=1e-12
    )

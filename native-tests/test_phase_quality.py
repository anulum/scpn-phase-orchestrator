# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Native phase quality boundary tests

"""Exercise the installed Rust extension; run in the kernel-built FFI CI lanes."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from spo_kernel import PyPhaseQualityScorer


@pytest.mark.parametrize("method", ["score", "collapse", "mask"])
@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(True),
        "0.8",
        b"0.8",
        complex(0.8, 0.1),
        np.timedelta64(1, "ms"),
        np.datetime64("2026-01-01"),
    ],
)
def test_native_quality_rejects_aliases(method: str, value: object) -> None:
    """Native methods inspect original scalar types before extracting f64."""
    scorer = PyPhaseQualityScorer()
    with pytest.raises(ValueError, match="quality"):
        if method == "score":
            scorer.score([value], [1.0])
        elif method == "collapse":
            scorer.is_collapsed([value])
        else:
            scorer.downweight_mask([value])


@pytest.mark.parametrize("value", [True, "1.0", np.timedelta64(1, "ms")])
def test_native_amplitude_rejects_aliases(value: object) -> None:
    """Native amplitude weighting shares the plain real source contract."""
    with pytest.raises(ValueError, match="amplitude"):
        PyPhaseQualityScorer().score([0.8], [value])


@pytest.mark.parametrize("argument", ["collapse_threshold", "min_quality"])
@pytest.mark.parametrize(
    "value",
    [True, np.bool_(True), "0.3", np.timedelta64(1, "ms"), np.nan, np.inf, -0.1, 1.1],
)
def test_native_thresholds_reject_invalid_source_or_domain(
    argument: str, value: object
) -> None:
    """Constructor validates source types, finiteness and the unit interval."""
    with pytest.raises(ValueError):
        PyPhaseQualityScorer(**{argument: value})


@pytest.mark.parametrize("dtype", ["float64", "float32", "int64", "uint64", "object"])
def test_native_quality_preserves_real_arrays(dtype: str) -> None:
    """Real scalar types in arrays remain accepted by the native scorer."""
    qualities = np.array([0, 1, 1]).astype(dtype)
    amplitudes = np.array([1, 2, 3]).astype(dtype)
    scorer = PyPhaseQualityScorer()
    assert scorer.score(qualities, amplitudes) == pytest.approx(5 / 6)
    assert not scorer.is_collapsed(qualities)
    np.testing.assert_array_equal(scorer.downweight_mask(qualities), [0.0, 1.0, 1.0])


def test_native_nonfinite_quality_policy_and_defaults() -> None:
    """Source checks preserve NaN policy, clamps, and constructor defaults."""
    scorer = PyPhaseQualityScorer()
    assert (
        str(inspect.signature(PyPhaseQualityScorer))
        == "(collapse_threshold=0.1, min_quality=0.3)"
    )
    assert scorer.score([np.nan, 0.8, 0.1], [1.0, 2.0, 1.0]) == pytest.approx(1.7 / 3)
    assert scorer.is_collapsed([np.nan, 0.0, 0.8])
    np.testing.assert_array_equal(
        scorer.downweight_mask([np.nan, 1.2, -0.1]), [0.0, 1.0, 0.0]
    )

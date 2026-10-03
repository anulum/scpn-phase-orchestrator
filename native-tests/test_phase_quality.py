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
import sys
from types import BuiltinMethodType, FrameType
from typing import cast

import numpy as np
import pytest
from spo_kernel import PyPhaseQualityScorer

from scpn_phase_orchestrator.oscillators.base import PhaseState
from scpn_phase_orchestrator.oscillators.quality import PhaseQualityScorer


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


def test_public_configured_and_override_calls_observe_actual_native() -> None:
    """Real C calls prove dispatch while analytical outputs prove correctness.

    Notes
    -----
    CPython suspends tracing inside profile callbacks; see
    https://docs.python.org/3.12/library/sys.html#sys.call_tracing
    The actual observer statements 120-122 and arcs (121, -118)/(121, 122)
    execute but remain untraced. The assertions below verify the exact real
    score/collapse/mask C calls, differing Python overrides, numerical outputs
    and original profiler restoration. No backend is replaced and no callback
    is called directly to pad coverage. The missing callback locations remain
    an explicit instrumentation qualification for independent whole review,
    not an unqualified claim that this test file has 100 percent coverage.
    """
    scorer = PhaseQualityScorer(collapse_threshold=0.7, min_quality=0.4)
    states = [
        PhaseState(0.0, 1.0, amplitude, quality, "P", f"n{i}")
        for i, (quality, amplitude) in enumerate([(0.1, 2.0), (0.6, 1.0), (0.9, 3.0)])
    ]
    calls: list[str] = []

    def observe(frame: FrameType, event: str, argument: object) -> None:
        """Record the installed extension's bound calls without substitution."""
        owner = getattr(argument, "__self__", None)
        if event == "c_call" and isinstance(owner, PyPhaseQualityScorer):
            calls.append(cast("BuiltinMethodType", argument).__name__)

    previous = sys.getprofile()
    sys.setprofile(observe)
    try:
        assert scorer.score(states) == pytest.approx(3.5 / 6.0)
        assert scorer.detect_collapse(states, threshold=0.7)
        np.testing.assert_array_equal(
            scorer.downweight_mask(states, min_quality=0.4), [0.0, 0.6, 0.9]
        )
        assert not scorer.detect_collapse(states, threshold=0.5)
        np.testing.assert_array_equal(
            scorer.downweight_mask(states, min_quality=0.8), [0.0, 0.0, 0.9]
        )
    finally:
        sys.setprofile(previous)
    assert calls == ["score", "is_collapsed", "downweight_mask"]
    assert sys.getprofile() is previous


@pytest.mark.parametrize("amplitude", [1e308, float(np.finfo(np.float64).max)])
def test_direct_native_finite_weighting_and_refusal_recovery(amplitude: float) -> None:
    """Native sums preserve finite means and recover after source rejection.

    Parameters
    ----------
    amplitude : float
        Finite weight whose repeated unscaled total exceeds float64.
    """
    scorer = PyPhaseQualityScorer()
    qualities = np.array([0.8, 0.2], dtype=np.float64)
    amplitudes = np.full(2, amplitude, dtype=np.float64)
    qualities.setflags(write=False)
    amplitudes.setflags(write=False)
    assert scorer.score(qualities, amplitudes) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="quality"):
        scorer.score(["0.8", 0.2], amplitudes)
    assert scorer.score(qualities, amplitudes) == pytest.approx(0.5)
    assert scorer.score([0.8, 0.2], [amplitude]) == pytest.approx(0.8)
    assert scorer.score([0.8], []) == 0.0
    assert scorer.score([np.nan], [amplitude]) == 0.0
    np.testing.assert_array_equal(qualities, [0.8, 0.2])
    np.testing.assert_array_equal(amplitudes, [amplitude] * 2)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Real WebGPU phase integration contracts

"""Execute generated WGSL in Chromium; require explicit installed runtimes.

Select this ``native_runtime`` module with ``SPO_PLAYWRIGHT_PACKAGE`` pointing to
the installed Node package and ``SPO_WEBGPU_BROWSER`` to Chromium. Missing or
broken runtimes fail the selected test; no fabricated adapter or skip is used.

The browser driver emits a fixed JSON object after actual shader readback, so
the Python decoder's non-object guard is structurally unreachable without
replacing the owned driver. Node is installed on this host; its missing-runtime
branch remains unexecuted. Actual installed browser execution covers the nearest
runtime boundary. No PATH or JavaScript bridge is forged to exercise absence.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from benchmarks.upde_webgpu_benchmark import benchmark_webgpu_phase_wrapping
from scpn_phase_orchestrator.experimental.accelerators.upde import _engine_webgpu
from scpn_phase_orchestrator.upde import engine

pytestmark = pytest.mark.native_runtime


def test_browser_repetitions_require_a_positive_count() -> None:
    """Refuse a real browser request with no measured calls before allocation."""
    with pytest.raises(ValueError, match="repeats must be a positive integer"):
        benchmark_webgpu_phase_wrapping(
            Path(os.environ["SPO_PLAYWRIGHT_PACKAGE"]),
            Path(os.environ["SPO_WEBGPU_BROWSER"]),
            repeats=0,
        )


@pytest.fixture(scope="module")
def browser_evidence() -> dict[str, object]:
    """Return outputs from an actual compiled shader and generated ES module.

    Returns
    -------
    dict[str, object]
        Real browser evidence from one owned Chromium process.
    """
    return benchmark_webgpu_phase_wrapping(
        Path(os.environ["SPO_PLAYWRIGHT_PACKAGE"]),
        Path(os.environ["SPO_WEBGPU_BROWSER"]),
        repeats=2,
    )


def test_browser_torus_and_buffer_preservation(
    browser_evidence: dict[str, object],
) -> None:
    """Preserve actual interior bits and caller buffers for one and three passes.

    Parameters
    ----------
    browser_evidence : dict[str, object]
        Outputs from the real WGSL compute pipeline.
    """
    cases = cast("list[dict[str, object]]", browser_evidence["cases"])
    assert len(cases) == 2
    assert not _engine_webgpu.is_webgpu_runtime_available()
    period = float(cast("float", browser_evidence["period"]))
    for case in cases:
        output = np.asarray(case["output"], dtype=np.float32)
        np.testing.assert_array_equal(output[:5], np.zeros(5))
        assert not any(cast("list[bool]", case["negativeZero"]))
        assert output[5] == case["interior"]
        assert output[6] == pytest.approx(0.27, rel=0.0, abs=6e-8)
        assert np.all((output >= 0.0) & (output < period))
        assert case["inputBitsBefore"] == case["inputBitsAfter"]


def test_browser_coupling_matches_public_reference(
    browser_evidence: dict[str, object],
) -> None:
    """Exercise genuine coupling, phase lag, drive, and multiple substeps.

    Parameters
    ----------
    browser_evidence : dict[str, object]
        Outputs from the real generated JavaScript runner.
    """
    previous = engine.ACTIVE_BACKEND
    engine.ACTIVE_BACKEND = "python"
    try:
        expected = engine.upde_run(
            np.array([0.1, 0.7]),
            np.array([0.2, -0.3]),
            np.array([[0.0, 0.8], [0.5, 0.0]]),
            np.array([[0.0, 0.1], [-0.2, 0.0]]),
            0.15,
            0.4,
            0.01,
            3,
            method="euler",
            n_substeps=2,
        )
    finally:
        engine.ACTIVE_BACKEND = previous
    # WGSL f32 sin permits absolute error 2**-11 on [-pi, pi]:
    # https://www.w3.org/TR/WGSL/#floating-point-accuracy
    # For this bounded six-pass fixture, row sum <= .8, |drive| = .15,
    # and the sup-norm derivative Lipschitz constant is <= 2*.8+.15.
    # Bound per-pass arithmetic by 20 units of binary32 rounding error: the
    # phase-domain operations are < 1, while derivative terms (up to 1.15)
    # enter the phase update multiplied by h=.005. This exceeds the actual
    # rounded-operation count. Propagate it independently of the observation.
    assert np.all((expected > 0.0) & (expected < 1.0))
    rounding_unit = float(np.finfo(np.float32).eps) / 2.0
    bound = rounding_unit
    h = 0.01 / 2.0
    for _ in range(6):
        bound = (1.0 + h * 1.75) * bound + h * 0.95 * 2.0**-11
        bound += 20.0 * rounding_unit
    actual = np.asarray(browser_evidence["coupled"], dtype=np.float64)
    assert float(np.max(np.abs(actual - expected))) <= bound


def test_browser_float32_control_refusal_and_recovery(
    browser_evidence: dict[str, object],
) -> None:
    """Reject actual conversion overflow/underflow and recover on the same device.

    Parameters
    ----------
    browser_evidence : dict[str, object]
        Refusals and subsequent successful device execution.
    """
    refusals = cast("list[dict[str, object]]", browser_evidence["refusals"])
    assert len(refusals) == 4
    assert "finite in float32" in str(refusals[0]["error"])
    for refusal in refusals[1:]:
        assert "positive and finite in float32" in str(refusal["error"])
    np.testing.assert_array_equal(
        np.asarray(browser_evidence["recovered"], dtype=np.float32),
        np.asarray(browser_evidence["coupled"], dtype=np.float32),
    )


def test_browser_computed_overflow_refusal_and_recovery(
    browser_evidence: dict[str, object],
) -> None:
    """Refuse genuine finite-input GPU arithmetic overflow and recover.

    Parameters
    ----------
    browser_evidence : dict[str, object]
        Actual readback refusal, caller bits and valid same-device retry.
    """
    refusal = cast("dict[str, object]", browser_evidence["numericalRefusal"])
    assert "result[0] is not finite" in str(refusal["error"])
    assert refusal["inputBitsBefore"] == refusal["inputBitsAfter"]
    np.testing.assert_array_equal(
        np.asarray(browser_evidence["recovered"], dtype=np.float32),
        np.asarray(browser_evidence["coupled"], dtype=np.float32),
    )


def test_browser_destroyed_device_refuses_readback(
    browser_evidence: dict[str, object],
) -> None:
    """Exercise failed mapping after destroying the actual generated backend.

    Parameters
    ----------
    browser_evidence : dict[str, object]
        Device-loss refusal from the real browser compute runner.
    """
    refusal = cast("dict[str, object]", browser_evidence["deviceLossRefusal"])
    assert refusal["reason"] == "destroyed"
    assert refusal["name"] == "AbortError"

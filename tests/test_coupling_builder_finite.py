# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Finite public coupling construction contracts

"""Exercise finite construction and refused-call recovery through public APIs."""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.coupling.knm import (
    SCPN_CALIBRATION_ANCHORS,
    SCPN_LAYER_TIMESCALES,
    CouplingBuilder,
)
from scpn_phase_orchestrator.upde.engine import UPDEEngine


@pytest.mark.parametrize("strength", [0.0, math.ulp(0.0), 0.45, np.finfo(float).max])
@pytest.mark.parametrize("decay", [0.0, math.ulp(0.0), 0.3, np.finfo(float).max])
def test_generic_and_amplitude_extremes(strength: float, decay: float) -> None:
    """Finite coefficients produce finite matrices under strict caller policies.

    Parameters
    ----------
    strength : float
        Finite binary64 coupling strength, including the case-specific extremes.
    decay : float
        Finite non-negative exponential decay per layer separation.
    """
    builder = CouplingBuilder()
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        state = builder.build_with_amplitude(4, strength, decay, strength, decay)
    assert state.knm_r is not None
    for matrix in (state.knm, state.knm_r):
        assert np.all(np.isfinite(matrix))
        np.testing.assert_array_equal(matrix, matrix.T)
        np.testing.assert_array_equal(np.diag(matrix), 0.0)
        for i in range(4):
            for j in range(4):
                expected = (
                    0.0
                    if i == j
                    else float(strength) * math.exp(-float(decay) * abs(i - j))
                )
                assert matrix[i, j] == pytest.approx(expected, rel=3e-15, abs=0.0)
    np.testing.assert_array_equal(state.alpha, 0.0)


@pytest.mark.parametrize("strength", [math.ulp(0.0), 0.45, np.finfo(float).max])
@pytest.mark.parametrize("decay", [0.0, 0.3, np.finfo(float).max])
def test_scpn_extremes_preserve_anchors(strength: float, decay: float) -> None:
    """The bounded SCPN model keeps published anchors with finite extremes.

    Parameters
    ----------
    strength : float
        Finite binary64 coupling strength, including the case-specific extremes.
    decay : float
        Finite non-negative exponential decay per layer separation.
    """
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        state = CouplingBuilder().build_scpn_physics(strength, decay)
    assert np.all(np.isfinite(state.knm))
    np.testing.assert_array_equal(state.knm, state.knm.T)
    np.testing.assert_array_equal(np.diag(state.knm), 0.0)
    for (i, j), expected in SCPN_CALIBRATION_ANCHORS.items():
        assert state.knm[i - 1, j - 1] == expected
    assert state.knm[0, 15] >= 0.05
    assert state.knm[4, 6] >= 0.15


@pytest.mark.parametrize("value", [math.ulp(0.0), np.finfo(float).max])
def test_public_timescale_extremes(value: float) -> None:
    """Positive finite exported timescales need no overflowing frequency ratio.

    Parameters
    ----------
    value : float
        Positive exported layer-six timescale in seconds, at a finite extreme.
    """
    previous = SCPN_LAYER_TIMESCALES[6]
    try:
        SCPN_LAYER_TIMESCALES[6] = value
        with warnings.catch_warnings(), np.errstate(all="raise"):
            warnings.simplefilter("error")
            state = CouplingBuilder().build_scpn_physics()
        assert np.all(np.isfinite(state.knm))
        assert 0.1 <= state.knm[5, 6] <= 0.5
    finally:
        SCPN_LAYER_TIMESCALES[6] = previous


@pytest.mark.parametrize("field", ["base_strength", "decay_alpha"])
def test_unrepresentable_coefficients_refuse_and_recover(field: str) -> None:
    """A genuine arbitrary precision integer is refused before matrix allocation.

    Parameters
    ----------
    field : str
        Name of the native output or scalar control replaced by this case.
    """
    builder = CouplingBuilder()
    controls = {"base_strength": 0.5, "decay_alpha": 0.2}
    controls[field] = 10**400
    with pytest.raises(ValueError, match=field):
        builder.build(4, **controls)
    actual = builder.build(4, 0.5, 0.2)
    assert actual.knm[0, 1] == pytest.approx(0.5 * math.exp(-0.2))


@pytest.mark.parametrize("n", [1 << (np.dtype(np.uintp).itemsize * 4), 10**400])
def test_unrepresentable_matrix_refuses_before_allocation(n: int) -> None:
    """Only guaranteed platform-overflow sizes are probed; no OOM is induced.

    Parameters
    ----------
    n : int
        Layer count selected by this case, including guaranteed capacity overflow.
    """
    builder = CouplingBuilder()
    with pytest.raises(ValueError, match="matrix|n_layers"):
        builder.build(n, 0.5, 0.2)
    assert builder.build(1, 0.5, 0.2).knm[0, 0] == 0.0


@pytest.mark.parametrize("payload", [[], None, 1, "matrix", {"matrix": None}])
def test_handshake_root_refusal_preserves_snapshot(
    tmp_path: Path, payload: object
) -> None:
    """Real JSON root errors preserve the source and allow the next valid overlay.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated directory for the real handshake JSON document.
    payload : object
        Original invalid producer field or JSON root; no coercion precedes admission.
    """
    builder = CouplingBuilder()
    state = builder.build_with_amplitude(3, 0.5, 0.2, 0.1, 0.3)
    original = state.knm.copy()
    path = tmp_path / "handshakes.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="handshake"):
        builder.apply_handshakes(state, path)
    np.testing.assert_array_equal(state.knm, original)
    path.write_text('{"matrix":[]}', encoding="utf-8")
    recovered = builder.apply_handshakes(state, path)
    np.testing.assert_array_equal(recovered.knm, original)
    assert not np.shares_memory(recovered.knm, state.knm)
    assert not np.shares_memory(recovered.alpha, state.alpha)
    assert recovered.knm_r is state.knm_r


def test_constructed_snapshot_consumed_by_real_rk4_engine() -> None:
    """Public construction, template replacement and integration preserve inputs."""
    builder = CouplingBuilder()
    state = builder.build(4, 0.5, 0.2)
    template = state.knm.copy()
    replacement = builder.switch_template(state, "measured", {"measured": template})
    original = replacement.knm.copy()
    template.fill(0.0)
    np.testing.assert_array_equal(replacement.knm, original)

    phases = np.array([0.0, 0.5, 1.0, 1.5])
    before = phases.copy()
    engine = UPDEEngine(4, dt=0.01, method="rk4")
    output = engine.run(
        phases, np.ones(4), replacement.knm, 0.0, 0.0, replacement.alpha, n_steps=10
    )
    assert np.all(np.isfinite(output))
    assert not np.array_equal(output, before)
    np.testing.assert_array_equal(phases, before)
    np.testing.assert_array_equal(replacement.knm, original)


def test_wider_template_refusal_preserves_and_recovers() -> None:
    """A real extended-precision array cannot overflow into an accepted template."""
    builder = CouplingBuilder()
    state = builder.build(3, 0.5, 0.2)
    original = state.knm.copy()
    with np.errstate(over="ignore"):
        value = np.longdouble(np.finfo(np.float64).max) * np.longdouble(2)
    template = np.zeros((3, 3), dtype=np.longdouble)
    template[0, 1] = value
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="finite"):
            builder.switch_template(
                state, "wide", {"wide": cast(NDArray[np.float64], template)}
            )
    np.testing.assert_array_equal(state.knm, original)
    recovered = builder.switch_template(state, "valid", {"valid": original})
    np.testing.assert_array_equal(recovered.knm, original)
    assert not np.shares_memory(recovered.knm, original)

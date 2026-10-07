# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Phase-SINDy tests

"""Exercise SINDy refusal, failure-state and array-protocol contracts.

Malformed upstream payloads below are explicit negative ABI controls. They
never provide successful numerical evidence or simulate kernel installation.
All successful fits use the implementation selected by the actual environment.
"""

from __future__ import annotations

import importlib.util
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_phase_orchestrator.autotune.sindy as sindy_mod
from scpn_phase_orchestrator.autotune.sindy import PhaseSINDy

FloatArray = NDArray[np.float64]


def _phase_table(samples: int = 12) -> FloatArray:
    """Provide finite nondegenerate phases for negative upstream-result controls."""
    times = np.linspace(0.0, 1.0, samples, dtype=np.float64)
    return np.column_stack(
        (
            0.6 * times,
            0.8 * times + 0.2 * np.sin(times),
            0.4 * times + 0.1 * np.cos(times),
        )
    )


@pytest.mark.parametrize(
    ("threshold", "max_iter", "match"),
    [
        (True, 1, "threshold"),
        (np.bool_(True), 1, "threshold"),
        ("0.1", 1, "threshold"),
        (-0.1, 1, "threshold"),
        (np.inf, 1, "threshold"),
        (np.nan, 1, "threshold"),
        (0.1, True, "max_iter"),
        (0.1, np.bool_(True), "max_iter"),
        (0.1, 0, "max_iter"),
        (0.1, -1, "max_iter"),
        (0.1, 1.2, "max_iter"),
        (0.1, "2", "max_iter"),
    ],
)
def test_phase_sindy_constructor_rejects_invalid_controls(
    threshold: object,
    max_iter: object,
    match: str,
) -> None:
    """Reject nonnumeric, boolean, nonfinite and out-of-domain public controls."""
    with pytest.raises(ValueError, match=match):
        PhaseSINDy(cast(float, threshold), cast(int, max_iter))


@pytest.mark.parametrize(
    ("phases", "dt", "match"),
    [
        (_phase_table(), True, "dt"),
        (_phase_table(), np.bool_(True), "dt"),
        (_phase_table(), "0.1", "dt"),
        (_phase_table(), 0.0, "dt"),
        (_phase_table(), -0.1, "dt"),
        (_phase_table(), np.inf, "dt"),
        (_phase_table(), np.nan, "dt"),
        (np.asarray([[False], [True]], dtype=object), 0.1, "boolean"),
        (np.asarray([[False], [True]], dtype=np.bool_), 0.1, "boolean"),
        (np.asarray([[0.0], [np.bool_(True)]], dtype=object), 0.1, "boolean"),
        (np.asarray([[0.0 + 1.0j], [1.0 + 0.0j]]), 0.1, "finite 2D"),
        (np.asarray([["x"], ["y"]], dtype=object), 0.1, "finite 2D"),
        ([0.0, 1.0, 2.0], 0.1, "2D"),
        (np.zeros((2, 2, 1)), 0.1, "2D"),
        (np.asarray([[np.nan], [1.0]]), 0.1, "finite"),
        (np.asarray([[np.inf], [1.0]]), 0.1, "finite"),
        (np.empty((1, 0)), 0.1, "at least two time samples"),
        (np.empty((0, 2)), 0.1, "at least two time samples"),
        (np.empty((4, 0)), 0.1, "at least two time samples"),
        (np.ones((2, 3)), 0.1, "derivative sample"),
    ],
)
def test_phase_sindy_fit_rejects_invalid_inputs(
    phases: object, dt: object, match: str
) -> None:
    """Input coercion and dimensional admission remain fail-closed through fit."""
    with pytest.raises(ValueError, match=match):
        PhaseSINDy().fit(cast(FloatArray, phases), cast(float, dt))


class _DtypeDependentInput:
    """Supply inconsistent public array-protocol views as an adversarial input.

    The object view contains finite numbers, but the default view contains
    booleans. This tests the second admission guard without changing either
    production guard or backend selection.
    """

    def __array__(
        self, dtype: object = None, copy: bool | None = None
    ) -> NDArray[np.object_] | NDArray[np.bool_]:
        """Return the requested object view or the inconsistent boolean view."""
        if dtype is not None:
            return np.array([[0.0], [1.0]], dtype=object)
        return np.array([[False], [True]], dtype=np.bool_)


def test_phase_sindy_rejects_inconsistent_array_protocol_boolean_view() -> None:
    """Reject an actual user array-protocol object without bypassing the alias guard."""
    with pytest.raises(ValueError, match="boolean"):
        PhaseSINDy().fit(cast(FloatArray, _DtypeDependentInput()), 0.1)


def test_phase_sindy_accepts_numpy_numeric_controls_and_numeric_strings() -> None:
    """Real scalar aliases and convertible phase values use the public numeric path."""
    model = PhaseSINDy(cast(float, np.float64(0.0)), cast(int, np.int64(2)))
    phases = cast(FloatArray, np.array([["0.0"], ["0.2"], ["0.4"]]))
    np.testing.assert_allclose(model.fit(phases, 0.1), [[2.0]], rtol=0.0, atol=1e-12)


def test_phase_sindy_get_equations_requires_fit() -> None:
    """Equation export fails until a fit has produced coefficients."""
    with pytest.raises(RuntimeError, match="before fit"):
        PhaseSINDy().get_equations()


@pytest.mark.parametrize("case", ["non-numeric", "wrong", "non-finite"])
def test_phase_sindy_rejects_upstream_abi_fault_and_preserves_fit(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    """Inject only rejected ABI results; genuine prior coefficients remain intact.

    These negative controls exercise defensive output validation. They are
    explicitly not observations of successful native or SciPy computation.
    Backend presence is never changed, and every prior fit is a real call.
    """
    model = PhaseSINDy(threshold=0.0)
    before = model.fit(np.array([[0.0], [0.2], [0.4]]), 0.1)
    equations = model.get_equations()
    native = importlib.util.find_spec("spo_kernel") is not None
    count = 9 if native else 3
    payload: object
    if case == "non-numeric":
        payload = ["not-a-number"]
    elif case == "wrong":
        payload = np.zeros(4, dtype=np.float64)
    else:
        payload = np.full(count, np.nan, dtype=np.float64)
    if native:

        def rejected_native(*args: object, **kwargs: object) -> object:
            """Return a deliberately malformed ABI payload for rejection only."""
            return payload

        monkeypatch.setattr(sindy_mod, "_rust_sindy_fit", rejected_native)
    else:

        def rejected_lstsq(*args: object, **kwargs: object) -> tuple[object]:
            """Supply a malformed least-squares payload for rejection only."""
            return (payload,)

        monkeypatch.setattr(sindy_mod, "lstsq", rejected_lstsq)
    with pytest.raises(ValueError, match=case):
        model.fit(_phase_table(), 0.1)
    np.testing.assert_array_equal(model.coefficients, before)
    assert model.get_equations() == equations


@pytest.mark.parametrize(
    ("phases", "dt", "match"),
    [
        (np.array([[0.0], [1.0]]), 1e-320, "derivatives"),
        (np.tile(np.array([-1e308, 1e308]), (3, 1)), 1.0, "feature library"),
        (np.array([[-1e308], [1e308]]), 1.0, "derivatives"),
        (
            np.array([[0.0, 1e-8], [0.1, 0.1 + 2e-8], [0.1, 0.1 + 3e-8]]),
            1e-309,
            "non-finite coefficients",
        ),
    ],
)
def test_phase_sindy_derived_overflow_preserves_previous_fit(
    phases: FloatArray,
    dt: float,
    match: str,
) -> None:
    """Finite raw inputs that overflow derived arithmetic cannot corrupt prior fit."""
    model = PhaseSINDy(threshold=0.0)
    before = model.fit(np.array([[0.0], [0.2], [0.4]]), 0.1)
    equations = model.get_equations()
    with pytest.raises(ValueError, match=match):
        model.fit(phases, dt)
    np.testing.assert_array_equal(model.coefficients, before)
    assert model.get_equations() == equations


@pytest.mark.parametrize(
    "phases", [np.empty((0, 1)), np.zeros((1, 1)), np.zeros((2, 2)), np.empty((4, 0))]
)
def test_phase_sindy_underdetermined_refit_clears_equation_state(
    phases: FloatArray,
) -> None:
    """Preserve the documented clearing contract for insufficient trajectories."""
    model = PhaseSINDy(threshold=0.0)
    model.fit(np.array([[0.0], [0.2], [0.4]]), 0.1)
    with pytest.raises(ValueError, match="at least"):
        model.fit(phases, 0.1)
    assert model.coefficients == []
    assert model.feature_names == []
    with pytest.raises(RuntimeError, match="before fit"):
        model.get_equations()

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Phase-SINDy tests

from __future__ import annotations

import builtins
import importlib
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_phase_orchestrator.autotune.sindy as sindy_mod
from scpn_phase_orchestrator.autotune.sindy import PhaseSINDy

FloatArray = NDArray[np.float64]


def _phase_table(samples: int = 12) -> FloatArray:
    times = np.linspace(0.0, 1.0, samples, dtype=np.float64)
    return np.column_stack(
        [
            0.6 * times,
            0.8 * times + 0.2 * np.sin(times),
            0.4 * times + 0.1 * np.cos(times),
        ]
    )


def test_phase_sindy_fits_python_path_and_formats_equations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sindy_mod, "_HAS_RUST", False)
    model = PhaseSINDy(threshold=0.0, max_iter=2)

    coefficients = model.fit(_phase_table(), 0.1)
    equations = model.get_equations()

    assert len(coefficients) == 3
    assert len(equations) == 3
    assert all(coefficient.shape == (3,) for coefficient in coefficients)
    assert all(equation.startswith("d(theta_") for equation in equations)


def test_phase_sindy_import_falls_back_when_rust_extension_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def blocked_import(
        name: str,
        globals_: object = None,
        locals_: object = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "spo_kernel":
            raise ImportError("blocked for test")
        return original_import(name, globals_, locals_, fromlist, level)

    with monkeypatch.context() as context:
        context.setattr(builtins, "__import__", blocked_import)
        reloaded = importlib.reload(sindy_mod)
        assert reloaded._HAS_RUST is False

    importlib.reload(sindy_mod)


@pytest.mark.parametrize(
    ("threshold", "max_iter", "match"),
    [
        (True, 1, "threshold"),
        (-0.1, 1, "threshold"),
        (np.inf, 1, "threshold"),
        (0.1, True, "max_iter"),
        (0.1, 0, "max_iter"),
        (0.1, 1.2, "max_iter"),
    ],
)
def test_phase_sindy_constructor_rejects_invalid_controls(
    threshold: object,
    max_iter: object,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        PhaseSINDy(cast(float, threshold), cast(int, max_iter))


@pytest.mark.parametrize(
    ("phases", "dt", "match"),
    [
        (_phase_table(), True, "dt"),
        (_phase_table(), 0.0, "dt"),
        (_phase_table(), np.inf, "dt"),
        (np.asarray([[False], [True]], dtype=object), 0.1, "boolean"),
        (np.asarray([[0.0 + 1.0j], [1.0 + 0.0j]]), 0.1, "finite 2D"),
        (np.asarray([["x"], ["y"]], dtype=object), 0.1, "finite 2D"),
        ([0.0, 1.0, 2.0], 0.1, "2D"),
        (np.asarray([[np.nan], [1.0]], dtype=np.float64), 0.1, "finite"),
        (np.empty((1, 0), dtype=np.float64), 0.1, "at least two time samples"),
        (np.ones((2, 3), dtype=np.float64), 0.1, "derivative sample"),
    ],
)
def test_phase_sindy_fit_rejects_invalid_inputs(
    phases: object,
    dt: object,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        PhaseSINDy().fit(cast(FloatArray, phases), cast(float, dt))


def test_phase_sindy_rejects_raw_boolean_arrays_after_alias_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sindy_mod, "_HAS_RUST", False)
    monkeypatch.setattr(sindy_mod, "_contains_boolean_alias", lambda _value: False)

    with pytest.raises(ValueError, match="boolean"):
        PhaseSINDy().fit(np.asarray([[False], [True]], dtype=np.bool_), 0.1)


def test_phase_sindy_get_equations_requires_fit() -> None:
    with pytest.raises(RuntimeError, match="before fit"):
        PhaseSINDy().get_equations()


def test_phase_sindy_rejects_bad_lstsq_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def wrong_size(*_args: object, **_kwargs: object) -> tuple[FloatArray]:
        return (np.asarray([1.0, 2.0], dtype=np.float64),)

    monkeypatch.setattr(sindy_mod, "_HAS_RUST", False)
    monkeypatch.setattr(sindy_mod, "lstsq", wrong_size)

    with pytest.raises(ValueError, match="wrong coefficient count"):
        PhaseSINDy().fit(_phase_table(), 0.1)


def test_phase_sindy_rejects_non_numeric_lstsq_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def non_numeric(*_args: object, **_kwargs: object) -> tuple[list[str]]:
        return (["not-a-number", "still-not"],)

    monkeypatch.setattr(sindy_mod, "_HAS_RUST", False)
    monkeypatch.setattr(sindy_mod, "lstsq", non_numeric)

    with pytest.raises(ValueError, match="non-numeric coefficients"):
        PhaseSINDy().fit(_phase_table(), 0.1)


def test_phase_sindy_rejects_non_finite_lstsq_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def non_finite(*_args: object, **_kwargs: object) -> tuple[FloatArray]:
        return (np.full(3, np.inf, dtype=np.float64),)

    monkeypatch.setattr(sindy_mod, "_HAS_RUST", False)
    monkeypatch.setattr(sindy_mod, "lstsq", non_finite)

    with pytest.raises(ValueError, match="non-finite coefficients"):
        PhaseSINDy().fit(_phase_table(), 0.1)


def test_phase_sindy_maps_valid_rust_coefficients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def rust_fit(
        _flat: FloatArray,
        _nodes: int,
        _samples: int,
        _dt: float,
        _threshold: float,
        _max_iter: int,
    ) -> FloatArray:
        return np.asarray(
            [
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
            ],
            dtype=np.float64,
        )

    monkeypatch.setattr(sindy_mod, "_HAS_RUST", True)
    monkeypatch.setattr(sindy_mod, "_rust_sindy_fit", rust_fit)

    coefficients = PhaseSINDy().fit(_phase_table(), 0.1)

    np.testing.assert_array_equal(coefficients[0], np.asarray([0.1, 0.2, 0.3]))
    np.testing.assert_array_equal(coefficients[1], np.asarray([0.5, 0.4, 0.6]))
    assert PhaseSINDy().threshold == pytest.approx(0.05)


@pytest.mark.parametrize(
    ("rust_result", "match"),
    [
        (["not-number"], "non-numeric"),
        ([1.0, 2.0], "wrong number"),
        ([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], "non-finite"),
    ],
)
def test_phase_sindy_rejects_invalid_rust_coefficients(
    monkeypatch: pytest.MonkeyPatch,
    rust_result: object,
    match: str,
) -> None:
    def rust_fit(
        _flat: FloatArray,
        _nodes: int,
        _samples: int,
        _dt: float,
        _threshold: float,
        _max_iter: int,
    ) -> object:
        return rust_result

    monkeypatch.setattr(sindy_mod, "_HAS_RUST", True)
    monkeypatch.setattr(sindy_mod, "_rust_sindy_fit", rust_fit)

    with pytest.raises(ValueError, match=match):
        PhaseSINDy().fit(_phase_table(), 0.1)

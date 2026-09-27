# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Native named-call compatibility

"""Exercise positional, keyword, mixed and refused calls on the installed kernel."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import TypeAlias, cast

import numpy as np
import pytest
import spo_kernel
from numpy.typing import NDArray

NativeCall: TypeAlias = Callable[..., object]

PARAMETERS: dict[str, tuple[str, ...]] = {
    "swarmalator_run_rust": (
        "pos_init",
        "phases_init",
        "omegas",
        "n",
        "dim",
        "dt",
        "a",
        "b",
        "j",
        "k",
        "n_steps",
    ),
    "delayed_kuramoto_run_rust": (
        "phases_init",
        "omegas",
        "knm_flat",
        "alpha_flat",
        "n",
        "zeta",
        "psi",
        "dt",
        "delay_steps",
        "n_steps",
    ),
    "inertial_step_rust": (
        "theta",
        "omega_dot",
        "power",
        "knm_flat",
        "inertia",
        "damping",
        "n",
        "dt",
    ),
    "inertial_run_rust": (
        "theta",
        "omega_dot",
        "power",
        "knm_flat",
        "inertia",
        "damping",
        "n",
        "dt",
        "n_steps",
    ),
    "basin_stability_rust": (
        "omegas",
        "knm_flat",
        "alpha_flat",
        "n",
        "dt",
        "n_transient",
        "n_measure",
        "n_samples",
        "r_threshold",
        "seed",
    ),
    "steady_state_r_rust": (
        "phases_init",
        "omegas",
        "knm_flat",
        "alpha_flat",
        "n",
        "k_scale",
        "dt",
        "n_transient",
        "n_measure",
    ),
    "trace_sync_transition_rust": (
        "omegas",
        "knm_flat",
        "alpha_flat",
        "n",
        "phases_init",
        "k_min",
        "k_max",
        "n_points",
        "dt",
        "n_transient",
        "n_measure",
    ),
    "find_critical_coupling_bif_rust": (
        "omegas",
        "knm_flat",
        "alpha_flat",
        "n",
        "phases_init",
        "dt",
        "n_transient",
        "n_measure",
        "tol",
    ),
    "torus_run_rust": (
        "phases",
        "omegas",
        "knm",
        "alpha",
        "_n",
        "zeta",
        "psi",
        "dt",
        "n_steps",
    ),
    "splitting_run_rust": (
        "phases",
        "omegas",
        "knm",
        "alpha",
        "_n",
        "zeta",
        "psi",
        "dt",
        "n_steps",
    ),
    "hypergraph_run_rust": (
        "phases",
        "omegas",
        "n",
        "edge_nodes",
        "edge_offsets",
        "edge_strengths",
        "pairwise_knm",
        "alpha",
        "zeta",
        "psi",
        "dt",
        "n_steps",
    ),
    "simplicial_run_rust": (
        "phases",
        "omegas",
        "knm",
        "alpha",
        "n",
        "zeta",
        "psi",
        "sigma2",
        "dt",
        "n_steps",
    ),
    "PyHypergraphStepper.step": (
        "phases",
        "omegas",
        "edges",
        "knm",
        "alpha",
        "zeta",
        "psi",
    ),
    "PyHypergraphStepper.run": (
        "phases",
        "omegas",
        "edges",
        "knm",
        "alpha",
        "zeta",
        "psi",
        "n_steps",
    ),
    "PySplittingStepper.run": (
        "phases",
        "omegas",
        "knm",
        "alpha",
        "zeta",
        "psi",
        "n_steps",
    ),
    "PySwarmalatorStepper.step": ("pos", "phases", "omegas", "a", "b", "j", "k"),
    "PySimplicialStepper.step": (
        "phases",
        "omegas",
        "knm",
        "alpha",
        "zeta",
        "psi",
        "sigma2",
    ),
    "PySimplicialStepper.run": (
        "phases",
        "omegas",
        "knm",
        "alpha",
        "zeta",
        "psi",
        "sigma2",
        "n_steps",
    ),
}


def _call(name: str) -> NativeCall:
    """Resolve a real free function or a fresh stateful stepper method."""
    if "." not in name:
        return cast(NativeCall, getattr(spo_kernel, name))
    owner, method = name.split(".")
    constructor = cast(NativeCall, getattr(spo_kernel, owner))
    instance = constructor(3, 2) if owner == "PySwarmalatorStepper" else constructor(3)
    return cast(NativeCall, getattr(instance, method))


def _values(name: str) -> dict[str, object]:
    """Build bounded deterministic numerical inputs for each native model."""
    phases = np.arange(3, dtype=np.float64) * 0.1
    knm = np.zeros(9)
    position = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    values: dict[str, object] = {
        "phases": phases,
        "phases_init": phases,
        "theta": phases,
        "omega_dot": np.zeros(3),
        "omegas": np.ones(3),
        "power": np.ones(3),
        "inertia": np.ones(3),
        "damping": np.ones(3),
        "knm": knm,
        "knm_flat": knm,
        "pairwise_knm": knm,
        "alpha": knm,
        "alpha_flat": knm,
        "n": 3,
        "_n": 3,
        "dim": 2,
        "dt": 0.01,
        "n_steps": 2,
        "delay_steps": 1,
        "n_transient": 2,
        "n_measure": 2,
        "n_samples": 2,
        "r_threshold": 0.2,
        "seed": 2026,
        "k_scale": 1.0,
        "k_min": 0.0,
        "k_max": 1.0,
        "n_points": 2,
        "tol": 0.1,
        "zeta": 0.0,
        "psi": 0.0,
        "sigma2": 0.1,
        "a": 1.0,
        "b": 1.0,
        "j": 0.1,
        "k": 0.1,
        "pos": position,
        "pos_init": position,
        "edges": [([0, 1, 2], 0.1)],
        "edge_nodes": np.array([0, 1, 2], dtype=np.int64),
        "edge_offsets": np.array([0], dtype=np.int64),
        "edge_strengths": np.array([0.1]),
    }
    return {key: values[key] for key in PARAMETERS[name]}


def _equivalent(actual: object, expected: object) -> None:
    """Compare returned numerical state recursively without backend substitution."""
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        np.testing.assert_allclose(
            cast(NDArray[np.float64], actual),
            cast(NDArray[np.float64], expected),
            rtol=0,
            atol=1e-12,
        )
    elif isinstance(expected, (tuple, list)):
        assert isinstance(actual, type(expected))
        left = cast(tuple[object, ...] | list[object], actual)
        right = cast(tuple[object, ...] | list[object], expected)
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _equivalent(a, b)
    elif isinstance(expected, float):
        assert isinstance(actual, (int, float))
        if np.isnan(expected):
            assert np.isnan(actual)
        else:
            assert actual == pytest.approx(expected, rel=0, abs=1e-12)
    else:
        assert actual == expected


@pytest.mark.parametrize("name", PARAMETERS)
def test_positional_keyword_and_mixed_parity(name: str) -> None:
    """Every legacy numerical call accepts all three binding forms identically."""
    kwargs = _values(name)
    ordered = list(kwargs.values())
    expected = _call(name)(*ordered)
    _equivalent(_call(name)(**kwargs), expected)
    first = PARAMETERS[name][0]
    _equivalent(
        _call(name)(ordered[0], **{k: v for k, v in kwargs.items() if k != first}),
        expected,
    )
    assert tuple(inspect.signature(_call(name)).parameters) == PARAMETERS[name]


@pytest.mark.parametrize("name", PARAMETERS)
@pytest.mark.parametrize(
    "failure", ["missing", "duplicate", "unknown", "too_many", "dtype"]
)
def test_invalid_call_refusal(name: str, failure: str) -> None:
    """Ambiguous, incomplete and ill-typed calls fail before numerical dispatch."""
    kwargs = _values(name)
    ordered = list(kwargs.values())
    first = PARAMETERS[name][0]
    with pytest.raises(TypeError):
        if failure == "missing":
            _call(name)(**{k: v for k, v in kwargs.items() if k != first})
        elif failure == "duplicate":
            _call(name)(*ordered, **{first: ordered[0]})
        elif failure == "unknown":
            _call(name)(**kwargs, unknown_argument=1)
        elif failure == "too_many":
            _call(name)(*ordered, 0)
        else:
            kwargs[first] = np.zeros(3, dtype=np.int64)
            _call(name)(**kwargs)

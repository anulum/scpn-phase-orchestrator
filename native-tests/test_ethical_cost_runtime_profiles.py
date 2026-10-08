# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Original ethical-cost FFI runtime contracts

"""Exercise compiled argument extraction and actual unavailable installations."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
from importlib import import_module
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from benchmarks.ethical_cost_benchmark import validate_profile_python
from scpn_phase_orchestrator.ssgf.ethical import EthicalKernel, FloatArray

pytestmark = pytest.mark.native_runtime


def _native() -> EthicalKernel:
    """Require the original compiled callable before an actual numerical call."""
    function = cast(
        "EthicalKernel", vars(import_module("spo_kernel"))["compute_ethical_cost_rust"]
    )
    assert inspect.isbuiltin(function)
    return function


@pytest.mark.parametrize("index", range(8))
@pytest.mark.parametrize(
    "alias", [True, np.bool_(False), "0.5", 1j, np.timedelta64(1, "s")]
)
def test_native_parameter_aliases_are_refused(index: int, alias: object) -> None:
    """All original scalar slots refuse aliases before numerical extraction."""
    parameters = [0.4, 0.3, 0.2, 0.1, 1.0, 0.2, 0.1, 5.0]
    parameters[index] = cast("float", alias)
    with pytest.raises(ValueError, match="real numbers"):
        _native()(np.zeros(2), np.zeros(4), 2, *parameters)


@pytest.mark.parametrize(
    "alias", [True, np.bool_(False), "2", 2.0, np.timedelta64(2, "s")]
)
def test_native_dimension_requires_original_integer(alias: object) -> None:
    """Dimension metadata cannot arrive through boolean, text or temporal coercion."""
    with pytest.raises(ValueError, match="non-boolean integer"):
        _native()(
            np.zeros(2),
            np.zeros(4),
            cast("int", alias),
            0.4,
            0.3,
            0.2,
            0.1,
            1.0,
            0.2,
            0.1,
            5.0,
        )


@pytest.mark.parametrize("count", [-1, 2**64])
def test_native_dimension_outside_usize_is_refused(count: int) -> None:
    """Original integral metadata outside the native range cannot index buffers."""
    with pytest.raises(OverflowError):
        _native()(
            np.zeros(0), np.zeros(0), count, 0.4, 0.3, 0.2, 0.1, 1.0, 0.2, 0.1, 5.0
        )


@pytest.mark.parametrize("size", [0, 1, 3, 5])
def test_native_flattened_cardinality_is_refused(size: int) -> None:
    """The original FFI refuses every nonmatching flattened matrix cardinality."""
    with pytest.raises(ValueError, match="dimensions must match"):
        _native()(
            np.zeros(2), np.zeros(size), 2, 0.4, 0.3, 0.2, 0.1, 1.0, 0.2, 0.1, 5.0
        )


def test_native_strided_buffer_refuses_and_public_call_recovers() -> None:
    """Direct slices refuse strides; the public adapter copies the same real views."""
    from scpn_phase_orchestrator.ssgf.ethical import compute_ethical_cost

    flattened = np.array([0.0, 99.0, 0.5, 98.0, 0.5, 97.0, 0.0, 96.0])[::2]
    assert not flattened.flags.c_contiguous
    with pytest.raises(ValueError):
        _native()(np.zeros(2), flattened, 2, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0)
    actual = compute_ethical_cost(
        np.zeros(2),
        flattened.reshape(2, 2),
        alpha_R=0.0,
        beta_K=1.0,
        gamma_Q=0.0,
        nu_S=0.0,
        kappa=0.0,
        R_min=0.0,
        connectivity_min=0.0,
    )
    assert actual.J_sec == pytest.approx(0.5, rel=1e-14)
    assert actual.c15_sec == pytest.approx(0.5, rel=1e-14)


@pytest.mark.parametrize("argument", ["phases", "knm"])
def test_native_unaligned_buffer_is_refused_before_borrowed_access(
    argument: str,
) -> None:
    """The original NumPy boundary refuses a real contiguous but unaligned array."""
    size = 2 if argument == "phases" else 4
    values = np.ndarray(
        (size,), dtype=np.float64, buffer=bytearray(size * 8 + 1), offset=1
    )
    assert values.flags.c_contiguous and not values.flags.aligned
    phases = values if argument == "phases" else np.zeros(2)
    matrix = values if argument == "knm" else np.zeros(4)
    with pytest.raises(ValueError):
        _native()(phases, matrix, 2, 0.4, 0.3, 0.2, 0.1, 1.0, 0.2, 0.1, 5.0)
    from scpn_phase_orchestrator.ssgf.ethical import compute_ethical_cost

    actual = compute_ethical_cost(phases, matrix.reshape(2, 2))
    assert actual.J_sec == pytest.approx(0.4, abs=1e-14)
    assert actual.phi_ethics == pytest.approx(0.01, abs=1e-14)
    assert actual.c15_sec == pytest.approx(0.61, abs=1e-14)
    assert actual.constraints_violated == 1


@pytest.mark.parametrize("name", ["phases", "knm"])
@pytest.mark.parametrize(
    "alias", [np.array([True, False]), np.array(["0", "1"]), np.array([0j, 1j])]
)
def test_native_typed_arrays_refuse_alias_dtypes(name: str, alias: object) -> None:
    """Borrowed float64 arrays refuse original boolean, text and complex data."""
    phases, matrix = np.zeros(2), np.zeros(4)
    if name == "phases":
        phases = cast("FloatArray", alias)
    else:
        matrix = cast("FloatArray", alias)
    with pytest.raises(TypeError):
        _native()(phases, matrix, 2, 0.4, 0.3, 0.2, 0.1, 1.0, 0.2, 0.1, 5.0)


def test_actual_absent_kernel_refuses_direct_owner_and_default_recovers() -> None:
    """An actual absent kernel refuses direct import and recovers through Python."""
    raw = os.environ.get("SPO_ETHICAL_PYTHON_PROFILE")
    assert raw is not None
    program = r"""
import hashlib, importlib.util, json
from pathlib import Path
import numpy as np
from scpn_phase_orchestrator.ssgf import ethical
assert importlib.util.find_spec('spo_kernel') is None
try:
    from spo_kernel import compute_ethical_cost_rust
except ModuleNotFoundError:
    missing = True
else:
    raise AssertionError('a genuinely absent owner is required')
cost = ethical.compute_ethical_cost(np.zeros(2), np.full((2, 2), -1e-16),
    alpha_R=0., beta_K=0., gamma_Q=1., nu_S=0., kappa=0.,
    R_min=0., connectivity_min=0.)
print(json.dumps(dict(missing=missing, values=[cost.J_sec, cost.phi_ethics,
    cost.c15_sec, cost.constraints_violated],
    source_sha256=hashlib.sha256(Path(ethical.__file__).read_bytes()).hexdigest())))
"""
    executable = validate_profile_python(Path(raw))
    result = subprocess.run(  # noqa: S603 - verified original Python binary; explicit profile, no shell.
        [str(executable), "-I", "-B", "-c", program],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    observed = cast("dict[str, object]", json.loads(result.stdout))
    assert observed["missing"] is True
    assert observed["values"] == [2.0, 0.0, -1.0, 0]
    module = import_module("scpn_phase_orchestrator.ssgf.ethical")
    source = module.__file__
    assert source is not None
    assert (
        observed["source_sha256"]
        == hashlib.sha256(Path(source).read_bytes()).hexdigest()
    )


@pytest.mark.parametrize("index", range(8))
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_native_empty_input_still_refuses_nonfinite_parameters(
    index: int, value: float
) -> None:
    """Typed native scalars must be finite before the empty numerical identity."""
    parameters = [0.4, 0.3, 0.2, 0.1, 1.0, 0.2, 0.1, 5.0]
    parameters[index] = value
    with pytest.raises(ValueError, match="must be finite"):
        _native()(np.zeros(0), np.zeros(0), 0, *parameters)
    assert _native()(
        np.zeros(0), np.zeros(0), 0, 0.4, 0.3, 0.2, 0.1, 1.0, 0.2, 0.1, 5.0
    ) == (0.0, 0.0, 1.0, 0)


def test_native_usize_square_overflow_is_refused_before_buffer_access() -> None:
    """A representable original native count cannot overflow its matrix cardinality."""
    maximum = int(np.iinfo(np.uintp).max)
    with pytest.raises(ValueError, match="dimension overflow"):
        _native()(
            np.zeros(0), np.zeros(0), maximum, 0.4, 0.3, 0.2, 0.1, 1.0, 0.2, 0.1, 5.0
        )


def test_native_finite_scalar_difference_overflow_is_refused() -> None:
    """Finite positive coupling and a negative bound cannot overflow a residual."""
    with pytest.raises(ValueError, match="constraint arithmetic"):
        _native()(
            np.zeros(1), np.array([1e308]), 1, 0.4, 0.3, 0.2, 0.1, 1.0, 0.2, 0.1, -1e308
        )


def test_native_finite_degrees_refuse_unrepresentable_connectivity() -> None:
    """The actual compiled core refuses lambda2=3*7e307 despite finite degrees."""
    matrix = np.full((3, 3), -7e307)
    np.fill_diagonal(matrix, 0.0)
    with pytest.raises(ValueError, match="eigenvalue arithmetic"):
        _native()(
            np.zeros(3), matrix.ravel(), 3, 0.4, 0.3, 0.2, 0.1, 1.0, 0.2, 0.1, 5.0
        )

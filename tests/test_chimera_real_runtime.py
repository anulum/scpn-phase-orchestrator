# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Public owner chimera numerical regressions

"""Exercise named public owners with analytic graphs and finite extreme phases."""

import cProfile
import hashlib
import json
import os
import subprocess
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor.chimera import (
    detect_chimera,
    local_order_parameter,
)


@pytest.mark.parametrize(
    "backend",
    [
        "python",
        *[
            pytest.param(owner, marks=pytest.mark.native_runtime)
            for owner in ("rust", "go", "julia", "mojo")
        ],
    ],
)
def test_admitted_self_residue_is_not_a_neighbour(backend: str) -> None:
    """An isolated oscillator remains incoherent at the diagonal tolerance."""
    p, k = np.array([2.0]), np.array([[1e-16]])
    np.testing.assert_array_equal(local_order_parameter(p, k, backend=backend), [0.0])
    state = detect_chimera(p, k, backend=backend)
    assert state.coherent_indices == []
    assert state.incoherent_indices == [0]
    assert state.chimera_index == 0.0


@pytest.mark.parametrize(
    "backend",
    [
        "python",
        *[
            pytest.param(owner, marks=pytest.mark.native_runtime)
            for owner in ("rust", "go", "julia", "mojo")
        ],
    ],
)
@pytest.mark.parametrize("connected", [False, True])
def test_extreme_finite_angles_need_no_pairwise_subtraction(
    backend: str, connected: bool
) -> None:
    """Both disconnected and single-neighbour graphs have exact analytic results."""
    p = np.array([1e308, -1e308])
    k = np.array([[0.0, float(connected)], [float(connected), 0.0]])
    with np.errstate(over="raise", invalid="raise"):
        values = local_order_parameter(p, k, backend=backend)
    np.testing.assert_allclose(values, [float(connected)] * 2, atol=1e-12, rtol=0)


@pytest.mark.parametrize(
    "backend",
    [
        "python",
        *[
            pytest.param(owner, marks=pytest.mark.native_runtime)
            for owner in ("rust", "go", "julia", "mojo")
        ],
    ],
)
def test_directed_positive_edges_are_unweighted_and_exclude_self(backend: str) -> None:
    """Very different positive weights count equally; negative edges never count."""
    p = np.array([0.0, 0.0, np.pi])
    k = np.array([[1e-16, 1e-300, 1e300], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        local_order_parameter(p, k, backend=backend),
        [0.0, 0.0, 1.0],
        atol=1e-12,
        rtol=0,
    )


FloatArray = NDArray[np.float64]


@pytest.mark.parametrize("backend", ["invalid", "", "Python"])
def test_unknown_owner_is_refused_even_for_empty_input(backend: str) -> None:
    """An empty identity cannot qualify an unsupported owner name."""
    with pytest.raises(ValueError, match="backend"):
        local_order_parameter(np.zeros(0), np.zeros((0, 0)), backend=backend)
    with pytest.raises(ValueError, match="backend"):
        detect_chimera(np.zeros(0), np.zeros((0, 0)), backend=backend)


def installed_absent_probe() -> dict[str, object]:
    """Measure the actual current installed Python-only profile in a child process.

    Returns
    -------
    dict[str, object]
        Original public availability, numerical results and missing-owner errors.

    Raises
    ------
    AssertionError
        If the required profile is missing, has an extension, resolves checkout
        sources, differs from the current monitor bytes, or fails the probe.
    """
    raw = os.environ.get("SPO_CHIMERA_ABSENT_PYTHON")
    assert raw is not None, (
        "SPO_CHIMERA_ABSENT_PYTHON must name a qualified installed profile"
    )
    python = Path(raw)
    assert python.is_file()
    script = """
import os
if os.environ.get("COVERAGE_PROCESS_START"):
    from coverage import process_startup
    process_startup()
import hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
from scpn_phase_orchestrator.monitor import chimera as c
assert importlib.util.find_spec("spo_kernel") is None
p = np.array([1e308, -1e308])
k = np.zeros((2, 2))
with np.errstate(over="raise", invalid="raise"):
    local = c.local_order_parameter(p, k)
state = c.detect_chimera(p, k)
missing = []
for owner in ("rust", "mojo", "julia", "go"):
    try:
        c.local_order_parameter(np.zeros(0), np.zeros((0, 0)), backend=owner)
    except ImportError:
        missing.append(owner)
    else:
        raise AssertionError("explicit missing owner silently succeeded: " + owner)
print(json.dumps(dict(prefix=sys.prefix, module=c.__file__,
    source_sha256=hashlib.sha256(Path(c.__file__).read_bytes()).hexdigest(),
    active=c.ACTIVE_BACKEND, available=c.AVAILABLE_BACKENDS,
    local=local.tolist(), incoherent=state.incoherent_indices,
    missing=missing)))
"""
    result = subprocess.run(
        [str(python), "-I", "-B", "-c", script],
        cwd=python.parent.parent,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    payload: object = json.loads(result.stdout)
    assert isinstance(payload, dict)
    record = cast("dict[str, object]", payload)
    from scpn_phase_orchestrator.monitor import chimera

    assert (
        record["source_sha256"]
        == hashlib.sha256(Path(chimera.__file__).read_bytes()).hexdigest()
    )
    assert Path(str(record["module"])).is_relative_to(python.parent.parent)
    assert record["active"] == "python" and record["available"] == ["python"]
    return record


@pytest.mark.parametrize("api", [local_order_parameter, detect_chimera])
def test_scalar_phase_is_not_promoted_to_a_population(
    api: object,
) -> None:
    """Dimension-preserving source conversion refuses a zero-dimensional phase."""
    function = cast("Callable[[NDArray[np.float64], NDArray[np.float64]], object]", api)
    with pytest.raises(ValueError, match="one-dimensional"):
        function(np.array(0.0), np.zeros((1, 1)))


def test_real_sequences_preserve_source_alias_refusal() -> None:
    """A Python sequence cannot hide a boolean through NumPy float promotion."""
    with pytest.raises(ValueError, match="boolean"):
        local_order_parameter(
            cast("NDArray[np.float64]", [0.0, True]),
            np.zeros((2, 2)),
            backend="python",
        )
    values = local_order_parameter(
        cast("NDArray[np.float64]", [0, 1.0]),
        cast("NDArray[np.float64]", [[0, 1], [1, 0]]),
        backend="python",
    )
    np.testing.assert_allclose(values, [1.0, 1.0], atol=1e-12, rtol=0)


def test_real_object_integer_overflow_is_a_measurement_refusal() -> None:
    """Plain numeric objects still must be representable in the Float64 domain."""
    with pytest.raises(ValueError, match="finite"):
        local_order_parameter(
            cast("NDArray[np.float64]", np.array([2**2048], dtype=object)),
            np.zeros((1, 1)),
            backend="python",
        )


def test_go_count_range_precedes_impossible_buffer_requirements() -> None:
    """The direct API refuses oversized C.int metadata using real empty buffers."""
    from scpn_phase_orchestrator.experimental.accelerators.monitor._chimera_go import (
        local_order_parameter_go,
    )

    with pytest.raises(ValueError, match="backend integer range"):
        local_order_parameter_go(np.zeros(0), np.zeros(0), 2**31)


@pytest.mark.native_runtime
@pytest.mark.parametrize("owner", ["go", "julia", "mojo"])
def test_legacy_direct_surface_preserves_empty_and_nonempty_contracts(
    owner: str,
) -> None:
    """Call compatibility imports with actual empty and coupled graphs."""
    module = import_module("scpn_phase_orchestrator.monitor._chimera_" + owner)
    function = cast(
        "Callable[[FloatArray, FloatArray, int], FloatArray]",
        vars(module)["local_order_parameter_" + owner],
    )
    assert function(np.zeros(0), np.zeros(0), 0).shape == (0,)
    np.testing.assert_allclose(
        function(np.zeros(2), np.array([0.0, 1.0, 1.0, 0.0]), 2),
        [1.0, 1.0],
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.native_runtime
def test_original_psychedelic_series_consumes_real_rust_monitor() -> None:
    """Exercise the original UPDE-to-series-to-monitor path with an analytic oracle."""
    from benchmarks.chimera_local_order_reference import scalar_local_order
    from scpn_phase_orchestrator.monitor import chimera
    from scpn_phase_orchestrator.monitor.psychedelic import (
        simulate_psychedelic_trajectory,
    )
    from scpn_phase_orchestrator.upde.engine import UPDEEngine

    assert chimera.ACTIVE_BACKEND == "rust"
    phases = np.array([0.0, 0.5, 2.0])
    omegas = np.array([0.1, 0.2, 0.3])
    coupling = np.ones((3, 3)) - np.eye(3)
    alpha = np.zeros((3, 3))
    originals = [values.copy() for values in (phases, omegas, coupling, alpha)]
    engine = UPDEEngine(3, dt=0.01)
    schedule = [0.0, 0.5, 1.0]
    profile = cProfile.Profile()
    profile.enable()
    records = simulate_psychedelic_trajectory(
        engine, phases, omegas, coupling, alpha, schedule, n_steps_per_level=2
    )
    profile.disable()
    native_calls = sum(
        entry.callcount
        for entry in profile.getstats()
        if isinstance(entry.code, str) and "detect_chimera_rust" in entry.code
    )
    assert native_calls >= len(schedule)
    assert len(records) == len(schedule)
    for record, reduction in zip(records, schedule, strict=True):
        actual_phases = cast(FloatArray, record["phases"])
        local = scalar_local_order(actual_phases, coupling * (1.0 - reduction))
        boundary = np.count_nonzero((local >= 0.3) & (local <= 0.7)) / 3
        assert record["chimera_index"] == boundary
        assert record["reduction_factor"] == reduction
        assert actual_phases.shape == (3,) and np.all(np.isfinite(actual_phases))
    for original, actual in zip(
        originals, (phases, omegas, coupling, alpha), strict=True
    ):
        np.testing.assert_array_equal(original, actual)


@pytest.mark.native_runtime
def test_genuine_installed_default_python_and_missing_owner_refusal() -> None:
    """Exercise the actual kernel-absent installation through default public APIs."""
    observed = installed_absent_probe()
    assert observed["active"] == "python"
    assert observed["local"] == [0.0, 0.0]
    assert observed["incoherent"] == [0, 1]
    assert observed["missing"] == ["rust", "mojo", "julia", "go"]

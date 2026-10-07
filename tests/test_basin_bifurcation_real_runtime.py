# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Real basin and coupling-sweep contracts

"""Original public runtimes measured against an independent scalar integrator."""

from __future__ import annotations

import cProfile
import importlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from benchmarks.kuramoto_trial_reference import scalar_trial
from scpn_phase_orchestrator.upde.basin_stability import (
    TrialKernel,
    basin_stability,
    multi_basin_stability,
    steady_state_r,
)
from scpn_phase_orchestrator.upde.bifurcation import (
    find_critical_coupling,
    trace_sync_transition,
)

FloatArray = NDArray[np.float64]
BACKENDS = tuple(os.environ.get("SPO_REQUIRED_BASIN_BACKENDS", "python").split(","))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("seed", (42, 1))
def test_superthreshold_lower_bracket_retains_interval_return_without_transition(
    backend: str, seed: int
) -> None:
    """Constant original trials preserve midpoint and historical flag semantics."""
    phases = np.random.default_rng(seed).uniform(0, 2 * np.pi, 4)
    expected = (
        math.hypot(
            sum(math.cos(float(value)) for value in phases),
            sum(math.sin(float(value)) for value in phases),
        )
        / 4
    )
    assert expected > 0.1
    omega = np.zeros(4)
    graph = np.zeros((4, 4))
    lower = steady_state_r(
        phases, omega, graph, n_transient=0, n_measure=1, backend=backend
    )
    assert lower == pytest.approx(expected, abs=2e-15)
    diagram = trace_sync_transition(
        omega,
        graph,
        K_range=(0.0, 20.0),
        n_points=3,
        n_transient=0,
        n_measure=1,
        seed=seed,
        backend=backend,
    )
    np.testing.assert_allclose(
        diagram.R_values, np.full(3, expected), rtol=0, atol=2e-15
    )
    assert diagram.K_critical is None
    assert all(point.stable is True for point in diagram.points)
    result = find_critical_coupling(
        omega,
        graph,
        n_transient=0,
        n_measure=1,
        tol=0.05,
        seed=seed,
        backend=backend,
    )
    assert result == 0.01953125


def _network(n: int) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return a signed directed graph with nonzero lags and self-couplings."""
    rng = np.random.default_rng(218 + n)
    return (
        rng.uniform(-2.0, 2.0, n),
        rng.uniform(-0.6, 0.6, n),
        rng.uniform(-0.7, 0.7, (n, n)),
        rng.uniform(-0.4, 0.4, (n, n)),
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n", (1, 2, 5))
@pytest.mark.parametrize(
    ("transient", "measure", "scale"),
    ((0, 1, 1.0), (3, 7, -0.8), (8, 0, 2.0), (2, 4, 0.0)),
)
def test_public_trial_matches_independent_euler(
    backend: str,
    n: int,
    transient: int,
    measure: int,
    scale: float,
) -> None:
    """Exercise orientation, lag sign, self-edges and post-step window semantics."""
    p, o, k, a = _network(n)
    expected = scalar_trial(
        p.tolist(),
        o.tolist(),
        k.tolist(),
        a.tolist(),
        scale=scale,
        dt=0.07,
        transient=transient,
        measure=measure,
    )
    original = p.copy()
    actual = steady_state_r(
        p,
        o,
        k,
        a,
        k_scale=scale,
        dt=0.07,
        n_transient=transient,
        n_measure=measure,
        backend=backend,
    )
    assert actual == pytest.approx(expected, abs=2e-14, rel=2e-14)
    np.testing.assert_array_equal(p, original)


@pytest.mark.parametrize("backend", BACKENDS)
def test_tiny_nonzero_edge_has_an_analytic_effect(backend: str) -> None:
    """A finite admitted case distinguishes the removed native cutoff."""
    actual = steady_state_r(
        np.array([0.0, math.pi / 2]),
        np.zeros(2),
        np.array([[0.0, 5e-31], [5e-31, 0.0]]),
        dt=1e30,
        n_transient=0,
        n_measure=1,
        backend=backend,
    )
    assert actual == pytest.approx(math.cos((math.pi / 2 - 1.0) / 2), abs=2e-15)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("failure", ("weight", "angle", "step"))
def test_finite_inputs_with_unrepresentable_arithmetic_fail_and_recover(
    backend: str, failure: str
) -> None:
    """Actual numerical faults propagate; a subsequent ordinary trial succeeds."""
    p = np.array([0.0, 1.0])
    o = np.zeros(2)
    k = np.zeros((2, 2))
    scale = 1.0
    dt = 1.0
    if failure == "weight":
        k[0, 1] = 1e308
        scale = 2.0
    elif failure == "angle":
        p[:] = [1e308, -1e308]
        k[0, 1] = 1.0
    else:
        o[:] = 1e308
        dt = 2.0
    with pytest.raises((ValueError, RuntimeError)):
        steady_state_r(
            p, o, k, k_scale=scale, dt=dt, n_transient=0, n_measure=1, backend=backend
        )
    assert (
        steady_state_r(
            np.zeros(2),
            np.zeros(2),
            np.zeros((2, 2)),
            n_transient=0,
            n_measure=1,
            backend=backend,
        )
        == 1.0
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_zero_edges_do_not_evaluate_unused_overflowing_angles(backend: str) -> None:
    """Disconnected finite phases do not require representable pair differences."""
    p = np.array([1e308, -1e308])
    expected = abs(math.cos(1e308))
    assert steady_state_r(
        p, np.zeros(2), np.zeros((2, 2)), n_transient=0, n_measure=1, backend=backend
    ) == pytest.approx(expected, abs=2e-14)


@pytest.mark.parametrize("backend", BACKENDS)
def test_paired_monte_carlo_and_thresholds_match_scalar_trials(backend: str) -> None:
    """Public NumPy RNG, inclusive threshold, counts and shared trials agree."""
    _, o, k, a = _network(3)
    rng = np.random.default_rng(29)
    expected = np.array(
        [
            scalar_trial(
                rng.uniform(0, 2 * np.pi, 3).tolist(),
                o.tolist(),
                k.tolist(),
                a.tolist(),
                dt=0.04,
                transient=2,
                measure=3,
            )
            for _ in range(8)
        ]
    )
    result = basin_stability(
        o,
        k,
        a,
        dt=0.04,
        n_transient=2,
        n_measure=3,
        n_samples=8,
        R_threshold=0.5,
        seed=29,
        backend=backend,
    )
    np.testing.assert_allclose(result.R_final, expected, atol=2e-14, rtol=2e-14)
    assert result.n_converged == int(np.count_nonzero(expected >= 0.5))
    assert result.n_converged / 8 == result.S_B
    results = multi_basin_stability(
        o,
        k,
        a,
        dt=0.04,
        n_transient=2,
        n_measure=3,
        n_samples=8,
        R_thresholds=(0.0, 0.5, 1.0),
        seed=29,
        backend=backend,
    )
    for threshold in (0.0, 0.5, 1.0):
        row = results[f"R>={threshold:.2f}"]
        np.testing.assert_allclose(row.R_final, expected, atol=2e-14, rtol=2e-14)
        assert row.n_converged == int(np.count_nonzero(expected >= threshold))
    assert results["R>=0.00"].S_B == 1.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_independent_grid_sweep_uses_the_selected_owner(backend: str) -> None:
    """Every public composite point starts from the same actual NumPy sample."""
    _, o, k, a = _network(3)
    np.fill_diagonal(k, 0.0)
    p = np.random.default_rng(17).uniform(0, 2 * np.pi, 3)
    grid = np.linspace(0, 3, 5)
    expected = [
        scalar_trial(
            p.tolist(),
            o.tolist(),
            k.tolist(),
            a.tolist(),
            scale=float(scale),
            dt=0.03,
            transient=2,
            measure=4,
        )
        for scale in grid
    ]
    profile = cProfile.Profile()
    profile.enable()
    result = trace_sync_transition(
        o,
        k,
        a,
        K_range=(0.0, 3.0),
        n_points=5,
        dt=0.03,
        n_transient=2,
        n_measure=4,
        seed=17,
        backend=backend,
    )
    profile.disable()
    np.testing.assert_allclose(result.K_values, grid, atol=0, rtol=0)
    np.testing.assert_allclose(result.R_values, expected, atol=2e-14, rtol=2e-14)
    codes = [str(entry.code) for entry in profile.getstats()]
    if backend == "rust":
        assert any(
            "built-in method spo_kernel.spo_kernel.trace_sync_transition_rust" in code
            for code in codes
        )
    else:
        needle = (
            "_python_steady_state_r"
            if backend == "python"
            else "steady_state_r_" + backend
        )
        assert any(needle in code for code in codes)


@pytest.mark.parametrize("backend", BACKENDS)
def test_empty_measurement_consumers_and_search(backend: str) -> None:
    """Actual zero-window composites yield zero R and no threshold crossing."""
    result = trace_sync_transition(
        np.zeros(2),
        K_range=(0.0, 2.0),
        n_points=3,
        n_transient=10,
        n_measure=0,
        backend=backend,
    )
    np.testing.assert_array_equal(result.R_values, np.zeros(3))
    assert result.K_critical is None
    assert math.isnan(
        find_critical_coupling(
            np.zeros(2), n_transient=10, n_measure=0, backend=backend
        )
    )
    basin = basin_stability(
        np.zeros(2),
        np.zeros((2, 2)),
        n_measure=0,
        n_samples=3,
        R_threshold=0.0,
        backend=backend,
    )
    assert basin.n_converged == 3 and basin.S_B == 1.0


def test_ambiguous_threshold_labels_are_rejected() -> None:
    """Legacy two-decimal labels cannot silently discard a distinct threshold."""
    with pytest.raises(ValueError, match="distinct two-decimal"):
        multi_basin_stability(
            np.zeros(2),
            np.zeros((2, 2)),
            R_thresholds=(0.301, 0.302),
            n_samples=0,
            backend="python",
        )


@pytest.mark.parametrize("operation", ("trial", "basin", "multi", "sweep", "search"))
def test_unknown_owner_is_rejected_even_for_empty_work(operation: str) -> None:
    """Named selection validates before an empty-window identity return."""
    with pytest.raises(ValueError, match="unknown basin backend"):
        if operation == "trial":
            steady_state_r(
                np.zeros(2),
                np.zeros(2),
                np.zeros((2, 2)),
                n_measure=0,
                backend="unknown",
            )
        elif operation == "basin":
            basin_stability(
                np.zeros(2), np.zeros((2, 2)), n_samples=0, backend="unknown"
            )
        elif operation == "multi":
            multi_basin_stability(
                np.zeros(2), np.zeros((2, 2)), n_samples=0, backend="unknown"
            )
        elif operation == "sweep":
            trace_sync_transition(np.zeros(2), n_measure=0, backend="unknown")
        else:
            find_critical_coupling(np.zeros(2), n_measure=0, backend="unknown")


@pytest.mark.parametrize("backend", BACKENDS)
def test_real_upcrossing_matches_two_oscillator_algebra(backend: str) -> None:
    """Actual Rust and delegated sweeps agree on the first threshold upcrossing."""
    phases = np.random.default_rng(92).uniform(0, 2 * np.pi, 2)
    delta = float(phases[1] - phases[0])
    expected = np.array(
        [
            abs(math.cos((delta - 2 * scale * math.sin(delta)) / 2))
            for scale in (0.0, 0.5, 1.0)
        ]
    )
    actual = trace_sync_transition(
        np.zeros(2),
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        K_range=(0.0, 1.0),
        n_points=3,
        dt=1.0,
        n_transient=0,
        n_measure=1,
        seed=92,
        backend=backend,
    )
    np.testing.assert_allclose(actual.R_values, expected, atol=2e-15, rtol=0)
    crossing = 0.5 * (0.1 - expected[0]) / (expected[1] - expected[0])
    assert actual.K_critical == pytest.approx(crossing, abs=2e-15)


@pytest.mark.parametrize("field", ("phases_init", "omegas", "knm", "alpha"))
@pytest.mark.parametrize("as_objects", (False, True))
def test_public_arrays_refuse_complex_values_without_discarding_them(
    field: str, as_objects: bool
) -> None:
    """A complex alias cannot become a valid real-valued trial after coercion."""
    values = {
        "phases_init": np.zeros(2),
        "omegas": np.zeros(2),
        "knm": np.zeros((2, 2)),
        "alpha": np.zeros((2, 2)),
    }
    hostile = values[field].astype(object if as_objects else np.complex128)
    hostile.flat[0] = 1 + 2j
    values[field] = hostile
    with pytest.raises(ValueError, match="real-valued, not complex"):
        steady_state_r(
            values["phases_init"],
            values["omegas"],
            values["knm"],
            values["alpha"],
            n_transient=0,
            n_measure=1,
            backend="python",
        )


@pytest.mark.parametrize(
    ("fraction", "count", "match"),
    [(0.5, 0, "n_converged must match"), (1.0, 1, "S_B must match")],
)
def test_result_records_refuse_inconsistent_classification(
    fraction: float, count: int, match: str
) -> None:
    """Well-shaped finite fields cannot publish contradictory sampling evidence."""
    from scpn_phase_orchestrator.upde.basin_stability import BasinStabilityResult

    with pytest.raises(ValueError, match=match):
        BasinStabilityResult(fraction, 2, count, np.array([0.25, 0.75]), 0.5)


def test_result_fraction_preserves_float32_rounding_and_canonicalizes_it() -> None:
    """A supplied low-precision fraction retains its exact classified meaning."""
    from scpn_phase_orchestrator.upde.basin_stability import BasinStabilityResult

    result = BasinStabilityResult(
        float(np.float32(1 / 3)), 3, 1, np.array([0.1, 0.2, 0.8]), 0.5
    )
    assert result.S_B == 1 / 3


def test_equal_repeated_thresholds_preserve_the_original_single_label() -> None:
    """An identical redundant threshold stays valid, without an ambiguous label."""
    result = multi_basin_stability(
        np.zeros(2),
        np.zeros((2, 2)),
        n_transient=0,
        n_measure=1,
        n_samples=2,
        R_thresholds=(0.5, 0.5),
        backend="python",
    )
    assert list(result) == ["R>=0.50"]


@pytest.mark.parametrize("field", ("phases_init", "knm"))
def test_actual_public_conversion_failures_are_reported_as_value_errors(
    field: str,
) -> None:
    """A hostile array protocol cannot escape ingress validation."""

    class FailedArray:
        """Negative control whose original NumPy coercion always fails."""

        def __array__(self, dtype: object = None, copy: object = None) -> FloatArray:
            """Deliberately reject conversion before the trial can run."""
            raise TypeError("unavailable array payload")

    hostile = FailedArray()
    with pytest.raises(ValueError, match="finite float array"):
        if field == "phases_init":
            steady_state_r(
                cast(FloatArray, hostile),
                np.zeros(2),
                np.zeros((2, 2)),
                backend="python",
            )
        else:
            steady_state_r(
                np.zeros(2), np.zeros(2), cast(FloatArray, hostile), backend="python"
            )


def test_actual_missing_rust_export_is_refused_without_an_identity_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse an actual installed export removed as a broken-ABI control."""
    import importlib

    from scpn_phase_orchestrator.upde import basin_stability as surface

    try:
        kernel = importlib.import_module("spo_kernel")
    except ImportError:
        with pytest.raises(ImportError):
            steady_state_r(
                np.zeros(2), np.zeros(2), np.zeros((2, 2)), n_measure=0, backend="rust"
            )
        return
    monkeypatch.delattr(kernel, "steady_state_r_rust")
    monkeypatch.setattr(surface, "_BACKEND_CACHE", {})
    with pytest.raises(ImportError, match="requested basin backend"):
        steady_state_r(
            np.zeros(2), np.zeros(2), np.zeros((2, 2)), n_measure=0, backend="rust"
        )


def test_checked_go_bridge_refuses_unrepresentable_step_metadata() -> None:
    """Original direct Go ingress rejects integer wrap before C invocation."""
    from scpn_phase_orchestrator.experimental.accelerators.upde import (
        _basin_stability_go as go_bridge,
    )

    with pytest.raises(ValueError, match="signed 64-bit"):
        go_bridge.steady_state_r_go(
            np.zeros(2), np.zeros(2), np.zeros(4), np.zeros(4), 2, 1.0, 0.01, 2**63, 0
        )


def test_an_actual_incompatible_go_library_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real shared library lacking the required trial ABI cannot be used."""
    from pathlib import Path

    from scpn_phase_orchestrator.experimental.accelerators.upde import (
        _basin_stability_go as bridge,
    )
    from scpn_phase_orchestrator.upde import basin_stability as surface

    artifact = Path(__file__).resolve().parents[1] / "go/libattnres.so"
    if not artifact.is_file():
        # The genuine absent-library path is exercised without fabricating one.
        artifact = Path(__file__).resolve().parents[1] / "go/missing-basin-fixture.so"
    monkeypatch.setattr(bridge, "_LIB_PATH", artifact)
    monkeypatch.setattr(bridge, "_LIB", None)
    monkeypatch.setattr(surface, "_BACKEND_CACHE", {})
    with pytest.raises(ImportError):
        steady_state_r(
            np.zeros(2),
            np.zeros(2),
            np.zeros((2, 2)),
            n_transient=0,
            n_measure=1,
            backend="go",
        )


def test_original_mojo_direct_zero_window_keeps_its_identity_contract() -> None:
    """The original public direct adapter returns its defined empty-window value."""
    from scpn_phase_orchestrator.experimental.accelerators.upde import (
        _basin_stability_mojo as bridge,
    )

    assert (
        bridge.steady_state_r_mojo(
            np.zeros(2), np.zeros(2), np.zeros(4), np.zeros(4), 2, 1.0, 0.01, 3, 0
        )
        == 0.0
    )


def test_actual_julia_syntax_failure_is_classified_as_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A real Julia parser failure cannot masquerade as an available owner."""
    from scpn_phase_orchestrator.experimental.accelerators.upde import (
        _basin_stability_julia as bridge,
    )
    from scpn_phase_orchestrator.upde import basin_stability as surface

    broken = tmp_path / "malformed_basin.jl"
    broken.write_text("module BasinStabilityJL\nfunction missing(\n", encoding="utf-8")
    monkeypatch.setattr(bridge, "_JULIA_FILE", broken)
    monkeypatch.setattr(bridge, "_JULIA_MODULE", None)
    monkeypatch.setattr(surface, "_BACKEND_CACHE", {})
    with pytest.raises(ImportError):
        steady_state_r(
            np.zeros(2),
            np.zeros(2),
            np.zeros((2, 2)),
            n_transient=0,
            n_measure=1,
            backend="julia",
        )


def test_non_julia_include_errors_preserve_the_original_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A deliberate Python include fault is not reclassified as Julia absence."""
    from scpn_phase_orchestrator.experimental.accelerators.upde import (
        _basin_stability_julia as bridge,
    )

    class BrokenMain:
        """Negative control that raises a Python error rather than JuliaError."""

        def include(self, path: str) -> None:
            """Fail the include transport deliberately before any numerical result."""
            raise LookupError("include transport fault")

    broken = tmp_path / "source.jl"
    broken.write_text("module BasinStabilityJL\nend\n", encoding="utf-8")
    monkeypatch.setattr(bridge, "_JULIA_FILE", broken)
    monkeypatch.setattr(bridge, "_JULIA_MODULE", None)
    monkeypatch.setattr(bridge, "require_julia_main", lambda: BrokenMain())
    # This error-classification negative control requires the original Julia
    # error type; genuine absence is separately observed through named requests.
    try:
        importlib.import_module("juliacall")
    except ImportError:
        with pytest.raises(ImportError):
            bridge.steady_state_r_julia(
                np.zeros(2), np.zeros(2), np.zeros(4), np.zeros(4), 2, 1.0, 0.01, 0, 1
            )
        return
    with pytest.raises(LookupError, match="include transport fault"):
        bridge.steady_state_r_julia(
            np.zeros(2), np.zeros(2), np.zeros(4), np.zeros(4), 2, 1.0, 0.01, 0, 1
        )


def test_non_julia_trial_errors_propagate_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deliberately failed trial transport cannot become a numerical value."""
    from scpn_phase_orchestrator.experimental.accelerators.upde import (
        _basin_stability_julia as bridge,
    )

    class BrokenTrial:
        """Negative control whose trial raises a Python transport error."""

        def steady_state_r(self, *_args: object) -> object:
            """Deliberately fail before returning a scalar observation."""
            raise LookupError("trial transport fault")

    monkeypatch.setattr(bridge, "_ensure", lambda: BrokenTrial())
    try:
        importlib.import_module("juliacall")
    except ImportError:
        with pytest.raises(ImportError):
            bridge.steady_state_r_julia(
                np.zeros(2), np.zeros(2), np.zeros(4), np.zeros(4), 2, 1.0, 0.01, 0, 1
            )
        return
    with pytest.raises(LookupError, match="trial transport fault"):
        bridge.steady_state_r_julia(
            np.zeros(2), np.zeros(2), np.zeros(4), np.zeros(4), 2, 1.0, 0.01, 0, 1
        )


def test_actual_missing_composite_export_uses_original_trial_computation() -> None:
    """A child removes the real export without replacing collected public classes."""
    from scpn_phase_orchestrator.upde import bifurcation as surface

    if importlib.util.find_spec("spo_kernel") is None:
        result = surface.trace_sync_transition(
            np.zeros(2), n_points=2, n_transient=0, n_measure=1, backend="python"
        )
        assert len(result.points) == 2
        return
    code = r"""
import cProfile,json,math
import numpy as np
import spo_kernel
if hasattr(spo_kernel,'find_critical_coupling_bif_rust'):
    del spo_kernel.find_critical_coupling_bif_rust
from scpn_phase_orchestrator.upde.bifurcation import trace_sync_transition
phases=np.random.default_rng(92).uniform(0,2*np.pi,2)
delta=float(phases[1]-phases[0])
with cProfile.Profile() as profile:
    result=trace_sync_transition(np.zeros(2),np.array([[0.,1.],[1.,0.]]),
        K_range=(0.,1.),n_points=3,dt=1.,n_transient=0,n_measure=1,
        seed=92,backend='rust')
profile.create_stats()
expected=[abs(math.cos((delta-2*scale*math.sin(delta))/2))for scale in(0.,.5,1.)]
np.testing.assert_allclose(result.R_values,expected,atol=2e-15,rtol=0)
names={key[2]for key in profile.stats}
assert any('spo_kernel.spo_kernel.steady_state_r_rust' in name for name in names)
assert '_python_steady_state_r'not in names
print(json.dumps({'values':result.R_values.tolist(),'original_rust':True}))
"""
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads(process.stdout.splitlines()[-1])
    assert report["original_rust"] is True
    assert len(report["values"]) == 3


@pytest.mark.parametrize("owner", ("go", "julia", "mojo"))
def test_legacy_public_accelerator_wrappers_execute_original_values(owner: str) -> None:
    """Established import paths still route actual values through original adapters."""
    module = importlib.import_module(
        "scpn_phase_orchestrator.upde._basin_stability_" + owner
    )
    fn = cast(
        "TrialKernel",
        getattr(module, "steady_state_r_" + owner),
    )
    from scpn_phase_orchestrator.upde.basin_stability import AVAILABLE_BACKENDS

    phases = np.array([0.0, math.pi / 2])
    omega = np.zeros(2)
    graph = np.array([0.0, 5e-31, 5e-31, 0.0])
    alpha = np.zeros(4)
    if owner not in AVAILABLE_BACKENDS:
        with pytest.raises(ImportError):
            fn(phases, omega, graph, alpha, 2, 1.0, 1e30, 0, 1)
        return
    actual = fn(phases, omega, graph, alpha, 2, 1.0, 1e30, 0, 1)
    assert actual == pytest.approx(math.cos((math.pi / 2 - 1) / 2), abs=2e-15)

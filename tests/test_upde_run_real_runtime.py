# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Actual UPDE run runtime contracts

"""Stateless UPDE runs through every actual owner, without substitution.

The oracles integrate the UPDE in plain Python loops with ``math.sin`` and
``math.fsum``: forward Euler, classical RK4, and Dormand–Prince 5(4) with
the documented PI step control (accept when the scaled error is at most 1,
grow by ``min(0.9·e^-1/5, 5)`` up to ten configured steps, otherwise shrink
by ``max(0.9·e^-1/4, 0.2)`` and retry at most four times). Phases are
wrapped into ``[0, 2π)`` after every outer step.

Owners are observed with :mod:`cProfile`: the Rust stepper's builtin
methods, or the Python entry function of each bridge. Child-process
profiles remove native owners by building a package path without them, so
their absence is real.
"""

from __future__ import annotations

import collections
import cProfile
import json
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeVar

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _engine_go as engine_go_mod,
)
from scpn_phase_orchestrator.upde import _run as run_mod
from scpn_phase_orchestrator.upde import engine as engine_mod

pytestmark = pytest.mark.native_runtime

ROOT = Path(__file__).resolve().parents[1]
TWO_PI = 2.0 * math.pi
OWNERS = ("python", "rust", "mojo", "julia", "go")
METHODS = ("euler", "rk4", "rk45")
N = 4
ZETA, PSI, DT = 0.3, 0.4, 0.01
ATOL, RTOL = 1e-6, 1e-3

FloatArray = NDArray[np.float64]
Matrix = list[list[float]]
T = TypeVar("T")

# Calls that prove each owner computed: (step marker, schedule marker).
_MARKERS = {
    "python": ("upde_run_python", "upde_run_omega_schedule_python"),
    "rust": (
        "<method 'run' of 'builtins.PyUPDEStepper' objects>",
        "<method 'run_omega_schedule' of 'builtins.PyUPDEStepper' objects>",
    ),
    "mojo": ("upde_run_mojo", "upde_run_omega_schedule_mojo"),
    "julia": ("upde_run_julia", "upde_run_omega_schedule_julia"),
    "go": ("upde_run_go", "upde_run_omega_schedule_go"),
}

# Dormand–Prince 5(4) tableau (Dormand & Prince 1980).
_DP_A = (
    (),
    (1 / 5,),
    (3 / 40, 9 / 40),
    (44 / 45, -56 / 15, 32 / 9),
    (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
    (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
)
_DP_B5 = (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84)
_DP_B4 = (
    5179 / 57600,
    0.0,
    7571 / 16695,
    393 / 640,
    -92097 / 339200,
    187 / 2100,
    1 / 40,
)


def _tolerance(owner: str) -> float:
    """Return the owner's absolute tolerance; Mojo crosses a text bridge."""
    return 1e-9 if owner == "mojo" else 1e-12


def _problem() -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return seeded phases, frequencies, zero-diagonal coupling and lags."""
    rng = np.random.default_rng(2026)
    phases = rng.uniform(0.0, TWO_PI, N)
    omegas = rng.normal(1.0, 0.3, N)
    knm = rng.uniform(0.0, 0.5, (N, N))
    np.fill_diagonal(knm, 0.0)
    alpha = rng.uniform(-0.3, 0.3, (N, N))
    return phases, omegas, knm, alpha


def _schedule(steps: int) -> FloatArray:
    """Return a seeded frequency schedule of ``steps`` rows."""
    return np.random.default_rng(7).normal(1.0, 0.3, (steps, N))


def _derivative(
    theta: Sequence[float], omegas: Sequence[float], knm: Matrix, alpha: Matrix
) -> list[float]:
    """Return ``ω_i + Σ_j K_ij sin(θ_j − θ_i − α_ij) + ζ sin(Ψ − θ_i)``."""
    return [
        omegas[i]
        + math.fsum(
            knm[i][j] * math.sin(theta[j] - theta[i] - alpha[i][j]) for j in range(N)
        )
        + ZETA * math.sin(PSI - theta[i])
        for i in range(N)
    ]


def _axpy(
    theta: Sequence[float], h: float, terms: Sequence[tuple[float, list[float]]]
) -> list[float]:
    """Return ``θ + h Σ c·k`` over the weighted stage list."""
    return [theta[i] + h * math.fsum(c * k[i] for c, k in terms) for i in range(N)]


def _wrap(value: float) -> float:
    """Return ``value`` on ``[0, 2π)`` with positive zero."""
    wrapped = value % TWO_PI
    return 0.0 if wrapped >= TWO_PI else wrapped + 0.0


def _rk4(
    theta: list[float], om: Sequence[float], knm: Matrix, alpha: Matrix, h: float
) -> list[float]:
    """Return one classical RK4 step."""
    k1 = _derivative(theta, om, knm, alpha)
    k2 = _derivative(_axpy(theta, 0.5 * h, [(1.0, k1)]), om, knm, alpha)
    k3 = _derivative(_axpy(theta, 0.5 * h, [(1.0, k2)]), om, knm, alpha)
    k4 = _derivative(_axpy(theta, h, [(1.0, k3)]), om, knm, alpha)
    return _axpy(theta, h / 6.0, [(1.0, k1), (2.0, k2), (2.0, k3), (1.0, k4)])


def _dp45(
    theta: list[float],
    om: Sequence[float],
    knm: Matrix,
    alpha: Matrix,
    dt_config: float,
    last_dt: float,
) -> tuple[list[float], float]:
    """Return one accepted Dormand–Prince step and the next trial step."""
    h = last_dt
    for _ in range(4):
        stages: list[list[float]] = []
        for row in _DP_A:
            stages.append(
                _derivative(
                    _axpy(theta, h, list(zip(row, stages, strict=False))),
                    om,
                    knm,
                    alpha,
                )
            )
        y5 = _axpy(theta, h, list(zip(_DP_B5, stages, strict=True)))
        stages.append(_derivative(y5, om, knm, alpha))
        y4 = _axpy(theta, h, list(zip(_DP_B4, stages, strict=True)))
        err = max(
            abs(y5[i] - y4[i]) / (ATOL + RTOL * max(abs(theta[i]), abs(y5[i])))
            for i in range(N)
        )
        if err <= 1.0:
            factor = min(0.9 * err**-0.2, 5.0) if err > 0.0 else 5.0
            return y5, min(h * factor, dt_config * 10.0)
        h *= max(0.9 * err**-0.25, 0.2)
    return y5, h


def _oracle(
    phases: FloatArray,
    omega_rows: Sequence[Sequence[float]],
    knm: FloatArray,
    alpha: FloatArray,
    method: str,
    n_substeps: int,
) -> list[float]:
    """Integrate one outer step per frequency row by plain loops."""
    theta = phases.tolist()
    k_rows, a_rows = knm.tolist(), alpha.tolist()
    h = DT / n_substeps
    last_dt = DT
    for om in omega_rows:
        if method == "rk45":
            theta, last_dt = _dp45(theta, om, k_rows, a_rows, DT, last_dt)
        else:
            for _ in range(n_substeps):
                if method == "rk4":
                    theta = _rk4(theta, om, k_rows, a_rows, h)
                else:
                    theta = _axpy(
                        theta, h, [(1.0, _derivative(theta, om, k_rows, a_rows))]
                    )
        theta = [_wrap(v) for v in theta]
    return theta


def _assert_on_circle(got: FloatArray, expected: Sequence[float], atol: float) -> None:
    """Check phases agree on the circle and lie on ``[0, 2π)``."""
    assert got.dtype == np.float64
    assert np.all((got >= 0.0) & (got < TWO_PI))
    gap = np.abs(np.angle(np.exp(1j * (got - np.asarray(expected)))))
    assert float(np.max(gap)) <= atol, (got.tolist(), list(expected))


def _profiled(call: Callable[[], T]) -> tuple[T, collections.Counter[str]]:
    """Run ``call`` under cProfile and count calls by function name."""
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        result = call()
    finally:
        profiler.disable()
    counts: collections.Counter[str] = collections.Counter()
    for entry in profiler.getstats():
        name = entry.code if isinstance(entry.code, str) else entry.code.co_name
        counts[name] += entry.callcount
    return result, counts


def _assert_served_by(counts: collections.Counter[str], owner: str, slot: int) -> None:
    """Check that ``owner`` computed once and no other owner computed."""
    assert counts[_MARKERS[owner][slot]] == 1, owner
    for other, markers in _MARKERS.items():
        if other != owner:
            assert counts[markers[slot]] == 0, other


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("backend", OWNERS)
def test_each_owner_integrates_the_oracle_trajectory(backend: str, method: str) -> None:
    """``upde_run(backend=...)`` runs on that owner and matches the oracle."""
    phases, omegas, knm, alpha = _problem()
    n_substeps = 1 if method == "rk45" else 2
    got, counts = _profiled(
        lambda: run_mod.upde_run(
            phases,
            omegas,
            knm,
            alpha,
            ZETA,
            PSI,
            DT,
            30,
            method,
            n_substeps,
            ATOL,
            RTOL,
            backend=backend,
        )
    )
    _assert_served_by(counts, backend, 0)
    expected = _oracle(phases, [omegas.tolist()] * 30, knm, alpha, method, n_substeps)
    _assert_on_circle(got, expected, _tolerance(backend))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("backend", OWNERS)
def test_each_owner_integrates_the_oracle_schedule(backend: str, method: str) -> None:
    """``upde_run_omega_schedule(backend=...)`` uses one frequency row per step."""
    phases, _, knm, alpha = _problem()
    schedule = _schedule(12)
    n_substeps = 1 if method == "rk45" else 3
    got, counts = _profiled(
        lambda: run_mod.upde_run_omega_schedule(
            phases,
            schedule,
            knm,
            alpha,
            ZETA,
            PSI,
            DT,
            method,
            n_substeps,
            ATOL,
            RTOL,
            backend=backend,
        )
    )
    _assert_served_by(counts, backend, 1)
    expected = _oracle(phases, schedule.tolist(), knm, alpha, method, n_substeps)
    _assert_on_circle(got, expected, _tolerance(backend))


@pytest.mark.parametrize("backend", OWNERS)
def test_each_owner_normalises_valid_array_likes(backend: str) -> None:
    """Nested lists and flat coupling lists are admitted as float64 arrays."""
    phases, omegas, knm, alpha = _problem()
    got = run_mod.upde_run(
        phases.tolist(),
        omegas.tolist(),
        knm.ravel().tolist(),
        alpha.ravel().tolist(),
        ZETA,
        PSI,
        DT,
        5,
        backend=backend,
    )
    expected = _oracle(phases, [omegas.tolist()] * 5, knm, alpha, "euler", 1)
    _assert_on_circle(got, expected, _tolerance(backend))


def test_engine_facade_passes_the_named_owner_through() -> None:
    """The public engine facade routes ``backend=`` to the named owner."""
    phases, omegas, knm, alpha = _problem()
    schedule = _schedule(4)
    got, counts = _profiled(
        lambda: engine_mod.upde_run(
            phases, omegas, knm, alpha, ZETA, PSI, DT, 4, backend="go"
        )
    )
    _assert_served_by(counts, "go", 0)
    _assert_on_circle(
        got, _oracle(phases, [omegas.tolist()] * 4, knm, alpha, "euler", 1), 1e-12
    )
    got, counts = _profiled(
        lambda: engine_mod.upde_run_omega_schedule(
            phases, schedule, knm, alpha, ZETA, PSI, DT, backend="julia"
        )
    )
    _assert_served_by(counts, "julia", 1)
    _assert_on_circle(
        got, _oracle(phases, schedule.tolist(), knm, alpha, "euler", 1), 1e-12
    )


@pytest.mark.parametrize("backend", OWNERS)
def test_each_owner_is_named_for_both_slots(backend: str) -> None:
    """``upde_run_backend`` names a loadable owner for steps and schedules."""
    assert run_mod.upde_run_backend("step", backend) == backend
    assert run_mod.upde_run_backend("schedule", backend) == backend
    assert engine_mod.upde_run_backend("step", backend) == backend


def test_auto_owner_is_the_active_backend() -> None:
    """``auto`` names the resolved active owner for both slots here."""
    active = run_mod.ACTIVE_BACKEND
    assert run_mod.upde_run_backend() == active
    assert run_mod.upde_run_backend("schedule") == active


def test_webgpu_has_no_schedule_kernel() -> None:
    """WebGPU is a step owner only; naming it for a schedule is refused."""
    phases, _, knm, alpha = _problem()
    with pytest.raises(RuntimeError, match="'webgpu' has no schedule kernel"):
        run_mod.upde_run_backend("schedule", "webgpu")
    with pytest.raises(RuntimeError, match="'webgpu' has no schedule kernel"):
        run_mod.upde_run_omega_schedule(
            phases, _schedule(2), knm, alpha, ZETA, PSI, DT, backend="webgpu"
        )


def test_named_webgpu_without_a_host_bridge_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ``SPO_WEBGPU_DISPATCH_BRIDGE`` a named WebGPU run is refused."""
    monkeypatch.delenv("SPO_WEBGPU_DISPATCH_BRIDGE", raising=False)
    monkeypatch.setattr(run_mod, "_BACKEND_CACHE", {})
    phases, omegas, knm, alpha = _problem()
    with pytest.raises(RuntimeError, match="UPDE backend 'webgpu' unavailable") as info:
        run_mod.upde_run(phases, omegas, knm, alpha, ZETA, PSI, DT, 1, backend="webgpu")
    assert "SPO_WEBGPU_DISPATCH_BRIDGE" in str(info.value.__cause__)


def test_unknown_names_are_refused() -> None:
    """Unknown owners and slots are ValueErrors before any computation."""
    phases, omegas, knm, alpha = _problem()
    with pytest.raises(ValueError, match="unknown UPDE backend"):
        run_mod.upde_run(phases, omegas, knm, alpha, ZETA, PSI, DT, 1, backend="cuda")
    with pytest.raises(ValueError, match="unknown UPDE backend"):
        run_mod.upde_run_omega_schedule(
            phases, _schedule(1), knm, alpha, ZETA, PSI, DT, backend="cuda"
        )
    with pytest.raises(ValueError, match="unknown UPDE backend"):
        run_mod.upde_run_backend("step", "cuda")
    with pytest.raises(ValueError, match="unknown UPDE run slot"):
        run_mod.upde_run_backend("spectrum")


def test_named_step_owner_is_loaded_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two named Go runs load the Go library binding once and compute twice."""
    monkeypatch.setattr(run_mod, "_BACKEND_CACHE", {})
    phases, omegas, knm, alpha = _problem()

    def twice() -> FloatArray:
        """Run the same Go integration twice."""
        run_mod.upde_run(phases, omegas, knm, alpha, ZETA, PSI, DT, 3, backend="go")
        return run_mod.upde_run(
            phases, omegas, knm, alpha, ZETA, PSI, DT, 3, backend="go"
        )

    got, counts = _profiled(twice)
    assert counts["_load_go_fn"] == 1
    assert counts["upde_run_go"] == 2
    assert {"go": engine_go_mod.upde_run_go} == run_mod._BACKEND_CACHE
    _assert_on_circle(
        got, _oracle(phases, [omegas.tolist()] * 3, knm, alpha, "euler", 1), 1e-12
    )


def test_schedule_owner_is_loaded_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """The schedule cache loads the Go schedule binding once."""
    monkeypatch.setattr(run_mod, "_SCHEDULE_BACKEND_CACHE", {})

    def twice() -> Callable[..., FloatArray]:
        """Load the Go schedule owner twice."""
        first = run_mod._load_schedule_backend("go")
        assert run_mod._load_schedule_backend("go") is first
        return first

    loaded, counts = _profiled(twice)
    assert loaded is engine_go_mod.upde_run_omega_schedule_go
    assert counts["_load_go_schedule_fn"] == 1


def test_go_schedule_loader_opens_the_shared_library() -> None:
    """The Go schedule loader opens the library and returns its bridge."""
    phases, _, knm, alpha = _problem()
    schedule = _schedule(5)

    loader, counts = _profiled(run_mod._load_go_schedule_fn)
    assert counts["_load_lib"] == 1
    assert loader is engine_go_mod.upde_run_omega_schedule_go
    got = loader(phases, schedule, knm, alpha, ZETA, PSI, DT, "rk4", 2, ATOL, RTOL)
    _assert_on_circle(
        got, _oracle(phases, schedule.tolist(), knm, alpha, "rk4", 2), 1e-12
    )


@pytest.mark.parametrize("method", METHODS)
def test_rust_loaders_drive_the_compiled_stepper(method: str) -> None:
    """The Rust loaders call ``PyUPDEStepper.run`` and ``run_omega_schedule``."""
    phases, omegas, knm, alpha = _problem()
    schedule = _schedule(6)
    step = run_mod._load_rust_fn()
    got, counts = _profiled(
        lambda: step(
            phases, omegas, knm, alpha, ZETA, PSI, DT, 6, method, 2, ATOL, RTOL
        )
    )
    assert counts[_MARKERS["rust"][0]] == 1
    substeps = 1 if method == "rk45" else 2
    _assert_on_circle(
        got,
        _oracle(phases, [omegas.tolist()] * 6, knm, alpha, method, substeps),
        1e-12,
    )
    run_schedule = run_mod._load_rust_schedule_fn()
    got, counts = _profiled(
        lambda: run_schedule(
            phases, schedule, knm, alpha, ZETA, PSI, DT, method, 2, ATOL, RTOL
        )
    )
    assert counts[_MARKERS["rust"][1]] == 1
    _assert_on_circle(
        got, _oracle(phases, schedule.tolist(), knm, alpha, method, substeps), 1e-12
    )


_CHILD = r"""
import json, sys
sys.path[:0] = [sys.argv[1], sys.argv[2]]
import numpy as np
from scpn_phase_orchestrator.upde import _run as run
out = {"file": run.__file__}
out["active"], out["available"] = run._resolve_backends()
refused = {}
for owner in ("rust", "webgpu", "mojo", "julia", "go"):
    try:
        run.upde_run_backend("step", owner)
        refused[owner] = None
    except RuntimeError as exc:
        refused[owner] = str(exc)
out["refused"] = refused
args = [np.asarray(v, dtype=np.float64) for v in json.loads(sys.argv[3])]
out["phases"] = run.upde_run(*args, 0.3, 0.4, 0.01, 3).tolist()
print(json.dumps(out))
"""


def _run_profile(
    tmp_path: Path, *, keep_rust: bool, keep_go: bool
) -> dict[str, object]:
    """Run the real resolver in an isolated interpreter with one profile.

    The package is an exported source copy; ``spo_kernel`` is left out of the
    package path unless ``keep_rust``, ``juliacall`` is always left out, no
    Mojo executable sits beside the export, and the Go library is linked
    beside it only when ``keep_go``.
    """
    site = Path(np.__file__).resolve().parents[1]
    profile = tmp_path / "site"
    profile.mkdir()
    for entry in site.iterdir():
        name = entry.name.lower()
        if name.startswith(("juliacall", "juliapkg", "scpn_phase_orchestrator")):
            continue
        if name.endswith(".pth") or (name.startswith("spo_kernel") and not keep_rust):
            continue
        (profile / entry.name).symlink_to(entry)
    export = tmp_path / "export"
    shutil.copytree(
        ROOT / "src" / "scpn_phase_orchestrator",
        export / "src" / "scpn_phase_orchestrator",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    if keep_go:
        (export / "go").mkdir()
        (export / "go" / "libupde_engine.so").symlink_to(
            ROOT / "go" / "libupde_engine.so"
        )
    phases, omegas, knm, alpha = _problem()
    payload = json.dumps([v.tolist() for v in (phases, omegas, knm, alpha)])
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path)}
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            _CHILD,
            str(export / "src"),
            str(profile),
            payload,
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env=env,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result: dict[str, object] = json.loads(completed.stdout.splitlines()[-1])
    assert Path(str(result["file"])).is_relative_to(export)
    phases_out = np.asarray(result["phases"], dtype=np.float64)
    _assert_on_circle(
        phases_out,
        _oracle(phases, [omegas.tolist()] * 3, knm, alpha, "euler", 1),
        1e-12,
    )
    return result


def test_resolver_skips_absent_owners_to_the_first_that_loads(tmp_path: Path) -> None:
    """Without Rust, WebGPU, Mojo and Julia, the resolver selects Go."""
    result = _run_profile(tmp_path, keep_rust=False, keep_go=True)
    assert result["active"] == "go"
    assert result["available"] == ["go", "python"]
    refused = result["refused"]
    assert isinstance(refused, dict)
    assert refused.pop("go") is None
    for owner, message in refused.items():
        assert message == f"UPDE backend {owner!r} unavailable"


def test_resolver_falls_back_to_python_without_native_owners(tmp_path: Path) -> None:
    """With every native owner absent, Python is active and alone."""
    result = _run_profile(tmp_path, keep_rust=False, keep_go=False)
    assert result["active"] == "python"
    assert result["available"] == ["python"]


def test_resolver_prefers_rust_when_it_is_installed(tmp_path: Path) -> None:
    """With the compiled kernel and Go both present, Rust leads the chain."""
    result = _run_profile(tmp_path, keep_rust=True, keep_go=True)
    assert result["active"] == "rust"
    assert result["available"] == ["rust", "go", "python"]

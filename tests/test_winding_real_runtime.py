# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Actual winding-number runtime contracts

"""Winding numbers through every actual owner, without substitution.

The oracle wraps each increment into ``(−π, π]`` with
:func:`math.remainder`, sums the increments with :func:`math.fsum` and
floors the turn count, in plain loops. Owners are observed with
:mod:`cProfile`: the Rust builtin, or the Python entry function of each
bridge. Child-process profiles remove the compiled kernel by building a
package path without it, so its absence is real.
"""

from __future__ import annotations

import collections
import cProfile
import functools
import json
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor import winding
from scpn_phase_orchestrator.monitor.winding import (
    winding_backend,
    winding_numbers,
    winding_vector,
)
from tests._runtime_owners import NEEDS_KERNEL, owner_params

ROOT = Path(__file__).resolve().parents[1]
OWNERS = owner_params(("python", "rust", "mojo", "julia", "go"))
T = TypeVar("T")

# Calls that prove each native owner counted.
_MARKERS = {
    "rust": "<built-in method spo_kernel.spo_kernel.winding_numbers>",
    "mojo": "winding_numbers_mojo",
    "julia": "winding_numbers_julia",
    "go": "winding_numbers_go",
}


def _oracle(history: list[list[float]]) -> list[int]:
    """Return ``floor(Σ_t wrap(Δθ) / 2π)`` per column with ``wrap ∈ (−π, π]``."""
    turns = []
    for column in range(len(history[0])):
        steps = []
        for before, after in zip(history, history[1:], strict=False):
            step = math.remainder(after[column] - before[column], 2 * math.pi)
            steps.append(math.pi if step == -math.pi else step)
        turns.append(math.floor(math.fsum(steps) / (2 * math.pi)))
    return turns


def _histories() -> list[np.ndarray]:
    """Return seeded drifting trajectories and exact-boundary edge cases."""
    rng = np.random.default_rng(2026)
    drift = np.cumsum(rng.normal([0.9, -0.6, 0.05, 2.0], 0.4, (300, 4)), axis=0)
    return [
        drift,
        np.mod(drift, 2 * np.pi),
        # Exact half-turns count forward.
        np.array([[0.0], [np.pi], [2 * np.pi], [3 * np.pi]]),
        # A small net backward rotation floors to −1, forward to 0.
        np.array([[0.0, 0.0], [-0.01, 0.01]]),
        np.zeros((2, 3)),
    ]


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


@pytest.mark.parametrize("backend", OWNERS)
def test_each_owner_counts_the_oracle_windings(backend: str) -> None:
    """Each named owner counts every history, and only that owner computes."""
    assert winding_backend(backend) == backend
    for history in _histories():
        got, counts = _profiled(
            functools.partial(winding_numbers, history, backend=backend)
        )
        assert got.dtype == np.int64
        assert got.tolist() == _oracle(history.tolist())
        for owner, marker in _MARKERS.items():
            assert counts[marker] == (1 if owner == backend else 0), owner


@pytest.mark.parametrize("backend", OWNERS)
def test_vector_alias_routes_the_named_owner(backend: str) -> None:
    """``winding_vector`` passes ``backend=`` to the same owner."""
    history = _histories()[0]
    got, counts = _profiled(lambda: winding_vector(history, backend=backend))
    assert got.tolist() == _oracle(history.tolist())
    if backend != "python":
        assert counts[_MARKERS[backend]] == 1


def test_auto_owner_is_the_active_backend() -> None:
    """``auto`` names and runs the import-time active owner here."""
    assert winding_backend() == winding.ACTIVE_BACKEND
    history = _histories()[0]
    got, counts = _profiled(lambda: winding_numbers(history))
    assert got.tolist() == _oracle(history.tolist())
    if winding.ACTIVE_BACKEND != "python":
        assert counts[_MARKERS[winding.ACTIVE_BACKEND]] == 1


def test_auto_without_a_loadable_listed_owner_uses_the_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A backend state that lists no loadable owner and no Python ends in Python."""
    monkeypatch.setattr(winding, "ACTIVE_BACKEND", "webgpu")
    monkeypatch.setattr(winding, "AVAILABLE_BACKENDS", ["webgpu"])
    assert winding_backend() == "python"
    history = _histories()[0]
    got, counts = _profiled(lambda: winding_numbers(history))
    assert got.tolist() == _oracle(history.tolist())
    assert not any(counts[marker] for marker in _MARKERS.values())


def test_unknown_owner_is_refused() -> None:
    """An unknown owner name is a ValueError before any counting."""
    with pytest.raises(ValueError, match="unknown winding backend"):
        winding_numbers(np.zeros((3, 2)), backend="cuda")
    with pytest.raises(ValueError, match="unknown winding backend"):
        winding_backend("cuda")


_CHILD = r"""
import json, sys
sys.path[:0] = [sys.argv[1], sys.argv[2]]
import numpy as np
from scpn_phase_orchestrator.monitor import winding as w
out = {"file": w.__file__, "available": w.AVAILABLE_BACKENDS}
out["owner"] = w.winding_backend()
history = np.array(json.loads(sys.argv[3]))
refused = {}
for owner in ("rust", "mojo", "julia", "go"):
    try:
        w.winding_numbers(history, backend=owner)
        refused[owner] = None
    except RuntimeError as exc:
        refused[owner] = str(exc)
out["refused"] = refused
out["auto"] = w.winding_numbers(history).tolist()
print(json.dumps(out))
"""

_PROFILE_HISTORY = [[0.0, 0.0], [2.0, -2.0], [4.0, -4.0], [6.5, -6.5]]


def _run_profile(tmp_path: Path, keep_rust: bool) -> dict[str, object]:
    """Run an isolated interpreter on an exported source copy with one profile.

    The export holds no ``go/``, ``mojo/`` or ``julia/`` assets beside the
    package and ``juliacall`` is left out, so those owners are genuinely
    absent; ``spo_kernel`` is present only when ``keep_rust``.
    """
    site = Path(np.__file__).resolve().parents[1]
    profile = tmp_path / ("rust-only" if keep_rust else "python-only")
    profile.mkdir()
    for entry in site.iterdir():
        name = entry.name.lower()
        if name.startswith(("juliacall", "juliapkg", "scpn_phase_orchestrator")):
            continue
        if name.endswith(".pth") or (name.startswith("spo_kernel") and not keep_rust):
            continue
        (profile / entry.name).symlink_to(entry)
    export = tmp_path / "export" / "src"
    shutil.copytree(
        ROOT / "src" / "scpn_phase_orchestrator",
        export / "scpn_phase_orchestrator",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            _CHILD,
            str(export),
            str(profile),
            json.dumps(_PROFILE_HISTORY),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env={"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path)},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result: dict[str, object] = json.loads(completed.stdout.splitlines()[-1])
    assert Path(str(result["file"])).is_relative_to(tmp_path / "export")
    assert result["auto"] == _oracle(_PROFILE_HISTORY)
    return result


def test_python_only_profile_refuses_every_native_owner(tmp_path: Path) -> None:
    """Without native owners the reference counts and every named owner refuses."""
    result = _run_profile(tmp_path, keep_rust=False)
    assert result["available"] == ["python"]
    assert result["owner"] == "python"
    refused = result["refused"]
    assert isinstance(refused, dict)
    for owner, message in refused.items():
        assert message == f"winding backend {owner!r} unavailable"


@NEEDS_KERNEL
def test_rust_only_profile_serves_rust_and_refuses_the_rest(tmp_path: Path) -> None:
    """With only the compiled kernel present, Rust serves ``auto``."""
    result = _run_profile(tmp_path, keep_rust=True)
    assert result["available"] == ["rust", "python"]
    assert result["owner"] == "rust"
    refused = result["refused"]
    assert isinstance(refused, dict)
    assert refused.pop("rust") is None
    for owner, message in refused.items():
        assert message == f"winding backend {owner!r} unavailable"

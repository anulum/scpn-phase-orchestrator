# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Actual EVS frequency-specificity runtime contracts

"""EVS frequency specificity through both actual owners, without substitution.

The oracle computes ITPC as ``|mean_k exp(iθ_kt)|`` with ``cmath`` in plain
loops. Child-process profiles remove the compiled kernel by building a
package path without it, so its absence is real.
"""

from __future__ import annotations

import cmath
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor import evs
from scpn_phase_orchestrator.monitor.evs import (
    EVSMonitor,
    frequency_specificity_backend,
)

pytestmark = pytest.mark.native_runtime

ROOT = Path(__file__).resolve().parents[1]
OWNERS = ("python", "rust")


def _oracle_specificity(
    trials: list[list[float]], target: float, control: float
) -> float:
    """Return mean target ITPC over mean control ITPC by plain loops."""

    def mean_itpc(rows: list[list[float]]) -> float:
        """Average ``|mean_k exp(iθ_kt)|`` over time points."""
        n = len(rows)
        values = [
            abs(sum(cmath.exp(1j * row[t]) for row in rows) / n)
            for t in range(len(rows[0]))
        ]
        return math.fsum(values) / len(values)

    scaled = [[v * control / target for v in row] for row in trials]
    target_mean, control_mean = mean_itpc(trials), mean_itpc(scaled)
    if control_mean < 1e-12:
        return math.inf if target_mean > 0 else 0.0
    return target_mean / control_mean


def _trials(seed: int, n_trials: int, n_time: int, jitter: float) -> np.ndarray:
    """Return phase-locked trials with Gaussian jitter."""
    rng = np.random.default_rng(seed)
    base = np.linspace(0.0, 4.0 * np.pi, n_time)
    return np.tile(base, (n_trials, 1)) + rng.normal(0.0, jitter, (n_trials, n_time))


@pytest.mark.parametrize("backend", OWNERS)
def test_each_owner_matches_the_independent_oracle(backend: str) -> None:
    """Each named owner serves the ratio and agrees with the plain-loop oracle."""
    assert frequency_specificity_backend(backend) == backend
    for seed, n_trials, n_time, jitter in ((1, 3, 5, 0.1), (2, 40, 64, 0.6)):
        trials = _trials(seed, n_trials, n_time, jitter)
        got = EVSMonitor._frequency_specificity(trials, 10.0, 23.0, backend=backend)
        expected = _oracle_specificity(trials.tolist(), 10.0, 23.0)
        assert got == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("backend", OWNERS)
def test_monitor_routes_its_owner_through_evaluate(backend: str) -> None:
    """``EVSMonitor(backend=...)`` gives the same verdict through either owner."""
    trials = _trials(7, 50, 100, 0.05)
    result = EVSMonitor(specificity_threshold=1.0, backend=backend).evaluate(
        trials, list(range(80, 100)), 10.0, 23.0
    )
    assert result.is_entrained
    assert result.specificity_ratio == pytest.approx(
        _oracle_specificity(trials.tolist(), 10.0, 23.0), rel=1e-12
    )


def test_unknown_owner_is_refused() -> None:
    """An unknown owner name is a ValueError, at construction and per call."""
    with pytest.raises(ValueError, match="unknown EVS backend"):
        EVSMonitor(backend="cuda")
    with pytest.raises(ValueError, match="unknown EVS backend"):
        frequency_specificity_backend("cuda")


_CHILD = r"""
import json, sys
sys.path[:0] = [sys.argv[1], sys.argv[2]]
from scpn_phase_orchestrator.monitor import evs
out = {"file": evs.__file__, "has_rust": evs._HAS_RUST}
out["owner"] = evs.frequency_specificity_backend()
try:
    evs.EVSMonitor(backend="rust")
    out["rust"] = None
except RuntimeError as exc:
    out["rust"] = str(exc)
trials = [[0.0, 1.0, 2.0], [0.1, 1.1, 2.1]]
out["ratio"] = evs.EVSMonitor._frequency_specificity(trials, 10.0, 23.0)
print(json.dumps(out))
"""


def _run_profile(tmp_path: Path, keep_rust: bool) -> dict[str, object]:
    """Run an isolated interpreter on an exported source copy with one profile."""
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
        [sys.executable, "-I", "-S", "-B", "-c", _CHILD, str(export), str(profile)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env={"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path)},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result: dict[str, object] = json.loads(completed.stdout.splitlines()[-1])
    return result


_PROFILE_EXPECTED = _oracle_specificity([[0.0, 1.0, 2.0], [0.1, 1.1, 2.1]], 10.0, 23.0)


def test_python_only_profile_refuses_rust(tmp_path: Path) -> None:
    """Without the compiled kernel, Python serves and a named Rust owner refuses."""
    result = _run_profile(tmp_path, keep_rust=False)
    assert Path(str(result["file"])).is_relative_to(tmp_path / "export")
    assert result["has_rust"] is False
    assert result["owner"] == "python"
    assert result["rust"] == "EVS backend 'rust' unavailable"
    assert result["ratio"] == pytest.approx(_PROFILE_EXPECTED, rel=1e-12)


def test_rust_only_profile_serves_rust(tmp_path: Path) -> None:
    """With the compiled kernel installed, Rust serves ``auto``."""
    result = _run_profile(tmp_path, keep_rust=True)
    assert result["has_rust"] is True
    assert result["owner"] == "rust"
    assert result["rust"] is None
    assert result["ratio"] == pytest.approx(_PROFILE_EXPECTED, rel=1e-12)


def test_auto_owner_matches_the_installed_kernel() -> None:
    """In this interpreter ``auto`` names Rust exactly when the kernel loaded."""
    assert frequency_specificity_backend() == ("rust" if evs._HAS_RUST else "python")

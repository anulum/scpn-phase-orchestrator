# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Actual entropy-production runtime contracts

"""Entropy production through every actual owner, without substitution.

The oracle is written independently of the package: a plain Python double
loop over ``math.sin`` with ``math.fsum`` sums. Child-process profiles remove
native owners by building a package path without them, so their absence is
real.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor import entropy_prod
from scpn_phase_orchestrator.monitor.entropy_prod import (
    entropy_production_backend,
    entropy_production_rate,
)

pytestmark = pytest.mark.native_runtime

ROOT = Path(__file__).resolve().parents[1]
OWNERS = ("python", "rust", "mojo", "julia", "go")
RELATIVE_TOLERANCE = 1e-12


def _oracle(
    phases: list[float],
    omegas: list[float],
    knm: list[list[float]],
    alpha: float,
    dt: float,
) -> float:
    """Return ``Σ_i (ω_i + α/N Σ_j K_ij sin(θ_j − θ_i))² · dt`` by plain loops."""
    n = len(phases)
    if n == 0 or dt == 0.0:
        return 0.0
    rates = [
        omegas[i]
        + alpha
        / n
        * math.fsum(knm[i][j] * math.sin(phases[j] - phases[i]) for j in range(n))
        for i in range(n)
    ]
    return math.fsum(r * r for r in rates) * dt


def _cases() -> list[tuple[list[float], list[float], list[list[float]], float, float]]:
    """Return seeded cases from one oscillator to a sixteen-oscillator network."""
    rng = np.random.default_rng(2026)
    cases = []
    for n, alpha, dt in (
        (1, 1.0, 0.01),
        (2, 0.5, 0.1),
        (7, 2.0, 0.003),
        (16, 1.3, 0.02),
    ):
        cases.append(
            (
                rng.uniform(-math.pi, math.pi, n).tolist(),
                rng.normal(0.0, 1.5, n).tolist(),
                rng.uniform(-1.0, 3.0, (n, n)).tolist(),
                alpha,
                dt,
            )
        )
    return cases


@pytest.mark.parametrize("backend", OWNERS)
def test_each_owner_matches_the_independent_oracle(backend: str) -> None:
    """Each named owner serves the call and agrees with the plain-loop oracle."""
    assert entropy_production_backend(backend) == backend
    for phases, omegas, knm, alpha, dt in _cases():
        rate = entropy_production_rate(
            np.array(phases),
            np.array(omegas),
            np.array(knm),
            alpha,
            dt,
            backend=backend,
        )
        expected = _oracle(phases, omegas, knm, alpha, dt)
        assert rate == pytest.approx(expected, rel=RELATIVE_TOLERANCE, abs=1e-300)


@pytest.mark.parametrize("backend", OWNERS)
def test_each_owner_returns_zero_at_a_locked_state(backend: str) -> None:
    """Equal phases and zero frequencies dissipate nothing in every owner."""
    rate = entropy_production_rate(
        np.full(5, 0.7), np.zeros(5), np.ones((5, 5)), 2.0, 0.01, backend=backend
    )
    assert rate == 0.0


@pytest.mark.parametrize("backend", OWNERS)
def test_each_owner_leaves_its_inputs_untouched(backend: str) -> None:
    """No owner writes into the caller's arrays."""
    phases, omegas, knm, alpha, dt = _cases()[2]
    arrays = (np.array(phases), np.array(omegas), np.array(knm))
    before = [a.copy() for a in arrays]
    entropy_production_rate(*arrays, alpha, dt, backend=backend)
    for after, original in zip(arrays, before, strict=True):
        np.testing.assert_array_equal(after, original)


def test_auto_selection_uses_the_active_owner() -> None:
    """``auto`` names the active owner and gives that owner's result."""
    assert entropy_production_backend() == entropy_prod.ACTIVE_BACKEND
    phases, omegas, knm, alpha, dt = _cases()[3]
    automatic = entropy_production_rate(phases, omegas, knm, alpha, dt)
    explicit = entropy_production_rate(
        phases, omegas, knm, alpha, dt, backend=entropy_prod.ACTIVE_BACKEND
    )
    assert automatic == explicit


_CHILD = r"""
import json, sys
sys.path[:0] = [sys.argv[1], sys.argv[2]]
from scpn_phase_orchestrator.monitor import entropy_prod as ep
out = {
    "file": ep.__file__,
    "active": ep.ACTIVE_BACKEND,
    "available": ep.AVAILABLE_BACKENDS,
    "owner": ep.entropy_production_backend(),
    "refused": {},
}
case = ([0.0, 1.0], [1.0, 2.0], [[0.0, 1.0], [1.0, 0.0]], 1.0, 0.1)
for owner in ("rust", "mojo", "julia", "go"):
    try:
        ep.entropy_production_rate(*case, backend=owner)
        out["refused"][owner] = None
    except RuntimeError as exc:
        out["refused"][owner] = str(exc)
out["rate"] = ep.entropy_production_rate(*case)
print(json.dumps(out))
"""


def _profile_site(tmp_path: Path, keep_rust: bool) -> Path:
    """Build a package path holding every installed entry except native owners."""
    site = Path(np.__file__).resolve().parents[1]
    profile = tmp_path / ("rust-only" if keep_rust else "python-only")
    profile.mkdir()
    excluded = ("juliacall", "juliapkg", "scpn_phase_orchestrator")
    for entry in site.iterdir():
        name = entry.name.lower()
        if name.startswith(excluded) or name.endswith(".pth"):
            continue
        if name.startswith("spo_kernel") and not keep_rust:
            continue
        (profile / entry.name).symlink_to(entry)
    return profile


def _run_profile(tmp_path: Path, keep_rust: bool) -> dict[str, object]:
    """Run an isolated interpreter on an exported source copy with one profile.

    The export holds no ``go/``, ``mojo/`` or ``julia/`` assets beside the
    package, so those owners are genuinely absent from the child.
    """
    site = _profile_site(tmp_path, keep_rust)
    export = tmp_path / "export" / "src"
    shutil.copytree(
        ROOT / "src" / "scpn_phase_orchestrator",
        export / "scpn_phase_orchestrator",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", _CHILD, str(export), str(site)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env={"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path)},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result: dict[str, object] = json.loads(completed.stdout.splitlines()[-1])
    return result


_PROFILE_EXPECTED = _oracle([0.0, 1.0], [1.0, 2.0], [[0.0, 1.0], [1.0, 0.0]], 1.0, 0.1)


def test_python_only_profile_refuses_every_native_owner(tmp_path: Path) -> None:
    """Without native owners the NumPy law serves and every named owner refuses."""
    result = _run_profile(tmp_path, keep_rust=False)
    assert Path(str(result["file"])).is_relative_to(tmp_path / "export")
    assert result["active"] == "python"
    assert result["available"] == ["python"]
    assert result["owner"] == "python"
    refused = result["refused"]
    assert isinstance(refused, dict)
    for owner, message in refused.items():
        assert message == f"entropy production backend {owner!r} unavailable"
    assert result["rate"] == pytest.approx(_PROFILE_EXPECTED, rel=RELATIVE_TOLERANCE)


def test_rust_only_profile_serves_rust_and_refuses_the_rest(tmp_path: Path) -> None:
    """With only Rust present, Rust serves ``auto`` and the other owners refuse."""
    result = _run_profile(tmp_path, keep_rust=True)
    assert result["active"] == "rust"
    assert result["available"] == ["rust", "python"]
    assert result["owner"] == "rust"
    refused = result["refused"]
    assert isinstance(refused, dict)
    assert refused["rust"] is None
    for owner in ("mojo", "julia", "go"):
        assert refused[owner] == f"entropy production backend {owner!r} unavailable"
    assert result["rate"] == pytest.approx(_PROFILE_EXPECTED, rel=RELATIVE_TOLERANCE)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Actual free-energy runtime contracts

"""Free-energy helpers through both actual owners, without substitution.

Oracles use the standard library only: ``math.exp`` with the documented
clamp, and ``statistics`` for the effective temperature. Child-process
profiles remove the compiled kernel by building a package path without it.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scpn_phase_orchestrator.ssgf import free_energy
from scpn_phase_orchestrator.ssgf.free_energy import (
    add_langevin_noise,
    boltzmann_weight,
    effective_temperature,
    free_energy_backend,
)

pytestmark = pytest.mark.native_runtime

ROOT = Path(__file__).resolve().parents[1]
OWNERS = ("python", "rust")


def _oracle_weight(u_total: float, temperature: float) -> float:
    """Return exp(-U/T) with the exponent clamped to [-700, 700]."""
    if temperature <= 0.0:
        return 1.0 if u_total <= 0.0 else 0.0
    return math.exp(max(-700.0, min(700.0, -u_total / temperature)))


def _oracle_temperature(costs: list[float]) -> float:
    """Return Var(U) / (2 |<U>|), or 0 for a short or zero-mean history."""
    if len(costs) < 2:
        return 0.0
    mean = statistics.fmean(costs)
    return 0.0 if abs(mean) < 1e-30 else statistics.variance(costs) / (2 * abs(mean))


@pytest.mark.parametrize("backend", OWNERS)
def test_each_owner_weights_by_the_boltzmann_law(backend: str) -> None:
    """Each owner matches the clamped Boltzmann oracle, including T = 0."""
    assert free_energy_backend(backend) == backend
    for u_total, temperature in (
        (2.0, 8.0),
        (-3.0, 0.5),
        (0.0, 1.0),
        (1.0, 0.0),
        (0.0, 0.0),
        (-1.0, 0.0),
        (1e10, 1e-10),
        (-1e10, 1e-10),
    ):
        got = boltzmann_weight(u_total, temperature, backend=backend)
        assert got == pytest.approx(_oracle_weight(u_total, temperature), rel=1e-15)


@pytest.mark.parametrize("backend", OWNERS)
def test_each_owner_estimates_the_effective_temperature(backend: str) -> None:
    """Each owner matches the ``statistics`` oracle on 1-D and 2-D histories."""
    rng = np.random.default_rng(2026)
    histories = [
        np.array([5.0, 5.0, 5.0]),
        np.array([1.0, -1.0, 1.0, -1.0]),
        10.0 + rng.standard_normal(500),
        rng.uniform(1.0, 4.0, (20, 3)),
    ]
    for costs in histories:
        got = effective_temperature(costs, backend=backend)
        expected = _oracle_temperature(costs.ravel().tolist())
        assert got == pytest.approx(expected, rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("backend", OWNERS)
def test_each_owner_adds_noise_of_the_langevin_variance(backend: str) -> None:
    """Each owner adds zero-mean noise of variance 2·T·dt and keeps the input."""
    z = np.full((120, 250), 3.0)
    got = add_langevin_noise(z, 0.8, 0.05, backend=backend)
    assert got.shape == z.shape
    # 30 000 draws: within 3 % of 0.08 and a mean shift below 0.01.
    assert float(np.var(got)) == pytest.approx(2 * 0.8 * 0.05, rel=0.03)
    assert abs(float(np.mean(got)) - 3.0) < 0.01
    assert np.all(z == 3.0)


def test_rust_refuses_a_caller_generator() -> None:
    """The Rust kernel seeds itself, so a caller ``rng`` is refused by name."""
    if not free_energy._HAS_RUST:
        pytest.skip("Rust backend not built")
    with pytest.raises(ValueError, match="rng is only supported by the python"):
        add_langevin_noise(
            np.zeros(3), 1.0, 0.1, rng=np.random.default_rng(), backend="rust"
        )


def test_named_rust_owner_without_the_kernel_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Naming Rust when its kernels are absent raises for every function."""
    monkeypatch.setattr(free_energy, "_HAS_RUST", False)
    for call in (
        lambda: free_energy_backend("rust"),
        lambda: boltzmann_weight(1.0, 1.0, backend="rust"),
        lambda: effective_temperature(np.ones(3), backend="rust"),
        lambda: add_langevin_noise(np.ones(3), 1.0, 0.1, backend="rust"),
    ):
        with pytest.raises(
            RuntimeError, match="free-energy backend 'rust' unavailable"
        ):
            call()
    assert free_energy_backend() == "python"


def test_unknown_owner_is_refused() -> None:
    """An unknown owner name is a ValueError."""
    with pytest.raises(ValueError, match="unknown free-energy backend"):
        boltzmann_weight(1.0, 1.0, backend="cuda")


_CHILD = r"""
import json, sys
sys.path[:0] = [sys.argv[1], sys.argv[2]]
from scpn_phase_orchestrator.ssgf import free_energy as fe
out = {"file": fe.__file__, "has_rust": fe._HAS_RUST, "owner": fe.free_energy_backend()}
try:
    fe.boltzmann_weight(2.0, 8.0, backend="rust")
    out["rust"] = None
except RuntimeError as exc:
    out["rust"] = str(exc)
out["weight"] = fe.boltzmann_weight(2.0, 8.0)
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


@pytest.mark.parametrize("keep_rust", [False, True], ids=["python-only", "rust-only"])
def test_profiles_select_the_installed_owner(tmp_path: Path, keep_rust: bool) -> None:
    """``auto`` follows the installation; a named Rust owner refuses without it."""
    result = _run_profile(tmp_path, keep_rust=keep_rust)
    assert Path(str(result["file"])).is_relative_to(tmp_path / "export")
    assert result["has_rust"] is keep_rust
    assert result["owner"] == ("rust" if keep_rust else "python")
    expected_refusal = None if keep_rust else "free-energy backend 'rust' unavailable"
    assert result["rust"] == expected_refusal
    assert result["weight"] == pytest.approx(math.exp(-0.25), rel=1e-15)

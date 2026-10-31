# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Actual amplitude-envelope runtime contracts

"""Amplitude envelope through every actual owner, without substitution.

The oracles are written independently of the package: the RMS envelope is a
plain loop over each trailing window with ``math.fsum``, and the modulation
depth uses ``max``/``min`` over a Python list. Child-process profiles remove
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

from scpn_phase_orchestrator.upde import envelope
from scpn_phase_orchestrator.upde.envelope import (
    envelope_backend,
    envelope_modulation_depth,
    extract_envelope,
)
from tests._runtime_owners import NEEDS_KERNEL, owner_params

ROOT = Path(__file__).resolve().parents[1]
OWNERS = owner_params(("python", "rust", "mojo", "julia", "go"))
RELATIVE_TOLERANCE = 1e-12


def _oracle_envelope(values: list[float], window: int) -> list[float]:
    """Return the trailing-window RMS by plain loops, front-padded with the first."""
    rms = [
        math.sqrt(math.fsum(v * v for v in values[i - window + 1 : i + 1]) / window)
        for i in range(window - 1, len(values))
    ]
    return [rms[0]] * (window - 1) + rms


def _cumulative_sum_tolerance(values: list[float], window: int) -> float:
    """Return the absolute error the O(T) cumulative-sum form may carry.

    Each window's sum of squares is a difference of two running sums, so its
    rounding error grows with the whole trace's sum of squares, not with the
    window's. Propagated through the square root this is at most
    ``T · eps · Σx² / (window · 2 · rms_min)``; the bound below uses the
    smallest positive sample for ``rms_min``.
    """
    squares = math.fsum(v * v for v in values)
    smallest = min((abs(v) for v in values if v != 0.0), default=1.0)
    return len(values) * sys.float_info.epsilon * squares / (2.0 * window * smallest)


def _oracle_depth(values: list[float]) -> float:
    """Return ``(max − min) / (max + min)``, or 0 for a non-positive sum."""
    high, low = max(values), min(values)
    return 0.0 if high + low <= 0.0 else (high - low) / (high + low)


def _signals() -> list[list[float]]:
    """Return seeded positive amplitude traces of several lengths."""
    rng = np.random.default_rng(2026)
    return [
        [3.0, 4.0],
        np.abs(rng.normal(1.0, 0.3, 37)).tolist(),
        np.abs(rng.normal(2.0, 1.0, 400)).tolist(),
    ]


@pytest.mark.parametrize("backend", OWNERS)
def test_each_owner_extracts_the_oracle_envelope(backend: str) -> None:
    """Each named owner serves the 1-D extraction and matches the oracle."""
    assert envelope_backend("extract", backend) == backend
    for signal in _signals():
        for window in (1, 2, len(signal) - 1):
            if window < 1 or window >= len(signal):
                continue
            got = extract_envelope(np.array(signal), window, backend=backend)
            assert got.dtype == np.float64
            np.testing.assert_allclose(
                got,
                _oracle_envelope(signal, window),
                rtol=RELATIVE_TOLERANCE,
                atol=_cumulative_sum_tolerance(signal, window),
            )


@pytest.mark.parametrize("backend", OWNERS)
def test_each_owner_measures_the_oracle_depth(backend: str) -> None:
    """Each named owner serves the modulation depth and matches the oracle."""
    assert envelope_backend("mod", backend) == backend
    for signal in [*_signals(), [0.0, 0.0, 0.0], [2.0, 2.0]]:
        depth = envelope_modulation_depth(np.array(signal), backend=backend)
        assert depth == pytest.approx(_oracle_depth(signal), rel=RELATIVE_TOLERANCE)


@pytest.mark.parametrize("backend", OWNERS)
def test_each_owner_leaves_its_input_untouched(backend: str) -> None:
    """No owner writes into the caller's array."""
    signal = np.array(_signals()[1])
    before = signal.copy()
    extract_envelope(signal, 5, backend=backend)
    envelope_modulation_depth(signal, backend=backend)
    np.testing.assert_array_equal(signal, before)


def test_auto_selection_uses_the_active_owner() -> None:
    """``auto`` names the probe-selected owner and gives its result."""
    assert envelope_backend("extract") == envelope.ACTIVE_BACKEND
    signal = np.array(_signals()[2])
    automatic = extract_envelope(signal, 9)
    explicit = extract_envelope(signal, 9, backend=envelope.ACTIVE_BACKEND)
    np.testing.assert_array_equal(automatic, explicit)


def test_unknown_names_are_refused() -> None:
    """Unknown owners and operations are ValueErrors."""
    with pytest.raises(ValueError, match="unknown envelope backend"):
        extract_envelope(np.ones(4), 2, backend="cuda")
    with pytest.raises(ValueError, match="unknown envelope backend"):
        envelope_modulation_depth(np.ones(4), backend="cuda")
    with pytest.raises(ValueError, match="unknown envelope operation"):
        envelope_backend("spectrum")


_CHILD = r"""
import json, sys
sys.path[:0] = [sys.argv[1], sys.argv[2]]
import numpy as np
from scpn_phase_orchestrator.upde import envelope as ev
out = {
    "file": ev.__file__,
    "available": ev.AVAILABLE_BACKENDS,
    "owners": {slot: ev.envelope_backend(slot) for slot in ("extract", "mod")},
    "refused": {},
}
signal = np.array([3.0, 4.0, 0.0, 5.0])
for owner in ("rust", "mojo", "julia", "go"):
    try:
        ev.extract_envelope(signal, 2, backend=owner)
        out["refused"][owner] = None
    except RuntimeError as exc:
        out["refused"][owner] = str(exc)
out["envelope"] = ev.extract_envelope(signal, 2).tolist()
out["depth"] = ev.envelope_modulation_depth(signal)
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


_PROFILE_SIGNAL = [3.0, 4.0, 0.0, 5.0]


def _assert_profile_values(result: dict[str, object]) -> None:
    """Check the child's envelope and depth against the oracles."""
    np.testing.assert_allclose(
        np.asarray(result["envelope"]),
        _oracle_envelope(_PROFILE_SIGNAL, 2),
        rtol=RELATIVE_TOLERANCE,
        atol=0.0,
    )
    assert result["depth"] == pytest.approx(_oracle_depth(_PROFILE_SIGNAL))


def test_python_only_profile_refuses_every_native_owner(tmp_path: Path) -> None:
    """Without native owners the NumPy law serves and every named owner refuses."""
    result = _run_profile(tmp_path, keep_rust=False)
    assert Path(str(result["file"])).is_relative_to(tmp_path / "export")
    assert result["available"] == ["python"]
    assert result["owners"] == {"extract": "python", "mod": "python"}
    refused = result["refused"]
    assert isinstance(refused, dict)
    for owner, message in refused.items():
        assert message == f"envelope backend {owner!r} unavailable"
    _assert_profile_values(result)


@NEEDS_KERNEL
def test_rust_only_profile_serves_rust_and_refuses_the_rest(tmp_path: Path) -> None:
    """With only Rust present, a named Rust call runs and the others refuse."""
    result = _run_profile(tmp_path, keep_rust=True)
    assert result["available"] == ["rust", "python"]
    refused = result["refused"]
    assert isinstance(refused, dict)
    assert refused["rust"] is None
    for owner in ("mojo", "julia", "go"):
        assert refused[owner] == f"envelope backend {owner!r} unavailable"
    _assert_profile_values(result)

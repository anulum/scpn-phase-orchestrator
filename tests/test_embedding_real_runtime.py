# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Actual delay-embedding runtime contracts

"""Delay-embedding regressions through every actual owner, without substitution.

The oracles below are written independently of the package: histogram bins
use ``bisect`` over explicitly built edges, mutual information sums exact
count ratios with ``math.fsum``, and the false-nearest-neighbour search is a
plain Python loop. Child-process profiles remove native owners by building a
package path without them, so their absence is real.
"""

from __future__ import annotations

import bisect
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor.embedding import (
    auto_embed,
    delay_embed,
    embedding_backend,
    mutual_information,
    nearest_neighbor_distances,
    optimal_delay,
    optimal_dimension,
)

pytestmark = pytest.mark.native_runtime

ROOT = Path(__file__).resolve().parents[1]
OWNERS = ("python", "rust", "mojo", "julia", "go")
MI_OWNERS = ("python", "mojo", "julia", "go")


def _oracle_bins(values: list[float], n_bins: int) -> list[int]:
    """Return NumPy-law histogram bins built from explicit edges."""
    lo, hi = min(values), max(values)
    step = (hi - lo) / n_bins
    edges = [k * step + lo for k in range(n_bins)] + [hi]
    return [min(bisect.bisect_right(edges, v) - 1, n_bins - 1) for v in values]


def _oracle_mi(signal: list[float], lag: int, n_bins: int) -> float:
    """Return the mutual information of exact bin counts."""
    t = len(signal) - lag
    x, y = signal[:t], signal[lag : lag + t]
    if max(x) == min(x) or max(y) == min(y):
        return 0.0
    pairs = list(zip(_oracle_bins(x, n_bins), _oracle_bins(y, n_bins), strict=True))
    joint: dict[tuple[int, int], int] = {}
    for pair in pairs:
        joint[pair] = joint.get(pair, 0) + 1
    cx = [sum(c for (a, _), c in joint.items() if a == i) for i in range(n_bins)]
    cy = [sum(c for (_, b), c in joint.items() if b == j) for j in range(n_bins)]
    return math.fsum(
        (c / t) * math.log((c / t) / ((cx[a] / t) * (cy[b] / t)))
        for (a, b), c in joint.items()
    )


def _oracle_first_minimum(signal: list[float], max_lag: int, n_bins: int) -> int:
    """Return the first strict local minimum of the oracle MI curve, else 1."""
    curve = [
        _oracle_mi(signal, lag, n_bins) for lag in range(min(max_lag, len(signal) // 2))
    ]
    for i in range(1, len(curve) - 1):
        if curve[i] < curve[i - 1] and curve[i] < curve[i + 1]:
            return i
    return 1


def _oracle_fnn(signal: list[float], delay: int, max_dim: int) -> int:
    """Return the Kennel-Brown-Abarbanel dimension with tolerances 15 and 2."""
    mean = math.fsum(signal) / len(signal)
    sigma = math.sqrt(math.fsum((v - mean) ** 2 for v in signal) / len(signal))
    for m in range(1, max_dim + 1):
        rows = len(signal) - (m - 1) * delay
        if len(signal) - m * delay <= 1:
            return m
        points = [[signal[i + k * delay] for k in range(m)] for i in range(rows)]
        false = valid = 0
        for i, p in enumerate(points):
            squared = [
                math.inf
                if j == i
                else sum((a - b) ** 2 for a, b in zip(p, q, strict=True))
                for j, q in enumerate(points)
            ]
            j = squared.index(min(squared))
            d = math.sqrt(squared[j])
            if d == 0 or i + m * delay >= len(signal) or j + m * delay >= len(signal):
                continue
            valid += 1
            extra = abs(signal[i + m * delay] - signal[j + m * delay])
            if extra / d > 15.0 or math.hypot(d, extra) / sigma > 2.0:
                false += 1
        if (false / valid if valid else 0.0) < 0.01:
            return m
    return max_dim


def _edge_rich_signals() -> list[list[float]]:
    """Return signals whose samples often lie exactly on histogram edges."""
    rng = np.random.default_rng(2026)
    signals = [[0.30000000000000004, 0.1, 0.2, 0.15000000000000002]]
    for scale in (0.1, 0.3, 1 / 3, 0.7, 2.5):
        signals.append((rng.integers(0, 9, size=64) * scale).tolist())
    signals.append(rng.normal(size=96).tolist())
    return signals


@pytest.mark.parametrize("backend", OWNERS)
def test_delay_embed_returns_exact_windows(backend: str) -> None:
    """Each owner returns ``signal[i + k*delay]`` exactly and leaves input intact."""
    signal = np.array([3.5, -1.0, 2.25, 9.0, 0.0, 7.75, -4.5])
    before = signal.copy()
    assert embedding_backend("de", backend) == backend
    expected = [[signal[i + k * 2] for k in range(3)] for i in range(3)]
    np.testing.assert_array_equal(delay_embed(signal, 2, 3, backend=backend), expected)
    np.testing.assert_array_equal(signal, before)


@pytest.mark.parametrize("backend", MI_OWNERS)
def test_mutual_information_counts_interior_edge_samples(backend: str) -> None:
    """Edge samples open their own bins, so lag 0 gives the entropy ``log 4``."""
    signal = [0.30000000000000004, 0.1, 0.2, 0.15000000000000002]
    assert _oracle_mi(signal, 0, 4) == math.log(4.0)
    assert mutual_information(signal, 0, 4, backend=backend) == math.log(4.0)


@pytest.mark.parametrize("backend", MI_OWNERS)
def test_mutual_information_bins_each_window_over_its_own_range(backend: str) -> None:
    """``x = [0, 1]`` and ``y = [10, 20]`` at lag 4 give ``log 2``."""
    value = mutual_information([0.0, 1.0, 0.0, 1.0, 10.0, 20.0], 4, 2, backend=backend)
    assert value == pytest.approx(math.log(2.0), rel=0.0, abs=1e-15)


@pytest.mark.parametrize("backend", MI_OWNERS)
def test_mutual_information_agrees_with_independent_counts(backend: str) -> None:
    """Every owner equals the exact-count oracle on edge-rich signals."""
    for signal in _edge_rich_signals():
        for lag, n_bins in ((0, 4), (1, 7), (3, 16)):
            expected = _oracle_mi(signal, lag, n_bins)
            actual = mutual_information(signal, lag, n_bins, backend=backend)
            assert actual == pytest.approx(expected, rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("backend", MI_OWNERS)
def test_mutual_information_of_a_constant_window_is_zero(backend: str) -> None:
    """A constant x window carries no information about y."""
    assert mutual_information([1.0, 1.0, 1.0, 5.0], 1, 4, backend=backend) == 0.0
    assert mutual_information([2.0, 3.0], 2, 4, backend=backend) == 0.0


@pytest.mark.parametrize("backend", MI_OWNERS)
def test_nearest_neighbours_on_a_line(backend: str) -> None:
    """Points 0, 1, 3, 7 have neighbours 1, 0, 1, 3 at distances 1, 1, 2, 4."""
    dist, idx = nearest_neighbor_distances(
        [[0.0], [1.0], [3.0], [7.0]], backend=backend
    )
    np.testing.assert_array_equal(dist, [1.0, 1.0, 2.0, 4.0])
    np.testing.assert_array_equal(idx, [1, 0, 1, 2])


@pytest.mark.parametrize("backend", MI_OWNERS)
def test_nearest_neighbour_ties_take_the_first_minimum(backend: str) -> None:
    """The middle of three equally spaced points picks its lower neighbour."""
    dist, idx = nearest_neighbor_distances(
        [[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]], backend=backend
    )
    np.testing.assert_array_equal(dist, [5.0, 5.0, 5.0])
    np.testing.assert_array_equal(idx, [1, 0, 1])


@pytest.mark.parametrize("backend", OWNERS)
def test_optimal_delay_matches_independent_first_minimum(backend: str) -> None:
    """Rust's native search and the composed searches equal the oracle curve."""
    rng = np.random.default_rng(7)
    for frequency in (0.21, 0.37):
        signal = (
            np.sin(frequency * np.arange(120)) + 0.05 * rng.normal(size=120)
        ).tolist()
        expected = _oracle_first_minimum(signal, 30, 12)
        assert optimal_delay(signal, 30, 12, backend=backend) == expected


@pytest.mark.parametrize("backend", OWNERS)
def test_optimal_dimension_matches_independent_false_neighbours(backend: str) -> None:
    """Every owner's FNN dimension equals the plain-loop oracle."""
    rng = np.random.default_rng(11)
    for signal in (np.sin(0.3 * np.arange(90)).tolist(), rng.normal(size=70).tolist()):
        assert optimal_dimension(signal, 2, 5, backend=backend) == _oracle_fnn(
            signal, 2, 5
        )


@pytest.mark.parametrize("backend", OWNERS)
def test_auto_embed_is_owner_independent(backend: str) -> None:
    """Every owner selects the same delay, dimension and trajectory."""
    signal = np.sin(np.linspace(0.0, 30.0, 300))
    reference = auto_embed(signal, 40, 6, backend="python")
    result = auto_embed(signal, 40, 6, backend=backend)
    assert (result.delay, result.dimension) == (reference.delay, reference.dimension)
    np.testing.assert_array_equal(result.trajectory, reference.trajectory)


_CHILD = r"""
import json, sys
sys.path[:0] = sys.argv[1:3]
import numpy as np
from scpn_phase_orchestrator.monitor import embedding as em
slots = ("de", "mi", "nn", "optimal_delay", "optimal_dimension")
out = {
    "file": em.__file__,
    "active": em.ACTIVE_BACKEND,
    "available": em.AVAILABLE_BACKENDS,
    "owners": {slot: em.embedding_backend(slot) for slot in slots},
    "refused": {},
}
for owner in ("rust", "mojo", "julia", "go"):
    try:
        em.delay_embed([0.0, 1.0, 2.0], 1, 2, backend=owner)
        out["refused"][owner] = None
    except RuntimeError as exc:
        out["refused"][owner] = str(exc)
signal = np.sin(0.3 * np.arange(90))
out["delay"] = em.optimal_delay(signal, 30, 12)
out["dimension"] = em.optimal_dimension(signal, 2, 5)
edge = [0.30000000000000004, 0.1, 0.2, 0.15000000000000002]
out["mi"] = em.mutual_information(edge, 0, 4)
print(json.dumps(out))
"""


def _profile_site(tmp_path: Path, keep_rust: bool) -> Path:
    """Build a package path holding every installed entry except native owners."""
    import numpy

    site = Path(numpy.__file__).resolve().parents[1]
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


def test_python_only_profile_refuses_absent_owners(tmp_path: Path) -> None:
    """Without native owners every operation and search runs the Python law."""
    result = _run_profile(tmp_path, keep_rust=False)
    assert Path(str(result["file"])).is_relative_to(tmp_path / "export")
    assert result["active"] == "python"
    assert result["available"] == ["python"]
    owners, refused = result["owners"], result["refused"]
    assert isinstance(owners, dict)
    assert isinstance(refused, dict)
    assert set(owners.values()) == {"python"}
    for owner, message in refused.items():
        assert message == f"embedding backend {owner!r} unavailable"
    signal = np.sin(0.3 * np.arange(90)).tolist()
    assert result["delay"] == _oracle_first_minimum(signal, 30, 12)
    assert result["dimension"] == _oracle_fnn(signal, 2, 5)
    assert result["mi"] == math.log(4.0)


def test_rust_only_profile_owns_embedding_and_searches(tmp_path: Path) -> None:
    """With only Rust present, MI and kNN fall to Python and the searches use Rust."""
    result = _run_profile(tmp_path, keep_rust=True)
    assert result["active"] == "rust"
    assert result["available"] == ["rust", "python"]
    assert result["owners"] == {
        "de": "rust",
        "mi": "python",
        "nn": "python",
        "optimal_delay": "rust",
        "optimal_dimension": "rust",
    }
    refused = result["refused"]
    assert isinstance(refused, dict)
    assert refused["rust"] is None
    signal = np.sin(0.3 * np.arange(90)).tolist()
    assert result["delay"] == _oracle_first_minimum(signal, 30, 12)
    assert result["dimension"] == _oracle_fnn(signal, 2, 5)

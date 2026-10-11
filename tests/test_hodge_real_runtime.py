# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Public Hodge runtime contracts

"""Public runtime selection and analytic coupling-flow contracts."""

from __future__ import annotations

import cProfile
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from scpn_phase_orchestrator.coupling import hodge_decomposition


@pytest.mark.parametrize("backend", [None, "python"])
def test_two_node_directed_coupling_has_exact_gradient(backend: str | None) -> None:
    """A single alternating edge carries only a potential difference."""
    phases = np.array([0.3, -0.2], dtype=np.float64)
    coupling = np.array([[0.0, 3.0], [1.0, 0.0]], dtype=np.float64)
    result = hodge_decomposition(coupling, phases, backend=backend)
    flow = 2.0 * np.sin(-0.5)
    expected = np.array([[0.0, flow], [-flow, 0.0]])
    np.testing.assert_allclose(result.gradient, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(result.flow, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(result.curl, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.harmonic, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.potential, [-flow / 2.0, flow / 2.0])
    assert result.betti_one == 0


@pytest.mark.native_runtime
def test_explicit_rust_repeated_public_calls_execute_original_kernel() -> None:
    """Both public calls must execute the actual built-in Rust operation."""
    coupling = np.array([[0.0, 3.0], [1.0, 0.0]], dtype=np.float64)
    phases = np.array([0.3, -0.2], dtype=np.float64)
    profile = cProfile.Profile()
    profile.enable()
    try:
        first = hodge_decomposition(coupling, phases, backend="rust")
        second = hodge_decomposition(coupling, phases, backend="rust")
    finally:
        profile.disable()
    calls = sum(
        entry.callcount
        for entry in profile.getstats()
        if isinstance(entry.code, str)
        and "<built-in method spo_kernel." in entry.code
        and "hodge_decomposition_rust>" in entry.code
    )
    assert calls == 2
    expected_flow = 2.0 * np.sin(-0.5)
    for result in (first, second):
        assert result.gradient[0, 1] == pytest.approx(expected_flow)
        np.testing.assert_allclose(result.gradient, result.flow, atol=1e-12)
        np.testing.assert_allclose(result.curl, 0.0, atol=1e-12)
        np.testing.assert_allclose(result.harmonic, 0.0, atol=1e-12)


def test_unknown_backend_is_refused_even_for_empty_input() -> None:
    """An empty problem must not conceal an invalid runtime request."""
    with pytest.raises(ValueError, match="unknown Hodge backend"):
        hodge_decomposition(np.zeros((0, 0)), np.array([]), backend="unknown")


@pytest.mark.parametrize("triangles", [None, []])
def test_python_triangle_circulation_preserves_topological_boundary(
    triangles: list[tuple[int, int, int]] | None,
) -> None:
    """Filling a cyclic triangle moves its circulation from harmonic to curl."""
    phases = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
    coupling = np.ones((3, 3)) - np.eye(3)
    result = hodge_decomposition(coupling, phases, triangles, backend="python")
    circulation = np.sqrt(3.0) / 2.0
    flow = circulation * np.array(
        [[0.0, 1.0, -1.0], [-1.0, 0.0, 1.0], [1.0, -1.0, 0.0]]
    )
    np.testing.assert_allclose(result.flow, flow, atol=1e-12)
    np.testing.assert_allclose(result.gradient, 0.0, atol=1e-12)
    if triangles is None:
        np.testing.assert_allclose(result.curl, flow, atol=1e-12)
        np.testing.assert_allclose(result.harmonic, 0.0, atol=1e-12)
        assert result.betti_one == 0
    else:
        np.testing.assert_allclose(result.curl, 0.0, atol=1e-12)
        np.testing.assert_allclose(result.harmonic, flow, atol=1e-12)
        assert result.betti_one == 1


@pytest.mark.native_runtime
@pytest.mark.parametrize("backend", ["rust", "go", "julia", "mojo"])
def test_named_backend_recovers_triangle_circulation(backend: str) -> None:
    """An explicitly selected native owner must return the filled curl flow."""
    phases = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
    coupling = np.ones((3, 3)) - np.eye(3)
    result = hodge_decomposition(coupling, phases, backend=backend)
    circulation = np.sqrt(3.0) / 2.0
    flow = circulation * np.array(
        [[0.0, 1.0, -1.0], [-1.0, 0.0, 1.0], [1.0, -1.0, 0.0]]
    )
    np.testing.assert_allclose(result.gradient, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.curl, flow, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(result.harmonic, 0.0, atol=1e-12)
    assert result.betti_one == 0


@pytest.mark.native_runtime
def test_installed_python_profile_refuses_absent_native_runtime() -> None:
    """A real separately installed Python package must refuse absent runtimes."""
    interpreter = os.environ.get("SPO_HODGE_ABSENT_PYTHON")
    assert interpreter is not None, "The genuine Python-only CI profile is required"
    from scpn_phase_orchestrator.coupling import hodge

    probe = """
import hashlib, json
from pathlib import Path
import numpy as np
from scpn_phase_orchestrator.coupling import hodge
assert hodge.AVAILABLE_BACKENDS == ['python'], hodge.AVAILABLE_BACKENDS
k = np.array([[0., 1.], [1., 0.]])
p = np.array([0., .4])
result = hodge.hodge_decomposition(k, p)
assert abs(result.gradient[0, 1] - np.sin(.4)) < 1e-12
refused = []
for owner in ('rust', 'go', 'julia', 'mojo'):
    try:
        hodge.hodge_decomposition(k, p, backend=owner)
    except ImportError:
        refused.append(owner)
    else:
        raise AssertionError(owner + ' silently fell back')
print(json.dumps(dict(refused=refused,
    source_sha256=hashlib.sha256(Path(hodge.__file__).read_bytes()).hexdigest())))
"""
    completed = subprocess.run(
        [interpreter, "-I", "-B", "-c", probe],
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    record = json.loads(completed.stdout.splitlines()[-1])
    assert record["refused"] == ["rust", "go", "julia", "mojo"]
    assert (
        record["source_sha256"]
        == hashlib.sha256(Path(hodge.__file__).read_bytes()).hexdigest()
    )

    benchmark = Path(__file__).resolve().parents[1] / "benchmarks/hodge_benchmark.py"
    required = subprocess.run(
        [
            interpreter,
            "-I",
            "-B",
            str(benchmark),
            "--parity-gate",
            "--sizes",
            "4",
            "--calls",
            "1",
            "--require-backends",
            "rust",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    assert required.returncode == 1, required.stdout + required.stderr
    report = json.loads(required.stdout)
    assert report["acceptance_passed"] == 0
    assert report["required_backends_present"] == 0
    assert report["python_reference_present"] == 1
    assert json.loads(report["required_backends_json"]) == ["rust"]
    backend_records = json.loads(report["backend_records_json"])
    assert [r["backend"] for r in backend_records if r["status"] == "available"] == [
        "python"
    ]


@pytest.mark.parametrize("backend", [None, "python", "rust", "go", "julia", "mojo"])
def test_public_hodge_refuses_nonfinite_computed_reference(
    backend: str | None,
) -> None:
    """Floating-point overflow cannot publish nonfinite reference components."""
    largest = np.finfo(np.float64).max
    coupling = np.array([[0.0, largest], [largest, 0.0]])
    phases = np.array([0.0, np.pi / 2.0])
    assert np.isfinite(coupling).all()
    assert np.isfinite(phases).all()
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ValueError, match="finite"),
    ):
        hodge_decomposition(coupling, phases, backend=backend)

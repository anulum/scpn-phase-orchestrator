# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Genuine installed phase attention dependency profiles

"""Compare original native and actually kernel-absent installed package owners."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from test_sindy_runtime_profiles import python_only as python_only

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.native_runtime

PROBE = r"""
import cProfile, hashlib, importlib.util, json
from pathlib import Path
import numpy as np
from scpn_phase_orchestrator.coupling import attention_residuals as surface
from scpn_phase_orchestrator.upde.engine import UPDEEngine
projection = np.eye(2).reshape(1,2,2)
coupling = np.array([[0.,-.3],[-.3,0.]])
phases = np.array([0.,np.pi/2])
with cProfile.Profile() as profile:
    matrix = surface.attnres_modulate(coupling, phases, w_q=projection,
        w_k=projection,w_v=projection,w_o=np.eye(2),n_heads=1)
profile.create_stats()
calls = sorted({item[2] for item in profile.stats})
frequency=np.array([.7,-.2]);alpha=np.array([[0.,.1],[.2,0.]])
actual=UPDEEngine(n_oscillators=2,dt=.01,method='euler').step(
    phases,frequency,matrix,.2,.5,alpha)
original=Path(surface.__file__).resolve()
native=importlib.util.find_spec('spo_kernel') is not None
print(json.dumps({'native':native,'active':surface.ACTIVE_BACKEND,
    'available':surface.AVAILABLE_BACKENDS,'coupling':matrix.tolist(),
    'phases':actual.tolist(),'calls':calls,'source':str(original),
    'source_sha256':hashlib.sha256(original.read_bytes()).hexdigest()}))
"""


def _run_profile(
    python: Path, probe: str, working: Path
) -> subprocess.CompletedProcess[str]:
    """Run a real installed interpreter without overriding import resolution."""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [str(python), "-B", "-c", probe],
        cwd=working,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_installed_native_and_absent_profiles_execute_original_laws(
    python_only: Path, tmp_path: Path
) -> None:
    """Observe original owners, source hashes and independent coupled Euler outputs."""
    expected_matrix = np.array([[0.0, -0.375], [-0.375, 0.0]])
    phases = np.array([0.0, np.pi / 2])
    omega = np.array([0.7, -0.2])
    alpha = np.array([[0.0, 0.1], [0.2, 0.0]])
    expected = phases.copy()
    for target in range(2):
        derivative = (
            omega[target]
            + sum(
                expected_matrix[target, source]
                * np.sin(phases[source] - phases[target] - alpha[target, source])
                for source in range(2)
            )
            + 0.2 * np.sin(0.5 - phases[target])
        )
        expected[target] = (phases[target] + 0.01 * derivative) % (2 * np.pi)
    source_hash = hashlib.sha256(
        (
            ROOT / "src/scpn_phase_orchestrator/coupling/attention_residuals.py"
        ).read_bytes()
    ).hexdigest()
    for python, native in ((Path(sys.executable), True), (python_only, False)):
        process = _run_profile(python, PROBE, tmp_path)
        assert process.returncode == 0, process.stdout + process.stderr
        report = json.loads(process.stdout)
        assert report["native"] is native
        assert report["source_sha256"] == source_hash
        assert report["active"] == ("rust" if native else "python")
        owner = (
            "<built-in method spo_kernel.spo_kernel.attnres_modulate_rust>"
            if native
            else "_python_fallback"
        )
        assert owner in report["calls"]
        np.testing.assert_allclose(
            report["coupling"], expected_matrix, rtol=0.0, atol=1e-12
        )
        np.testing.assert_allclose(report["phases"], expected, rtol=0.0, atol=2e-12)
        if not native:
            assert report["available"] == ["python"]


def test_absent_profile_refuses_explicit_compiled_owners(
    python_only: Path, tmp_path: Path
) -> None:
    """Real missing dependencies cannot return a different named owner's result."""
    probe = r"""
import json
import numpy as np
from scpn_phase_orchestrator.coupling.attention_residuals import attnres_modulate
results=[]
for name in ['rust','go','julia','mojo']:
    try:
        attnres_modulate(np.array([[0.,.3],[.3,0.]]),np.array([.1,.7]),backend=name)
    except ImportError:
        results.append(name)
print(json.dumps(results))
"""
    process = _run_profile(python_only, probe, tmp_path)
    assert process.returncode == 0, process.stdout + process.stderr
    assert json.loads(process.stdout) == ["rust", "go", "julia", "mojo"]

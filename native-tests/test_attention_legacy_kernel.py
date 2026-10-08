# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Historical Rust ABI refusal and recovery

"""Reject the actual pre-AttnRes kernel and restore the current numerical owner."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.native_runtime

PROBE = r"""
import os, sys
from pathlib import Path
from juliacall import Main
from coverage import Coverage
assert Main is not None
measurement = Coverage(data_file=os.environ['COVERAGE_FILE'], data_suffix=True,
                      branch=True, config_file=str(Path(
                          os.environ['SPO_BRANCH_RECORDING_ROOT']) / 'pyproject.toml'))
measurement.set_option('run:patch', [])
measurement.start()
try:
    import hashlib, importlib.metadata, json, math
    import numpy as np
    import spo_kernel
    import scpn_phase_orchestrator as package
    from scpn_phase_orchestrator.coupling import attention_residuals as surface
    assert Path(package.__file__).resolve().is_relative_to(Path(sys.prefix))
    mode = sys.argv[1]
    assert importlib.metadata.version('spo-kernel') == (
        '0.5.0' if mode == 'legacy' else '0.5.11')
    assert hasattr(spo_kernel, 'attnres_modulate_rust') is (mode == 'current')
    projection = np.eye(2).reshape(1,2,2)
    def call(owner):
        return surface.attnres_modulate(
            np.array([[0.,.3],[.3,0.]]), np.array([0.,math.pi/2]),
            w_q=projection,w_k=projection,w_v=projection,w_o=np.eye(2),
            n_heads=1,backend=owner)
    reference = np.array([[0.,.375],[.375,0.]])
    np.testing.assert_allclose(call(None), reference, rtol=0, atol=1e-12)
    try:
        result = call('rust')
    except ImportError as error:
        assert mode == 'legacy'
        assert 'lacks attnres_modulate_rust' in str(error)
    else:
        assert mode == 'current'
        np.testing.assert_allclose(result, reference, rtol=0, atol=1e-12)
    binaries = list(Path(spo_kernel.__file__).parent.glob('*.so'))
    assert len(binaries) == 1
    print(json.dumps({'mode':mode, 'active':surface.ACTIVE_BACKEND,
                      'binary':str(binaries[0]),
                      'sha256':hashlib.sha256(binaries[0].read_bytes()).hexdigest()}))
finally:
    measurement.stop()
    measurement.save()
"""


def test_original_historical_kernel_export_refusal_and_recovery() -> None:
    """Use the untouched historical SDK, with real automatic and explicit calls."""
    import scpn_phase_orchestrator as package

    assert package.__file__ is not None
    assert Path(package.__file__).resolve().is_relative_to(Path(sys.prefix))
    legacy = Path(os.environ["SPO_ATTENTION_LEGACY_KERNEL_WHEEL"])
    current = Path(os.environ["SPO_ATTENTION_CURRENT_KERNEL_WHEEL"])
    assert legacy.is_file() and current.is_file()
    wheels = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (legacy, current)
    }
    environment = {**os.environ, "SPO_BRANCH_RECORDING_ROOT": str(ROOT)}
    environment.pop("COVERAGE_PROCESS_START", None)

    def install(wheel: Path) -> None:
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--force-reinstall",
                str(wheel),
            ],
            env=environment,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        assert process.returncode == 0, process.stdout + process.stderr

    def probe(mode: str) -> dict[str, object]:
        process = subprocess.run(
            [sys.executable, "-I", "-B", "-c", PROBE, mode],
            cwd=Path(sys.prefix),
            env=environment,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        assert process.returncode == 0, process.stdout + process.stderr
        result: object = json.loads(process.stdout.splitlines()[-1])
        assert isinstance(result, dict)
        return result

    before = probe("current")
    try:
        install(legacy)
        assert probe("legacy")["mode"] == "legacy"
    finally:
        install(current)
    after = probe("current")
    assert before["sha256"] == after["sha256"]
    assert all(
        hashlib.sha256(path.read_bytes()).hexdigest() == digest
        for path, digest in wheels.items()
    )

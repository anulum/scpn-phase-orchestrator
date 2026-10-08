# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Installed native asset refusal and recovery

"""Exercise original public Julia owners with real missing or truncated assets."""

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
measurement = Coverage(
    data_file=os.environ['COVERAGE_FILE'], data_suffix=True, branch=True,
    config_file=str(Path(os.environ['SPO_BRANCH_RECORDING_ROOT']) / 'pyproject.toml'),
)
measurement.set_option('run:patch', [])
measurement.start()
try:
    import json, math
    import numpy as np
    import scpn_phase_orchestrator as package
    assert Path(package.__file__).resolve().is_relative_to(Path(sys.prefix))
    owner, expected = sys.argv[1:]
    if owner == 'attnres':
        from scpn_phase_orchestrator.coupling.attention_residuals import (
            attnres_modulate,
        )
        def call():
            projection = np.eye(2).reshape(1,2,2)
            return attnres_modulate(np.array([[0.,.3],[.3,0.]]),
                                   np.array([0.,math.pi/2]), w_q=projection,
                                   w_k=projection, w_v=projection, w_o=np.eye(2),
                                   n_heads=1, backend='julia')
        reference = np.array([[0.,.375],[.375,0.]])
    elif owner == 'basin_stability':
        from scpn_phase_orchestrator.upde.basin_stability import steady_state_r
        def call():
            return steady_state_r(np.array([.1,.7]), np.zeros(2),
                                  np.zeros((2,2)), n_transient=0,
                                  n_measure=1, backend='julia')
        reference = math.cos(.3)
    elif owner == 'chimera':
        from scpn_phase_orchestrator.monitor.chimera import local_order_parameter
        def call():
            return local_order_parameter(np.array([.1,.7]),
                                        np.array([[0.,1.],[1.,0.]]), backend='julia')
        reference = np.ones(2)
    else:
        raise AssertionError('unknown required owner')
    try:
        result = call()
    except ImportError as error:
        assert expected == 'refused', repr(error)
        print(json.dumps({'status':'refused', 'module':package.__file__,
                          'error':str(error), 'cause':str(error.__cause__)}))
    else:
        assert expected == 'computed', 'broken asset silently computed'
        values = np.asarray(result)
        assert values.size and np.all(np.isfinite(values))
        np.testing.assert_allclose(values, reference, rtol=0, atol=1e-12)
        print(json.dumps({'status':'computed', 'module':package.__file__}))
finally:
    measurement.stop()
    measurement.save()
"""


@pytest.mark.parametrize("owner", ["attnres", "basin_stability", "chimera"])
@pytest.mark.parametrize("fault", ["missing", "truncated"])
def test_installed_julia_asset_refusal_and_real_recovery(
    owner: str, fault: str
) -> None:
    """Refuse an actual filesystem fault, restore original bytes and compute again."""
    asset_root = Path(os.environ["SPO_BRANCH_ASSET_ROOT"]).resolve()
    assert asset_root.is_relative_to(Path(sys.prefix).resolve())
    asset = asset_root / "julia" / (owner + ".jl")
    original = asset.read_bytes()
    assert original == (ROOT / "julia" / (owner + ".jl")).read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    environment = {**os.environ, "SPO_BRANCH_RECORDING_ROOT": str(ROOT)}
    environment.pop("COVERAGE_PROCESS_START", None)

    def probe(expected: str) -> dict[str, object]:
        process = subprocess.run(
            [sys.executable, "-I", "-B", "-c", PROBE, owner, expected],
            cwd=Path(sys.prefix),
            env=environment,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        assert process.returncode == 0, process.stdout + process.stderr
        value: object = json.loads(process.stdout.splitlines()[-1])
        assert isinstance(value, dict)
        return value

    assert probe("computed")["status"] == "computed"
    try:
        if fault == "missing":
            asset.unlink()
        else:
            end = original.rfind(b"end")
            assert end > 0
            asset.write_bytes(original[:end])
        assert probe("refused")["status"] == "refused"
    finally:
        asset.write_bytes(original)
    assert hashlib.sha256(asset.read_bytes()).hexdigest() == digest
    assert probe("computed")["status"] == "computed"

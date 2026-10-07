# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Genuine installed basin dependency profiles

"""Original native and actually absent wheels, without import or flag overrides."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from test_sindy_runtime_profiles import python_only as python_only

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.native_runtime
SOURCES = (
    "upde/basin_stability.py",
    "upde/bifurcation.py",
    "upde/_basin_stability_validation.py",
    "experimental/accelerators/upde/_basin_stability_go.py",
    "experimental/accelerators/upde/_basin_stability_julia.py",
    "experimental/accelerators/upde/_basin_stability_mojo.py",
)
PROBE = r"""
import cProfile, hashlib, importlib.util, json, math
from pathlib import Path
import numpy as np
from scpn_phase_orchestrator.upde import basin_stability as surface
from scpn_phase_orchestrator.upde.bifurcation import (
    trace_sync_transition, find_critical_coupling,
)
with cProfile.Profile() as profile:
    value=surface.steady_state_r(np.array([0.,math.pi/2]),np.zeros(2),
        np.array([[0.,5e-31],[5e-31,0.]]),dt=1e30,n_transient=0,n_measure=1)
    diagram=trace_sync_transition(np.zeros(2),n_points=3,n_measure=0)
    critical=find_critical_coupling(np.zeros(2),n_measure=0)
profile.create_stats()
original=Path(surface.__file__).resolve();root=original.parents[1]
sources=['upde/basin_stability.py','upde/bifurcation.py','upde/_basin_stability_validation.py',
'experimental/accelerators/upde/_basin_stability_go.py',
'experimental/accelerators/upde/_basin_stability_julia.py',
'experimental/accelerators/upde/_basin_stability_mojo.py']
print(json.dumps({'native':importlib.util.find_spec('spo_kernel') is not None,
'active':surface.ACTIVE_BACKEND,'available':surface.AVAILABLE_BACKENDS,'value':value,
'R_values':diagram.R_values.tolist(),'no_crossing':math.isnan(critical),
'calls':sorted({key[2] for key in profile.stats}), 'source':str(original),
'hashes':{name:hashlib.sha256((root/name).read_bytes()).hexdigest()
    for name in sources}}))
"""


def _run(python: Path, probe: str, working: Path) -> subprocess.CompletedProcess[str]:
    """Run an actual interpreter away from source without import overrides."""
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


def test_original_native_and_absent_installed_consumers(
    python_only: Path,
    tmp_path: Path,
) -> None:
    """Actual owner calls and all six installed files qualify the same law."""
    hashes = {
        name: hashlib.sha256(
            (ROOT / "src/scpn_phase_orchestrator" / name).read_bytes()
        ).hexdigest()
        for name in SOURCES
    }
    for python, native in ((Path(sys.executable), True), (python_only, False)):
        process = _run(python, PROBE, tmp_path)
        assert process.returncode == 0, process.stdout + process.stderr
        report = json.loads(process.stdout)
        assert report["native"] is native
        assert report["hashes"] == hashes
        assert report["active"] == ("rust" if native else "python")
        assert report["value"] == pytest.approx(
            math.cos((math.pi / 2 - 1.0) / 2), abs=2e-15
        )
        assert report["R_values"] == [0.0, 0.0, 0.0]
        assert report["no_crossing"] is True
        owner = (
            "<built-in method spo_kernel.spo_kernel.steady_state_r_rust>"
            if native
            else "_python_steady_state_r"
        )
        assert owner in report["calls"]
        if native:
            assert (
                "<built-in method spo_kernel.spo_kernel.trace_sync_transition_rust>"
                in report["calls"]
            )
        else:
            assert report["available"] == ["python"]


def test_absent_named_owners_never_fall_back_even_for_empty_work(
    python_only: Path,
    tmp_path: Path,
) -> None:
    """Genuine absence is exercised through all five public consumers."""
    probe = r"""
import json
import numpy as np
from scpn_phase_orchestrator.upde.basin_stability import (
    steady_state_r, basin_stability, multi_basin_stability,
)
from scpn_phase_orchestrator.upde.bifurcation import (
    trace_sync_transition, find_critical_coupling,
)
results=[]
for owner in ['rust','go','julia','mojo']:
    calls=[lambda:steady_state_r(np.zeros(2),np.zeros(2),np.zeros((2,2)),n_measure=0,backend=owner),
        lambda:basin_stability(np.zeros(2),np.zeros((2,2)),n_samples=0,backend=owner),
        lambda:multi_basin_stability(np.zeros(2),np.zeros((2,2)),n_samples=0,backend=owner),
        lambda:trace_sync_transition(np.zeros(2),n_points=2,n_measure=0,backend=owner),
        lambda:find_critical_coupling(np.zeros(2),n_measure=0,backend=owner)]
    for index, call in enumerate(calls):
        try:call()
        except ImportError:results.append([owner,index])
print(json.dumps(results))
"""
    process = _run(python_only, probe, tmp_path)
    assert process.returncode == 0, process.stdout + process.stderr
    assert json.loads(process.stdout) == [
        [owner, index]
        for owner in ("rust", "go", "julia", "mojo")
        for index in range(5)
    ]


def test_original_compiled_bindings_validate_metadata_before_native_indexing(
    tmp_path: Path,
) -> None:
    """Actual compiled Python exports reject malformed shapes and boolean metadata."""
    probe = r"""
import json, math
import numpy as np
import spo_kernel
p=np.array([0.,math.pi/2]);o=np.zeros(2);k=np.array([0.,5e-31,5e-31,0.]);a=np.zeros(4)
trial_value=spo_kernel.steady_state_r_rust(p,o,k,a,2,1.,1e30,0,1)
assert abs(trial_value-math.cos((math.pi/2-1)/2))<2e-15
assert spo_kernel.steady_state_r_rust(p,o,k,a,2,1.,.01,9,0)==0
cases=[('steady',spo_kernel.steady_state_r_rust,[p,o,k,a,2,1.,.01,0,1],range(4,9)),
 ('basin',spo_kernel.basin_stability_rust,[o,k,a,2,.01,0,1,2,.5,29],range(3,10)),
 ('trace',spo_kernel.trace_sync_transition_rust,[o,k,a,2,p,0.,2.,3,.01,0,1],[3,5,6,7,8,9,10]),
 ('search',spo_kernel.find_critical_coupling_bif_rust,[o,k,a,2,p,.01,0,1,.1],[3,5,6,7,8])]
rows=[]
for label,fn,args,indices in cases:
 for index in indices:
  for boolean in [True,np.bool_(True)]:
   hostile=list(args);hostile[index]=boolean
   try:fn(*hostile)
   except TypeError:rows.append([label,'boolean',index])
   else:raise AssertionError((label,index,'boolean accepted'))
 for index,value in enumerate(args):
  if isinstance(value,np.ndarray):
   hostile=list(args);hostile[index]=value[:-1]
   try:fn(*hostile)
   except ValueError:rows.append([label,'shape',index])
   else:raise AssertionError((label,index,'shape accepted'))
 for index,value in enumerate(args):
  if isinstance(value,np.ndarray):
   hostile=list(args);hostile[index]=value.copy();hostile[index][0]=float('nan')
   try:fn(*hostile)
   except ValueError:rows.append([label,'finite',index])
   else:raise AssertionError((label,index,'nonfinite accepted'))
# Valid original positional/keyword binding remains a live compiled computation.
kw=spo_kernel.steady_state_r_rust(p,omegas=o,knm_flat=k,alpha_flat=a,n=2,
 k_scale=1.,dt=1e30,n_transient=0,n_measure=1)
assert kw==trial_value
gap=math.pi-0.1
signed_grid,signed_values,signed_cross=spo_kernel.trace_sync_transition_rust(
 o,np.array([0.,1e-308,1e-308,0.]),a,2,np.array([0.,gap]),
 -1e308,1e308,2,1.,0,1)
low=abs(math.cos((gap+2*(1e308*1e-308)*math.sin(gap))/2))
high=abs(math.cos((gap-2*(1e308*1e-308)*math.sin(gap))/2))
np.testing.assert_allclose(signed_values,[low,high],rtol=0,atol=2e-15)
fraction=(0.1-low)/(high-low)
expected=(1-fraction)*-1e308+fraction*1e308
assert math.isfinite(signed_cross) and abs(signed_cross-expected)/1e308<2e-15
print(json.dumps({'value':trial_value,'negative_controls':len(rows),'cases':rows}))
"""
    process = _run(Path(sys.executable), probe, tmp_path)
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads(process.stdout)
    assert report["negative_controls"] == 78
    assert report["value"] == pytest.approx(math.cos((math.pi / 2 - 1) / 2), abs=2e-15)


def test_original_mojo_protocol_or_genuine_owner_absence() -> None:
    """Exercise the compiled request boundary, or assert actual named-owner refusal."""
    from scpn_phase_orchestrator.upde.basin_stability import (
        AVAILABLE_BACKENDS,
        steady_state_r,
    )

    required = os.environ.get("SPO_REQUIRED_BASIN_BACKENDS", "python").split(",")
    if "mojo" not in AVAILABLE_BACKENDS:
        assert "mojo" not in required, "required Mojo executable is unavailable"
        with pytest.raises(ImportError, match="requested basin backend 'mojo'"):
            steady_state_r(
                np.array([0.0]),
                np.zeros(1),
                np.zeros((1, 1)),
                n_transient=0,
                n_measure=1,
                backend="mojo",
            )
        return

    executable = ROOT / "mojo/basin_stability_mojo"
    assert executable.is_file()
    buffers = "0 1 0 0 0 1 1 0 0 0 0 0"
    valid = f"STEADY 2 1 0.1 0 1 {buffers}"
    malformed = (
        "STEADY",
        "UNKNOWN",
        "STEADY 0 1 0.1 0 1",
        "STEADY -2 1 0.1 0 1",
        valid.rsplit(" ", 1)[0],
        valid + " 0",
        valid.replace("STEADY 2 ", "STEADY 2x ", 1),
        valid.replace("STEADY 2 1 ", "STEADY 2 1x ", 1),
        valid.replace("0.1", "nan", 1),
        valid.replace("0.1 0 1", "0.1 -1 1", 1),
    )
    for request in malformed:
        process = subprocess.run(
            [str(executable)],
            input=request + "\n",
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert process.returncode != 0, request

    for request, expected in (
        (valid, math.cos((1.0 - 0.2 * math.sin(1.0)) / 2)),
        (valid.replace("0.1 0 1", "0.1 99 0", 1), 0.0),
        (
            "STEADY 2 1 1e30 0 1 0 "
            + repr(math.pi / 2)
            + " 0 0 0 5e-31 5e-31 0 0 0 0 0",
            math.cos((math.pi / 2 - 1) / 2),
        ),
    ):
        process = subprocess.run(
            [str(executable)],
            input=request + "\n",
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert process.returncode == 0, process.stdout + process.stderr
        assert float(process.stdout) == pytest.approx(expected, abs=2e-15)

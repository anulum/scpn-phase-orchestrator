# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Installed public connectome comparison

"""Measure original public generators in distinct verified installed profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np

from benchmarks.connectome_reference import reference_connectome

PROFILE_PROGRAM = r"""
import cProfile, hashlib, importlib, importlib.util, inspect, json, sys, time
from pathlib import Path
import numpy as np
from scpn_phase_orchestrator.coupling import connectome
from scpn_phase_orchestrator.coupling import _connectome_validation as admission
data=json.loads(sys.argv[1]);owner=data['owner'];kind=data.get('kind','synthetic')
spec=importlib.util.find_spec('spo_kernel')
assert (spec is not None)==(owner=='rust')
native=None
if spec is not None:
    import spo_kernel
    native=spo_kernel.load_hcp_connectome_rust
    assert inspect.isbuiltin(native)
if data.get('allocation_budget'):
    import resource
    assert sys.platform=='linux'
    status=Path('/proc/self/status').read_text().splitlines()
    virtual=int(next(s.split()[1] for s in status if s.startswith('VmSize:')))*1024
    resource.setrlimit(resource.RLIMIT_AS,
        (virtual+data['allocation_budget'],virtual+data['allocation_budget']))
n=data['n_regions'];seed=data.get('seed',42)
def call(seed_value):
    if kind=='hcp':return connectome.load_neurolib_hcp(n)
    return connectome.load_hcp_connectome(n,seed_value)
profile=cProfile.Profile();profile.enable();error=None;matrix=None
try:matrix=call(seed)
except (ValueError,TypeError,ImportError) as exc:
    error=dict(type=type(exc).__name__,message=str(exc))
finally:profile.disable()
native_calls=sum(e.callcount for e in profile.getstats()
    if isinstance(e.code,str) and 'load_hcp_connectome_rust' in e.code)
record=dict(owner=owner,kind=kind,prefix=sys.prefix,python=sys.version,
    numpy=np.__version__,module=connectome.__file__,
    admission_module=admission.__file__,error=error,
    source_sha256=hashlib.sha256(Path(connectome.__file__).read_bytes()).hexdigest(),
    admission_sha256=hashlib.sha256(Path(admission.__file__).read_bytes()).hexdigest(),
    native_calls=native_calls,matrix=None)
if data.get('allocation_budget'):
    recovery=connectome.load_hcp_connectome(2,42)
    record['recovery_matrix']=recovery.tolist()
if native is not None:
    binary=Path(importlib.import_module(native.__module__).__file__)
    record.update(binary=str(binary),
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
if matrix is not None:
    record.update(matrix=matrix.tolist(),dtype=str(matrix.dtype),
        contiguous=bool(matrix.flags.c_contiguous))
    if kind=='synthetic':
        original=matrix.copy();matrix[0,1]=-999.0;again=call(seed)
        assert np.array_equal(original,again)
        assert not np.shares_memory(matrix,again)
        record['independent_copy']=True
    if kind=='hcp':
        import neurolib
        import importlib.metadata
        record['neurolib_version']=importlib.metadata.version('neurolib')
        record['dataset_root']=str(Path(neurolib.__file__).parent/
            'data/datasets/hcp/subjects')
    batches=data.get('batches',0);iterations=data.get('iterations',1)
    if batches:
        assert kind=='synthetic' and batches>=20 and iterations>=1
        mode=data['mode'];assert mode in {'cold','warm'}
        timings=[];means=[]
        for batch in range(batches):
            chunk=[]
            for iteration in range(iterations):
                value=seed+1+batch*iterations+iteration if mode=='cold' else seed
                start=time.perf_counter_ns();call(value)
                chunk.append(time.perf_counter_ns()-start)
            timings.extend(chunk);means.append(sum(chunk)/len(chunk))
        record.update(mode=mode,batch_means_ns=means,call_timings_ns=timings,
            seed_schedule='seed+1..seed+batches*iterations' if mode=='cold'
                else 'fixed seed, already cached')
print(json.dumps(record))
"""


def run_profile(executable: Path, parameters: dict[str, object]) -> dict[str, object]:
    """Call the unchanged public loader inside a qualified installation.

    Parameters
    ----------
    executable : pathlib.Path
        Original venv interpreter, whose bytes must equal this trusted Python.
    parameters : dict[str, object]
        Owner, count, seed, loader kind and optional repeated timing request.

    Returns
    -------
    dict[str, object]
        Original results, native observations, identities and optional timings.

    Raises
    ------
    ValueError
        If the executable, installed source or producer response is unqualified.
    """
    trusted = Path(sys.executable).resolve()
    if (
        not executable.is_file()
        or hashlib.sha256(executable.read_bytes()).digest()
        != hashlib.sha256(trusted.read_bytes()).digest()
    ):
        raise ValueError("profile executable must match the trusted Python binary")
    result = subprocess.run(  # noqa: S603 -- same trusted Python bytes, fixed -I/-B program
        [str(executable), "-I", "-B", "-c", PROFILE_PROGRAM, json.dumps(parameters)],
        cwd=executable.parent.parent,
        capture_output=True,
        text=True,
        check=False,
        timeout=240,
    )
    if result.returncode:
        raise ValueError(result.stdout + result.stderr)
    record = cast("dict[str, object]", json.loads(result.stdout))
    from scpn_phase_orchestrator.coupling import _connectome_validation, connectome

    prefix = Path(str(record["prefix"])).resolve()
    for key in ("module", "admission_module"):
        if not Path(str(record[key])).resolve().is_relative_to(prefix):
            raise ValueError("profile must execute its physically installed package")
    if "binary" in record and not Path(str(record["binary"])).resolve().is_relative_to(
        prefix
    ):
        raise ValueError("profile must execute its physically installed kernel")
    for key, module in [
        ("source_sha256", connectome),
        ("admission_sha256", _connectome_validation),
    ]:
        source = module.__file__
        if source is None:
            raise ValueError("production source path is unavailable")
        if record[key] != hashlib.sha256(Path(source).read_bytes()).hexdigest():
            raise ValueError("profile source differs from current production source")
    return record


def compare_profiles(python: Path, rust: Path) -> dict[str, object]:
    """Compare cold generation and cached copies for both real installed owners.

    Parameters
    ----------
    python : pathlib.Path
        Genuine kernel-absent installed profile.
    rust : pathlib.Path
        Original compiled-kernel installed profile.

    Returns
    -------
    dict[str, object]
        Source-bound raw individual and batch timings, with structural oracles.
    """
    rows = []
    for n_regions in (16, 64, 256):
        for mode in ("cold", "warm"):
            for owner, executable in [("python", python), ("rust", rust)]:
                record = run_profile(
                    executable,
                    {
                        "owner": owner,
                        "n_regions": n_regions,
                        "seed": 42,
                        "mode": mode,
                        "batches": 20,
                        "iterations": 5 if mode == "cold" else 100,
                    },
                )
                assert record["error"] is None
                assert record["native_calls"] == (1 if owner == "rust" else 0)
                actual = np.asarray(record.pop("matrix"), dtype=np.float64)
                np.testing.assert_allclose(
                    actual,
                    reference_connectome(n_regions, 42, owner),
                    atol=3e-14,
                    rtol=3e-14,
                )
                means = cast("list[float]", record["batch_means_ns"])
                record.update(
                    n_regions=n_regions,
                    median_batch_ns=statistics.median(means),
                    repetitions=len(cast("list[int]", record["call_timings_ns"])),
                )
                rows.append(record)
    return {
        "rows": rows,
        "shared_host": True,
        "performance_claim": "local diagnostic, no controlled speed-up",
        "noise_parity": (
            "same structural law; PCG64 Gaussian and LCG uniform are distinct"
        ),
    }


def main() -> int:
    """Run the operator-selected installed comparison and print strict JSON.

    Returns
    -------
    int
        Zero after both real owners and every numerical oracle complete.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--rust", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(compare_profiles(args.python, args.rust), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

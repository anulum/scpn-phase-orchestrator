# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Embedded native kernel profile contract

"""Load the genuine Rust library through an actual embedded Linux CPython."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest
from coverage import Coverage

pytestmark = pytest.mark.native_runtime
ROOT = Path(__file__).resolve().parents[1]


def test_embedded_native_library_without_file_is_refused(tmp_path: Path) -> None:
    """Refuse unidentifiable built-in native loading after real numerical steps."""
    import spo_kernel.spo_kernel as extension

    assert sys.platform == "linux", "this is a Linux CPython embedding contract"
    compiler = shutil.which("cc")
    config = shutil.which(
        f"python{sys.version_info.major}.{sys.version_info.minor}-config"
    )
    assert compiler is not None and config is not None
    flags = shlex.split(
        subprocess.check_output(
            [config, "--includes", "--ldflags", "--embed"], text=True, timeout=10
        )
    )
    executable = tmp_path / "embedded-native"
    subprocess.run(
        [
            compiler,
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-Wpedantic",
            str(ROOT / "native-tests/helpers/embed_kernel.c"),
            "-o",
            str(executable),
            *flags,
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"SPO_ENV", "SPO_API_KEY", "SPO_RATE_LIMIT_PER_MINUTE"}
    }
    library_dir = sysconfig.get_config_var("LIBDIR")
    assert isinstance(library_dir, str)
    env["LD_LIBRARY_PATH"] = library_dir + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    active = Coverage.current()
    if active is not None:
        data_file = active.get_option("run:data_file")
        assert isinstance(data_file, str)
        env["SPO_KERNEL_COVERAGE_FILE"] = data_file
        # The child writes a parallel data file beside the active one. Coverage
        # refuses to combine branch data with statement data, so the child
        # measures in the mode of the run that will combine it.
        env["SPO_KERNEL_COVERAGE_BRANCH"] = (
            "1" if active.get_option("run:branch") else "0"
        )
    probe = tmp_path / "embedded-probe.py"
    program = (
        f"import sys; sys.path.insert(0,{str(ROOT / 'src')!r})\n"
        + """
import json,os,sys
import spo_kernel
import spo_kernel.spo_kernel as extension
from scpn_phase_orchestrator.runtime.kernel import verify_kernel
calls=[]
def observe(frame,event,value):
    if event!="c_return":
        return
    owner=getattr(value,"__self__",None)
    if (isinstance(owner,
        (spo_kernel.PyUPDEStepper,spo_kernel.PyStuartLandauStepper))
        and getattr(value,"__name__","")=="step"):
        calls.append(type(owner).__name__+".step")
measurement=None
if "SPO_KERNEL_COVERAGE_FILE" in os.environ:
    from coverage import Coverage
    measurement=Coverage(source=[],include=["*/src/scpn_phase_orchestrator/runtime/kernel.py"],
                         data_file=os.environ["SPO_KERNEL_COVERAGE_FILE"],data_suffix=True,
                         branch=os.environ["SPO_KERNEL_COVERAGE_BRANCH"]=="1")
    measurement.start()
previous=sys.getprofile()
sys.setprofile(observe)
try:
    verify_kernel()
except ImportError as exc:
    result={"refused":str(exc)}
else:
    raise AssertionError("an unidentifiable native module qualified")
finally:
    sys.setprofile(previous)
    if measurement is not None:
        measurement.stop()
        measurement.save()
result.update({"origin":extension.__spec__.origin,"file":getattr(extension,"__file__",None),
               "completed_calls":calls,"profile_restored":sys.getprofile() is previous})
from scpn_phase_orchestrator.runtime.cli import main
import click
try:
    main(["serve",sys.argv[1],"--port",sys.argv[2]],standalone_mode=False)
except click.ClickException as exc:
    result["cli_refused"]=str(exc)
else:
    raise AssertionError("required-native listener accepted the unidentifiable module")
print(json.dumps(result,sort_keys=True))
"""
    )
    program = program.replace(
        "sys.argv[1]", repr(str(ROOT / "domainpacks/minimal_domain/binding_spec.yaml"))
    )
    program = program.replace("sys.argv[2]", repr(str(port)))
    probe.write_text(
        "\n".join(Path(__file__).read_text().splitlines()[:7]) + "\n" + program
    )
    native_path = Path(extension.__file__).resolve()
    native_digest = hashlib.sha256(native_path.read_bytes()).hexdigest()
    result = subprocess.run(
        [str(executable), str(native_path), sys.executable, str(probe)],
        capture_output=True,
        text=True,
        env=env,
        check=True,
        timeout=60,
    )
    observed = json.loads(result.stdout)
    assert observed["origin"] == "built-in" and observed["file"] is None
    assert observed["completed_calls"] == [
        "PyUPDEStepper.step",
        "PyStuartLandauStepper.step",
    ]
    assert observed["profile_restored"] is True
    assert (
        observed["refused"]
        == observed["cli_refused"]
        == "spo-kernel has no native library path"
    )
    assert hashlib.sha256(native_path.read_bytes()).hexdigest() == native_digest
    with socket.socket() as stopped:
        assert stopped.connect_ex(("127.0.0.1", port)) != 0

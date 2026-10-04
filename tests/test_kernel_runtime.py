# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Real native kernel verification contracts

"""Exercise public numerical verification while observing real PyO3 calls."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from types import FrameType

import numpy as np
import pytest
from coverage import Coverage

from scpn_phase_orchestrator.runtime.kernel import verify_kernel
from scpn_phase_orchestrator.upde.engine import UPDEEngine

pytestmark = pytest.mark.native_runtime
ROOT = Path(__file__).resolve().parents[1]


def test_verification_executes_both_native_consumers() -> None:
    """Observe the actual native calls and bind the report to their loaded binary."""
    import spo_kernel

    calls: list[str] = []

    def observe(_frame: FrameType, event: str, value: object) -> None:
        """Record native step calls without changing the numerical implementation."""
        owner = getattr(value, "__self__", None)
        if event == "c_call" and isinstance(
            owner, (spo_kernel.PyUPDEStepper, spo_kernel.PyStuartLandauStepper)
        ):
            calls.append(
                type(owner).__name__ + "." + str(getattr(value, "__name__", ""))
            )

    previous = sys.getprofile()
    sys.setprofile(observe)
    try:
        report = verify_kernel()
    finally:
        sys.setprofile(previous)
    assert "PyUPDEStepper.step" in calls
    assert "PyStuartLandauStepper.step" in calls
    assert report.version == version("spo-kernel")
    assert Path(report.extension).is_file()
    assert (
        report.sha256 == hashlib.sha256(Path(report.extension).read_bytes()).hexdigest()
    )
    assert sys.getprofile() is previous


@pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
def test_public_native_step_wraps_an_uncoupled_trajectory(method: str) -> None:
    """Compare native motion across the phase boundary to its closed-form solution."""
    dt = 0.01
    phase = np.array([2 * np.pi - 0.002, 0.001])
    omega = np.array([1.0, -0.5])
    zero = np.zeros((2, 2))
    engine = UPDEEngine(2, dt, method)
    assert engine.backend == "rust"
    after = engine.step(phase, omega, zero, alpha=zero)
    np.testing.assert_allclose(after, (phase + dt * omega) % (2 * np.pi), atol=1e-13)
    assert engine.time == pytest.approx(dt)


def test_kernel_installed_after_import_requires_process_restart(tmp_path: Path) -> None:
    """Refuse a real process that selected NumPy before the native wheel existed."""
    environment = tmp_path / "late-install"
    subprocess.run(
        [sys.executable, "-m", "venv", str(environment)], check=True, timeout=60
    )
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    minor = "py311" if sys.version_info[:2] == (3, 11) else "py312"
    lock = (
        f"server-lock-windows-{minor}.txt"
        if os.name == "nt"
        else "server-lock-py311.txt"
        if minor == "py311"
        else "server-lock.txt"
    )
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--require-hashes",
            "--no-deps",
            "-r",
            str(ROOT / "requirements" / lock),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    tools = tmp_path / "coverage-tools.txt"
    block: list[str] = []
    for line in (ROOT / "requirements/dev-lock.txt").read_text().splitlines():
        if line.startswith("coverage==") or (
            block and (not line or line[0].isspace() or line.startswith("#"))
        ):
            block.append(line)
        elif block:
            break
    assert block and any("--hash=sha256:" in line for line in block)
    tools.write_text(
        "\n".join((ROOT / "requirements/server-lock.txt").read_text().splitlines()[:7])
        + "\n"
        + "\n".join(block)
        + "\n"
    )
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--require-hashes",
            "--no-deps",
            "-r",
            str(tools),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    wheels = tmp_path / "wheels"
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"SPO_ENV", "SPO_API_KEY", "SPO_RATE_LIMIT_PER_MINUTE"}
    }
    env["CARGO_BUILD_JOBS"] = "1"
    env["LLVM_PROFILE_FILE"] = str(tmp_path / "kernel-build-%p.profraw")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "maturin",
            "build",
            "--release",
            "-m",
            str(ROOT / "spo-kernel/crates/spo-ffi/Cargo.toml"),
            "--out",
            str(wheels),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    built = list(wheels.glob("*.whl"))
    assert len(built) == 1
    env["PYTHONPATH"] = str(ROOT / "src")
    active = Coverage.current()
    if active is not None:
        data_file = active.get_option("run:data_file")
        assert isinstance(data_file, str)
        env["SPO_KERNEL_COVERAGE_FILE"] = data_file
    probe = tmp_path / "late-install.py"
    program = """
import importlib.util,json,os,subprocess,sys
from pathlib import Path
from scpn_phase_orchestrator.upde.engine import UPDEEngine
assert importlib.util.find_spec("spo_kernel") is None
before=UPDEEngine(2,.01).backend
assert before=="numpy"
subprocess.run([sys.executable,"-m","pip","install","--no-deps",sys.argv[1]],
               check=True,capture_output=True,text=True,timeout=60)
import spo_kernel.spo_kernel as native
from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine
from scpn_phase_orchestrator.runtime.kernel import verify_kernel
measurement=None
if "SPO_KERNEL_COVERAGE_FILE" in os.environ:
    from coverage import Coverage
    measurement=Coverage(source=[],include=["*/src/scpn_phase_orchestrator/runtime/kernel.py"],
                         data_file=os.environ["SPO_KERNEL_COVERAGE_FILE"],data_suffix=True,branch=True)
    measurement.start()
try:
    verify_kernel()
except RuntimeError as exc:
    result={"refused":str(exc)}
else:
    raise AssertionError("a cached NumPy consumer qualified as native")
finally:
    if measurement is not None:
        measurement.stop()
        measurement.save()
result.update({"before":before,"after_phase":UPDEEngine(2,.01).backend,
               "after_amplitude":StuartLandauEngine(2,.01).backend,
               "native":str(Path(native.__file__).resolve()),"prefix":sys.prefix})
print(json.dumps(result,sort_keys=True))
"""
    probe.write_text(
        "\n".join(Path(__file__).read_text().splitlines()[:7]) + "\n" + program
    )
    result = subprocess.run(
        [str(python), str(probe), str(built[0])],
        capture_output=True,
        text=True,
        env=env,
        check=True,
        timeout=90,
    )
    observed = json.loads(result.stdout)
    assert observed["before"] == observed["after_phase"] == "numpy"
    assert observed["after_amplitude"] == "rust"
    assert "selected NumPy" in observed["refused"]
    assert Path(observed["native"]).is_relative_to(environment)
    assert Path(observed["prefix"]) == environment
    recovered = subprocess.run(
        [
            str(python),
            "-c",
            "import json; "
            "from scpn_phase_orchestrator.runtime.kernel import verify_kernel; "
            "from scpn_phase_orchestrator.upde.engine import UPDEEngine; "
            "from scpn_phase_orchestrator.upde.stuart_landau import "
            "StuartLandauEngine; "
            "report=verify_kernel(); "
            "print(json.dumps({'native':report.extension, "
            "'phase':UPDEEngine(2,.01).backend, "
            "'amplitude':StuartLandauEngine(2,.01).backend}))",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert json.loads(recovered.stdout) == {
        "native": observed["native"],
        "phase": "rust",
        "amplitude": "rust",
    }

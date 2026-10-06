# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Genuine kernel-absent deployment contracts

"""Verify required-native refusal in a separately installed Python-only profile."""

from __future__ import annotations

import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml
from coverage import Coverage

pytestmark = pytest.mark.native_runtime
if sys.platform == "win32":
    PROCESS_GROUP = subprocess.CREATE_NEW_PROCESS_GROUP
    STOP_SIGNAL = signal.CTRL_BREAK_EVENT
else:
    PROCESS_GROUP = 0
    STOP_SIGNAL = signal.SIGTERM

ROOT = Path(__file__).resolve().parents[1]
BINDING = ROOT / "domainpacks/minimal_domain/binding_spec.yaml"
CLI = "from scpn_phase_orchestrator.runtime.cli import main; main()"
# The child runs in an environment of its own, so the parent's measurement does
# not reach it. When the parent measures, the child measures the server module
# itself, in the parent's mode, into a parallel data file beside the parent's.
MEASURED = """
import os
measurement = None
if "SPO_SERVER_COVERAGE_FILE" in os.environ:
    from coverage import Coverage
    measurement = Coverage(
        source=[],
        include=["*/src/scpn_phase_orchestrator/runtime/server.py"],
        data_file=os.environ["SPO_SERVER_COVERAGE_FILE"],
        data_suffix=True,
        branch=os.environ["SPO_SERVER_COVERAGE_BRANCH"] == "1",
    )
    measurement.start()
try:
    {body}
finally:
    if measurement is not None:
        measurement.stop()
        measurement.save()
"""


def _child_environment() -> dict[str, str]:
    """Return the child's environment: the checkout source and no server settings."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"SPO_ENV", "SPO_API_KEY", "SPO_RATE_LIMIT_PER_MINUTE"}
    }
    env["PYTHONPATH"] = str(ROOT / "src")
    active = Coverage.current()
    if active is not None:
        data_file = active.get_option("run:data_file")
        assert isinstance(data_file, str)
        env["SPO_SERVER_COVERAGE_FILE"] = data_file
        # Coverage refuses to combine branch data with statement data.
        env["SPO_SERVER_COVERAGE_BRANCH"] = (
            "1" if active.get_option("run:branch") else "0"
        )
    return env


def _install(python: Path, requirements: Path, *, timeout: int) -> None:
    """Install one hash-locked requirement file into the child's environment."""
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--require-hashes",
            "--no-deps",
            "-r",
            str(requirements),
        ],
        check=True,
        timeout=timeout,
        capture_output=True,
        text=True,
    )


def _install_locked_coverage(python: Path, tmp_path: Path) -> None:
    """Give a measured child the coverage release that the parent's lock pins."""
    if Coverage.current() is None:
        return
    block: list[str] = []
    for line in (ROOT / "requirements/dev-lock.txt").read_text().splitlines():
        if line.startswith("coverage==") or (
            block and (not line or line[0].isspace() or line.startswith("#"))
        ):
            block.append(line)
        elif block:
            break
    assert block and any("--hash=sha256:" in line for line in block)
    tools = tmp_path / "coverage-tools.txt"
    tools.write_text(
        "\n".join((ROOT / "requirements/server-lock.txt").read_text().splitlines()[:7])
        + "\n"
        + "\n".join(block)
        + "\n"
    )
    _install(python, tools, timeout=90)


@pytest.mark.parametrize("amplitude", [False, True])
def test_real_python_only_profile_refuses_required_kernel(
    tmp_path: Path, amplitude: bool
) -> None:
    """Install the real server profile without Rust and check both admission modes."""
    binding = yaml.safe_load(BINDING.read_text())
    if not amplitude:
        binding.pop("amplitude")
    spec_path = tmp_path / "binding.yaml"
    spec_path.write_text(yaml.safe_dump(binding))
    environment = tmp_path / "python-only"
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
    _install(python, ROOT / "requirements" / lock, timeout=180)
    _install_locked_coverage(python, tmp_path)
    env = _child_environment()
    probe = subprocess.run(
        [
            str(python),
            "-c",
            "import importlib.util, json; "
            "from scpn_phase_orchestrator.binding.loader import load_binding_spec; "
            "from scpn_phase_orchestrator.runtime.server import SimulationState; "
            f"spec=load_binding_spec({str(spec_path)!r}); "
            "sim=SimulationState(spec); "
            "print(json.dumps({'kernel':"
            "importlib.util.find_spec('spo_kernel') is not None,"
            "'backend':sim.engine.backend,"
            "'amplitude_backend':sim.sl_engine.backend if sim.sl_engine else None,"
            "'step':sim.step()['step']}))",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert json.loads(probe.stdout) == {
        "kernel": False,
        "backend": "numpy",
        "amplitude_backend": "numpy" if amplitude else None,
        "step": 1,
    }
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    refused = subprocess.run(
        [
            str(python),
            "-c",
            MEASURED.format(body=CLI),
            "serve",
            str(spec_path),
            "--require-kernel",
            "--port",
            str(port),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert refused.returncode == 1
    assert "Error: spo-kernel required: simulation selected NumPy" in refused.stderr
    assert "Uvicorn running" not in refused.stderr
    with socket.socket() as stopped:
        assert stopped.connect_ex(("127.0.0.1", port)) != 0
    with (tmp_path / "python-server.log").open("w") as log:
        allowed = subprocess.Popen(
            [
                str(python),
                "-c",
                CLI,
                "serve",
                str(spec_path),
                "--allow-python",
                "--port",
                str(port),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=PROCESS_GROUP,
        )
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        try:
            deadline = time.monotonic() + 30
            while True:
                assert allowed.poll() is None, (
                    tmp_path / "python-server.log"
                ).read_text()
                try:
                    connection.request("GET", "/api/config")
                    response = connection.getresponse()
                    config = json.loads(response.read())
                    break
                except (ConnectionError, TimeoutError):
                    connection.close()
                    assert time.monotonic() < deadline
                    time.sleep(0.1)
            assert config["backend"] == "numpy"
            assert config["kernel_required"] is False and config["kernel"] is None
            connection.request("POST", "/api/step")
            response = connection.getresponse()
            assert response.status == 200
            state = json.loads(response.read())
            assert state["step"] == 1 and 0 <= state["R_global"] <= 1
        finally:
            connection.close()
            allowed.send_signal(STOP_SIGNAL)
            try:
                allowed.wait(timeout=10)
            except subprocess.TimeoutExpired:
                allowed.kill()
                allowed.wait(timeout=5)
        # Uvicorn re-raises the captured stop signal after its complete graceful
        # shutdown. The default action ends the process with -SIGTERM on POSIX
        # and, in the Windows C runtime, with exit code 3.
        expected_exit = {0, 3} if sys.platform == "win32" else {0, -signal.SIGTERM}
        assert allowed.returncode in expected_exit
        assert (
            "Application shutdown complete"
            in (tmp_path / "python-server.log").read_text()
        )
        with socket.socket() as stopped:
            assert stopped.connect_ex(("127.0.0.1", port)) != 0


@pytest.mark.skipif(os.name == "nt", reason="the base runtime lock is built for POSIX")
def test_real_profile_without_the_web_framework_refuses_to_build_the_app(
    tmp_path: Path,
) -> None:
    """Install the base runtime alone and ask it for the HTTP application.

    The base runtime lock holds the numerical packages and no web framework.
    The application factory must name the missing framework instead of failing
    somewhere inside its first use.
    """
    environment = tmp_path / "base-runtime"
    subprocess.run(
        [sys.executable, "-m", "venv", str(environment)], check=True, timeout=60
    )
    python = environment / "bin/python"
    _install(python, ROOT / "requirements/runtime-lock.txt", timeout=180)
    _install_locked_coverage(python, tmp_path)
    program = (
        "import importlib.util, json, sys\n"
        "from scpn_phase_orchestrator.runtime.server import create_app\n"
        "assert importlib.util.find_spec('fastapi') is None\n"
        "try:\n"
        "    create_app(sys.argv[1])\n"
        "except ImportError as error:\n"
        "    print(json.dumps({'refused': str(error), "
        "'cause': type(error.__cause__).__name__}))\n"
        "else:\n"
        "    raise AssertionError('an application was built without its framework')"
    )
    indented = program.replace("\n", "\n    ")
    result = subprocess.run(
        [str(python), "-c", MEASURED.format(body=indented), str(BINDING)],
        env=_child_environment(),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "refused": "fastapi not installed. pip install fastapi uvicorn",
        "cause": "ModuleNotFoundError",
    }

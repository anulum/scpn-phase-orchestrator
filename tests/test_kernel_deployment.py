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

pytestmark = pytest.mark.native_runtime
if sys.platform == "win32":
    PROCESS_GROUP = subprocess.CREATE_NEW_PROCESS_GROUP
    STOP_SIGNAL = signal.CTRL_BREAK_EVENT
else:
    PROCESS_GROUP = 0
    STOP_SIGNAL = signal.SIGTERM

ROOT = Path(__file__).resolve().parents[1]
BINDING = ROOT / "domainpacks/minimal_domain/binding_spec.yaml"


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
        timeout=180,
        capture_output=True,
        text=True,
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"SPO_ENV", "SPO_API_KEY", "SPO_RATE_LIMIT_PER_MINUTE"}
    }
    env["PYTHONPATH"] = str(ROOT / "src")
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
            "from scpn_phase_orchestrator.runtime.cli import main; main()",
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
                "from scpn_phase_orchestrator.runtime.cli import main; main()",
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
        # Uvicorn can re-emit SIGTERM after its complete graceful shutdown.
        expected_exit = {0} if sys.platform == "win32" else {0, -signal.SIGTERM}
        assert allowed.returncode in expected_exit
        assert (
            "Application shutdown complete"
            in (tmp_path / "python-server.log").read_text()
        )
        with socket.socket() as stopped:
            assert stopped.connect_ex(("127.0.0.1", port)) != 0

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Foreground CLI and native HTTP contracts

"""Run the registered serve command and real HTTP simulation operations."""

from __future__ import annotations

import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from scpn_phase_orchestrator.binding.loader import load_binding_spec
from scpn_phase_orchestrator.runtime.server import SimulationState

pytestmark = pytest.mark.native_runtime
if sys.platform == "win32":
    PROCESS_GROUP = subprocess.CREATE_NEW_PROCESS_GROUP
    STOP_SIGNAL = signal.CTRL_BREAK_EVENT
else:
    PROCESS_GROUP = 0
    STOP_SIGNAL = signal.SIGTERM

ROOT = Path(__file__).resolve().parents[1]
CLI = "from scpn_phase_orchestrator.runtime.cli import main; main()"


def test_serve_refuses_an_invalid_binding_before_listening(tmp_path: Path) -> None:
    """Reject an actual binding with a zero sample period before opening its port."""
    binding = yaml.safe_load(
        (ROOT / "domainpacks/minimal_domain/binding_spec.yaml").read_text()
    )
    binding["sample_period_s"] = 0.0
    spec_path = tmp_path / "invalid-binding.yaml"
    spec_path.write_text(yaml.safe_dump(binding))
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"SPO_ENV", "SPO_API_KEY", "SPO_RATE_LIMIT_PER_MINUTE"}
    }
    refused = subprocess.run(
        [sys.executable, "-c", CLI, "serve", str(spec_path), "--port", str(port)],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=False,
    )
    assert refused.returncode == 1
    assert "Error:" in refused.stderr and "positive" in refused.stderr
    assert "Uvicorn running" not in refused.stderr
    with socket.socket() as stopped:
        assert stopped.connect_ex(("127.0.0.1", port)) != 0


@pytest.mark.parametrize("amplitude", [False, True])
def test_foreground_server_uses_native_engine_and_stops(
    tmp_path: Path, amplitude: bool
) -> None:
    """Check actual CLI startup, numerical HTTP output and graceful termination."""
    binding = yaml.safe_load(
        (ROOT / "domainpacks/minimal_domain/binding_spec.yaml").read_text()
    )
    if not amplitude:
        binding.pop("amplitude")
    spec_path = tmp_path / "binding.yaml"
    spec_path.write_text(yaml.safe_dump(binding))
    reference = SimulationState(load_binding_spec(spec_path))
    theta = reference.phases.copy()
    dt = reference.spec.sample_period_s
    offsets = theta[None, :] - theta[:, None] - reference.coupling.alpha
    derivative = (
        reference.omegas
        + np.sum(reference.coupling.knm * np.sin(offsets), axis=1)
        + reference.zeta * np.sin(reference.psi_target - theta)
    )
    expected_theta = (theta + dt * derivative) % (2 * np.pi)
    expected_r = round(float(abs(np.mean(np.exp(1j * expected_theta)))), 4)
    expected_amplitude = None
    if amplitude:
        assert reference.mu is not None and reference.spec.amplitude is not None
        assert reference.coupling.knm_r is not None
        radius = np.sqrt(reference.mu)
        growth = (reference.mu - radius**2) * radius
        coupling = np.sum(
            reference.coupling.knm_r * radius[None, :] * np.cos(offsets), axis=1
        )
        expected_amplitude = round(
            float(
                np.mean(
                    np.maximum(
                        radius
                        + dt * (growth + reference.spec.amplitude.epsilon * coupling),
                        0.0,
                    )
                )
            ),
            4,
        )
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]

    def request(path: str, method: str = "GET") -> dict[str, Any]:
        """Read a real HTTP response from the foreground server."""
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        try:
            connection.request(method, path)
            response = connection.getresponse()
            assert response.status == 200
            result: dict[str, Any] = json.loads(response.read())
            return result
        finally:
            connection.close()

    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"SPO_ENV", "SPO_API_KEY", "SPO_RATE_LIMIT_PER_MINUTE"}
    }
    native_calls = tmp_path / "native-calls.txt"
    observed_cli = (
        "import sys; from pathlib import Path; import spo_kernel\n"
        "def observe(frame,event,value):\n"
        " if event!='c_return' or frame.f_globals.get('__name__','') not in "
        "('scpn_phase_orchestrator.upde.engine',"
        "'scpn_phase_orchestrator.upde.stuart_landau'): return\n"
        " owner=getattr(value,'__self__',None)\n"
        " if isinstance(owner,"
        "(spo_kernel.PyUPDEStepper,spo_kernel.PyStuartLandauStepper)) "
        "and getattr(value,'__name__','')=='step':\n"
        f"  with Path({str(native_calls)!r}).open('a') as recorded:\n"
        "   recorded.write(type(owner).__name__+':'+str(id(owner))+chr(10))\n"
        "sys.setprofile(observe)\n" + CLI
    )
    with (tmp_path / "server.log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                observed_cli,
                "serve",
                str(spec_path),
                "--port",
                str(port),
                "--require-kernel",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            creationflags=PROCESS_GROUP,
        )
        try:
            deadline = time.monotonic() + 30
            while True:
                assert process.poll() is None, (tmp_path / "server.log").read_text()
                try:
                    health = request("/api/health")
                    break
                except (ConnectionError, TimeoutError):
                    assert time.monotonic() < deadline
                    time.sleep(0.1)
            assert health["status"] == "healthy"
            config = request("/api/config")
            assert config["backend"] == "rust" and config["kernel_required"] is True
            assert config["kernel"]["version"] == version("spo-kernel")
            assert len(config["kernel"]["sha256"]) == 64
            assert request("/api/state")["step"] == 0
            stepped = request("/api/step", "POST")
            assert stepped["step"] == 1
            assert stepped["R_global"] == pytest.approx(expected_r, abs=1e-4)
            if expected_amplitude is not None:
                assert stepped["mean_amplitude"] == pytest.approx(
                    expected_amplitude, abs=1e-4
                )
            assert request("/api/reset", "POST")["step"] == 0
        finally:
            process.send_signal(STOP_SIGNAL)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        # Uvicorn can re-emit SIGTERM after its complete graceful shutdown.
        expected_exit = {0} if sys.platform == "win32" else {0, -signal.SIGTERM}
        assert process.returncode in expected_exit, (
            tmp_path / "server.log"
        ).read_text()
        assert "Application shutdown complete" in (tmp_path / "server.log").read_text()
        with socket.socket() as stopped:
            assert stopped.connect_ex(("127.0.0.1", port)) != 0

    calls = native_calls.read_text().splitlines()
    assert len(calls) == 3, calls
    expected_consumer = "PyStuartLandauStepper" if amplitude else "PyUPDEStepper"
    assert calls[-1].startswith(expected_consumer + ":")
    assert calls[-1] not in calls[:2]

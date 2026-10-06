# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — stateless UPDE dispatcher validation contracts

"""Fail-closed contracts for public stateless UPDE dispatch."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_phase_orchestrator.upde._engine_validation as engine_validation
from scpn_phase_orchestrator.upde import _run as run_mod

FloatArray = NDArray[np.float64]


def _payload() -> list[Any]:
    """Return a valid public ``upde_run`` argument payload."""
    return [
        np.array([0.1, 0.2], dtype=np.float64),
        np.array([1.0, 1.2], dtype=np.float64),
        np.array([[0.0, 0.3], [0.3, 0.0]], dtype=np.float64),
        np.zeros((2, 2), dtype=np.float64),
        0.0,
        0.0,
        0.01,
        1,
        "euler",
        1,
        1e-6,
        1e-3,
    ]


def _identity_backend(phases: FloatArray, *_args: object) -> FloatArray:
    """Return a copy of the supplied phases for dispatch-boundary tests."""
    return phases.copy()


def test_core_engine_validator_is_directly_linked() -> None:
    """Keep the core-owned validator visible to module-linkage checks."""
    assert callable(engine_validation.validate_upde_backend_inputs)
    assert callable(engine_validation.validate_upde_backend_output)
    assert callable(engine_validation.validate_upde_schedule_backend_inputs)


@pytest.mark.parametrize(
    ("index", "replacement", "match"),
    [
        (0, np.array(["0.1", "0.2"]), "phases"),
        (0, np.array([], dtype=np.float64), "phases"),
        (1, np.array(["1.0", "1.2"]), "omegas"),
        (2, np.array([["0", "0.3"], ["0.3", "0"]]), "knm"),
        (2, np.zeros(3, dtype=np.float64), "knm"),
        (3, np.array([["0", "0"], ["0", "0"]]), "alpha"),
        (3, np.zeros((1, 2, 2), dtype=np.float64), "alpha"),
        (4, "0.0", "zeta"),
        (5, True, "psi"),
        (6, "0.01", "dt"),
        (7, "1", "n_steps"),
        (8, 1, "method"),
        (9, True, "n_substeps"),
        (9, 1.5, "n_substeps"),
        (10, "1e-6", "atol"),
        (11, "1e-3", "rtol"),
    ],
)
def test_public_run_rejects_aliases_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    index: int,
    replacement: object,
    match: str,
) -> None:
    """Reject coercible aliases before an optional backend sees them."""
    dispatched = False

    def _backend(*_args: object) -> FloatArray:
        nonlocal dispatched
        dispatched = True
        return np.zeros(2, dtype=np.float64)

    payload = _payload()
    payload[index] = replacement
    monkeypatch.setattr(run_mod, "_dispatch", lambda: _backend)

    with pytest.raises((TypeError, ValueError), match=match):
        run_mod.upde_run(*payload)

    assert not dispatched


def test_public_schedule_rejects_numeric_string_aliases_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a coercible frequency schedule before backend selection."""
    payload = _payload()
    dispatched = False

    def _backend(*_args: object) -> FloatArray:
        nonlocal dispatched
        dispatched = True
        return np.zeros(2, dtype=np.float64)

    monkeypatch.setattr(run_mod, "_dispatch_schedule", lambda: _backend)

    with pytest.raises(TypeError, match="omega_schedule"):
        run_mod.upde_run_omega_schedule(
            payload[0],
            np.array([["1.0", "1.2"]]),
            payload[2],
            payload[3],
            payload[4],
            payload[5],
            payload[6],
            payload[8],
            payload[9],
            payload[10],
            payload[11],
        )

    assert not dispatched


@pytest.mark.parametrize(
    "backend_output",
    [
        np.array(["0.1", "0.2"]),
        np.array([True, False]),
        np.array([0.1, np.bool_(True)], dtype=object),
        np.array([0.1, np.inf]),
        np.array([0.1]),
        np.array([[0.1, 0.2]]),
        np.array([0.1, 2.0 * np.pi]),
        np.array([-1e-9, 0.2]),
    ],
)
def test_public_run_rejects_malformed_backend_outputs(
    monkeypatch: pytest.MonkeyPatch,
    backend_output: NDArray[Any],
) -> None:
    """Do not publish malformed optional-backend phase evidence."""
    monkeypatch.setattr(
        run_mod,
        "_dispatch",
        lambda: lambda *_args: backend_output,
    )

    with pytest.raises((TypeError, ValueError)):
        run_mod.upde_run(*_payload())


def test_public_schedule_rejects_numeric_string_backend_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply the output contract to the schedule dispatcher too."""
    payload = _payload()
    monkeypatch.setattr(
        run_mod,
        "_dispatch_schedule",
        lambda: lambda *_args: np.array(["0.1", "0.2"]),
    )

    with pytest.raises(TypeError, match="result"):
        run_mod.upde_run_omega_schedule(
            payload[0],
            np.array([[1.0, 1.2]], dtype=np.float64),
            payload[2],
            payload[3],
            payload[4],
            payload[5],
            payload[6],
            payload[8],
            payload[9],
            payload[10],
            payload[11],
        )


def test_public_run_normalises_valid_array_likes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep ordinary real-valued array-like inputs supported."""
    payload = _payload()
    payload[0] = payload[0].tolist()
    payload[1] = payload[1].tolist()
    payload[2] = payload[2].ravel().tolist()
    payload[3] = payload[3].ravel().tolist()
    monkeypatch.setattr(run_mod, "_dispatch", lambda: _identity_backend)

    result = run_mod.upde_run(*payload)

    np.testing.assert_allclose(result, np.array([0.1, 0.2]))
    assert result.dtype == np.float64


@pytest.mark.parametrize("schedule", [False, True])
def test_public_run_uses_python_when_no_listed_backend_loads(
    monkeypatch: pytest.MonkeyPatch, schedule: bool
) -> None:
    """A backend list without the Python entry still ends in the Python integrator.

    The public backend state names one optional backend and omits ``python``.
    Where that backend cannot be loaded the dispatcher runs out of candidates;
    where it can, it computes the same step. Either way the result is the
    Python reference.
    """
    payload = _payload()
    monkeypatch.setattr(run_mod, "ACTIVE_BACKEND", "python")
    monkeypatch.setattr(run_mod, "AVAILABLE_BACKENDS", ["python"])
    if schedule:
        arguments = (
            payload[0],
            np.array([[1.0, 1.2]], dtype=np.float64),
            *payload[2:7],
            *payload[8:12],
        )
        reference = run_mod.upde_run_omega_schedule(*arguments)
    else:
        reference = run_mod.upde_run(*payload)

    monkeypatch.setattr(run_mod, "ACTIVE_BACKEND", "mojo")
    monkeypatch.setattr(run_mod, "AVAILABLE_BACKENDS", ["mojo", "mojo"])
    if schedule:
        result = run_mod.upde_run_omega_schedule(*arguments)
    else:
        result = run_mod.upde_run(*payload)

    np.testing.assert_allclose(result, reference, rtol=1e-12, atol=1e-12)


def test_backend_state_is_resolved_once_when_two_readers_arrive_together() -> None:
    """A reader that waited for another reader's resolution takes its result.

    In a fresh interpreter the main thread holds the resolution lock, as a
    first reader does. A second thread then reads the public attribute, finds
    no state and waits. The main thread publishes the resolved state and
    releases the lock; the waiting reader must return that state without
    resolving again.
    """
    program = """
import json, threading, time
import scpn_phase_orchestrator.upde._run as run
assert "ACTIVE_BACKEND" not in vars(run)
resolutions = []
resolve = run._resolve_backends
seen = {}
def read():
    seen["active"] = run.ACTIVE_BACKEND
    seen["available"] = list(run.AVAILABLE_BACKENDS)
with run._BACKEND_STATE_LOCK:
    reader = threading.Thread(target=read)
    reader.start()
    time.sleep(0.5)
    assert reader.is_alive() and not seen
    active, available = resolve()
    resolutions.append(active)
    run.ACTIVE_BACKEND, run.AVAILABLE_BACKENDS = active, available
reader.join(timeout=30)
assert not reader.is_alive()
report = {"seen": seen, "published": [active, available]}
report["resolutions"] = len(resolutions)
print(json.dumps(report))
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed["seen"] == {
        "active": observed["published"][0],
        "available": observed["published"][1],
    }
    assert observed["published"][1][-1] == "python"
    assert observed["resolutions"] == 1

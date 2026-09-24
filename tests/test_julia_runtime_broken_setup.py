# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Broken Julia set-up tests

"""Prove a broken Julia set-up disables the Julia backend instead of the package.

``juliacall`` reports an unusable Julia installation with ``ValueError`` or a bare
``Exception`` rather than ``ImportError``. The backend resolvers only treat
``ImportError`` (and a few runtime errors) as "backend unavailable", so before the
runtime probes converted those errors, importing any module with a Julia backend
failed outright. Each case runs in a fresh interpreter whose environment points
``juliacall`` at an executable that does not exist, which is a real
misconfiguration, not a substituted module.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytest.importorskip("juliacall")

_BROKEN_JULIA = {**os.environ, "PYTHON_JULIACALL_EXE": "/nonexistent/julia"}

_PROBES = (
    "scpn_phase_orchestrator.coupling._julia_runtime",
    "scpn_phase_orchestrator.monitor._julia_runtime",
    "scpn_phase_orchestrator.upde._julia_runtime",
    "scpn_phase_orchestrator.experimental.accelerators._julia_runtime",
)


def _run(source: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-W", "ignore", "-c", source],
        capture_output=True,
        text=True,
        timeout=300.0,
        env=_BROKEN_JULIA,
        check=False,
    )


@pytest.mark.parametrize("probe", _PROBES)
def test_runtime_probe_reports_a_broken_setup_as_import_error(probe: str) -> None:
    name = (
        "require_juliacall_main"
        if not probe.endswith("accelerators._julia_runtime")
        else "require_julia_main"
    )
    completed = _run(
        f"import {probe} as m\n"
        "try:\n"
        f"    m.{name}()\n"
        "except ImportError as exc:\n"
        "    print('ImportError', exc)\n"
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("ImportError juliacall could not start Julia")


@pytest.mark.parametrize(
    "module",
    [
        "scpn_phase_orchestrator.monitor.transfer_entropy",
        "scpn_phase_orchestrator.coupling.spectral",
        "scpn_phase_orchestrator.upde.order_params",
    ],
)
def test_modules_with_a_julia_backend_still_import(module: str) -> None:
    completed = _run(f"import {module} as m\nprint('julia' in m.AVAILABLE_BACKENDS)\n")
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False"

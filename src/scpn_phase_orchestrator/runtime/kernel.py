# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Native kernel execution verification

"""Verify the installed native kernel through public numerical consumers."""

from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np

from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine


@dataclass(frozen=True)
class KernelVerification:
    """Identity of a native binary that passed public numerical checks.

    Attributes
    ----------
    version : str
        Installed spo-kernel distribution version.
    extension : str
        Absolute path of the loaded extension.
    sha256 : str
        SHA-256 of the loaded native library.
    """

    version: str
    extension: str
    sha256: str


def verify_kernel() -> KernelVerification:
    """Execute native phase and amplitude steps with analytical oracles.

    Both engines integrate uncoupled oscillators: phase advances by omega * dt
    and a Stuart-Landau amplitude at its unit equilibrium stays one. The public
    engines must select Rust before either result can qualify native operation.
    This check does not advance a caller's simulation.

    Returns
    -------
    KernelVerification
        Installed binary identity after both real numerical operations pass.

    Raises
    ------
    ImportError
        The native extension cannot be loaded or has no identifiable library file.
    RuntimeError
        A public engine selected the NumPy fallback.
    AssertionError
        A native result disagrees with its analytical solution.
    """
    extension = importlib.import_module("spo_kernel.spo_kernel")
    phase = UPDEEngine(2, dt=0.01, method="rk4")
    amplitude = StuartLandauEngine(2, dt=0.01, method="rk4")
    if phase.backend != "rust" or amplitude.backend != "rust":
        raise RuntimeError("spo-kernel required: a numerical engine selected NumPy")

    initial = np.array([0.2, 1.1], dtype=np.float64)
    omega = np.array([1.0, -0.5], dtype=np.float64)
    zero = np.zeros((2, 2), dtype=np.float64)
    expected = initial + 0.01 * omega
    result = phase.step(initial, omega, zero, alpha=zero)
    np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-13)
    result_amplitude = amplitude.step(
        np.concatenate((initial, np.ones(2))),
        omega,
        np.ones(2),
        zero,
        zero,
        0.0,
        0.0,
        alpha=zero,
    )
    np.testing.assert_allclose(
        result_amplitude,
        np.concatenate((expected, np.ones(2))),
        rtol=0.0,
        atol=1e-13,
    )
    binary_path = getattr(extension, "__file__", None)
    if binary_path is None:
        raise ImportError("spo-kernel has no native library path")
    binary = Path(binary_path).resolve()
    return KernelVerification(
        version=version("spo-kernel"),
        extension=str(binary),
        sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
    )

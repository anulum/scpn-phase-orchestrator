# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Cybernetic closure validation contracts

"""
Validation contracts for CyberneticClosure constructor, run, and step phase-vector
boundaries.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.ssgf.carrier import GeometryCarrier
from scpn_phase_orchestrator.ssgf.closure import CyberneticClosure


def test_u1_cybernetic_closure_rejects_negative_max_steps() -> None:
    carrier = GeometryCarrier(n_oscillators=4, z_dim=2, lr=0.1, seed=1)
    with pytest.raises(ValueError, match="non-negative integer"):
        CyberneticClosure(carrier=carrier, max_steps=-1)


def test_u1_cybernetic_closure_run_rejects_negative_outer_steps() -> None:
    carrier = GeometryCarrier(n_oscillators=4, z_dim=2, lr=0.1, seed=1)
    closure = CyberneticClosure(carrier=carrier)
    with pytest.raises(ValueError, match="non-negative integer"):
        closure.run(np.zeros(4, dtype=float), -1)


def test_u1_cybernetic_closure_run_rejects_boolean_outer_steps() -> None:
    carrier = GeometryCarrier(n_oscillators=4, z_dim=2, lr=0.1, seed=1)
    closure = CyberneticClosure(carrier=carrier)
    with pytest.raises(TypeError, match="non-negative integer"):
        closure.run(np.zeros(4, dtype=float), True)  # type: ignore[arg-type]


def test_u1_cybernetic_closure_run_rejects_non_vector_phases() -> None:
    carrier = GeometryCarrier(n_oscillators=4, z_dim=2, lr=0.1, seed=1)
    closure = CyberneticClosure(carrier=carrier)
    with pytest.raises(ValueError, match="1D vector"):
        closure.run(np.zeros((2, 2), dtype=float), 0)


def test_u1_cybernetic_closure_run_rejects_boolean_phase_vector() -> None:
    carrier = GeometryCarrier(n_oscillators=4, z_dim=2, lr=0.1, seed=1)
    closure = CyberneticClosure(carrier=carrier)
    with pytest.raises(ValueError, match="boolean dtype"):
        closure.run(np.array([True, False, True, False], dtype=bool), 0)


def test_u1_cybernetic_closure_step_rejects_mismatched_phase_length() -> None:
    carrier = GeometryCarrier(n_oscillators=4, z_dim=2, lr=0.1, seed=1)
    closure = CyberneticClosure(carrier=carrier)
    with pytest.raises(ValueError, match="oscillator count"):
        closure.step(np.zeros(3, dtype=float))


def test_u1_cybernetic_closure_step_rejects_boolean_phase_vector() -> None:
    carrier = GeometryCarrier(n_oscillators=4, z_dim=2, lr=0.1, seed=1)
    closure = CyberneticClosure(carrier=carrier)
    with pytest.raises(ValueError, match="boolean dtype"):
        closure.step(np.array([True, False, True, False], dtype=bool))

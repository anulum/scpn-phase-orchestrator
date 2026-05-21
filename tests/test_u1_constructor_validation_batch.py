# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — U1 constructor validation batch tests

from __future__ import annotations

import pytest
import numpy as np

from scpn_phase_orchestrator.actuation.constraints import ActionProjector
from scpn_phase_orchestrator.coupling.templates import KnmTemplate, KnmTemplateSet
from scpn_phase_orchestrator.monitor.boundaries import BoundaryObserver
from scpn_phase_orchestrator.runtime.audit_logger import AuditLogger
from scpn_phase_orchestrator.ssgf.carrier import GeometryCarrier
from scpn_phase_orchestrator.ssgf.closure import CyberneticClosure


def test_u1_action_projector_rejects_non_finite_rate_limit() -> None:
    with pytest.raises(ValueError, match="finite >= 0"):
        ActionProjector(rate_limits={"K": float("nan")}, value_bounds={"K": (0.0, 1.0)})


def test_u1_knm_template_set_rejects_non_square_template() -> None:
    reg = KnmTemplateSet()
    with pytest.raises(ValueError, match="square"):
        reg.add(
            KnmTemplate(
                name="bad",
                knm=np.ones((2, 3), dtype=float),
                alpha=np.ones((2, 3), dtype=float),
                description="invalid",
            )
        )


def test_u1_boundary_observer_rejects_inverted_bounds() -> None:
    with pytest.raises(TypeError, match="BoundaryDef"):
        BoundaryObserver([object()])  # type: ignore[list-item]


def test_u1_audit_logger_rejects_directory_path(tmp_path) -> None:
    with pytest.raises(Exception, match="directory"):
        AuditLogger(tmp_path)


def test_u1_geometry_carrier_rejects_non_positive_latent_dim() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        GeometryCarrier(n_oscillators=4, z_dim=0, lr=0.1)


def test_u1_cybernetic_closure_rejects_negative_max_steps() -> None:
    carrier = GeometryCarrier(n_oscillators=4, z_dim=2, lr=0.1, seed=1)
    with pytest.raises(ValueError, match="non-negative integer"):
        CyberneticClosure(carrier=carrier, max_steps=-1)

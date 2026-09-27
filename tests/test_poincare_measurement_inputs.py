# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — poincare measurement ingress

"""Verify original source-type refusal through monitor.poincare public calls."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

import scpn_phase_orchestrator.monitor.poincare as module
from tests.measurement_samples import source_alias


@pytest.mark.parametrize("kind", ["text", "duration", "object_duration"])
def test_original_measurement_types_are_refused(kind: str) -> None:
    """An otherwise valid numerical shape cannot erase text or duration units."""
    with pytest.raises(ValueError):
        module.poincare_section(source_alias(kind, (16, 2)), np.array([1.0, 0.0]))


@pytest.mark.parametrize("backend", ["go", "julia", "mojo"])
@pytest.mark.parametrize("kind", ["text", "duration", "object_duration"])
def test_direct_backend_refuses_original_measurement_types(
    backend: str, kind: str
) -> None:
    """Each actual language bridge refuses aliases before invoking its kernel."""
    bridge = importlib.import_module(
        f"scpn_phase_orchestrator.experimental.accelerators.monitor._poincare_{backend}"
    )
    operation = getattr(bridge, "poincare_section_" + backend)
    with pytest.raises(ValueError):
        operation(source_alias(kind, (48,)), 16, 3, np.array([1.0, 0.0, 0.0]), 0.0, 0)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — recurrence measurement ingress

"""Verify original source-type refusal through monitor.recurrence public calls."""

from __future__ import annotations

import importlib

import pytest

import scpn_phase_orchestrator.monitor.recurrence as module
from tests.measurement_samples import source_alias


@pytest.mark.parametrize("kind", ["text", "duration", "object_duration"])
def test_original_measurement_types_are_refused(kind: str) -> None:
    """An otherwise valid numerical shape cannot erase text or duration units."""
    with pytest.raises(ValueError):
        module.recurrence_matrix(source_alias(kind, (8, 2)), 0.1)


@pytest.mark.parametrize("backend", ["go", "julia", "mojo"])
@pytest.mark.parametrize("kind", ["text", "duration", "object_duration"])
def test_direct_backend_refuses_original_measurement_types(
    backend: str, kind: str
) -> None:
    """Each actual language bridge refuses aliases before invoking its kernel."""
    bridge = importlib.import_module(
        f"scpn_phase_orchestrator.experimental.accelerators.monitor._recurrence_{backend}"
    )
    operation = getattr(bridge, "recurrence_matrix_" + backend)
    with pytest.raises(ValueError):
        operation(source_alias(kind, (48,)), 16, 3, 0.1, False)

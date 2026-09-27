# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — itpc measurement ingress

"""Verify original source-type refusal through monitor.itpc public calls."""

from __future__ import annotations

import importlib

import pytest

import scpn_phase_orchestrator.monitor.itpc as module
from tests.measurement_samples import source_alias


@pytest.mark.parametrize("kind", ["text", "duration", "object_duration"])
def test_original_measurement_types_are_refused(kind: str) -> None:
    """An otherwise valid numerical shape cannot erase text or duration units."""
    with pytest.raises(ValueError):
        module.compute_itpc(source_alias(kind, (3, 16)))


@pytest.mark.parametrize("backend", ["go", "julia", "mojo"])
@pytest.mark.parametrize("kind", ["text", "duration", "object_duration"])
def test_direct_backend_refuses_original_measurement_types(
    backend: str, kind: str
) -> None:
    """Each actual language bridge refuses aliases before invoking its kernel."""
    bridge = importlib.import_module(
        f"scpn_phase_orchestrator.experimental.accelerators.monitor._itpc_{backend}"
    )
    operation = getattr(bridge, "compute_itpc_" + backend)
    with pytest.raises(ValueError):
        operation(source_alias(kind, (12,)), 3, 4)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Private accelerator namespace tests

"""Regression tests for private experimental accelerator namespace exports."""

from __future__ import annotations

import importlib

import pytest

PRIVATE_NAMESPACES = (
    "scpn_phase_orchestrator.experimental",
    "scpn_phase_orchestrator.experimental.accelerators",
    "scpn_phase_orchestrator.experimental.accelerators.coupling",
    "scpn_phase_orchestrator.experimental.accelerators.monitor",
    "scpn_phase_orchestrator.experimental.accelerators.upde",
)


@pytest.mark.parametrize("module_name", PRIVATE_NAMESPACES)
def test_private_namespace_has_no_wildcard_exports(module_name: str) -> None:
    module = importlib.import_module(module_name)

    assert module.__all__ == ()


def test_private_backend_submodule_remains_explicitly_importable() -> None:
    module = importlib.import_module(
        "scpn_phase_orchestrator.experimental.accelerators.upde._engine_go"
    )

    assert module.__name__.endswith("._engine_go")

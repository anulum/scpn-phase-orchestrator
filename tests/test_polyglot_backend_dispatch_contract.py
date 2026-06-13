# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Polyglot backend dispatch contract tests

from __future__ import annotations

import importlib
from collections.abc import Callable
from types import ModuleType

import pytest

REPRESENTATIVE_RESOLVE_MODULES = (
    "scpn_phase_orchestrator.coupling.hodge",
    "scpn_phase_orchestrator.monitor.transfer_entropy",
    "scpn_phase_orchestrator.upde._run",
    "scpn_phase_orchestrator.upde.envelope",
)


DISPATCH_MODULES: tuple[tuple[str, tuple[object, ...]], ...] = (
    ("scpn_phase_orchestrator.coupling.hodge", ()),
    ("scpn_phase_orchestrator.monitor.dimension", ("correlation_integral",)),
    ("scpn_phase_orchestrator.monitor.transfer_entropy", ("matrix",)),
    ("scpn_phase_orchestrator.upde.envelope", ("extract",)),
    ("scpn_phase_orchestrator.upde._run", ()),
)


def _module(name: str) -> ModuleType:
    return importlib.import_module(name)


@pytest.mark.parametrize("module_name", REPRESENTATIVE_RESOLVE_MODULES)
@pytest.mark.parametrize("fault_type", (ValueError, TypeError))
def test_backend_resolution_propagates_backend_contract_faults(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    fault_type: type[Exception],
) -> None:
    """Backend discovery must not hide physics or ABI contract faults."""

    module = _module(module_name)

    def raise_contract_fault(_backend: str) -> object:
        raise fault_type("backend contract fault")

    monkeypatch.setattr(module, "_load_backend", raise_contract_fault)
    with pytest.raises(fault_type, match="backend contract fault"):
        module._resolve_backends()


@pytest.mark.parametrize("module_name,args", DISPATCH_MODULES)
@pytest.mark.parametrize("fault_type", (ValueError, TypeError))
def test_backend_dispatch_propagates_backend_contract_faults(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    args: tuple[object, ...],
    fault_type: type[Exception],
) -> None:
    """Runtime dispatch must not silently demote invalid compiled kernels."""

    module = _module(module_name)

    def raise_contract_fault(_backend: str) -> object:
        raise fault_type("backend contract fault")

    monkeypatch.setattr(module, "ACTIVE_BACKEND", "rust")
    monkeypatch.setattr(module, "AVAILABLE_BACKENDS", ["rust", "python"])
    monkeypatch.setattr(module, "_load_backend", raise_contract_fault)
    with pytest.raises(fault_type, match="backend contract fault"):
        module._dispatch(*args)


def test_schedule_dispatch_propagates_backend_contract_faults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scheduled UPDE dispatch has the same fail-closed backend contract."""

    module = _module("scpn_phase_orchestrator.upde._run")

    def raise_contract_fault(_backend: str) -> Callable[..., object]:
        raise ValueError("schedule backend contract fault")

    monkeypatch.setattr(module, "ACTIVE_BACKEND", "rust")
    monkeypatch.setattr(module, "AVAILABLE_BACKENDS", ["rust", "python"])
    monkeypatch.setattr(module, "_load_schedule_backend", raise_contract_fault)
    with pytest.raises(ValueError, match="schedule backend contract fault"):
        module._dispatch_schedule()

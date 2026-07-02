# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — monitor/coupling Julia runtime contracts

"""Regression tests for monitor and coupling optional Julia runtime probes."""

from __future__ import annotations

import sys
from collections.abc import Callable
from types import ModuleType

import pytest

from scpn_phase_orchestrator.coupling import (
    attention_residuals,
    spatial_modulator,
    spectral,
)
from scpn_phase_orchestrator.coupling import hodge as coupling_hodge
from scpn_phase_orchestrator.coupling._julia_runtime import (
    require_juliacall_main as require_coupling_juliacall_main,
)
from scpn_phase_orchestrator.monitor import (
    chimera,
    dimension,
    embedding,
    entropy_prod,
    itpc,
    koopman_edmd,
    lyapunov,
    npe,
    opt_entropy,
    pid,
    poincare,
    psychedelic,
    recurrence,
    transfer_entropy,
    twin_confidence,
    winding,
)
from scpn_phase_orchestrator.monitor._julia_runtime import (
    require_juliacall_main as require_monitor_juliacall_main,
)

Loader = Callable[[], object]
RuntimeProbe = Callable[[], ModuleType]

JULIA_LOADERS: tuple[tuple[str, Loader], ...] = (
    ("coupling.attention_residuals", attention_residuals._load_julia),
    ("coupling.hodge", coupling_hodge._load_julia_fn),
    ("coupling.spatial_modulator", spatial_modulator._load_julia_fn),
    ("coupling.spectral", spectral._load_julia_primitive),
    ("monitor.chimera", chimera._load_julia_fn),
    ("monitor.dimension", dimension._load_julia_fns),
    ("monitor.embedding", embedding._load_julia_fns),
    ("monitor.entropy_prod", entropy_prod._load_julia_fn),
    ("monitor.itpc", itpc._load_julia_fns),
    ("monitor.koopman_edmd", koopman_edmd._load_julia_fns),
    ("monitor.lyapunov", lyapunov._load_julia_fn),
    ("monitor.npe", npe._load_julia_fns),
    ("monitor.opt_entropy", opt_entropy._load_julia_fns),
    ("monitor.pid", pid._load_julia_fn),
    ("monitor.poincare", poincare._load_julia_fns),
    ("monitor.psychedelic", psychedelic._load_julia_fn),
    ("monitor.recurrence", recurrence._load_julia_fns),
    ("monitor.transfer_entropy", transfer_entropy._load_julia_fns),
    ("monitor.twin_confidence", twin_confidence._load_julia),
    ("monitor.winding", winding._load_julia_fn),
)


@pytest.mark.parametrize(
    ("loader_name", "loader"),
    JULIA_LOADERS,
    ids=[name for name, _loader in JULIA_LOADERS],
)
def test_public_julia_loaders_reject_partial_juliacall_runtime(
    loader_name: str,
    loader: Loader,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public Julia loaders must reject `juliacall` when `Main` is absent."""
    monkeypatch.setitem(sys.modules, "juliacall", ModuleType("juliacall"))

    with pytest.raises(ImportError, match="juliacall.*Main|Main unavailable"):
        loader()

    assert loader_name


@pytest.mark.parametrize(
    ("probe_name", "probe"),
    (
        ("coupling", require_coupling_juliacall_main),
        ("monitor", require_monitor_juliacall_main),
    ),
)
def test_julia_runtime_probe_returns_complete_runtime(
    probe_name: str,
    probe: RuntimeProbe,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime probes must return the complete `juliacall` module unchanged."""
    stub = ModuleType("juliacall")
    stub.Main = object()
    monkeypatch.setitem(sys.modules, "juliacall", stub)

    assert probe() is stub
    assert probe_name

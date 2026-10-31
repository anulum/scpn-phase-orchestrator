# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Dispatch fallback-chain tests for upde_run

"""Backend dispatch-chain contracts for the UPDE run dispatcher."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde import _run as run_mod

FloatArray = NDArray[np.float64]
BackendFn = Callable[..., FloatArray]


class TestDispatchFallbackChain:
    def test_dispatch_falls_back_to_python_when_loader_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, int] = {"rust": 0}

        def _fail_rust() -> BackendFn:
            calls["rust"] += 1
            raise ImportError("missing rust backend")

        monkeypatch.setattr(run_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(run_mod, "AVAILABLE_BACKENDS", ["rust", "python"])
        monkeypatch.setattr(run_mod, "_LOADERS", {"rust": _fail_rust})
        # Empty the cache last. monkeypatch reads ACTIVE_BACKEND before setting
        # it, and the module resolves that name lazily by loading the real
        # owners into the cache, so an earlier empty cache would be refilled.
        monkeypatch.setattr(run_mod, "_BACKEND_CACHE", {})

        backend = run_mod._dispatch()
        assert backend is None
        assert calls["rust"] == 1

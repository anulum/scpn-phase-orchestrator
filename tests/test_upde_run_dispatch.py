# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Dispatch fallback-chain tests for upde_run

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.upde import _run as run_mod


class TestDispatchFallbackChain:
    def test_dispatch_falls_back_to_python_when_loader_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, int] = {"rust": 0}

        def _fail_rust():
            calls["rust"] += 1
            raise ImportError("missing rust backend")

        monkeypatch.setattr(run_mod, "_BACKEND_CACHE", {})
        monkeypatch.setattr(run_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(run_mod, "AVAILABLE_BACKENDS", ["rust", "python"])
        monkeypatch.setattr(run_mod, "_LOADERS", {"rust": _fail_rust})

        backend = run_mod._dispatch()
        assert backend is None
        assert calls["rust"] == 1

    def test_dispatch_uses_cached_loader_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, int] = {"go": 0}

        def _ok_go():
            calls["go"] += 1

            def _backend(
                phases,
                omegas,
                knm,
                alpha,
                zeta,
                psi,
                dt,
                n_steps,
                method,
                n_substeps,
                atol,
                rtol,
            ):
                return np.asarray(phases, dtype=np.float64)

            return _backend

        monkeypatch.setattr(run_mod, "_BACKEND_CACHE", {})
        monkeypatch.setattr(run_mod, "ACTIVE_BACKEND", "go")
        monkeypatch.setattr(run_mod, "AVAILABLE_BACKENDS", ["go", "python"])
        monkeypatch.setattr(run_mod, "_LOADERS", {"go": _ok_go})

        first = run_mod._dispatch()
        second = run_mod._dispatch()

        assert first is not None
        assert second is not None
        assert calls["go"] == 1

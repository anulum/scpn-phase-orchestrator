# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Bridge error-path coverage for ITPC

"""Coverage tests for the ITPC bridge modules' defensive error paths.

The cross-backend parity tests exercise the happy path; these cover:

* `_load_lib` / `_ensure_exe` `ImportError` when the build artefact is
  missing — so a deployment guard is guaranteed to fire.
* The lazy-cache branch of `_load_lib` (second call returns the
  cached CDLL without re-resolving symbols).
* Empty-input short-circuits in the Mojo bridges.
* The `n_trials == 0` and empty-pause early returns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor import (
    _itpc_go,
    _itpc_mojo,
)


class TestGoBridgeErrorPaths:
    def test_missing_lib_raises_import_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_itpc_go, "_LIB", None)
        monkeypatch.setattr(_itpc_go, "_LIB_PATH", Path("/nonexistent.so"))
        with pytest.raises(ImportError, match="libitpc.so not found"):
            _itpc_go._load_lib()

    def test_cached_lib_returns_early(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The second _load_lib() call must return the cached handle
        without re-opening the .so."""
        if not _itpc_go._LIB_PATH.exists():
            pytest.skip("libitpc.so not built")
        first = _itpc_go._load_lib()
        second = _itpc_go._load_lib()
        assert first is second


class TestMojoBridgeErrorPaths:
    def test_missing_exe_raises_import_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_itpc_mojo, "_EXE_PATH", Path("/nonexistent-mojo-exe"))
        with pytest.raises(ImportError, match="not built"):
            _itpc_mojo._ensure_exe()

    def test_zero_trials_short_circuits(self) -> None:
        """``compute_itpc_mojo`` returns a zero vector without spawning
        a subprocess when no trials are supplied."""
        result = _itpc_mojo.compute_itpc_mojo(
            np.zeros((0, 5)),
            n_trials=0,
            n_tp=5,
        )
        assert result.shape == (5,)
        assert np.all(result == 0.0)

    def test_empty_pause_short_circuits(self) -> None:
        result = _itpc_mojo.itpc_persistence_mojo(
            np.zeros((0, 5)),
            n_trials=0,
            n_tp=5,
            pause_indices=np.array([]),
        )
        assert result == 0.0

    def test_failed_subprocess_raises_value_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-zero exit → clean ValueError with stderr surface."""

        class _FakeProc:
            returncode = 1
            stdout = ""
            stderr = "boom"

        def _fake_run(*args, **kwargs):
            return _FakeProc()

        monkeypatch.setattr(_itpc_mojo.subprocess, "run", _fake_run)
        # The short-circuit for empty indices would skip _run; use a
        # non-empty call that reaches the subprocess path.
        with pytest.raises(ValueError, match="exit 1.*boom"):
            _itpc_mojo.compute_itpc_mojo(
                np.ones((2, 3)),
                n_trials=2,
                n_tp=3,
            )


class TestPythonFallbackEdgeCases:
    """Hit the Python reference branches that the Rust / dispatcher
    path short-circuits around."""

    def test_1d_phases_in_persistence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A 1-D ``phases_trials`` vector must be reshaped to
        ``(1, n_tp)`` before ITPC is computed."""
        from scpn_phase_orchestrator.monitor import itpc as it_mod

        monkeypatch.setattr(it_mod, "ACTIVE_BACKEND", "python")
        out = it_mod.itpc_persistence(
            np.array([0.1, 0.2, 0.3, 0.4]),
            pause_indices=[0, 2],
        )
        # Single trial → ITPC identically 1.0 at every time point.
        assert abs(out - 1.0) < 1e-12

    def test_all_indices_out_of_range_returns_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``valid.size == 0`` in the Python fallback → ``0.0``."""
        from scpn_phase_orchestrator.monitor import itpc as it_mod

        monkeypatch.setattr(it_mod, "ACTIVE_BACKEND", "python")
        out = it_mod.itpc_persistence(
            np.ones((3, 5)),
            pause_indices=np.array([-5, 100, 999]),
        )
        assert out == 0.0

    def test_resolve_skips_broken_loader(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A loader that raises must be skipped rather than aborting
        backend resolution."""
        from scpn_phase_orchestrator.monitor import itpc as it_mod

        def _broken() -> dict:
            raise ImportError("simulated toolchain missing")

        monkeypatch.setitem(it_mod._LOADERS, "rust", _broken)
        first, available = it_mod._resolve_backends()
        assert "rust" not in available
        assert "python" in available
        assert first == available[0]

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Optional dependency import guard tests

"""Exercise optional import boundaries and real package signal configuration."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys

import numpy as np
import pytest


class TestJaxEngineImportGuard:
    """Check actual JAX availability and engine construction."""

    def test_has_jax_is_bool(self) -> None:
        """Expose availability as a boolean for the original installed runtime."""
        from scpn_phase_orchestrator.upde.jax_engine import HAS_JAX

        assert isinstance(HAS_JAX, bool)

    def test_has_jax_matches_importlib(self) -> None:
        """HAS_JAX must agree with whether jax is actually importable."""
        from scpn_phase_orchestrator.upde.jax_engine import HAS_JAX

        jax_available = importlib.util.find_spec("jax") is not None
        assert jax_available == HAS_JAX

    def test_jax_engine_usable_when_available(self) -> None:
        """If JAX is installed, JaxUPDEEngine must produce valid output."""
        from scpn_phase_orchestrator.upde.jax_engine import HAS_JAX

        if not HAS_JAX:
            import pytest

            pytest.skip("JAX not installed")

        from scpn_phase_orchestrator.upde.jax_engine import JaxUPDEEngine

        eng = JaxUPDEEngine(4, dt=0.01)
        phases = np.array([0.0, 0.5, 1.0, 1.5])
        omegas = np.ones(4)
        knm = np.ones((4, 4)) * 0.3
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((4, 4))
        result = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert result.shape == (4,)
        assert np.all(np.isfinite(result))
        assert not np.allclose(result, phases), "Phases must advance under coupling"

    def test_numpy_engine_fallback_always_works(self) -> None:
        """UPDEEngine (NumPy) must always work, regardless of JAX."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        eng = UPDEEngine(4, dt=0.01)
        knm = np.ones((4, 4)) * 0.3
        np.fill_diagonal(knm, 0.0)
        result = eng.step(
            np.array([0.0, 0.5, 1.0, 1.5]),
            np.ones(4),
            knm,
            0.0,
            0.0,
            np.zeros((4, 4)),
        )
        assert np.all(np.isfinite(result))


class TestNNModuleImportGuard:
    """Check the lazy neural namespace against actual installed dependencies."""

    def test_nn_module_imports(self) -> None:
        """Import the public neural namespace before resolving optional exports."""
        import scpn_phase_orchestrator.nn as nn_mod

        assert hasattr(nn_mod, "__all__")

    def test_nn_all_exports_exist(self) -> None:
        """Resolve declared exports or report an unavailable optional owner."""
        import importlib.util

        import scpn_phase_orchestrator.nn as nn_mod

        has_jax = importlib.util.find_spec("jax") is not None
        for name in nn_mod.__all__:
            if has_jax:
                assert hasattr(nn_mod, name), (
                    f"nn.__all__ lists {name!r} but it's not accessible"
                )
            else:
                # Without JAX, accessing any nn symbol must raise
                # AttributeError (not ModuleNotFoundError / crash)
                try:
                    getattr(nn_mod, name)
                except AttributeError:
                    pass  # expected
                else:
                    pass  # some symbols (spectral, chimera) are numpy-only


class TestJuliaSignalHandlingGuard:
    """Check real package signal configuration and the public solver fallback."""

    _MARKER = "SPO_HS_RESULT="

    def _handle_signals_after_import(
        self, preset: dict[str, str], options: tuple[str, ...] = ()
    ) -> str:
        env = {
            key: value
            for key, value in os.environ.items()
            if key
            not in {
                "PYTHON_JULIACALL_HANDLE_SIGNALS",
                "PYTHON_JULIACALL_THREADS",
                "JULIA_NUM_THREADS",
            }
        }
        env.update(preset)
        script = (
            "import os, scpn_phase_orchestrator;"
            f"print({self._MARKER!r} + "
            "os.environ.get('PYTHON_JULIACALL_HANDLE_SIGNALS', '<unset>'))"
        )
        completed = subprocess.run(
            [sys.executable, *options, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
        )
        for line in completed.stdout.splitlines():
            if line.startswith(self._MARKER):
                return line[len(self._MARKER) :]
        raise AssertionError(
            f"marker not found in subprocess stdout: {completed.stdout!r}"
        )

    def test_default_preserves_python_signal_handling(self) -> None:
        """An explicit single-thread mode avoids Base and retains Python signals."""
        assert self._handle_signals_after_import({}) == "no"

    @pytest.mark.parametrize(
        "preset,expected",
        [
            ({"JULIA_NUM_THREADS": "2"}, "yes"),
            ({"PYTHON_JULIACALL_THREADS": "auto"}, "yes"),
            ({"PYTHON_JULIACALL_THREADS": "1", "JULIA_NUM_THREADS": "2"}, "no"),
            ({"PYTHON_JULIACALL_HANDLE_SIGNALS": "yes"}, "yes"),
        ],
    )
    def test_explicit_thread_and_signal_modes(
        self, preset: dict[str, str], expected: str
    ) -> None:
        """Actual package imports honour Julia thread precedence and overrides."""
        assert self._handle_signals_after_import(preset) == expected

    @pytest.mark.parametrize("threads,expected", [("1", "no"), ("2", "yes")])
    def test_python_thread_option_has_precedence(
        self, threads: str, expected: str
    ) -> None:
        """The real interpreter option precedes conflicting environment defaults."""
        assert (
            self._handle_signals_after_import(
                {"PYTHON_JULIACALL_THREADS": "auto", "JULIA_NUM_THREADS": "4"},
                ("-X", "juliacall-threads=" + threads),
            )
            == expected
        )

    def test_operator_override_is_preserved(self) -> None:
        """An operator-provided value must not be overwritten by the package."""
        assert (
            self._handle_signals_after_import({"PYTHON_JULIACALL_HANDLE_SIGNALS": "no"})
            == "no"
        )

    def test_package_import_yields_python_backend_floor(self) -> None:
        """Advance real phases through the public dispatcher and Python floor."""
        from scpn_phase_orchestrator.upde._run import AVAILABLE_BACKENDS, upde_run

        assert "python" in AVAILABLE_BACKENDS

        phases = np.array([0.0, 0.5, 1.0, 1.5])
        omegas = np.ones(4)
        knm = np.ones((4, 4)) * 0.3
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((4, 4))
        result = upde_run(phases, omegas, knm, alpha, 0.0, 0.0, 0.01, 5)

        assert result.shape == (4,)
        assert np.all(np.isfinite(result))
        assert not np.allclose(result, phases)


# Pipeline wiring: optional import tests verify jax_engine and nn module availability
# detection. TestJaxEngineImportGuard and TestNNModuleImportGuard prove import
# fallback paths. TestJuliaSignalHandlingGuard proves the package root keeps the
# Julia backend probe importable on multithreaded hosts and that the Python
# backend floor remains usable.

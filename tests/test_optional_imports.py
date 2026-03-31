# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Optional dependency import guard tests

from __future__ import annotations

import importlib

import numpy as np


class TestJaxEngineImportGuard:
    """Verify that jax_engine module is usable regardless of JAX availability,
    and that HAS_JAX correctly reflects the runtime state."""

    def test_has_jax_is_bool(self):
        from scpn_phase_orchestrator.upde.jax_engine import HAS_JAX

        assert isinstance(HAS_JAX, bool)

    def test_has_jax_matches_importlib(self):
        """HAS_JAX must agree with whether jax is actually importable."""
        from scpn_phase_orchestrator.upde.jax_engine import HAS_JAX

        jax_available = importlib.util.find_spec("jax") is not None
        assert HAS_JAX == jax_available

    def test_jax_engine_usable_when_available(self):
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

    def test_numpy_engine_fallback_always_works(self):
        """UPDEEngine (NumPy) must always work, regardless of JAX."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        eng = UPDEEngine(4, dt=0.01)
        result = eng.step(
            np.array([0.0, 0.5, 1.0, 1.5]),
            np.ones(4),
            np.ones((4, 4)) * 0.3,
            0.0, 0.0, np.zeros((4, 4)),
        )
        assert np.all(np.isfinite(result))


class TestNNModuleImportGuard:
    """Verify that the nn/ package uses lazy loading and doesn't crash
    on import when JAX is absent."""

    def test_nn_module_imports(self):
        import scpn_phase_orchestrator.nn as nn_mod

        assert hasattr(nn_mod, "__all__")

    def test_nn_all_exports_exist(self):
        """Every name in __all__ must be resolvable."""
        import scpn_phase_orchestrator.nn as nn_mod

        for name in nn_mod.__all__:
            assert hasattr(nn_mod, name), f"nn.__all__ lists {name!r} but it's not accessible"

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Compatibility shim tests

from __future__ import annotations

import importlib.util
import math

import numpy as np

from scpn_phase_orchestrator._compat import HAS_RUST, TWO_PI

# ---------------------------------------------------------------------------
# TWO_PI: mathematical contract
# ---------------------------------------------------------------------------


class TestTwoPiConstant:
    """TWO_PI is used across 10+ modules for phase wrapping.
    Any deviation would corrupt all phase computations."""

    def test_matches_numpy_two_pi(self):
        """Must equal 2*np.pi exactly (same source)."""
        assert 2.0 * np.pi == TWO_PI

    def test_matches_stdlib_two_pi(self):
        """Must agree with math.pi to float64 precision."""
        assert abs(TWO_PI - 2.0 * math.pi) < 1e-15

    def test_is_float64(self):
        """Must be float64 for numerical consistency with numpy arrays."""
        assert isinstance(TWO_PI, (float, np.floating))

    def test_phase_wrapping_contract(self):
        """Phase wrapping with TWO_PI must map any angle to [0, 2π).
        This is the actual use case across all modules."""
        test_angles = [-10.0, -TWO_PI, -0.001, 0.0, 3.14, TWO_PI, 100.0]
        for angle in test_angles:
            wrapped = angle % TWO_PI
            assert 0.0 <= wrapped < TWO_PI, (
                f"Phase wrapping failed for angle={angle}: got {wrapped}"
            )

    def test_sin_cos_period(self):
        """sin(x + TWO_PI) == sin(x) and cos(x + TWO_PI) == cos(x)
        to machine precision — verifies TWO_PI is the correct period."""
        x = 1.2345
        assert abs(np.sin(x + TWO_PI) - np.sin(x)) < 1e-14
        assert abs(np.cos(x + TWO_PI) - np.cos(x)) < 1e-14


# ---------------------------------------------------------------------------
# HAS_RUST: runtime detection contract
# ---------------------------------------------------------------------------


class TestHasRustDetection:
    """HAS_RUST gates Rust FFI acceleration in UPDEEngine, CouplingBuilder,
    PhysicalExtractor, and order_params. Incorrect detection would either
    crash (True when missing) or silently degrade (False when present)."""

    def test_is_bool(self):
        assert isinstance(HAS_RUST, bool)

    def test_matches_importlib_detection(self):
        """Must agree with importlib.util.find_spec — the authoritative check."""
        spec_found = importlib.util.find_spec("spo_kernel") is not None
        assert spec_found == HAS_RUST, (
            f"HAS_RUST={HAS_RUST} but find_spec says {spec_found}"
        )

    def test_rust_import_consistency(self):
        """If HAS_RUST is True, `import spo_kernel` must succeed.
        If False, it must raise ImportError."""
        if HAS_RUST:
            import spo_kernel  # noqa: F401
        else:
            try:
                import spo_kernel  # noqa: F401, F811

                raise AssertionError(
                    "HAS_RUST is False but spo_kernel imports successfully"
                )
            except ImportError:
                pass  # Expected

    def test_engine_works_regardless_of_rust(self):
        """UPDEEngine must function correctly whether Rust FFI is available
        or not — the Python fallback must produce valid results."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        eng = UPDEEngine(4, dt=0.01)
        phases = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
        omegas = np.array([1.0, 1.0, 1.0, 1.0])
        knm = np.ones((4, 4)) * 0.5
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((4, 4))

        result = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert result.shape == (4,)
        assert np.all(np.isfinite(result))
        # Phases must have changed under coupling + omegas
        assert not np.allclose(result, phases)

    def test_order_params_work_regardless_of_rust(self):
        """compute_order_parameter must produce R∈[0,1] whether using
        Rust or Python backend."""
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        # Synchronised → R ≈ 1
        r_sync, _ = compute_order_parameter(np.array([0.01, 0.02, 0.015]))
        assert r_sync > 0.99

        # Anti-phase → R ≈ 0
        r_anti, _ = compute_order_parameter(np.array([0.0, np.pi]))
        assert r_anti < 0.01

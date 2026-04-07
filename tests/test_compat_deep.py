# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Compatibility shim tests

from __future__ import annotations

import math

import numpy as np
from numpy.testing import assert_allclose

from scpn_phase_orchestrator._compat import HAS_RUST, TWO_PI

# ── TWO_PI numerical precision ─────────────────────────────────────────


class TestTwoPiPrecision:
    """TWO_PI is the fundamental constant for all phase arithmetic.
    Any error here propagates to every module in the orchestrator."""

    def test_exact_float64_representation(self):
        """TWO_PI must be exactly 2.0 * np.pi (same IEEE 754 float64 bits)."""
        assert 2.0 * np.pi == TWO_PI

    def test_bit_identical_to_stdlib(self):
        """Must be bit-identical to 2.0 * math.pi (both are float64)."""
        # math.pi and np.pi should be the same IEEE 754 value
        assert 2.0 * math.pi == TWO_PI

    def test_known_decimal_approximation(self):
        """Must agree with the known decimal expansion to 15 significant figures."""
        assert_allclose(TWO_PI, 6.283185307179586, atol=1e-15)

    def test_positive(self):
        assert TWO_PI > 0.0

    def test_greater_than_six(self):
        assert TWO_PI > 6.0

    def test_less_than_seven(self):
        assert TWO_PI < 7.0


# ── TWO_PI trigonometric identities ────────────────────────────────────


class TestTwoPiTrigIdentities:
    """Verify TWO_PI satisfies fundamental trigonometric identities
    that the engine and order parameter code rely upon."""

    def test_sin_period(self):
        """sin(x + TWO_PI) = sin(x) for several test points."""
        for x in [0.0, 0.5, 1.0, np.pi, 4.5, -2.3]:
            assert_allclose(np.sin(x + TWO_PI), np.sin(x), atol=1e-14)

    def test_cos_period(self):
        """cos(x + TWO_PI) = cos(x)."""
        for x in [0.0, 0.5, 1.0, np.pi, 4.5, -2.3]:
            assert_allclose(np.cos(x + TWO_PI), np.cos(x), atol=1e-14)

    def test_exp_period(self):
        """exp(i·TWO_PI) = 1 (Euler's identity)."""
        z = np.exp(1j * TWO_PI)
        assert_allclose(z.real, 1.0, atol=1e-14)
        assert_allclose(z.imag, 0.0, atol=1e-14)

    def test_half_period_is_pi(self):
        assert_allclose(TWO_PI / 2, np.pi, atol=1e-15)

    def test_quarter_period(self):
        """sin(TWO_PI / 4) = 1 (quarter-turn on unit circle)."""
        assert_allclose(np.sin(TWO_PI / 4), 1.0, atol=1e-14)


# ── TWO_PI phase wrapping contract ─────────────────────────────────────


class TestTwoPiWrapping:
    """Phase wrapping x % TWO_PI must always produce values in [0, TWO_PI)."""

    def test_positive_angles(self):
        for x in [0.0, 1.0, 3.14, TWO_PI - 1e-15, TWO_PI, TWO_PI + 1.0, 100.0]:
            wrapped = x % TWO_PI
            assert 0.0 <= wrapped < TWO_PI

    def test_negative_angles(self):
        for x in [-0.001, -1.0, -np.pi, -TWO_PI, -100.0]:
            wrapped = x % TWO_PI
            assert 0.0 <= wrapped < TWO_PI

    def test_array_wrapping(self):
        """Vectorised wrapping on a numpy array."""
        angles = np.array([-10.0, -1.0, 0.0, 3.14, TWO_PI, 20.0])
        wrapped = angles % TWO_PI
        assert np.all(wrapped >= 0.0)
        assert np.all(wrapped < TWO_PI)

    def test_wrapping_preserves_order_parameter(self):
        """Wrapping phases should not change the order parameter R."""
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        phases_raw = np.array([0.1, TWO_PI + 0.1, -TWO_PI + 0.1])
        phases_wrapped = phases_raw % TWO_PI
        r_raw, _ = compute_order_parameter(phases_wrapped)
        assert_allclose(r_raw, 1.0, atol=1e-13)


# ── HAS_RUST contract ──────────────────────────────────────────────────


class TestHasRust:
    def test_is_bool_type(self):
        assert isinstance(HAS_RUST, bool)

    def test_consistent_with_importlib(self):
        """HAS_RUST must match what importlib says at runtime."""
        import importlib.util

        expected = importlib.util.find_spec("spo_kernel") is not None
        assert expected == HAS_RUST

    def test_module_level_constant(self):
        """HAS_RUST is a module-level constant, not a function."""
        assert not callable(HAS_RUST)


# ── __all__ exports ─────────────────────────────────────────────────────


class TestExports:
    def test_all_contains_two_pi(self):
        from scpn_phase_orchestrator import _compat

        assert "TWO_PI" in _compat.__all__

    def test_all_contains_has_rust(self):
        from scpn_phase_orchestrator import _compat

        assert "HAS_RUST" in _compat.__all__

    def test_all_is_complete(self):
        """__all__ should list exactly the public symbols."""
        from scpn_phase_orchestrator import _compat

        assert set(_compat.__all__) == {"TWO_PI", "HAS_RUST"}

    def test_no_private_symbols_exported(self):
        """No underscore-prefixed names in __all__."""
        from scpn_phase_orchestrator import _compat

        for name in _compat.__all__:
            assert not name.startswith("_")

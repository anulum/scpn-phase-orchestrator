# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for differentiable BOLD signal generator

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax", reason="JAX required")
jnp = pytest.importorskip("jax.numpy", reason="JAX required")

from scpn_phase_orchestrator.nn.bold import (
    balloon_windkessel_step,
    bold_from_neural,
    bold_signal,
)

N = 4
DT = 0.001  # 1ms neural timestep


@pytest.fixture()
def impulse():
    """Single impulse at t=0, then silence. 10s at 1kHz = 10000 samples."""
    T = 10000
    return jnp.zeros((T, N)).at[0, :].set(1.0)


@pytest.fixture()
def sustained():
    """Sustained activity for 2s, then silence for 8s."""
    T = 10000
    return jnp.zeros((T, N)).at[:2000, :].set(0.5)


class TestBalloonWindkesselStep:
    def test_output_shapes(self):
        s = jnp.zeros(N)
        f = jnp.ones(N)
        v = jnp.ones(N)
        q = jnp.ones(N)
        x = jnp.ones(N) * 0.5
        ns, nf, nv, nq = balloon_windkessel_step(s, f, v, q, x, DT)
        assert ns.shape == (N,)
        assert nf.shape == (N,)
        assert nv.shape == (N,)
        assert nq.shape == (N,)

    def test_resting_state_stable(self):
        """Zero neural input keeps hemodynamic variables near resting."""
        s = jnp.zeros(N)
        f = jnp.ones(N)
        v = jnp.ones(N)
        q = jnp.ones(N)
        x = jnp.zeros(N)
        for _ in range(100):
            s, f, v, q = balloon_windkessel_step(s, f, v, q, x, DT)
        assert jnp.allclose(f, 1.0, atol=0.01)
        assert jnp.allclose(v, 1.0, atol=0.01)

    def test_positive_input_increases_flow(self):
        """Neural input should increase blood flow."""
        s = jnp.zeros(N)
        f = jnp.ones(N)
        v = jnp.ones(N)
        q = jnp.ones(N)
        x = jnp.ones(N)
        for _ in range(100):
            s, f, v, q = balloon_windkessel_step(s, f, v, q, x, DT)
        assert jnp.all(f > 1.0)


class TestBOLDSignal:
    def test_resting_near_zero(self):
        """At resting state (v=1, q=1), BOLD should be near 0."""
        v = jnp.ones(N)
        q = jnp.ones(N)
        y = bold_signal(v, q)
        assert jnp.allclose(y, 0.0, atol=0.01)

    def test_output_shape(self):
        v = jnp.ones((10, N))
        q = jnp.ones((10, N))
        y = bold_signal(v, q)
        assert y.shape == (10, N)


class TestBOLDFromNeural:
    def test_output_shape(self, impulse):
        bold = bold_from_neural(impulse, DT, dt_bold=0.5)
        expected_T = 10000 // 500  # 10s at 2Hz = 20 samples
        assert bold.shape == (expected_T, N)

    def test_impulse_response_peaks(self, impulse):
        """BOLD response to impulse should peak around 4-6s."""
        bold = bold_from_neural(impulse, DT, dt_bold=0.5)
        # Peak should be in samples 8-12 (4-6s at 2Hz)
        peak_idx = jnp.argmax(jnp.abs(bold[:, 0]))
        peak_time = float(peak_idx) * 0.5
        assert 3.0 <= peak_time <= 8.0

    def test_zero_input_near_zero_bold(self):
        """No neural input should produce near-zero BOLD change."""
        neural = jnp.zeros((5000, N))
        bold = bold_from_neural(neural, DT, dt_bold=0.5)
        assert jnp.allclose(bold, 0.0, atol=0.01)

    def test_sustained_input_produces_response(self, sustained):
        bold = bold_from_neural(sustained, DT, dt_bold=0.5)
        assert jnp.max(jnp.abs(bold)) > 0.001

    def test_differentiable(self, impulse):
        """BOLD generator should be differentiable w.r.t. neural input."""

        def loss_fn(neural):
            bold = bold_from_neural(neural, DT, dt_bold=0.5)
            return jnp.sum(bold**2)

        grad = jax.grad(loss_fn)(impulse)
        assert grad.shape == impulse.shape
        assert jnp.isfinite(grad).all()

    def test_different_dt_bold(self, impulse):
        """Different TR should change output length."""
        bold_2hz = bold_from_neural(impulse, DT, dt_bold=0.5)
        bold_1hz = bold_from_neural(impulse, DT, dt_bold=1.0)
        assert bold_2hz.shape[0] == 2 * bold_1hz.shape[0]

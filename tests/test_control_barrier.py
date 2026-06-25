# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Verified neural CBF safety filter tests

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.actuation.control_barrier import (
    BarrierCertificate,
    ControlBarrierFilter,
    NeuralBarrier,
)


def _linear_barrier(threshold: float = 0.3) -> NeuralBarrier:
    """h(x) = x - threshold; safe set x >= threshold."""
    return NeuralBarrier(weights=(np.array([[1.0]]),), biases=(np.array([-threshold]),))


def _relu_net(seed: int = 0, in_dim: int = 2, hidden: int = 5) -> NeuralBarrier:
    rng = np.random.default_rng(seed)
    return NeuralBarrier(
        weights=(
            rng.standard_normal((hidden, in_dim)),
            rng.standard_normal((1, hidden)),
        ),
        biases=(rng.standard_normal(hidden), rng.standard_normal(1)),
    )


class TestNeuralBarrierValidation:
    @pytest.mark.parametrize(
        ("weights", "biases", "match"),
        [
            ((), (), "at least one layer"),
            ((np.zeros((2, 2)),), (np.zeros(2), np.zeros(1)), "same layer count"),
            ((np.zeros(3),), (np.zeros(3),), "must be 2-D"),
            ((np.zeros((2, 2)),), (np.zeros(3),), "bias shape"),
            ((np.array([[np.inf]]),), (np.zeros(1),), "finite"),
            (
                (np.zeros((4, 2)), np.zeros((1, 3))),
                (np.zeros(4), np.zeros(1)),
                "expects input",
            ),
            ((np.zeros((2, 2)),), (np.zeros(2),), "single scalar"),
        ],
    )
    def test_rejects_malformed_layers(
        self, weights: Any, biases: Any, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            NeuralBarrier(weights=weights, biases=biases)

    def test_input_dim(self) -> None:
        assert _relu_net(in_dim=3).input_dim == 3


class TestBarrierValue:
    def test_linear_value(self) -> None:
        b = _linear_barrier(0.3)
        assert b.value(np.array([0.5])) == pytest.approx(0.2)
        assert b.value(np.array([0.3])) == pytest.approx(0.0)
        assert b.value(np.array([0.1])) == pytest.approx(-0.2)

    def test_relu_clips_negative_hidden(self) -> None:
        # Hidden layer that would go negative is zeroed by ReLU.
        b = NeuralBarrier(
            weights=(np.array([[1.0]]), np.array([[1.0]])),
            biases=(np.array([-1.0]), np.array([0.0])),
        )
        assert b.value(np.array([0.5])) == pytest.approx(0.0)  # relu(0.5-1)=0
        assert b.value(np.array([2.0])) == pytest.approx(1.0)  # relu(2-1)=1

    @pytest.mark.parametrize(
        ("state", "match"),
        [
            (np.array([[0.0]]), "one-dimensional"),
            (np.array([0.0, 1.0]), "does not match"),
            (np.array([np.nan]), "finite"),
            (np.array([1.0j]), "real-valued"),
            (np.array([True]), "boolean"),
        ],
    )
    def test_value_rejects_bad_state(self, state: np.ndarray, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            _linear_barrier().value(state)


class TestBarrierGradient:
    def test_linear_gradient(self) -> None:
        np.testing.assert_allclose(_linear_barrier().gradient(np.array([0.7])), [1.0])

    @pytest.mark.parametrize("seed", [0, 1, 7])
    def test_gradient_matches_finite_difference(self, seed: int) -> None:
        net = _relu_net(seed)
        rng = np.random.default_rng(100 + seed)
        x = rng.standard_normal(2)
        grad = net.gradient(x)
        eps = 1e-6
        fd = np.zeros(2)
        for i in range(2):
            xp, xm = x.copy(), x.copy()
            xp[i] += eps
            xm[i] -= eps
            fd[i] = (net.value(xp) - net.value(xm)) / (2 * eps)
        np.testing.assert_allclose(grad, fd, atol=1e-4)


class TestIntervalBounds:
    def test_linear_bounds_exact(self) -> None:
        lo, hi = _linear_barrier(0.3).interval_bounds(np.array([0.4]), np.array([0.9]))
        assert lo == pytest.approx(0.1)
        assert hi == pytest.approx(0.6)

    @pytest.mark.parametrize("seed", [0, 2, 5])
    def test_ibp_is_sound_over_samples(self, seed: int) -> None:
        net = _relu_net(seed)
        lo = np.array([-1.0, -1.0])
        hi = np.array([1.0, 1.0])
        ib_lo, ib_hi = net.interval_bounds(lo, hi)
        rng = np.random.default_rng(seed)
        samples = rng.uniform(lo, hi, size=(4000, 2))
        values = np.array([net.value(s) for s in samples])
        assert ib_lo <= values.min() + 1e-9
        assert ib_hi >= values.max() - 1e-9

    def test_rejects_inverted_box(self) -> None:
        with pytest.raises(ValueError, match=">= lower"):
            _linear_barrier().interval_bounds(np.array([1.0]), np.array([0.0]))


class TestFilterValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"gamma": 0.0}, "gamma"),
            ({"gamma": 1.5}, "gamma"),
            ({"gamma": np.nan}, "gamma"),
            ({"gamma": "fast"}, "gamma"),
            ({"control_lo": 1.0, "control_hi": 0.0}, "control_lo"),
            ({"control_effect": np.array([1.0, 2.0])}, "control_effect dimension"),
        ],
    )
    def test_rejects_bad_config(self, kwargs: dict[str, Any], match: str) -> None:
        base = {
            "barrier": _linear_barrier(),
            "gamma": 0.5,
            "control_lo": 0.0,
            "control_hi": 1.0,
            "control_effect": np.array([1.0]),
        }
        base.update(kwargs)
        with pytest.raises(ValueError, match=match):
            ControlBarrierFilter(**base)


class TestFilter:
    def _filter(self, **kw: Any) -> ControlBarrierFilter:
        base = {
            "barrier": _linear_barrier(0.3),
            "gamma": 0.5,
            "control_lo": 0.0,
            "control_hi": 1.0,
            "control_effect": np.array([1.0]),
        }
        base.update(kw)
        return ControlBarrierFilter(**base)

    def test_safe_nominal_passes_through(self) -> None:
        # Deep in the safe set, no drift: nominal control is admitted unchanged.
        u, intervened = self._filter().filter(0.2, np.array([0.9]), np.array([0.0]))
        assert u == pytest.approx(0.2)
        assert intervened is False

    def test_unsafe_drift_forces_control(self) -> None:
        # x=0.35 (h=0.05), drift -0.1 -> needs u >= 0.075 (grad=1, gamma=0.5).
        u, intervened = self._filter().filter(0.0, np.array([0.35]), np.array([-0.1]))
        assert u == pytest.approx(0.075)
        assert intervened is True

    def test_nominal_clipped_to_bounds(self) -> None:
        u, _ = self._filter().filter(5.0, np.array([0.9]), np.array([0.0]))
        assert u == pytest.approx(1.0)

    def test_negative_control_effect_projects_down(self) -> None:
        # control_effect -1: increasing u lowers x. Safe set still x>=0.3.
        # drift +0.1 pushes x up (safe); a too-high nominal u would push down.
        filt = self._filter(control_effect=np.array([-1.0]), control_lo=-1.0)
        u, intervened = filt.filter(1.0, np.array([0.35]), np.array([0.0]))
        # need -u >= -gamma*h = -0.025 -> u <= 0.025
        assert u <= 0.025 + 1e-9
        assert intervened is True

    def test_rejects_wrong_drift_dimension(self) -> None:
        with pytest.raises(ValueError, match="drift dimension"):
            self._filter().filter(0.0, np.array([0.5]), np.array([0.1, 0.2]))

    def test_rejects_non_numeric_drift(self) -> None:
        with pytest.raises(ValueError, match="real float array"):
            self._filter().filter(0.0, np.array([0.5]), np.array(["x"], dtype=object))

    def test_rejects_non_real_nominal_control(self) -> None:
        with pytest.raises(ValueError, match="nominal_control"):
            self._filter().filter("go", np.array([0.5]), np.array([0.0]))

    def test_no_control_authority_returns_best_effort(self) -> None:
        # control_effect orthogonal-ish: grad·g = 0 -> no authority.
        filt = self._filter(control_effect=np.array([0.0]))
        u, _ = filt.filter(0.4, np.array([0.25]), np.array([-0.5]))
        assert filt.control_lo <= u <= filt.control_hi


class TestForwardInvarianceVerification:
    def _filter(self, **kw: Any) -> ControlBarrierFilter:
        base = {
            "barrier": _linear_barrier(0.3),
            "gamma": 0.5,
            "control_lo": 0.0,
            "control_hi": 1.0,
            "control_effect": np.array([1.0]),
        }
        base.update(kw)
        return ControlBarrierFilter(**base)

    def test_strong_control_certifies(self) -> None:
        filt = self._filter()
        cert = filt.verify_forward_invariance(
            np.array([0.0]),
            np.array([1.0]),
            np.array([-0.1]),
            np.array([0.1]),
            cells_per_axis=20,
            boundary_shell=0.25,
        )
        assert cert.verified is True
        assert cert.boundary_cells > 0
        assert cert.worst_margin >= 0.0
        assert cert.filter_digest == filt.filter_digest
        assert len(cert.verification_digest) == 64

    def test_weak_control_fails_soundly(self) -> None:
        # Tiny control authority cannot overcome a strong downward drift.
        cert = self._filter(control_hi=0.001).verify_forward_invariance(
            np.array([0.0]),
            np.array([1.0]),
            np.array([-0.5]),
            np.array([-0.5]),
            cells_per_axis=20,
            boundary_shell=0.25,
        )
        assert cert.verified is False
        assert cert.worst_margin < 0.0

    def test_box_entirely_inside_safe_set_has_no_boundary(self) -> None:
        cert = self._filter().verify_forward_invariance(
            np.array([2.0]),
            np.array([3.0]),
            np.array([-0.1]),
            np.array([0.1]),
            cells_per_axis=8,
            boundary_shell=0.25,
        )
        assert cert.boundary_cells == 0
        assert cert.verified is True
        assert cert.worst_margin == float("inf")

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"state_hi": np.array([-1.0])}, ">= state_lo"),
            ({"drift_lo": np.array([0.1, 0.2])}, "drift bounds"),
            ({"drift_hi": np.array([-1.0])}, "drift_hi"),
            ({"cells_per_axis": 0}, "cells_per_axis"),
            ({"cells_per_axis": True}, "cells_per_axis"),
            ({"boundary_shell": -1.0}, "boundary_shell"),
        ],
    )
    def test_rejects_bad_inputs(self, kwargs: dict[str, Any], match: str) -> None:
        base = {
            "state_lo": np.array([0.0]),
            "state_hi": np.array([1.0]),
            "drift_lo": np.array([-0.1]),
            "drift_hi": np.array([0.1]),
        }
        base.update(kwargs)
        with pytest.raises(ValueError, match=match):
            self._filter().verify_forward_invariance(**base)

    def test_certificate_to_dict(self) -> None:
        filt = self._filter()
        cert = filt.verify_forward_invariance(
            np.array([0.0]), np.array([1.0]), np.array([-0.1]), np.array([0.1])
        )
        data = cert.to_dict()
        assert set(data) == {
            "verified",
            "cells_checked",
            "boundary_cells",
            "worst_margin",
            "boundary_shell",
            "gamma",
            "filter_digest",
            "verification_digest",
        }
        assert isinstance(cert, BarrierCertificate)

    def test_filter_digest_changes_with_safety_envelope(self) -> None:
        assert self._filter().filter_digest != self._filter(gamma=1.0).filter_digest

    def test_validate_certificate_rejects_wrong_filter(self) -> None:
        cert = self._filter(gamma=1.0).verify_forward_invariance(
            np.array([0.0]), np.array([1.0]), np.array([-0.1]), np.array([0.1])
        )
        with pytest.raises(ValueError, match="does not match"):
            self._filter().validate_certificate(cert)


class TestPipelineWiring:
    def test_coherence_floor_barrier_on_engine_order_parameter(self) -> None:
        """A coherence-floor CBF filters a coupling action off a real R(t)."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        # Barrier h(R) = R - 0.2: keep coherence above 0.2.
        barrier = _linear_barrier(0.2)
        # Coupling raises coherence: dR/dK > 0, model sensitivity ~0.5 per unit.
        filt = ControlBarrierFilter(
            barrier=barrier,
            gamma=0.5,
            control_lo=0.0,
            control_hi=2.0,
            control_effect=np.array([0.5]),
        )
        n = 12
        eng = UPDEEngine(n, dt=0.02)
        rng = np.random.default_rng(3)
        phases = rng.uniform(0.0, 2.0 * np.pi, n)
        omegas = rng.normal(0.0, 0.8, n)
        knm = 0.2 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        for _ in range(40):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        r, _psi = compute_order_parameter(phases)
        # Filter a "reduce coupling" proposal against the coherence floor.
        safe_u, _ = filt.filter(-0.5, np.array([r]), np.array([0.0]))
        assert filt.control_lo <= safe_u <= filt.control_hi
        assert np.isfinite(safe_u)

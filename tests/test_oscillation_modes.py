# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Inter-area oscillation mode estimation tests

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor.oscillation_modes import (
    DEFAULT_DAMPING_THRESHOLD,
    OscillationMode,
    estimate_oscillation_modes,
)

_FS = 50.0
_N = 500


def _damped(
    fd: float, zeta: float, amp: float, phase: float, n: int = _N
) -> np.ndarray:
    """A single damped sinusoid sampled at _FS: damped freq fd, damping ratio zeta."""
    t = np.arange(n) / _FS
    wd = 2.0 * np.pi * fd
    wn = wd / np.sqrt(1.0 - zeta**2)
    sigma = zeta * wn
    return amp * np.exp(-sigma * t) * np.cos(wd * t + phase)


class TestOscillationMode:
    def test_to_dict(self) -> None:
        mode = OscillationMode(
            frequency_hz=0.5,
            damping_ratio=0.04,
            amplitude=1.0,
            phase_rad=0.1,
            poorly_damped=False,
        )
        data = mode.to_dict()
        assert data == {
            "frequency_hz": 0.5,
            "damping_ratio": 0.04,
            "amplitude": 1.0,
            "phase_rad": 0.1,
            "poorly_damped": False,
        }


class TestCleanRecovery:
    def test_recovers_two_modes_exactly(self) -> None:
        signal = _damped(0.5, 0.05, 1.0, 0.3) + _damped(1.2, 0.10, 0.6, -0.7)
        modes = estimate_oscillation_modes(signal, _FS)
        assert len(modes) == 2
        first, second = modes  # sorted by descending amplitude
        assert first.frequency_hz == pytest.approx(0.5, abs=1e-3)
        assert first.damping_ratio == pytest.approx(0.05, abs=1e-3)
        assert first.amplitude == pytest.approx(1.0, abs=1e-3)
        assert first.phase_rad == pytest.approx(0.3, abs=1e-2)
        assert second.frequency_hz == pytest.approx(1.2, abs=1e-3)
        assert second.damping_ratio == pytest.approx(0.10, abs=1e-3)

    def test_single_mode(self) -> None:
        modes = estimate_oscillation_modes(_damped(0.8, 0.07, 1.0, 0.0), _FS)
        assert modes[0].frequency_hz == pytest.approx(0.8, abs=1e-3)
        assert modes[0].damping_ratio == pytest.approx(0.07, abs=1e-3)

    def test_pure_decay_has_zero_frequency_full_damping(self) -> None:
        t = np.arange(_N) / _FS
        signal = 2.0 * np.exp(-1.5 * t)
        modes = estimate_oscillation_modes(signal, _FS)
        assert modes[0].frequency_hz == pytest.approx(0.0, abs=1e-6)
        assert modes[0].damping_ratio == pytest.approx(1.0, abs=1e-9)

    def test_constant_signal_is_dc_mode(self) -> None:
        # A constant fits a degenerate DC pole (z ~ 1): zero frequency, finite
        # (near-marginal) damping. It is not a real oscillation mode.
        modes = estimate_oscillation_modes(np.full(_N, 3.0), _FS)
        assert modes[0].frequency_hz == pytest.approx(0.0, abs=1e-6)
        assert np.isfinite(modes[0].damping_ratio)

    def test_all_zero_signal_returns_no_modes(self) -> None:
        assert estimate_oscillation_modes(np.zeros(_N), _FS) == ()


class TestStability:
    def test_poorly_damped_mode_flagged(self) -> None:
        # 2% damping is below the 3% screening threshold.
        modes = estimate_oscillation_modes(_damped(0.6, 0.02, 1.0, 0.0), _FS)
        assert modes[0].damping_ratio == pytest.approx(0.02, abs=1e-3)
        assert modes[0].poorly_damped is True

    def test_unstable_growing_mode_has_negative_damping(self) -> None:
        t = np.arange(_N) / _FS
        signal = np.exp(0.05 * t) * np.cos(2.0 * np.pi * 0.5 * t)
        modes = estimate_oscillation_modes(signal, _FS)
        assert modes[0].damping_ratio < 0.0
        assert modes[0].poorly_damped is True

    def test_custom_threshold(self) -> None:
        signal = _damped(0.5, 0.05, 1.0, 0.0)
        relaxed = estimate_oscillation_modes(signal, _FS, damping_threshold=0.01)
        strict = estimate_oscillation_modes(signal, _FS, damping_threshold=0.08)
        assert relaxed[0].poorly_damped is False
        assert strict[0].poorly_damped is True


class TestNoiseRobustness:
    def test_recovers_modes_under_noise(self) -> None:
        signal = _damped(0.5, 0.05, 1.0, 0.3) + _damped(1.2, 0.10, 0.6, -0.7)
        rng = np.random.default_rng(0)
        noisy = signal + rng.normal(0.0, 0.01, signal.shape[0])
        modes = estimate_oscillation_modes(noisy, _FS, model_order=4)
        freqs = sorted(m.frequency_hz for m in modes if m.amplitude > 0.1)
        assert freqs[0] == pytest.approx(0.5, abs=2e-2)
        assert freqs[1] == pytest.approx(1.2, abs=2e-2)


class TestParameters:
    def test_explicit_model_order_caps(self) -> None:
        signal = _damped(0.5, 0.05, 1.0, 0.0)
        modes = estimate_oscillation_modes(signal, _FS, model_order=2)
        assert len(modes) >= 1

    def test_model_order_one_gives_real_pole(self) -> None:
        # A real Hankel keeps conjugate pairs together; order=1 yields one real
        # (zero-frequency) pole rather than a lone complex one.
        signal = _damped(0.5, 0.05, 1.0, 0.0)
        modes = estimate_oscillation_modes(signal, _FS, model_order=1)
        assert len(modes) == 1
        assert modes[0].frequency_hz == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("factor", [0.25, 0.33, 0.5])
    def test_pencil_factor_variation(self, factor: float) -> None:
        signal = _damped(0.7, 0.06, 1.0, 0.0)
        modes = estimate_oscillation_modes(signal, _FS, pencil_factor=factor)
        assert modes[0].frequency_hz == pytest.approx(0.7, abs=1e-2)

    def test_energy_floor_prunes_spurious_modes(self) -> None:
        signal = _damped(0.5, 0.05, 1.0, 0.0) + _damped(1.5, 0.08, 0.5, 0.0)
        many = estimate_oscillation_modes(signal, _FS, model_order=8, energy_floor=0.0)
        pruned = estimate_oscillation_modes(
            signal, _FS, model_order=8, energy_floor=0.2
        )
        assert len(pruned) <= len(many)
        assert len(pruned) == 2

    def test_modes_sorted_by_descending_amplitude(self) -> None:
        signal = _damped(0.5, 0.05, 0.4, 0.0) + _damped(1.5, 0.08, 1.0, 0.0)
        modes = estimate_oscillation_modes(signal, _FS)
        amps = [m.amplitude for m in modes]
        assert amps == sorted(amps, reverse=True)


class TestValidation:
    @pytest.mark.parametrize(
        ("signal", "match"),
        [
            (np.ones((4, 2)), "one-dimensional"),
            (np.ones(4) + 1j, "real-valued"),
            (np.array([True, False, True, False]), "boolean"),
            (np.array([1.0, np.nan, 2.0, 3.0]), "finite"),
            (np.ones(3), "at least 4"),
            (np.array(["a", "b", "c", "d"], dtype=object), "real float array"),
        ],
    )
    def test_rejects_bad_signal(self, signal: Any, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            estimate_oscillation_modes(signal, _FS)

    @pytest.mark.parametrize(
        ("fs", "match"),
        [
            (0.0, "fs must be positive"),
            (-5.0, "fs must be positive"),
            (np.nan, "fs must be finite"),
            (True, "fs must be a finite real"),
        ],
    )
    def test_rejects_bad_fs(self, fs: Any, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            estimate_oscillation_modes(_damped(0.5, 0.05, 1.0, 0.0), fs)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"pencil_factor": 0.0}, "pencil_factor must lie"),
            ({"pencil_factor": 1.0}, "pencil_factor must lie"),
            ({"pencil_factor": True}, "pencil_factor must be a finite real"),
            ({"model_order": 0}, "model_order must be a positive"),
            ({"model_order": 2.5}, "model_order must be a positive"),
            ({"model_order": True}, "model_order must be a positive"),
            ({"damping_threshold": True}, "damping_threshold must be a finite real"),
            ({"energy_floor": -0.1}, "energy_floor must be non-negative"),
        ],
    )
    def test_rejects_bad_parameters(self, kwargs: dict[str, Any], match: str) -> None:
        signal = _damped(0.5, 0.05, 1.0, 0.0)
        with pytest.raises(ValueError, match=match):
            estimate_oscillation_modes(signal, _FS, **kwargs)

    def test_default_threshold_constant(self) -> None:
        assert DEFAULT_DAMPING_THRESHOLD == 0.03


class TestPipelineWiring:
    def test_estimates_modes_from_engine_ringdown(self) -> None:
        """A perturbed Kuramoto network rings down; estimate its dominant mode."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        n = 16
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(5)
        phases = 0.2 * rng.standard_normal(n)  # near-synchronised
        omegas = 0.3 * rng.standard_normal(n)
        knm = 3.0 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        trace = []
        for _ in range(600):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
            r, _psi = compute_order_parameter(phases)
            trace.append(r)
        ringdown = np.asarray(trace) - float(np.mean(trace[-100:]))
        modes = estimate_oscillation_modes(ringdown, 1.0 / 0.01, model_order=6)
        assert modes  # at least one mode recovered
        assert all(np.isfinite(m.damping_ratio) for m in modes)
        assert all(m.frequency_hz >= 0.0 for m in modes)

# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — sc-neurocore bridge tests

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from scpn_phase_orchestrator._compat import HAS_RUST
from scpn_phase_orchestrator.adapters.neurocore_bridge import (
    HAS_NEUROCORE,
    NeurocoreBridge,
)
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState


def _make_state(r_values: list[float]) -> UPDEState:
    layers = [LayerState(R=r, psi=0.0) for r in r_values]
    return UPDEState(
        layers=layers,
        cross_layer_alignment=np.eye(len(layers)),
        stability_proxy=float(np.mean(r_values)),
        regime_id="nominal",
    )


class TestNeurocoreBridge:
    def test_init(self):
        bridge = NeurocoreBridge(n_layers=4, neurons_per_layer=4)
        assert bridge._n_total == 16

    def test_backend_selection(self):
        bridge = NeurocoreBridge(n_layers=2, neurons_per_layer=4)
        if HAS_RUST:
            assert bridge.backend == "rust"
        else:
            assert bridge.backend == "numpy"

    def test_step_returns_rates(self):
        bridge = NeurocoreBridge(n_layers=3, neurons_per_layer=4)
        state = _make_state([0.9, 0.5, 0.1])
        rates = bridge.step(state, n_substeps=50)
        assert rates.shape == (3,)
        assert np.all(rates >= 0.0)

    def test_high_coherence_higher_rate(self):
        bridge = NeurocoreBridge(
            n_layers=2,
            neurons_per_layer=8,
            current_scale=3.0,
        )
        state = _make_state([0.95, 0.1])
        rates = bridge.step(state, n_substeps=100)
        assert rates[0] >= rates[1]

    def test_rates_to_actions(self):
        bridge = NeurocoreBridge(n_layers=2)
        actions = bridge.rates_to_actions(np.array([100.0, 10.0]))
        assert len(actions) == 1
        assert actions[0].scope == "layer_0"
        assert actions[0].knob == "K"

    def test_step_and_act(self):
        bridge = NeurocoreBridge(n_layers=3, neurons_per_layer=4)
        state = _make_state([0.9, 0.9, 0.9])
        actions = bridge.step_and_act(state, n_substeps=50)
        assert isinstance(actions, list)

    def test_get_neuron_states(self):
        bridge = NeurocoreBridge(n_layers=2, neurons_per_layer=2)
        states = bridge.get_neuron_states()
        assert len(states) == 4
        assert "v" in states[0]

    def test_reset(self):
        bridge = NeurocoreBridge(n_layers=2, neurons_per_layer=2)
        state = _make_state([0.9, 0.9])
        bridge.step(state, n_substeps=10)
        bridge.reset()
        assert bridge._step_count == 0 or bridge.backend == "rust"
        assert np.all(bridge._spike_counts == 0) or bridge.backend == "rust"

    def test_invalid_backend(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            NeurocoreBridge(n_layers=2, backend="nonexistent")


class TestNeurocoreBridgeNumpy:
    """Force numpy backend to verify Python fallback."""

    def test_numpy_step(self):
        bridge = NeurocoreBridge(n_layers=3, neurons_per_layer=8, backend="numpy")
        assert bridge.backend == "numpy"
        state = _make_state([0.9, 0.5, 0.1])
        rates = bridge.step(state, n_substeps=50)
        assert rates.shape == (3,)
        assert np.all(rates >= 0.0)

    def test_numpy_coherence_ordering(self):
        bridge = NeurocoreBridge(
            n_layers=2, neurons_per_layer=100, current_scale=3.0, backend="numpy"
        )
        state = _make_state([0.95, 0.1])
        rates = bridge.step(state, n_substeps=100)
        assert rates[0] >= rates[1]

    def test_numpy_get_neuron_states(self):
        bridge = NeurocoreBridge(n_layers=2, neurons_per_layer=4, backend="numpy")
        states = bridge.get_neuron_states()
        assert len(states) == 8
        assert "v" in states[0]
        assert "refractory" in states[0]

    def test_numpy_reset(self):
        bridge = NeurocoreBridge(n_layers=2, neurons_per_layer=4, backend="numpy")
        state = _make_state([0.9, 0.9])
        bridge.step(state, n_substeps=10)
        bridge.reset()
        assert bridge._step_count == 0
        assert np.all(bridge._spike_counts == 0)


@pytest.mark.skipif(not HAS_RUST, reason="spo_kernel not installed")
class TestNeurocoreBridgeRust:
    """Rust backend tests."""

    def test_rust_step(self):
        bridge = NeurocoreBridge(n_layers=3, neurons_per_layer=8, backend="rust")
        assert bridge.backend == "rust"
        state = _make_state([0.9, 0.5, 0.1])
        rates = bridge.step(state, n_substeps=50)
        assert rates.shape == (3,)
        assert np.all(rates >= 0.0)

    def test_rust_coherence_ordering(self):
        bridge = NeurocoreBridge(
            n_layers=2, neurons_per_layer=100, current_scale=3.0, backend="rust"
        )
        state = _make_state([0.95, 0.1])
        rates = bridge.step(state, n_substeps=100)
        assert rates[0] >= rates[1]

    def test_rust_get_neuron_states(self):
        bridge = NeurocoreBridge(n_layers=2, neurons_per_layer=4, backend="rust")
        states = bridge.get_neuron_states()
        assert len(states) == 8
        assert "v" in states[0]

    def test_rust_reset(self):
        bridge = NeurocoreBridge(n_layers=2, neurons_per_layer=4, backend="rust")
        state = _make_state([0.9, 0.9])
        bridge.step(state, n_substeps=10)
        bridge.reset()
        # Rust manages its own state — verify step after reset
        rates = bridge.step(state, n_substeps=10)
        assert rates.shape == (2,)


class TestNeurocoreBridgeScale:
    """Scale validation: SPO supervisor controls sc-neurocore populations at N=1000+."""

    def test_scale_1000_neurons(self):
        bridge = NeurocoreBridge(n_layers=10, neurons_per_layer=100)
        assert bridge._n_total == 1000
        state = _make_state([0.8] * 10)
        rates = bridge.step(state, n_substeps=20)
        assert rates.shape == (10,)
        assert np.all(rates >= 0.0)

    def test_scale_sustained_stepping(self):
        bridge = NeurocoreBridge(n_layers=10, neurons_per_layer=100, current_scale=2.5)
        state = _make_state([0.9, 0.7, 0.5, 0.3, 0.1, 0.8, 0.6, 0.4, 0.2, 0.95])
        for _ in range(5):
            rates = bridge.step(state, n_substeps=100)
        assert rates.shape == (10,)
        assert rates[0] > rates[4]

    def test_scale_actions_at_scale(self):
        bridge = NeurocoreBridge(n_layers=10, neurons_per_layer=100, current_scale=3.0)
        state = _make_state([0.95] * 10)
        actions = bridge.step_and_act(state, n_substeps=200)
        assert isinstance(actions, list)
        for a in actions:
            assert a.knob == "K"
            assert a.value > 0

    def test_scale_5000_neurons(self):
        bridge = NeurocoreBridge(n_layers=10, neurons_per_layer=500, current_scale=2.5)
        assert bridge._n_total == 5000
        state = _make_state([0.9, 0.7, 0.5, 0.3, 0.1, 0.8, 0.6, 0.4, 0.2, 0.95])
        rates = bridge.step(state, n_substeps=100)
        assert rates.shape == (10,)
        assert rates[0] > rates[4]

    def test_scale_10000_neurons(self):
        bridge = NeurocoreBridge(n_layers=10, neurons_per_layer=1000, current_scale=2.5)
        assert bridge._n_total == 10000
        state = _make_state([0.9, 0.7, 0.5, 0.3, 0.1, 0.8, 0.6, 0.4, 0.2, 0.95])
        t0 = time.perf_counter()
        rates = bridge.step(state, n_substeps=100)
        elapsed = time.perf_counter() - t0
        assert rates.shape == (10,)
        assert rates[0] > rates[4]
        # Rust should complete in <100ms, numpy in <2s
        if bridge.backend == "rust":
            assert elapsed < 0.5, f"Rust N=10000 took {elapsed:.3f}s (expected <0.5s)"
        else:
            assert elapsed < 5.0, f"Numpy N=10000 took {elapsed:.3f}s (expected <5s)"


class TestNeurocoreBridgeScaleTiming:
    """Benchmark timing across backends and neuron counts."""

    @pytest.mark.parametrize(
        "n_per,label",
        [(100, "1k"), (500, "5k"), (1000, "10k")],
    )
    def test_timing_default_backend(self, n_per, label):
        bridge = NeurocoreBridge(
            n_layers=10,
            neurons_per_layer=n_per,
            current_scale=2.5,
        )
        state = _make_state(list(np.linspace(0.1, 0.95, 10)))
        t0 = time.perf_counter()
        rates = bridge.step(state, n_substeps=100)
        elapsed = time.perf_counter() - t0
        n_total = 10 * n_per
        ns_per = (elapsed / (n_total * 100)) * 1e9
        # Just verify it completes and produces valid output
        assert rates.shape == (10,)
        assert np.all(rates >= 0.0)
        print(
            f"\n  [{bridge.backend}] N={n_total}: "
            f"{elapsed:.4f}s, {ns_per:.0f} ns/neuron/substep",
        )

    @pytest.mark.parametrize("backend", ["numpy"])
    def test_timing_numpy_10k(self, backend):
        bridge = NeurocoreBridge(
            n_layers=10, neurons_per_layer=1000, current_scale=2.5, backend=backend
        )
        state = _make_state(list(np.linspace(0.1, 0.95, 10)))
        t0 = time.perf_counter()
        rates = bridge.step(state, n_substeps=100)
        elapsed = time.perf_counter() - t0
        assert rates.shape == (10,)
        print(f"\n  [numpy] N=10000: {elapsed:.4f}s")

    @pytest.mark.skipif(not HAS_RUST, reason="spo_kernel not installed")
    def test_timing_rust_10k(self):
        bridge = NeurocoreBridge(
            n_layers=10, neurons_per_layer=1000, current_scale=2.5, backend="rust"
        )
        state = _make_state(list(np.linspace(0.1, 0.95, 10)))
        t0 = time.perf_counter()
        rates = bridge.step(state, n_substeps=100)
        elapsed = time.perf_counter() - t0
        assert rates.shape == (10,)
        # Shared CI runners (especially macOS) have high variance — use 500ms
        # budget on CI, tight 100ms locally.
        budget = 0.5 if os.environ.get("CI") else 0.1
        assert elapsed < budget, (
            f"Rust N=10000 took {elapsed:.3f}s (expected <{budget * 1000:.0f}ms)"
        )
        print(f"\n  [rust] N=10000: {elapsed:.4f}s")


@pytest.mark.skipif(not HAS_NEUROCORE, reason="sc-neurocore not installed")
class TestNeurocoreBridgeScalar:
    """Scalar backend cross-validation against sc-neurocore."""

    def test_scalar_step(self):
        bridge = NeurocoreBridge(n_layers=2, neurons_per_layer=4, backend="scalar")
        assert bridge.backend == "scalar"
        state = _make_state([0.9, 0.1])
        rates = bridge.step(state, n_substeps=50)
        assert rates.shape == (2,)

    def test_scalar_coherence_ordering(self):
        bridge = NeurocoreBridge(
            n_layers=2, neurons_per_layer=8, current_scale=3.0, backend="scalar"
        )
        state = _make_state([0.95, 0.1])
        rates = bridge.step(state, n_substeps=100)
        assert rates[0] >= rates[1]


def test_has_neurocore_flag():
    assert isinstance(HAS_NEUROCORE, bool)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Runtime SimulationState Kuramoto contracts

"""Runtime SimulationState contracts for Kuramoto stepping and bounded global order."""

from __future__ import annotations

import pytest

from scpn_phase_orchestrator.binding.types import (
    ActuatorMapping,
    BindingSpec,
    CouplingSpec,
    DriverSpec,
    HierarchyLayer,
    ObjectivePartition,
    OscillatorFamily,
)


class TestSimulationStateKuramotoStep:
    """Verify that SimulationState in non-amplitude mode (pure Kuramoto)
    produces physically correct phase evolution."""

    @pytest.fixture()
    def sim(self):
        from scpn_phase_orchestrator.runtime.server import SimulationState

        spec = BindingSpec(
            name="test",
            version="1.0.0",
            safety_tier="research",
            sample_period_s=0.01,
            control_period_s=0.1,
            layers=[HierarchyLayer(name="L1", index=0, oscillator_ids=["o1", "o2"])],
            oscillator_families={
                "phys": OscillatorFamily(
                    channel="P",
                    extractor_type="hilbert",
                    config={},
                ),
            },
            coupling=CouplingSpec(base_strength=0.45, decay_alpha=0.3, templates={}),
            drivers=DriverSpec(physical={}, informational={}, symbolic={}),
            objectives=ObjectivePartition(good_layers=[0], bad_layers=[]),
            boundaries=[],
            actuators=[
                ActuatorMapping(
                    name="K_g", knob="K", scope="global", limits=(0.0, 1.0)
                ),
            ],
        )
        return SimulationState(spec)

    def test_non_amplitude_mode_flag(self, sim):
        """Non-amplitude spec must produce Kuramoto mode, not Stuart-Landau."""
        assert not sim.amplitude_mode

    def test_step_returns_valid_state(self, sim):
        """Single step must return step counter and R_global in [0, 1]."""
        result = sim.step()
        assert result["step"] == 1
        assert "R_global" in result
        assert 0.0 <= result["R_global"] <= 1.0

    def test_multi_step_advances_phase(self, sim):
        """10 consecutive steps must advance the state — R should vary,
        step counter must increment monotonically."""
        results = [sim.step() for _ in range(10)]
        steps = [r["step"] for r in results]
        assert steps == list(range(1, 11)), "Step counter must increment by 1 each call"
        r_values = [r["R_global"] for r in results]
        # With coupling, R should not be exactly constant (phases evolve)
        assert not all(r == r_values[0] for r in r_values), (
            "R_global should vary across steps under coupling"
        )

    def test_step_r_bounded(self, sim):
        """R must stay in [0, 1] across 50 integration steps."""
        for _ in range(50):
            result = sim.step()
            r = result["R_global"]
            s = result["step"]
            assert 0.0 <= r <= 1.0, f"R={r:.4f} at step {s}"

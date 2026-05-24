# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Initial phase channel-resolution contracts

"""Channel-resolution contracts for oscillator initial phase generation."""

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.binding.types import (
    BindingSpec,
    CouplingSpec,
    DriverSpec,
    HierarchyLayer,
    ObjectivePartition,
    OscillatorFamily,
)


class TestInitPhasesChannelResolution:
    """Verify that initial phase extraction correctly resolves channels
    and falls back to uniform random for unknown channels."""

    def _make_spec(self, channel, family_name="test_fam"):
        return BindingSpec(
            name="test",
            version="1.0.0",
            safety_tier="research",
            sample_period_s=0.01,
            control_period_s=0.1,
            layers=[
                HierarchyLayer(
                    name="L1",
                    index=0,
                    oscillator_ids=["o1", "o2"],
                    family=family_name,
                )
            ],
            oscillator_families={
                family_name: OscillatorFamily(
                    channel=channel,
                    extractor_type="hilbert",
                    config={},
                ),
            },
            coupling=CouplingSpec(base_strength=0.45, decay_alpha=0.3, templates={}),
            drivers=DriverSpec(physical={}, informational={}, symbolic={}),
            objectives=ObjectivePartition(good_layers=[0], bad_layers=[]),
            boundaries=[],
            actuators=[],
        )

    def test_unknown_channel_falls_back_to_uniform(self):
        """Channel 'X' (not P/I/S) must produce phases in [0, 2π)."""
        from scpn_phase_orchestrator.oscillators.init_phases import (
            extract_initial_phases,
        )

        spec = self._make_spec("X")
        phases = extract_initial_phases(spec, np.array([1.0, 1.0]))
        assert phases.shape == (2,)
        assert np.all(phases >= 0.0) and np.all(phases < 2 * np.pi), (
            f"Phases must be in [0, 2π), got {phases}"
        )

    def test_physical_channel_produces_valid_phases(self):
        """Channel 'P' must produce phases in [0, 2π)."""
        from scpn_phase_orchestrator.oscillators.init_phases import (
            extract_initial_phases,
        )

        spec = self._make_spec("P")
        phases = extract_initial_phases(spec, np.array([5.0, 5.0]))
        assert phases.shape == (2,)
        assert np.all(phases >= 0.0) and np.all(phases < 2 * np.pi)

    def test_resolve_channel_empty_families_defaults_to_P(self):
        """With no families defined, channel resolution must default to 'P'."""
        from scpn_phase_orchestrator.oscillators.init_phases import _resolve_channel

        assert _resolve_channel(None, {}, 0) == "P"
        assert _resolve_channel(None, {}, 5) == "P"

    def test_get_n_states_no_symbolic_family_defaults_to_4(self):
        """Without any symbolic family, n_states must default to 4."""
        from scpn_phase_orchestrator.oscillators.init_phases import _get_n_states

        families = {
            "phys": OscillatorFamily(channel="P", extractor_type="hilbert", config={}),
        }
        assert _get_n_states(families) == 4

    def test_get_n_states_from_symbolic_config(self):
        """Symbolic family with explicit n_states must use that value."""
        from scpn_phase_orchestrator.oscillators.init_phases import _get_n_states

        families = {
            "sym": OscillatorFamily(
                channel="S",
                extractor_type="symbolic",
                config={"n_states": 8},
            ),
        }
        assert _get_n_states(families) == 8

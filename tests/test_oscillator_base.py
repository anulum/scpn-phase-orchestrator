# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Oscillator base class tests: PhaseState, PhaseExtractor

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.oscillators.base import PhaseExtractor, PhaseState

# ── PhaseState construction and field access ────────────────────────────


class TestPhaseStateConstruction:
    def test_basic_construction(self):
        ps = PhaseState(
            theta=1.5, omega=2.0, amplitude=0.8, quality=0.9, channel="P", node_id="n1"
        )
        assert ps.theta == 1.5
        assert ps.omega == 2.0
        assert ps.amplitude == 0.8
        assert ps.quality == 0.9
        assert ps.channel == "P"
        assert ps.node_id == "n1"

    def test_zero_values(self):
        ps = PhaseState(
            theta=0.0, omega=0.0, amplitude=0.0, quality=0.0, channel="I", node_id="z"
        )
        assert ps.theta == 0.0
        assert ps.omega == 0.0
        assert ps.amplitude == 0.0
        assert ps.quality == 0.0

    def test_boundary_theta_zero(self):
        """θ = 0 is a valid phase at the lower boundary."""
        ps = PhaseState(
            theta=0.0, omega=1.0, amplitude=1.0, quality=1.0, channel="P", node_id="b0"
        )
        assert ps.theta == 0.0

    def test_boundary_theta_near_two_pi(self):
        """θ just below 2π is valid."""
        ps = PhaseState(
            theta=2 * np.pi - 1e-10,
            omega=1.0,
            amplitude=1.0,
            quality=1.0,
            channel="P",
            node_id="b2pi",
        )
        assert ps.theta < 2 * np.pi

    def test_negative_omega(self):
        """Negative frequencies are physically valid (backward rotation)."""
        ps = PhaseState(
            theta=1.0,
            omega=-3.5,
            amplitude=1.0,
            quality=0.5,
            channel="S",
            node_id="neg",
        )
        assert ps.omega == -3.5

    def test_all_channels(self):
        """P, I, and S channels should all be accepted."""
        for ch in ("P", "I", "S"):
            ps = PhaseState(
                theta=0.0,
                omega=0.0,
                amplitude=0.0,
                quality=0.0,
                channel=ch,
                node_id="x",
            )
            assert ps.channel == ch


# ── PhaseState as dataclass ─────────────────────────────────────────────


class TestPhaseStateDataclass:
    def test_is_dataclass(self):
        """PhaseState should be a dataclass."""
        import dataclasses

        assert dataclasses.is_dataclass(PhaseState)

    def test_equality(self):
        """Two PhaseState with identical fields should be equal."""
        a = PhaseState(
            theta=1.0, omega=2.0, amplitude=3.0, quality=0.5, channel="P", node_id="x"
        )
        b = PhaseState(
            theta=1.0, omega=2.0, amplitude=3.0, quality=0.5, channel="P", node_id="x"
        )
        assert a == b

    def test_inequality_different_theta(self):
        a = PhaseState(
            theta=1.0, omega=2.0, amplitude=3.0, quality=0.5, channel="P", node_id="x"
        )
        b = PhaseState(
            theta=1.1, omega=2.0, amplitude=3.0, quality=0.5, channel="P", node_id="x"
        )
        assert a != b

    def test_inequality_different_node_id(self):
        a = PhaseState(
            theta=1.0, omega=2.0, amplitude=3.0, quality=0.5, channel="P", node_id="x"
        )
        b = PhaseState(
            theta=1.0, omega=2.0, amplitude=3.0, quality=0.5, channel="P", node_id="y"
        )
        assert a != b

    def test_fields_mutable(self):
        """PhaseState is a regular (non-frozen) dataclass — fields are mutable."""
        ps = PhaseState(
            theta=0.0, omega=0.0, amplitude=0.0, quality=0.0, channel="P", node_id="x"
        )
        ps.theta = 3.14
        assert ps.theta == 3.14

    def test_repr_contains_fields(self):
        ps = PhaseState(
            theta=1.0, omega=2.0, amplitude=3.0, quality=0.5, channel="P", node_id="n1"
        )
        r = repr(ps)
        assert "theta=1.0" in r
        assert "node_id='n1'" in r


# ── PhaseState in collections ──────────────────────────────────────────


class TestPhaseStateCollections:
    def test_list_of_phase_states(self):
        states = [
            PhaseState(
                theta=i * 0.5,
                omega=1.0,
                amplitude=1.0,
                quality=0.9,
                channel="P",
                node_id=f"n{i}",
            )
            for i in range(10)
        ]
        assert len(states) == 10
        assert states[5].theta == pytest.approx(2.5)

    def test_sorting_by_theta(self):
        """PhaseStates can be sorted by theta for analysis."""
        states = [
            PhaseState(
                theta=3.0,
                omega=0.0,
                amplitude=0.0,
                quality=0.0,
                channel="P",
                node_id="a",
            ),
            PhaseState(
                theta=1.0,
                omega=0.0,
                amplitude=0.0,
                quality=0.0,
                channel="P",
                node_id="b",
            ),
            PhaseState(
                theta=2.0,
                omega=0.0,
                amplitude=0.0,
                quality=0.0,
                channel="P",
                node_id="c",
            ),
        ]
        sorted_states = sorted(states, key=lambda s: s.theta)
        assert [s.node_id for s in sorted_states] == ["b", "c", "a"]


# ── PhaseExtractor abstract contract ───────────────────────────────────


class TestPhaseExtractorAbstract:
    def test_cannot_instantiate(self):
        """PhaseExtractor is abstract — direct instantiation should fail."""
        with pytest.raises(TypeError):
            PhaseExtractor()

    def test_must_implement_extract(self):
        """A subclass missing extract() cannot be instantiated."""

        class Incomplete(PhaseExtractor):
            def quality_score(self, phase_states):
                return 0.0

        with pytest.raises(TypeError):
            Incomplete()

    def test_must_implement_quality_score(self):
        """A subclass missing quality_score() cannot be instantiated."""

        class Incomplete(PhaseExtractor):
            def extract(self, signal, sample_rate):
                return []

        with pytest.raises(TypeError):
            Incomplete()

    def test_complete_subclass_works(self):
        """A complete subclass can be instantiated and called."""

        class MockExtractor(PhaseExtractor):
            def extract(self, signal, sample_rate):
                return [
                    PhaseState(
                        theta=0.0,
                        omega=sample_rate,
                        amplitude=float(np.max(np.abs(signal))),
                        quality=0.9,
                        channel="P",
                        node_id="mock",
                    )
                ]

            def quality_score(self, phase_states):
                if not phase_states:
                    return 0.0
                return sum(s.quality for s in phase_states) / len(phase_states)

        ext = MockExtractor()
        signal = np.sin(np.linspace(0, 2 * np.pi, 100))
        states = ext.extract(signal, 1000.0)
        assert len(states) == 1
        assert states[0].omega == 1000.0
        score = ext.quality_score(states)
        assert score == pytest.approx(0.9)

    def test_quality_score_empty_list(self):
        """quality_score with empty list should not raise."""

        class SafeExtractor(PhaseExtractor):
            def extract(self, signal, sample_rate):
                return []

            def quality_score(self, phase_states):
                if not phase_states:
                    return 0.0
                return sum(s.quality for s in phase_states) / len(phase_states)

        ext = SafeExtractor()
        assert ext.quality_score([]) == 0.0


# ── PhaseExtractor interface contract ──────────────────────────────────


class TestPhaseExtractorContract:
    """Verify that the abstract interface defines the correct method signatures."""

    def test_extract_signature(self):
        """extract() has expected signature."""
        import inspect

        sig = inspect.signature(PhaseExtractor.extract)
        params = list(sig.parameters.keys())
        assert "signal" in params
        assert "sample_rate" in params

    def test_quality_score_signature(self):
        """quality_score() accepts (phase_states: list[PhaseState]) → float."""
        import inspect

        sig = inspect.signature(PhaseExtractor.quality_score)
        params = list(sig.parameters.keys())
        assert "phase_states" in params

    def test_extract_is_abstract(self):
        """extract must be marked @abstractmethod."""
        assert getattr(PhaseExtractor.extract, "__isabstractmethod__", False)

    def test_quality_score_is_abstract(self):
        """quality_score must be marked @abstractmethod."""
        assert getattr(PhaseExtractor.quality_score, "__isabstractmethod__", False)


# ── PhaseState edge values ─────────────────────────────────────────────


class TestPhaseStateEdgeValues:
    def test_very_large_omega(self):
        ps = PhaseState(
            theta=0.0,
            omega=1e12,
            amplitude=1.0,
            quality=1.0,
            channel="P",
            node_id="fast",
        )
        assert ps.omega == 1e12

    def test_very_small_amplitude(self):
        ps = PhaseState(
            theta=0.0,
            omega=1.0,
            amplitude=1e-15,
            quality=1.0,
            channel="P",
            node_id="weak",
        )
        assert ps.amplitude == 1e-15

    def test_quality_boundary_zero(self):
        ps = PhaseState(
            theta=0.0, omega=0.0, amplitude=0.0, quality=0.0, channel="P", node_id="q0"
        )
        assert ps.quality == 0.0

    def test_quality_boundary_one(self):
        ps = PhaseState(
            theta=0.0, omega=0.0, amplitude=0.0, quality=1.0, channel="P", node_id="q1"
        )
        assert ps.quality == 1.0

    def test_empty_node_id(self):
        """Empty string is technically valid for node_id."""
        ps = PhaseState(
            theta=0.0, omega=0.0, amplitude=0.0, quality=0.0, channel="P", node_id=""
        )
        assert ps.node_id == ""

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for binding-channel initial phase extraction

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.binding.types import (
    ActuatorMapping,
    BindingSpec,
    BoundaryDef,
    CouplingSpec,
    DriverSpec,
    HierarchyLayer,
    ObjectivePartition,
    OscillatorFamily,
)
from scpn_phase_orchestrator.oscillators.init_phases import (
    _get_n_states,
    _resolve_channel,
    extract_initial_phases,
)

TWO_PI = 2.0 * np.pi


def _make_pis_spec() -> BindingSpec:
    """BindingSpec with all three channel types (P, I, S)."""
    layers = [
        HierarchyLayer(name="L1", index=0, oscillator_ids=["osc0", "osc1", "osc2"]),
    ]
    families = {
        "phys": OscillatorFamily(channel="P", extractor_type="hilbert", config={}),
        "info": OscillatorFamily(channel="I", extractor_type="event", config={}),
        "symb": OscillatorFamily(
            channel="S", extractor_type="ring", config={"n_states": 5}
        ),
    }
    coupling = CouplingSpec(base_strength=0.45, decay_alpha=0.3, templates={})
    drivers = DriverSpec(physical={"freq": 10.0}, informational={}, symbolic={})
    objectives = ObjectivePartition(good_layers=[0], bad_layers=[])
    boundaries = [
        BoundaryDef(
            name="R_floor", variable="R", lower=0.2, upper=None, severity="hard"
        ),
    ]
    actuators = [
        ActuatorMapping(name="K_global", knob="K", scope="global", limits=(0.0, 1.0)),
    ]
    return BindingSpec(
        name="test-pis",
        version="1.0.0",
        safety_tier="research",
        sample_period_s=0.01,
        control_period_s=0.1,
        layers=layers,
        oscillator_families=families,
        coupling=coupling,
        drivers=drivers,
        objectives=objectives,
        boundaries=boundaries,
        actuators=actuators,
    )


def test_output_shape_matches_omegas():
    spec = _make_pis_spec()
    omegas = np.array([1.0, 2.0, 3.0])
    phases = extract_initial_phases(spec, omegas, seed=42)
    assert phases.shape == (3,)


def test_phases_in_valid_range():
    spec = _make_pis_spec()
    omegas = np.array([1.0, 2.0, 3.0])
    phases = extract_initial_phases(spec, omegas, seed=42)
    assert np.all(phases >= 0.0)
    assert np.all(phases < TWO_PI)


def test_deterministic_with_same_seed():
    spec = _make_pis_spec()
    omegas = np.array([1.0, 2.0, 3.0])
    a = extract_initial_phases(spec, omegas, seed=99)
    b = extract_initial_phases(spec, omegas, seed=99)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_differ():
    spec = _make_pis_spec()
    omegas = np.array([1.0, 2.0, 3.0])
    a = extract_initial_phases(spec, omegas, seed=1)
    b = extract_initial_phases(spec, omegas, seed=2)
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("seed", [True, -1, 1.5, "42"])
def test_extract_initial_phases_rejects_invalid_seed(seed: object):
    spec = _make_pis_spec()
    omegas = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="seed must be a non-negative integer"):
        extract_initial_phases(spec, omegas, seed=seed)


def test_extract_initial_phases_accepts_numpy_integer_seed():
    spec = _make_pis_spec()
    omegas = np.array([1.0, 2.0, 3.0])
    phases = extract_initial_phases(spec, omegas, seed=np.int64(42))
    assert phases.shape == (3,)


@pytest.mark.parametrize(
    "omegas",
    [
        np.array([1.0, float("nan"), 3.0]),
        np.array([1.0, float("inf"), 3.0]),
        np.array([True, False, True]),
        np.array([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]),
        np.array(["1.0", "2.0", "3.0"], dtype=object),
    ],
)
def test_extract_initial_phases_rejects_invalid_omegas(omegas: object):
    spec = _make_pis_spec()
    with pytest.raises(ValueError, match="omegas must be finite"):
        extract_initial_phases(spec, omegas)


@pytest.mark.parametrize(
    "omegas",
    [
        np.array([1.0, 2.0]),
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([[1.0, 2.0, 3.0]]),
    ],
)
def test_extract_initial_phases_rejects_omega_shape_mismatch(omegas: object):
    spec = _make_pis_spec()
    with pytest.raises(ValueError, match="omegas length"):
        extract_initial_phases(spec, omegas)


def test_named_channels_route_by_extractor_semantics():
    layers = [
        HierarchyLayer(
            name="L1",
            index=0,
            oscillator_ids=["osc0", "osc1", "osc2", "osc3"],
        ),
    ]
    families = {
        "fast": OscillatorFamily(channel="Q", extractor_type="hilbert", config={}),
        "events": OscillatorFamily(channel="C", extractor_type="event", config={}),
        "modes": OscillatorFamily(
            channel="M", extractor_type="ring", config={"n_states": 7}
        ),
        "edge": OscillatorFamily(
            channel="edge-node",
            extractor_type="graph",
            config={},
        ),
    }
    spec = BindingSpec(
        name="test-n-channel",
        version="1.0.0",
        safety_tier="research",
        sample_period_s=0.01,
        control_period_s=0.1,
        layers=layers,
        oscillator_families=families,
        coupling=CouplingSpec(base_strength=0.45, decay_alpha=0.3, templates={}),
        drivers=DriverSpec(
            physical={},
            informational={},
            symbolic={},
            extra={"Q": {"zeta": 0.1}, "C": {"zeta": 0.2}},
        ),
        objectives=ObjectivePartition(good_layers=[0], bad_layers=[]),
        boundaries=[],
        actuators=[],
    )
    phases = extract_initial_phases(spec, np.array([1.0, 2.0, 3.0, 4.0]), seed=42)

    assert phases.shape == (4,)
    assert np.all(phases >= 0.0)
    assert np.all(phases < TWO_PI)


def test_get_n_states_from_symbolic_config():
    families = {
        "sym": OscillatorFamily(
            channel="S",
            extractor_type="ring",
            config={"n_states": 8},
        ),
    }
    assert _get_n_states(families) == 8


def test_get_n_states_no_symbolic_family_defaults_to_4():
    families = {
        "phys": OscillatorFamily(channel="P", extractor_type="hilbert", config={}),
    }
    assert _get_n_states(families) == 4


@pytest.mark.parametrize("n_states", [True, 4.0, "4", 1])
def test_get_n_states_rejects_invalid_symbolic_config(n_states: object):
    families = {
        "sym": OscillatorFamily(
            channel="S",
            extractor_type="ring",
            config={"n_states": n_states},
        ),
    }
    with pytest.raises(ValueError, match="n_states"):
        _get_n_states(families)


def test_unknown_channel_and_extractor_falls_back_to_random_phase():
    layers = [
        HierarchyLayer(
            name="L1",
            index=0,
            oscillator_ids=["osc0"],
            family="unknown",
        ),
    ]
    families = {
        "unknown": OscillatorFamily(
            channel="X",
            extractor_type="bespoke",
            config={},
        ),
    }
    spec = BindingSpec(
        name="test-random-fallback",
        version="1.0.0",
        safety_tier="research",
        sample_period_s=0.01,
        control_period_s=0.1,
        layers=layers,
        oscillator_families=families,
        coupling=CouplingSpec(base_strength=0.45, decay_alpha=0.3, templates={}),
        drivers=DriverSpec(physical={}, informational={}, symbolic={}),
        objectives=ObjectivePartition(good_layers=[0], bad_layers=[]),
        boundaries=[],
        actuators=[],
    )

    phases = extract_initial_phases(spec, np.array([1.0]), seed=7)

    assert phases.shape == (1,)
    assert 0.0 <= phases[0] < TWO_PI


def test_resolve_channel_uses_explicit_family_binding():
    families = {
        "events": OscillatorFamily(channel="I", extractor_type="event", config={}),
    }

    assert _resolve_channel("events", families, osc_idx=0) == "I"


def test_symbolic_path_is_batched_into_single_extract_call(
    monkeypatch: pytest.MonkeyPatch,
):
    class _BatchProbeSymbolicExtractor:
        calls: list[np.ndarray] = []

        def __init__(self, n_states: int, node_id: str):
            self.n_states = n_states
            self.node_id = node_id

        def extract(self, signal: np.ndarray, sample_rate: float):
            _BatchProbeSymbolicExtractor.calls.append(
                np.asarray(signal, dtype=np.int64)
            )
            return [
                type(
                    "State",
                    (),
                    {
                        "theta": float(
                            (int(idx) % self.n_states) * (2.0 * np.pi / self.n_states)
                        )
                    },
                )()
                for idx in np.asarray(signal, dtype=np.int64)
            ]

    layers = [
        HierarchyLayer(
            name="Lsym",
            index=0,
            oscillator_ids=["s0", "s1", "s2", "s3"],
            family="symb",
        ),
    ]
    families = {
        "symb": OscillatorFamily(
            channel="S",
            extractor_type="ring",
            config={"n_states": 7},
        ),
    }
    spec = BindingSpec(
        name="test-symbolic-batch",
        version="1.0.0",
        safety_tier="research",
        sample_period_s=0.01,
        control_period_s=0.1,
        layers=layers,
        oscillator_families=families,
        coupling=CouplingSpec(base_strength=0.45, decay_alpha=0.3, templates={}),
        drivers=DriverSpec(physical={}, informational={}, symbolic={}),
        objectives=ObjectivePartition(good_layers=[0], bad_layers=[]),
        boundaries=[],
        actuators=[],
    )
    monkeypatch.setattr(
        "scpn_phase_orchestrator.oscillators.init_phases.SymbolicExtractor",
        _BatchProbeSymbolicExtractor,
    )

    phases = extract_initial_phases(spec, np.array([1.0, 2.0, 3.0, 4.0]), seed=42)
    assert phases.shape == (4,)
    assert len(_BatchProbeSymbolicExtractor.calls) == 1
    assert _BatchProbeSymbolicExtractor.calls[0].shape == (4,)


class TestInitPhasesPipelineWiring:
    """Pipeline: extract_initial_phases → engine simulation."""

    def test_extracted_phases_drive_engine(self):
        """extract_initial_phases → UPDEEngine: proves extraction
        output is valid engine input."""
        from scpn_phase_orchestrator.coupling import CouplingBuilder
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import (
            compute_order_parameter,
        )

        spec = _make_pis_spec()
        n = 3
        omegas = np.array([1.0, 2.0, 3.0])
        phases = extract_initial_phases(spec, omegas, seed=42)
        assert phases.shape == (n,)

        cs = CouplingBuilder().build(n, 0.5, 0.3)
        eng = UPDEEngine(n, dt=0.01)
        for _ in range(100):
            phases = eng.step(
                phases,
                omegas,
                cs.knm,
                0.0,
                0.0,
                cs.alpha,
            )
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0

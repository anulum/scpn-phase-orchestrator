# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Event and symbolic signal ingress

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.binding.loader import load_binding_spec
from scpn_phase_orchestrator.oscillators.factory import build_extractor
from scpn_phase_orchestrator.oscillators.init_phases import extract_initial_phases


@pytest.mark.parametrize("algorithm", ["event", "ring", "graph"])
@pytest.mark.parametrize(
    "dtype", ["timedelta64[ms]", "timedelta64[ns]", "datetime64[ms]", "U", "S"]
)
def test_discrete_factory_rejects_temporal_and_text_arrays(
    algorithm: str, dtype: str
) -> None:
    signal = np.array([0, 1, 2, 3]).astype(dtype)
    with pytest.raises(ValueError, match="signal"):
        build_extractor(algorithm, n_states=4).extract(signal, 1.0)


@pytest.mark.parametrize("algorithm", ["event", "ring", "graph"])
def test_discrete_factory_rejects_boolean_sequence_promotion(algorithm: str) -> None:
    signal = cast("NDArray[np.float64]", [0, True, 2, 3])
    with pytest.raises(ValueError, match="signal"):
        build_extractor(algorithm, n_states=4).extract(signal, 1.0)


def test_event_seconds_and_ring_indices_preserve_expected_frequencies() -> None:
    events = np.arange(4).astype(np.float64) / 10.0
    event = build_extractor("event").extract(events, 0.0)[0]
    ring = build_extractor("ring", n_states=4).extract(
        np.arange(4).astype(np.float64).astype("int64"), 1.0
    )[-1]
    assert event.omega == pytest.approx(20 * np.pi)
    assert ring.theta == pytest.approx(3 * np.pi / 2)
    assert ring.omega == pytest.approx(np.pi / 2)


@pytest.mark.parametrize("dtype", ["timedelta64[ms]", "timedelta64[ns]"])
def test_binding_initialisation_rejects_duration_frequencies(dtype: str) -> None:
    spec = load_binding_spec("domainpacks/minimal_domain/binding_spec.yaml")
    count = sum(len(layer.oscillator_ids) for layer in spec.layers)
    omegas = np.ones(count).astype(dtype)
    with pytest.raises(ValueError, match="omegas"):
        extract_initial_phases(spec, omegas, seed=42)


def test_binding_initialisation_preserves_seeded_numeric_phases() -> None:
    spec = load_binding_spec("domainpacks/minimal_domain/binding_spec.yaml")
    count = sum(len(layer.oscillator_ids) for layer in spec.layers)
    omegas = np.arange(1, count + 1).astype(np.float64)
    phases = extract_initial_phases(spec, omegas, seed=42)
    repeated = extract_initial_phases(spec, omegas, seed=42)
    np.testing.assert_array_equal(phases, repeated)
    assert phases.shape == (count,)
    assert np.all(np.isfinite(phases))
    assert np.all((phases >= 0) & (phases < 2 * np.pi))

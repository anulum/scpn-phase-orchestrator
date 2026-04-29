# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Binding-channel phase initialization

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.binding.types import BindingSpec, OscillatorFamily
from scpn_phase_orchestrator.oscillators.informational import InformationalExtractor
from scpn_phase_orchestrator.oscillators.physical import PhysicalExtractor
from scpn_phase_orchestrator.oscillators.symbolic import SymbolicExtractor

__all__ = ["extract_initial_phases"]

TWO_PI = 2.0 * np.pi
_PHYSICAL_EXTRACTORS = frozenset({"hilbert", "wavelet", "zero_crossing"})
_INFORMATIONAL_EXTRACTORS = frozenset({"event"})
_SYMBOLIC_EXTRACTORS = frozenset({"ring", "graph"})


def extract_initial_phases(
    spec: BindingSpec,
    omegas: NDArray,
    seed: int = 42,
) -> NDArray:
    """Extract initial phases from channels defined in binding_spec.

    For each oscillator, generates a synthetic signal matching the family
    channel or extractor semantics and extracts the phase. Falls back to
    random phase if extraction fails.

    Returns (n_osc,) array of initial phases in [0, 2*pi).
    """
    rng = np.random.default_rng(seed)
    n_osc = len(omegas)
    phases = np.zeros(n_osc)
    t = np.linspace(0, 1.0, 256)

    families = spec.oscillator_families
    osc_idx = 0
    for layer in spec.layers:
        for osc_id in layer.oscillator_ids:
            omega = omegas[osc_idx]
            family = _resolve_family(layer.family, families, osc_idx)
            channel = family.channel if family is not None else "P"
            extractor_type = family.extractor_type if family is not None else "hilbert"

            if channel == "P" or extractor_type in _PHYSICAL_EXTRACTORS:
                signal = np.sin(omega * TWO_PI * t) + rng.normal(0, 0.1, len(t))
                p_ext = PhysicalExtractor(node_id=osc_id)
                states = p_ext.extract(signal, sample_rate=256.0)
                phases[osc_idx] = states[-1].theta if states else rng.uniform(0, TWO_PI)

            elif channel == "I" or extractor_type in _INFORMATIONAL_EXTRACTORS:
                n_events = max(3, int(omega * 10))
                timestamps = np.sort(rng.uniform(0, 1.0, n_events))
                i_ext = InformationalExtractor(node_id=osc_id)
                states = i_ext.extract(timestamps, sample_rate=1.0)
                phases[osc_idx] = states[-1].theta if states else rng.uniform(0, TWO_PI)

            elif channel == "S" or extractor_type in _SYMBOLIC_EXTRACTORS:
                n_states = _get_n_states(families)
                state_idx = rng.integers(0, n_states)
                s_ext = SymbolicExtractor(n_states=n_states, node_id=osc_id)
                states = s_ext.extract(np.array([state_idx]), sample_rate=1.0)
                phases[osc_idx] = states[0].theta if states else rng.uniform(0, TWO_PI)

            else:
                phases[osc_idx] = rng.uniform(0, TWO_PI)

            osc_idx += 1

    return phases


def _resolve_channel(
    layer_family: str | None,
    families: dict[str, OscillatorFamily],
    osc_idx: int,
) -> str:
    """Resolve channel from explicit family binding or round-robin fallback."""
    family = _resolve_family(layer_family, families, osc_idx)
    if family is not None:
        return family.channel
    return "P"


def _resolve_family(
    layer_family: str | None,
    families: dict[str, OscillatorFamily],
    osc_idx: int,
) -> OscillatorFamily | None:
    """Resolve family from explicit layer binding or round-robin fallback."""
    if layer_family is not None and layer_family in families:
        return families[layer_family]
    ordered = [families[key] for key in sorted(families)]
    if not ordered:
        return None
    return ordered[osc_idx % len(ordered)]


def _get_n_states(families: dict[str, OscillatorFamily]) -> int:
    """Find n_states from the first symbolic-family extractor."""
    for fam in families.values():
        if fam.channel == "S" or fam.extractor_type in _SYMBOLIC_EXTRACTORS:
            cfg = fam.config or {}
            return int(cfg.get("n_states", 4))
    return 4

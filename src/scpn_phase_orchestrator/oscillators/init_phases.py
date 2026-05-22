# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Binding-channel phase initialization

"""Synthetic binding-aware initial phase generation.

`extract_initial_phases` uses the binding's oscillator families to generate
small deterministic synthetic signals per P/I/S channel, extract their phases,
and produce a finite initial phase vector for UPDE startup. It validates omega
length, seed, symbolic state counts, and extractor families before falling back
to seeded random phases for unsupported channel families.
"""

from __future__ import annotations

from numbers import Integral
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.binding.types import BindingSpec, OscillatorFamily
from scpn_phase_orchestrator.oscillators.informational import InformationalExtractor
from scpn_phase_orchestrator.oscillators.physical import PhysicalExtractor
from scpn_phase_orchestrator.oscillators.symbolic import SymbolicExtractor

__all__ = ["extract_initial_phases"]

FloatArray: TypeAlias = NDArray[np.float64]

TWO_PI = 2.0 * np.pi
_PHYSICAL_EXTRACTORS = frozenset({"hilbert", "wavelet", "zero_crossing"})
_INFORMATIONAL_EXTRACTORS = frozenset({"event"})
_SYMBOLIC_EXTRACTORS = frozenset({"ring", "graph"})


def _validate_n_states(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("n_states must be an integer >= 2")
    n_states = int(value)
    if n_states < 2:
        raise ValueError(f"n_states must be >= 2, got {n_states}")
    return n_states


def _oscillator_count(spec: BindingSpec) -> int:
    return sum(len(layer.oscillator_ids) for layer in spec.layers)


def _validate_omegas(value: object, *, expected_count: int) -> FloatArray:
    omegas = np.asarray(value)
    dtype = omegas.dtype
    if (
        np.issubdtype(dtype, np.bool_)
        or np.issubdtype(dtype, np.complexfloating)
        or not np.issubdtype(dtype, np.number)
    ):
        raise ValueError("omegas must be finite")
    if omegas.ndim != 1 or len(omegas) != expected_count:
        raise ValueError(
            f"omegas length must match oscillator count {expected_count}, "
            f"got shape {omegas.shape}"
        )
    parsed = omegas.astype(np.float64, copy=False)
    if not np.all(np.isfinite(parsed)):
        raise ValueError("omegas must be finite")
    return parsed


def _validate_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("seed must be a non-negative integer")
    seed = int(value)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def extract_initial_phases(
    spec: BindingSpec,
    omegas: FloatArray,
    seed: int = 42,
) -> FloatArray:
    """Extract initial phases from channels defined in binding_spec.

    For each oscillator, generates a synthetic signal matching the family
    channel or extractor semantics and extracts the phase. Falls back to
    random phase if extraction fails.

    Returns (n_osc,) array of initial phases in [0, 2*pi).
    """
    omegas = _validate_omegas(omegas, expected_count=_oscillator_count(spec))
    seed = _validate_seed(seed)
    rng = np.random.default_rng(seed)
    n_osc = len(omegas)
    phases = np.zeros(n_osc)
    t = np.linspace(0, 1.0, 256)

    families = spec.oscillator_families
    physical_extractor = PhysicalExtractor(node_id="init_physical")
    informational_extractor = InformationalExtractor(node_id="init_informational")
    symbolic_n_states = _get_n_states(families)
    symbolic_extractor = SymbolicExtractor(
        n_states=symbolic_n_states,
        node_id="init_symbolic",
    )
    symbolic_pending: list[tuple[int, int]] = []

    osc_idx = 0
    for layer in spec.layers:
        for _osc_id in layer.oscillator_ids:
            omega = omegas[osc_idx]
            family = _resolve_family(layer.family, families, osc_idx)
            channel = family.channel if family is not None else "P"
            extractor_type = family.extractor_type if family is not None else "hilbert"

            if channel == "P" or extractor_type in _PHYSICAL_EXTRACTORS:
                signal = np.sin(omega * TWO_PI * t) + rng.normal(0, 0.1, len(t))
                states = physical_extractor.extract(signal, sample_rate=256.0)
                phases[osc_idx] = states[-1].theta if states else rng.uniform(0, TWO_PI)

            elif channel == "I" or extractor_type in _INFORMATIONAL_EXTRACTORS:
                n_events = max(3, int(omega * 10))
                timestamps = np.sort(rng.uniform(0, 1.0, n_events))
                states = informational_extractor.extract(timestamps, sample_rate=1.0)
                phases[osc_idx] = states[-1].theta if states else rng.uniform(0, TWO_PI)

            elif channel == "S" or extractor_type in _SYMBOLIC_EXTRACTORS:
                state_idx = int(rng.integers(0, symbolic_n_states))
                symbolic_pending.append((osc_idx, state_idx))

            else:
                phases[osc_idx] = rng.uniform(0, TWO_PI)

            osc_idx += 1

    if symbolic_pending:
        symbolic_indices = np.array(
            [state_idx for _, state_idx in symbolic_pending],
            dtype=np.float64,
        )
        symbolic_states = symbolic_extractor.extract(symbolic_indices, sample_rate=1.0)
        for (phase_idx, _), state in zip(
            symbolic_pending, symbolic_states, strict=True
        ):
            phases[phase_idx] = state.theta

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
            return _validate_n_states(cfg.get("n_states", 4))
    return 4

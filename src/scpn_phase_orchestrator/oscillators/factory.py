# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Extractor construction

"""Construct the phase extractor named by a binding ``extractor_type``.

`build_extractor` maps a domainpack ``extractor_type`` (a channel alias such as
``physical``/``informational``/``symbolic`` or a canonical algorithm name such as
``hilbert``/``wavelet``/``zero_crossing``/``event``/``ring``/``graph``) to the
concrete `PhaseExtractor` that implements it. Aliases are resolved through
`resolve_extractor_type`; an unknown type raises ``ValueError`` (fail-closed)
rather than silently degrading to a default algorithm.
"""

from __future__ import annotations

from scpn_phase_orchestrator.binding.types import resolve_extractor_type
from scpn_phase_orchestrator.oscillators.base import PhaseExtractor
from scpn_phase_orchestrator.oscillators.informational import InformationalExtractor
from scpn_phase_orchestrator.oscillators.physical import PhysicalExtractor
from scpn_phase_orchestrator.oscillators.symbolic import SymbolicExtractor
from scpn_phase_orchestrator.oscillators.wavelet import WaveletExtractor
from scpn_phase_orchestrator.oscillators.zero_crossing import ZeroCrossingExtractor

__all__ = ["build_extractor"]


def build_extractor(
    extractor_type: str,
    *,
    node_id: str = "extractor",
    n_states: int = 2,
) -> PhaseExtractor:
    """Build the `PhaseExtractor` for a binding ``extractor_type``.

    Parameters
    ----------
    extractor_type : str
        A channel alias (``physical``/``informational``/``symbolic``) or a
        canonical algorithm name (``hilbert``/``wavelet``/``zero_crossing``/
        ``event``/``ring``/``graph``).
    node_id : str
        Identifier stamped onto the extractor's emitted `PhaseState` records.
    n_states : int
        Number of discrete states for symbolic (``ring``/``graph``) extractors;
        ignored by the continuous and event extractors.

    Returns
    -------
    PhaseExtractor
        The extractor implementing ``extractor_type``.

    Raises
    ------
    ValueError
        If ``extractor_type`` does not resolve to a known algorithm.
    """
    algorithm = resolve_extractor_type(extractor_type)
    if algorithm == "hilbert":
        return PhysicalExtractor(node_id=node_id)
    if algorithm == "wavelet":
        return WaveletExtractor(node_id=node_id)
    if algorithm == "zero_crossing":
        return ZeroCrossingExtractor(node_id=node_id)
    if algorithm == "event":
        return InformationalExtractor(node_id=node_id)
    if algorithm in ("ring", "graph"):
        return SymbolicExtractor(n_states=n_states, node_id=node_id, mode=algorithm)
    raise ValueError(
        f"unknown extractor_type {extractor_type!r} (resolved {algorithm!r}); "
        "expected one of hilbert, wavelet, zero_crossing, event, ring, graph "
        "or an alias physical/informational/symbolic"
    )

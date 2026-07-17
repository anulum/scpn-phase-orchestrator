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

from collections.abc import Mapping
from typing import Any

from scpn_phase_orchestrator.binding.types import resolve_extractor_type
from scpn_phase_orchestrator.oscillators.base import PhaseExtractor
from scpn_phase_orchestrator.oscillators.informational import InformationalExtractor
from scpn_phase_orchestrator.oscillators.physical import PhysicalExtractor
from scpn_phase_orchestrator.oscillators.symbolic import SymbolicExtractor
from scpn_phase_orchestrator.oscillators.wavelet import WaveletExtractor
from scpn_phase_orchestrator.oscillators.zero_crossing import ZeroCrossingExtractor

__all__ = ["build_extractor"]

_PHYSICAL_CONFIG_KEYS = ("band", "filter_order", "edge_trim")


def _physical_kwargs(config: Mapping[str, object] | None) -> dict[str, Any]:
    """Return the `PhysicalExtractor` keyword arguments named in ``config``.

    Only the recognised band-pass keys are forwarded; the extractor validates the
    values (and rejects malformed ones) so a binding's ``config`` cannot silently
    misconfigure the filter. Values are ``Any`` — they originate from arbitrary
    binding YAML/JSON and are narrowed by the extractor's own runtime validators.
    """
    if not config:
        return {}
    return {key: config[key] for key in _PHYSICAL_CONFIG_KEYS if key in config}


def build_extractor(
    extractor_type: str,
    *,
    node_id: str = "extractor",
    n_states: int = 2,
    config: Mapping[str, object] | None = None,
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
    config : Mapping[str, object] | None
        Optional oscillator-family ``config`` from the binding spec. For the
        physical/``hilbert`` extractor its ``band``/``filter_order``/``edge_trim``
        keys select the opt-in zero-phase band-pass and edge-trim; other extractors
        ignore it.

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
        return PhysicalExtractor(node_id=node_id, **_physical_kwargs(config))
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

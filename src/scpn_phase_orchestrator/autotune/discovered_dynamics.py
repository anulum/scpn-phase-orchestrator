# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — operator-facing discovered-dynamics record

"""Operator-facing record of dynamics discovered by phase-SINDy.

Where :mod:`scpn_phase_orchestrator.autotune.discovery` emits the raw evidence
blocks and :mod:`scpn_phase_orchestrator.autotune.sindy_confidence` judges how
far to trust a fit, this module presents the result the way an operator reads
it: the recovered equations, the per-node coupling edges, and — inseparably —
the honest confidence verdict that says how much weight the structure carries.

The recovered equations are never shown without their posture. A skipped or
weak fit still produces a record, but its confidence marks it ``refused`` or
``insufficient_evidence`` so the equations cannot be mistaken for a validated
model. Every record carries a canonical-JSON SHA-256 content hash for a
tamper-evident provenance trail.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash
from scpn_phase_orchestrator.autotune.sindy_confidence import (
    DEFAULT_SINDY_CONFIDENCE_POLICY,
    SindyConfidence,
    SindyConfidencePolicy,
    classify_phase_sindy_block,
)


@dataclass(frozen=True)
class DiscoveredDynamics:
    """A discovered phase-dynamics model paired with its honest confidence.

    Parameters
    ----------
    library : str
        The feature library the fit used, e.g. the Kuramoto sine-difference
        library.
    status : str
        The fit status from the discovery block (``"fitted"`` or a skip
        reason).
    equations : tuple of str
        Human-readable recovered equations, one per node; empty when no fit was
        performed.
    coupling_edges : tuple of Mapping
        Per-node coupling edges (``source``, ``target``, ``coefficient``,
        ``abs_coefficient``); empty when no fit was performed.
    confidence : SindyConfidence
        The honest tier and discovery posture for the fit.
    """

    library: str
    status: str
    equations: tuple[str, ...]
    coupling_edges: tuple[Mapping[str, Any], ...]
    confidence: SindyConfidence

    def _canonical_payload(self) -> dict[str, Any]:
        """Return the hash-bearing record content, excluding the hash itself."""
        return {
            "library": self.library,
            "status": self.status,
            "equations": list(self.equations),
            "coupling_edges": [dict(edge) for edge in self.coupling_edges],
            "confidence": self.confidence.to_audit_record(),
        }

    @property
    def content_hash(self) -> str:
        """Canonical-JSON SHA-256 digest of the record content.

        Returns
        -------
        str
            Lowercase hexadecimal SHA-256 digest of the canonical payload.
        """
        return canonical_record_hash(self._canonical_payload())

    def to_audit_record(self) -> dict[str, Any]:
        """Return the complete JSON-safe record, including the content hash.

        Returns
        -------
        dict
            The canonical payload with the ``content_hash`` provenance field
            appended.
        """
        record = self._canonical_payload()
        record["content_hash"] = self.content_hash
        return record


def discovered_dynamics_from_block(
    block: Mapping[str, Any],
    *,
    policy: SindyConfidencePolicy = DEFAULT_SINDY_CONFIDENCE_POLICY,
) -> DiscoveredDynamics:
    """Build an operator-facing record from a phase-SINDy evidence block.

    Parameters
    ----------
    block : Mapping
        A ``phase_sindy`` evidence block as emitted by the discovery report.
    policy : SindyConfidencePolicy, optional
        Thresholds separating a credible discovery from weak evidence.

    Returns
    -------
    DiscoveredDynamics
        The recovered equations and coupling edges paired with the honest
        confidence verdict.
    """
    confidence = classify_phase_sindy_block(block, policy=policy)
    equations = tuple(str(equation) for equation in block.get("equations", ()))
    coupling_edges = tuple(dict(edge) for edge in block.get("coupling_edges", ()))
    return DiscoveredDynamics(
        library=str(block.get("library", "")),
        status=str(block.get("status", "")),
        equations=equations,
        coupling_edges=coupling_edges,
        confidence=confidence,
    )

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Meta-orchestration exports

"""Public facade for cross-domain meta-transfer audit utilities.

The meta package exports policy records, training summaries, transfer proposal
types, and audit-log loaders used by cross-domain meta-orchestration workflows.
This initializer performs no filesystem scans or training by itself; concrete
functions in ``meta.transfer`` own JSONL parsing, validation, proposal ranking,
and any derived audit evidence.
"""

from __future__ import annotations

from scpn_phase_orchestrator.meta.transfer import (
    CrossDomainMetaTransfer,
    MetaPolicyRecord,
    MetaTrainingSummary,
    MetaTransferProposal,
    records_from_audit_directory,
    records_from_audit_jsonl,
)

__all__ = [
    "CrossDomainMetaTransfer",
    "MetaPolicyRecord",
    "MetaTrainingSummary",
    "MetaTransferProposal",
    "records_from_audit_directory",
    "records_from_audit_jsonl",
]

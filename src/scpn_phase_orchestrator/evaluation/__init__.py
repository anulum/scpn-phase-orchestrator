# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — honest early-warning detector auditing

"""Public facade for the detector-agnostic early-warning skill auditor.

An early-warning detector is only useful if it fires on genuine pre-transition
segments more often than its own false-alarm rate on transition-free nulls. This
package makes that honest comparison for *any* detector — the SCPN suite, an
AR(1)/Kendall-τ baseline, or a black-box deep classifier — from the per-segment
scores each emits, and seals the verdict into a tamper-evident record.

* :func:`audit_detector` / :func:`audit_scoring_detector` — audit event-vs-null
  skill at a matched false-alarm rate with a label-permutation p-value.
* :class:`DetectorAudit` — the verdict: calibrated threshold, achieved false
  alarm, detection rate, and permutation significance.
* :func:`seal_detector_audit` / :class:`AuditRecord` — bind a verdict to its
  corpus provenance under a SHA-256 content hash.
* Skill primitives (:func:`calibrate_score_threshold`,
  :func:`permutation_significance_from_alarms`, :func:`surrogate_rank_pvalue`)
  for callers composing their own harness.

The package-level surface is import-only; validation lives in the concrete
modules.
"""

from __future__ import annotations

from scpn_phase_orchestrator.evaluation.auditor import (
    DEFAULT_ALPHA,
    DetectorAudit,
    audit_detector,
    audit_scoring_detector,
)
from scpn_phase_orchestrator.evaluation.record import (
    AUDIT_DISCLAIMER,
    AUDIT_FRAMEWORK,
    AuditRecord,
    seal_detector_audit,
)
from scpn_phase_orchestrator.evaluation.skill import (
    DEFAULT_PERMUTATION_SEED,
    DEFAULT_PERMUTATIONS,
    DEFAULT_TARGET_FALSE_ALARM,
    PermutationSignificance,
    benjamini_hochberg,
    calibrate_score_threshold,
    matched_false_alarm_rate,
    permutation_significance_from_alarms,
    surrogate_rank_pvalue,
)

__all__ = [
    "AUDIT_DISCLAIMER",
    "AUDIT_FRAMEWORK",
    "DEFAULT_ALPHA",
    "DEFAULT_PERMUTATIONS",
    "DEFAULT_PERMUTATION_SEED",
    "DEFAULT_TARGET_FALSE_ALARM",
    "AuditRecord",
    "DetectorAudit",
    "PermutationSignificance",
    "audit_detector",
    "audit_scoring_detector",
    "benjamini_hochberg",
    "calibrate_score_threshold",
    "matched_false_alarm_rate",
    "permutation_significance_from_alarms",
    "seal_detector_audit",
    "surrogate_rank_pvalue",
]

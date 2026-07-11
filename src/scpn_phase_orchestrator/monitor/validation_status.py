# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Monitor external-validation posture registry

"""Declare the honest external-validation posture of every monitor family.

SPO ships dozens of dynamical monitors, but only one detection niche has been
checked against an independent ground truth: **grid modal damping**. To keep a
broad monitor gallery from reading as a broad set of *field-ready detectors*,
this module records, for each public monitor module, a machine-readable
:class:`MonitorValidationStatus` and a citable ``basis`` for that status.

The three tiers restate the repository's own ``README`` §*Evidence status*
verbatim in structured form, so the registry cannot quietly overclaim:

``EXTERNALLY_VALIDATED``
    The detector clears a matched-false-alarm operating point *plus* a
    permutation significance test on an **independent, real-data** corpus — the
    only tier that carries field evidence. Today that is the grid modal
    envelope-growth detector and its causal streaming form
    (study §3.5–3.6: 36/90 real PSML transitions led at permutation
    ``p = 0.0001``, held-out ``24/45`` at ``p = 0.0002``).

``SYNTHETIC_ONLY``
    The detector recovers an analytic or planted ground truth on **synthetic**
    data (the eigenvalue-regime map), yet is demonstrated **at chance** on real
    data under the same honest test. The generic early-warning suite and the
    matrix-pencil modal estimator sit here.

``RESEARCH``
    An exploratory diagnostic with **no** external- or synthetic-reference
    validation record. This is the conservative default: a monitor is never
    promoted above ``RESEARCH`` without a citable study section.

The registry is the single source of truth. It is exported from
:mod:`scpn_phase_orchestrator.monitor`, surfaced in the API reference and the
*Monitor validation status* guide, and guarded by a test that fails closed if a
newly added monitor module is left unclassified (see
:data:`NON_MONITOR_MODULES`).
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

__all__ = [
    "MONITOR_VALIDATION",
    "NON_MONITOR_MODULES",
    "MonitorValidationRecord",
    "MonitorValidationStatus",
    "monitors_by_status",
    "validation_record",
    "validation_summary",
]


class MonitorValidationStatus(enum.Enum):
    """The external-validation tier a monitor family is honestly entitled to.

    The three members are ordered from strongest to weakest evidence. Their
    string values (``"external"``, ``"synthetic-only"``, ``"research"``) are the
    stable, machine-readable tokens used across the API, the documentation, and
    any Studio surface, so a downstream consumer can filter on them without
    parsing prose.
    """

    EXTERNALLY_VALIDATED = "external"
    SYNTHETIC_ONLY = "synthetic-only"
    RESEARCH = "research"


@dataclass(frozen=True, slots=True)
class MonitorValidationRecord:
    """The validation posture of a single monitor family.

    Parameters
    ----------
    monitor:
        The monitor module name (the stem of the ``.py`` file under
        ``scpn_phase_orchestrator/monitor/``), used as the registry key.
    display_name:
        A short human-readable label for documentation and Studio surfaces.
    status:
        The :class:`MonitorValidationStatus` the monitor is entitled to.
    basis:
        A one-line, citable justification for ``status`` — a study section, a
        reference, or an explicit statement that no validation record exists.
    evidence:
        A pointer to the evidence backing ``basis`` (a study document, the
        ``README`` evidence-status section, or an empty string when the basis is
        simply the absence of a validation record).

    Raises
    ------
    ValueError
        If ``monitor``, ``display_name`` or ``basis`` is empty or blank, so a
        record can never be silently underspecified.
    TypeError
        If ``status`` is not a :class:`MonitorValidationStatus`.
    """

    monitor: str
    display_name: str
    status: MonitorValidationStatus
    basis: str
    evidence: str

    def __post_init__(self) -> None:
        """Validate the record fields, failing closed on an empty specification."""
        if not isinstance(self.status, MonitorValidationStatus):
            msg = (
                "status must be a MonitorValidationStatus, "
                f"got {type(self.status).__name__}"
            )
            raise TypeError(msg)
        for field_name in ("monitor", "display_name", "basis"):
            value = getattr(self, field_name)
            if not value or not value.strip():
                msg = f"{field_name} must be a non-empty string"
                raise ValueError(msg)


# Public ``monitor/`` modules that are not detector/monitor families and so are
# not classified: deterministic fixture and replay-record generators that supply
# review evidence rather than run as deployable detectors, plus this registry
# module itself. The classification drift-guard subtracts this set from the
# discovered module list, so a genuinely new monitor cannot be silently omitted
# — it is either classified below or explicitly listed here.
NON_MONITOR_MODULES: frozenset[str] = frozenset(
    {
        "validation_status",
        "hybrid_order_examples",
        "self_model_examples",
        "information_replay_cyber_industrial",
        "information_replay_infrastructure",
        "information_replay_physiology",
    }
)

_STUDY = "docs/studies/early_warning_matched_false_alarm.md"
_README_EVIDENCE = "README.md §Evidence status"
_NO_RECORD = (
    "exploratory diagnostic; no external- or synthetic-reference validation record"
)

# The validation posture of every public monitor family, ordered strongest
# evidence first and alphabetically within a tier. Every promotion above
# ``RESEARCH`` cites the study section that earns it; ``RESEARCH`` records state
# plainly that no external- or synthetic-reference validation record exists.
_RECORDS: tuple[MonitorValidationRecord, ...] = (
    # --- Externally validated on independent real data ---
    MonitorValidationRecord(
        monitor="grid_modal_growth",
        display_name="Grid modal envelope-growth detector",
        status=MonitorValidationStatus.EXTERNALLY_VALIDATED,
        basis=(
            "leads 36/90 real PSML growing-instability transitions at "
            "permutation p = 0.0001 (held-out 24/45, p = 0.0002) at a matched "
            "false alarm; the eigenvalue growth rate is a checkable physical "
            "quantity, not a proxy"
        ),
        evidence=f"{_STUDY} §3.5",
    ),
    MonitorValidationRecord(
        monitor="grid_modal_stream",
        display_name="Grid modal-growth streaming monitor",
        status=MonitorValidationStatus.EXTERNALLY_VALIDATED,
        basis=(
            "the certified grid modal-growth detector re-certified for causal "
            "streaming on the real PSML corpus with a hash-sealed operating "
            "point; leads growing-instability transitions above chance"
        ),
        evidence=f"{_STUDY} §3.6",
    ),
    # --- Validated on synthetic ground truth; at chance on real data ---
    MonitorValidationRecord(
        monitor="critical_slowing_down",
        display_name="Critical slowing down",
        status=MonitorValidationStatus.SYNTHETIC_ONLY,
        basis=(
            "generic early-warning member (rising variance and lag-one "
            "autocorrelation); recovers the analytic eigenvalue on the synthetic "
            "transition suite but is at chance on real data in all four domains"
        ),
        evidence=f"{_STUDY} §2.1, §3.1",
    ),
    MonitorValidationRecord(
        monitor="early_warning_suite",
        display_name="Domain-adaptable early-warning suite",
        status=MonitorValidationStatus.SYNTHETIC_ONLY,
        basis=(
            "the neutral-observable harness hosting the generic detector "
            "members; the suite recovers the synthetic regime map but no member "
            "reaches significance on real data at a matched false alarm"
        ),
        evidence=f"{_STUDY} §2.1–2.2",
    ),
    MonitorValidationRecord(
        monitor="ensemble_warning",
        display_name="Ensemble early warning",
        status=MonitorValidationStatus.SYNTHETIC_ONLY,
        basis=(
            "weighted fusion of the generic early-warning suite; inherits the "
            "suite's synthetic-only posture and is at chance on real data"
        ),
        evidence=f"{_STUDY} §2.1",
    ),
    MonitorValidationRecord(
        monitor="opt_entropy",
        display_name="Ordinal-pattern transition entropy",
        status=MonitorValidationStatus.SYNTHETIC_ONLY,
        basis=(
            "generic early-warning member (ordinal-pattern transition entropy "
            "of the phase field); recovers the synthetic regime map but is at "
            "chance on real data"
        ),
        evidence=f"{_STUDY} §2.1",
    ),
    MonitorValidationRecord(
        monitor="oscillation_modes",
        display_name="Inter-area oscillation modes (matrix pencil)",
        status=MonitorValidationStatus.SYNTHETIC_ONLY,
        basis=(
            "matrix-pencil modal damping recovers a planted growth rate exactly "
            "on synthetic data, yet on the short real PSML windows it is at "
            "chance (held-out p = 0.39–0.80)"
        ),
        evidence=f"{_STUDY} §3.5",
    ),
    MonitorValidationRecord(
        monitor="synchronisation",
        display_name="Rising synchronisation",
        status=MonitorValidationStatus.SYNTHETIC_ONLY,
        basis=(
            "generic early-warning member (robust z-score of the Kuramoto order "
            "parameter); recovers the synthetic regime map but is at chance on "
            "real data"
        ),
        evidence=f"{_STUDY} §2.1",
    ),
    # --- Exploratory diagnostics: no external- or synthetic-reference record ---
    MonitorValidationRecord(
        monitor="boundaries",
        display_name="Boundary observer",
        status=MonitorValidationStatus.RESEARCH,
        basis=(
            "structural compartment and event-bus safety observer; deterministic "
            "checks, not an empirical detector with a validation record"
        ),
        evidence="",
    ),
    MonitorValidationRecord(
        monitor="chimera",
        display_name="Chimera-state detection",
        status=MonitorValidationStatus.RESEARCH,
        basis=_NO_RECORD,
        evidence=_README_EVIDENCE,
    ),
    MonitorValidationRecord(
        monitor="coherence",
        display_name="Coherence partition monitor",
        status=MonitorValidationStatus.RESEARCH,
        basis=_NO_RECORD,
        evidence=_README_EVIDENCE,
    ),
    MonitorValidationRecord(
        monitor="dimension",
        display_name="Correlation dimension",
        status=MonitorValidationStatus.RESEARCH,
        basis=_NO_RECORD,
        evidence=_README_EVIDENCE,
    ),
    MonitorValidationRecord(
        monitor="embedding",
        display_name="Delay embedding",
        status=MonitorValidationStatus.RESEARCH,
        basis=_NO_RECORD,
        evidence=_README_EVIDENCE,
    ),
    MonitorValidationRecord(
        monitor="entropy_prod",
        display_name="Entropy-production rate",
        status=MonitorValidationStatus.RESEARCH,
        basis=_NO_RECORD,
        evidence=_README_EVIDENCE,
    ),
    MonitorValidationRecord(
        monitor="evs",
        display_name="EVS and phase-locking metrics",
        status=MonitorValidationStatus.RESEARCH,
        basis=_NO_RECORD,
        evidence=_README_EVIDENCE,
    ),
    MonitorValidationRecord(
        monitor="explosive_sync",
        display_name="Explosive-sync early warning",
        status=MonitorValidationStatus.RESEARCH,
        basis=(
            "exploratory early-warning variant for explosive synchronisation; "
            "no external- or synthetic-reference validation record"
        ),
        evidence=_README_EVIDENCE,
    ),
    MonitorValidationRecord(
        monitor="hybrid_order",
        display_name="Hybrid classical-quantum order parameter",
        status=MonitorValidationStatus.RESEARCH,
        basis=(
            "local quantum co-simulation evidence only; not QPU execution and "
            "not an externally validated detector"
        ),
        evidence=_README_EVIDENCE,
    ),
    MonitorValidationRecord(
        monitor="information_integration",
        display_name="Integrated-information proxy",
        status=MonitorValidationStatus.RESEARCH,
        basis=(
            "approximate integrated-information proxy over phase trajectories; "
            "engineering proxy, no external- or synthetic-reference validation record"
        ),
        evidence=_README_EVIDENCE,
    ),
    MonitorValidationRecord(
        monitor="itpc",
        display_name="Inter-trial phase coherence",
        status=MonitorValidationStatus.RESEARCH,
        basis=_NO_RECORD,
        evidence=_README_EVIDENCE,
    ),
    MonitorValidationRecord(
        monitor="koopman_edmd",
        display_name="EDMD with control",
        status=MonitorValidationStatus.RESEARCH,
        basis=(
            "data-driven linear predictor; no external- or synthetic-reference "
            "validation record as an early-warning detector"
        ),
        evidence=_README_EVIDENCE,
    ),
    MonitorValidationRecord(
        monitor="lyapunov",
        display_name="Lyapunov stability monitor",
        status=MonitorValidationStatus.RESEARCH,
        basis=_NO_RECORD,
        evidence=_README_EVIDENCE,
    ),
    MonitorValidationRecord(
        monitor="merge_window",
        display_name="Merge-window lock monitor",
        status=MonitorValidationStatus.RESEARCH,
        basis=(
            "deterministic phase-and-space lock monitor; structural check, not "
            "an empirical detector with a validation record"
        ),
        evidence="",
    ),
    MonitorValidationRecord(
        monitor="modal_participation",
        display_name="Modal participation and damping",
        status=MonitorValidationStatus.RESEARCH,
        basis=(
            "model-based small-signal modal analysis (Kundur 1994); exact on the "
            "specified network model, not an empirical detector validated against "
            "an independent reference"
        ),
        evidence=_README_EVIDENCE,
    ),
    MonitorValidationRecord(
        monitor="npe",
        display_name="Normalised persistent entropy",
        status=MonitorValidationStatus.RESEARCH,
        basis=_NO_RECORD,
        evidence=_README_EVIDENCE,
    ),
    MonitorValidationRecord(
        monitor="phase_koopman",
        display_name="Phase-autoencoder Koopman observables",
        status=MonitorValidationStatus.RESEARCH,
        basis=_NO_RECORD,
        evidence=_README_EVIDENCE,
    ),
    MonitorValidationRecord(
        monitor="pid",
        display_name="Partial information decomposition",
        status=MonitorValidationStatus.RESEARCH,
        basis=_NO_RECORD,
        evidence=_README_EVIDENCE,
    ),
    MonitorValidationRecord(
        monitor="poincare",
        display_name="Poincaré-section crossings",
        status=MonitorValidationStatus.RESEARCH,
        basis=_NO_RECORD,
        evidence=_README_EVIDENCE,
    ),
    MonitorValidationRecord(
        monitor="psychedelic",
        display_name="Psychedelic phase-dispersion diagnostics",
        status=MonitorValidationStatus.RESEARCH,
        basis=(
            "research-only phase-dispersion simulation utilities; no external- or "
            "synthetic-reference validation record"
        ),
        evidence=_README_EVIDENCE,
    ),
    MonitorValidationRecord(
        monitor="recurrence",
        display_name="Recurrence quantification analysis",
        status=MonitorValidationStatus.RESEARCH,
        basis=_NO_RECORD,
        evidence=_README_EVIDENCE,
    ),
    MonitorValidationRecord(
        monitor="self_model",
        display_name="Self-model discrepancy monitor",
        status=MonitorValidationStatus.RESEARCH,
        basis=(
            "deterministic self-model discrepancy monitor with auditable "
            "evidence; no external- or synthetic-reference validation record"
        ),
        evidence=_README_EVIDENCE,
    ),
    MonitorValidationRecord(
        monitor="session_start",
        display_name="Session-start validation gate",
        status=MonitorValidationStatus.RESEARCH,
        basis=(
            "input validation gate for extractor, imprint and coherence inputs; "
            "deterministic check, not an empirical detector with a validation record"
        ),
        evidence="",
    ),
    MonitorValidationRecord(
        monitor="sleep_staging",
        display_name="Sleep staging",
        status=MonitorValidationStatus.RESEARCH,
        basis=(
            "phase-synchrony sleep-staging helpers evaluated on SleepEDF; not in "
            "the README external-validation set for early-warning detection"
        ),
        evidence=_README_EVIDENCE,
    ),
    MonitorValidationRecord(
        monitor="transfer_entropy",
        display_name="Phase transfer entropy",
        status=MonitorValidationStatus.RESEARCH,
        basis=_NO_RECORD,
        evidence=_README_EVIDENCE,
    ),
    MonitorValidationRecord(
        monitor="twin_confidence",
        display_name="Digital-twin confidence scoring",
        status=MonitorValidationStatus.RESEARCH,
        basis=(
            "online model–observation divergence scoring; engineering surface, "
            "no external- or synthetic-reference validation record"
        ),
        evidence=_README_EVIDENCE,
    ),
    MonitorValidationRecord(
        monitor="twin_conformal_gate",
        display_name="Twin conformal admission gate",
        status=MonitorValidationStatus.RESEARCH,
        basis=(
            "coverage-valid admission gate over the twin-confidence stream; "
            "engineering surface, no external- or synthetic-reference validation record"
        ),
        evidence=_README_EVIDENCE,
    ),
    MonitorValidationRecord(
        monitor="winding",
        display_name="Winding-number tracker",
        status=MonitorValidationStatus.RESEARCH,
        basis=_NO_RECORD,
        evidence=_README_EVIDENCE,
    ),
)


def _build_registry(
    records: tuple[MonitorValidationRecord, ...],
) -> Mapping[str, MonitorValidationRecord]:
    """Return a read-only registry keyed by monitor name, guarding consistency.

    Parameters
    ----------
    records:
        The monitor validation records, ordered strongest evidence first.

    Returns
    -------
    Mapping[str, MonitorValidationRecord]
        A read-only mapping from monitor name to its record, preserving the
        declaration order.

    Raises
    ------
    ValueError
        If two records share a monitor name, or if any
        :class:`MonitorValidationStatus` tier is unrepresented, so the registry
        cannot silently lose a tier or double-declare a monitor.
    """
    registry: dict[str, MonitorValidationRecord] = {}
    for record in records:
        if record.monitor in registry:
            msg = f"duplicate monitor validation record for {record.monitor!r}"
            raise ValueError(msg)
        registry[record.monitor] = record
    represented = {record.status for record in records}
    if represented != set(MonitorValidationStatus):
        missing = sorted(
            status.value for status in set(MonitorValidationStatus) - represented
        )
        msg = f"monitor validation registry is missing tier(s): {missing}"
        raise ValueError(msg)
    return MappingProxyType(registry)


MONITOR_VALIDATION: Mapping[str, MonitorValidationRecord] = _build_registry(_RECORDS)
"""Read-only registry of every classified monitor's validation posture."""


def validation_record(monitor: str) -> MonitorValidationRecord:
    """Return the validation record for a monitor, failing closed if unknown.

    Parameters
    ----------
    monitor:
        The monitor module name (for example ``"grid_modal_growth"``).

    Returns
    -------
    MonitorValidationRecord
        The record declaring the monitor's validation posture.

    Raises
    ------
    KeyError
        If ``monitor`` is not a classified monitor family, listing the known
        monitors so the caller cannot fall through to a silent default.
    """
    try:
        return MONITOR_VALIDATION[monitor]
    except KeyError:
        known = ", ".join(sorted(MONITOR_VALIDATION))
        msg = f"unknown monitor {monitor!r}; known monitors: {known}"
        raise KeyError(msg) from None


def monitors_by_status(
    status: MonitorValidationStatus,
) -> tuple[MonitorValidationRecord, ...]:
    """Return every monitor record at a validation tier, ordered by name.

    Parameters
    ----------
    status:
        The :class:`MonitorValidationStatus` to filter on.

    Returns
    -------
    tuple[MonitorValidationRecord, ...]
        The matching records, sorted by monitor name for determinism (possibly
        empty).

    Raises
    ------
    TypeError
        If ``status`` is not a :class:`MonitorValidationStatus`.
    """
    if not isinstance(status, MonitorValidationStatus):
        msg = f"status must be a MonitorValidationStatus, got {type(status).__name__}"
        raise TypeError(msg)
    return tuple(
        sorted(
            (
                record
                for record in MONITOR_VALIDATION.values()
                if record.status is status
            ),
            key=lambda record: record.monitor,
        )
    )


def validation_summary() -> Mapping[MonitorValidationStatus, int]:
    """Return the count of classified monitors at each validation tier.

    Returns
    -------
    Mapping[MonitorValidationStatus, int]
        A read-only mapping from every :class:`MonitorValidationStatus` member
        to the number of monitors at that tier, including tiers with a zero
        count so the shape is stable.
    """
    counts: dict[MonitorValidationStatus, int] = dict.fromkeys(
        MonitorValidationStatus, 0
    )
    for record in MONITOR_VALIDATION.values():
        counts[record.status] += 1
    return MappingProxyType(counts)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Curated STL specification catalogue for phase fields

"""Curated STL safety specifications for Kuramoto-type phase fields.

Each :class:`PhaseFieldSpecification` names one runtime property of the phase
field — an order-parameter floor, a coupling bound, a chimera-index ceiling, a
Sakaguchi phase-lag bound, and a winding-stability bound — as a single-signal
STL formula that the builtin monitor backend can evaluate without ``rtamt``.

The thresholds are documented **engineering defaults** with a physical
rationale, not empirically fitted constants: a user selects and, where needed,
retunes them for a specific deployment. Robustness from :meth:`evaluate`
measures runtime signal margin; it is **not** a formal proof of correctness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .monitor import STLMonitor, STLTraceResult

_TEMPORAL_OPERATORS = frozenset({"always", "eventually"})
_COMPARISON_OPERATORS = frozenset({">=", ">", "<=", "<", "=="})
_SEVERITIES = frozenset({"soft", "hard"})


def _render_threshold(threshold: float) -> str:
    """Return a stable decimal rendering of a predicate threshold.

    Parameters
    ----------
    threshold : float
        The finite predicate threshold to render.

    Returns
    -------
    str
        ``"10.0"`` for integral values, otherwise a ``:.16g`` decimal
        rendering; both forms are accepted by the builtin predicate grammar.
    """
    if threshold.is_integer():
        return f"{int(threshold)}.0"
    return f"{threshold:.16g}"


@dataclass(frozen=True)
class PhaseFieldSpecification:
    """A named single-signal STL property of a Kuramoto-type phase field.

    Parameters
    ----------
    name : str
        Stable catalogue key, e.g. ``"order_parameter_floor"``.
    signal : str
        Trace key the property constrains, e.g. ``"R"``.
    temporal_op : str
        Temporal operator, ``"always"`` or ``"eventually"``.
    comparison : str
        Predicate comparison operator: one of ``>=``, ``>``, ``<=``, ``<``,
        ``==``.
    threshold : float
        Finite predicate threshold.
    rationale : str
        Physical or engineering justification for the property and threshold.
    severity : str
        Escalation tier, ``"soft"`` (default) or ``"hard"``.

    Raises
    ------
    ValueError
        If any field is empty or outside its permitted set, or if the
        threshold is not finite.
    """

    name: str
    signal: str
    temporal_op: str
    comparison: str
    threshold: float
    rationale: str
    severity: str = "soft"

    def __post_init__(self) -> None:
        _require_non_empty(self.name, "name")
        _require_non_empty(self.signal, "signal")
        _require_non_empty(self.rationale, "rationale")
        if self.temporal_op not in _TEMPORAL_OPERATORS:
            raise ValueError(
                f"temporal_op must be one of {sorted(_TEMPORAL_OPERATORS)}, "
                f"got {self.temporal_op!r}"
            )
        if self.comparison not in _COMPARISON_OPERATORS:
            raise ValueError(
                f"comparison must be one of {sorted(_COMPARISON_OPERATORS)}, "
                f"got {self.comparison!r}"
            )
        if self.severity not in _SEVERITIES:
            raise ValueError(
                f"severity must be one of {sorted(_SEVERITIES)}, got {self.severity!r}"
            )
        if not math.isfinite(self.threshold):
            raise ValueError(f"threshold must be finite, got {self.threshold!r}")

    @property
    def spec(self) -> str:
        """Return the builtin-compatible STL formula for this property.

        Returns
        -------
        str
            An STL string such as ``"always (R >= 0.3)"`` that both the builtin
            and the rtamt backends of :class:`~.monitor.STLMonitor` accept.
        """
        rendered = _render_threshold(self.threshold)
        return f"{self.temporal_op} ({self.signal} {self.comparison} {rendered})"

    def monitor(self) -> STLMonitor:
        """Return a fresh :class:`~.monitor.STLMonitor` for this property.

        Returns
        -------
        STLMonitor
            A monitor bound to :attr:`spec`.
        """
        return STLMonitor(self.spec)

    def evaluate(self, trace: dict[str, list[float]]) -> STLTraceResult:
        """Evaluate this property over *trace* and return its robustness record.

        Parameters
        ----------
        trace : dict[str, list[float]]
            Signal trace keyed by variable name; must include :attr:`signal`.

        Returns
        -------
        STLTraceResult
            Robustness value plus audit metadata (spec, satisfied, backend).

        Raises
        ------
        ValueError
            If the trace is empty, ragged, or numerically invalid.
        """
        return self.monitor().evaluate_result(trace)


def _require_non_empty(value: str, name: str) -> None:
    """Raise ``ValueError`` unless *value* is a non-empty string.

    Parameters
    ----------
    value : str
        Candidate string.
    name : str
        Field name used in the error message.

    Raises
    ------
    ValueError
        If *value* is not a non-empty string.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


# ──────────────────────────────────────────────────
# Curated catalogue
# ──────────────────────────────────────────────────

# Sakaguchi frustration limit: for phase-lag alpha >= pi/2 the pairwise
# coupling stops being attractive and global synchronisation is destroyed
# (Sakaguchi & Kuramoto 1986). Half of pi is the honest safety ceiling.
_SAKAGUCHI_FRUSTRATION_LIMIT = math.pi / 2.0


PHASE_FIELD_SPECIFICATIONS: tuple[PhaseFieldSpecification, ...] = (
    PhaseFieldSpecification(
        name="order_parameter_floor",
        signal="R",
        temporal_op="always",
        comparison=">=",
        threshold=0.3,
        rationale=(
            "Kuramoto order parameter R must stay above the partial-coherence "
            "floor for the field to remain functionally synchronised; below it "
            "the ensemble is effectively incoherent (Kuramoto 1984)."
        ),
        severity="hard",
    ),
    PhaseFieldSpecification(
        name="coupling_gain_ceiling",
        signal="K",
        temporal_op="always",
        comparison="<=",
        threshold=10.0,
        rationale=(
            "Global coupling gain K is bounded to keep the closed loop away "
            "from the overdriven regime where finite-step integration and "
            "actuation saturation destabilise the field."
        ),
        severity="hard",
    ),
    PhaseFieldSpecification(
        name="chimera_index_ceiling",
        signal="chimera_index",
        temporal_op="always",
        comparison="<=",
        threshold=0.5,
        rationale=(
            "The chimera index is the fraction of oscillators sitting in the "
            "incoherent domain; capping it at one half keeps the coherent "
            "domain in the majority and flags runaway symmetry breaking."
        ),
        severity="soft",
    ),
    PhaseFieldSpecification(
        name="phase_lag_bound",
        signal="phase_lag",
        temporal_op="always",
        comparison="<=",
        threshold=_SAKAGUCHI_FRUSTRATION_LIMIT,
        rationale=(
            "The Sakaguchi phase lag alpha is held below pi/2, the frustration "
            "limit beyond which the pairwise coupling turns repulsive and "
            "global synchronisation cannot form (Sakaguchi & Kuramoto 1986)."
        ),
        severity="soft",
    ),
    PhaseFieldSpecification(
        name="winding_stability",
        signal="winding_number",
        temporal_op="always",
        comparison="<=",
        threshold=1.0,
        rationale=(
            "The winding-number magnitude (absolute topological charge of the "
            "phase field) is capped so at most one twist is tolerated; growth "
            "beyond it signals accumulating phase slips."
        ),
        severity="soft",
    ),
)


_CATALOGUE_INDEX: dict[str, PhaseFieldSpecification] = {
    spec.name: spec for spec in PHASE_FIELD_SPECIFICATIONS
}


def phase_field_specification_names() -> tuple[str, ...]:
    """Return the catalogue keys in their curated order.

    Returns
    -------
    tuple[str, ...]
        The ``name`` of every specification in
        :data:`PHASE_FIELD_SPECIFICATIONS`.
    """
    return tuple(spec.name for spec in PHASE_FIELD_SPECIFICATIONS)


def phase_field_specification(name: str) -> PhaseFieldSpecification:
    """Return the curated specification registered under *name*.

    Parameters
    ----------
    name : str
        A catalogue key from :func:`phase_field_specification_names`.

    Returns
    -------
    PhaseFieldSpecification
        The matching specification.

    Raises
    ------
    KeyError
        If *name* is not a registered catalogue key.
    """
    try:
        return _CATALOGUE_INDEX[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown phase-field specification {name!r}; "
            f"known: {phase_field_specification_names()}"
        ) from exc

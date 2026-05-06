# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Strange-loop supervisor monitor

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import isfinite
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.actuation.mapper import ControlAction

__all__ = [
    "StrangeLoopAssessment",
    "StrangeLoopSupervisor",
]

FloatArray: TypeAlias = NDArray[np.float64]
_KNOB_INDEX = {"K": 0, "alpha": 1, "zeta": 2, "Psi": 3}


@dataclass(frozen=True)
class StrangeLoopAssessment:
    """Audit-ready metrics for supervisor self-control dynamics."""

    control_phase: float
    control_coherence: float
    drift_score: float
    oscillation_score: float
    overcontrol_score: float
    recommended_actions: tuple[ControlAction, ...]

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-serialisable record for supervisor audit logs."""
        return {
            "control_phase": self.control_phase,
            "control_coherence": self.control_coherence,
            "drift_score": self.drift_score,
            "oscillation_score": self.oscillation_score,
            "overcontrol_score": self.overcontrol_score,
            "recommended_actions": [
                {
                    "knob": action.knob,
                    "scope": action.scope,
                    "value": action.value,
                    "ttl_s": action.ttl_s,
                    "justification": action.justification,
                }
                for action in self.recommended_actions
            ],
        }


class StrangeLoopSupervisor:
    """Treat supervisor action history as a self-referential control channel.

    The monitor embeds each recent action bundle into the four native control
    knobs `(K, alpha, zeta, Psi)`. It then measures whether the supervisor is
    drifting, oscillating, or over-actuating and emits conservative damping
    recommendations for an outer policy gate to approve.
    """

    def __init__(
        self,
        *,
        history_size: int = 12,
        drift_threshold: float = 0.25,
        oscillation_threshold: float = 0.5,
        overcontrol_threshold: float = 0.2,
        damping_gain: float = 0.05,
        ttl_s: float = 3.0,
    ) -> None:
        if history_size < 2:
            raise ValueError("history_size must be >= 2")
        for name, value in {
            "drift_threshold": drift_threshold,
            "oscillation_threshold": oscillation_threshold,
            "overcontrol_threshold": overcontrol_threshold,
            "damping_gain": damping_gain,
            "ttl_s": ttl_s,
        }.items():
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        self._history: deque[FloatArray] = deque(maxlen=history_size)
        self._drift_threshold = float(drift_threshold)
        self._oscillation_threshold = float(oscillation_threshold)
        self._overcontrol_threshold = float(overcontrol_threshold)
        self._damping_gain = float(damping_gain)
        self._ttl_s = float(ttl_s)
        self.last_assessment: StrangeLoopAssessment | None = None

    def observe(self, actions: list[ControlAction]) -> StrangeLoopAssessment:
        """Record one supervisor action bundle and assess self-control state."""
        vector = _actions_to_vector(actions)
        self._history.append(vector)
        assessment = self._assess()
        self.last_assessment = assessment
        return assessment

    def reset(self) -> None:
        """Clear action history and the cached assessment."""
        self._history.clear()
        self.last_assessment = None

    def _assess(self) -> StrangeLoopAssessment:
        matrix = np.vstack(self._history)
        latest = matrix[-1]
        magnitude = float(np.linalg.norm(latest))
        control_phase = float(np.mod(np.arctan2(latest[3], latest[0]), 2.0 * np.pi))
        control_coherence = _control_coherence(matrix)
        drift_score = _drift_score(matrix)
        oscillation_score = _oscillation_score(matrix)
        overcontrol_score = magnitude
        recommendations = self._recommend(
            drift_score=drift_score,
            oscillation_score=oscillation_score,
            overcontrol_score=overcontrol_score,
        )
        return StrangeLoopAssessment(
            control_phase=control_phase,
            control_coherence=control_coherence,
            drift_score=drift_score,
            oscillation_score=oscillation_score,
            overcontrol_score=overcontrol_score,
            recommended_actions=tuple(recommendations),
        )

    def _recommend(
        self,
        *,
        drift_score: float,
        oscillation_score: float,
        overcontrol_score: float,
    ) -> list[ControlAction]:
        reasons = []
        if drift_score > self._drift_threshold:
            reasons.append("policy drift")
        if oscillation_score > self._oscillation_threshold:
            reasons.append("control-loop oscillation")
        if overcontrol_score > self._overcontrol_threshold:
            reasons.append("over-control")
        if not reasons:
            return []
        reason = ", ".join(reasons)
        return [
            ControlAction(
                knob="zeta",
                scope="global",
                value=self._damping_gain,
                ttl_s=self._ttl_s,
                justification=f"strange-loop damping: {reason}",
            ),
            ControlAction(
                knob="K",
                scope="global",
                value=-0.5 * self._damping_gain,
                ttl_s=self._ttl_s,
                justification=f"strange-loop coupling trim: {reason}",
            ),
        ]


def _actions_to_vector(actions: list[ControlAction]) -> FloatArray:
    vector = np.zeros(4, dtype=np.float64)
    for action in actions:
        if action.knob not in _KNOB_INDEX:
            continue
        if not isfinite(action.value):
            raise ValueError("action values must be finite")
        vector[_KNOB_INDEX[action.knob]] += float(action.value)
    return vector


def _control_coherence(matrix: FloatArray) -> float:
    norms = np.linalg.norm(matrix, axis=1)
    active = norms > 0.0
    if not np.any(active):
        return 1.0
    unit = matrix[active] / norms[active, None]
    mean_vector = np.mean(unit, axis=0)
    return float(np.clip(np.linalg.norm(mean_vector), 0.0, 1.0))


def _drift_score(matrix: FloatArray) -> float:
    if matrix.shape[0] < 2:
        return 0.0
    return float(np.mean(np.linalg.norm(np.diff(matrix, axis=0), axis=1)))


def _oscillation_score(matrix: FloatArray) -> float:
    if matrix.shape[0] < 3:
        return 0.0
    deltas = np.diff(matrix, axis=0)
    signs = np.sign(deltas)
    flips = signs[1:] * signs[:-1] < 0.0
    active = np.abs(deltas[1:]) + np.abs(deltas[:-1]) > 0.0
    if not np.any(active):
        return 0.0
    return float(np.count_nonzero(flips & active) / np.count_nonzero(active))

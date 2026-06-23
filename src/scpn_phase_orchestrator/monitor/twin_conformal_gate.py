# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Conformal twin-confidence admission gate

"""Coverage-valid admission gate over the twin-confidence stream.

The twin-confidence score (:mod:`scpn_phase_orchestrator.monitor.twin_confidence`)
quantifies model–observation disagreement, but a raw score gives no statistical
guarantee. This module wraps the stream in a distribution-free conformal gate:
it learns, from a trusted nominal calibration window, a threshold on a
nonconformity score (the composite z-deviation) such that nominal ticks fall
inside the band with probability ``1 − target_miscoverage``, then admits a tick
only when its score stays inside the band.

Because the twin's behaviour is non-stationary, the threshold adapts online by
Adaptive Conformal Inference (Gibbs & Candès, 2021): the effective miscoverage
``alpha_t`` is nudged up when a tick is covered and down when it is missed, so
the long-run empirical miscoverage tracks the target. The gate is optionally
*regime conditioned* — it keeps a separate calibration set and ``alpha_t`` per
detected regime (sync / chimera / chaotic), which SPO already classifies — so the
band is appropriate to the current dynamical regime rather than a global average.

This is a review-only safety observable: a flagged tick signals the twin has
drifted beyond its calibrated nominal band and that autonomy should narrow; it
never actuates. The computation is lightweight online statistics (one sorted
calibration array per regime, O(1) per update), so it has no compute hot path and
no multi-language backend.

References
----------
Gibbs, I. & Candès, E. (2021). Adaptive conformal inference under distribution
shift. NeurIPS. Regime conditioning follows the change-point/transition-conformal
direction (e.g. arXiv:2509.02844); only the established ACI core is implemented
here, with regime conditioning as the SPO-specific adaptation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from math import ceil, isfinite
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from scpn_phase_orchestrator.monitor.twin_confidence import TwinConfidenceScore

FloatArray = NDArray[np.float64]

__all__ = [
    "ConformalDecision",
    "ConformalGateConfig",
    "TwinConformalGate",
    "confidence_nonconformity",
]

_DEFAULT_REGIME = "default"


def _finite_real(value: object, *, name: str) -> float:
    """Return ``value`` as a finite real float, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def confidence_nonconformity(score: TwinConfidenceScore) -> float:
    """Return the nonconformity score used by the gate for a confidence score.

    The composite one-sided z-deviation is already a non-negative "how far from
    nominal" quantity, which is exactly the nonconformity scale the conformal gate
    expects.

    Parameters
    ----------
    score : TwinConfidenceScore
        A scored twin-confidence tick.

    Returns
    -------
    float
        The composite z-deviation as a nonconformity score.
    """
    return float(score.composite_z)


@dataclass(frozen=True)
class ConformalGateConfig:
    """Configuration for the conformal admission gate.

    Attributes
    ----------
    target_miscoverage : float
        Desired long-run fraction of nominal ticks falling outside the band
        (``alpha``); in ``(0, 1)``, default ``0.1`` (90% coverage).
    adaptation_rate : float
        Adaptive Conformal Inference step size (``gamma``); in ``(0, 1]``,
        default ``0.02``.
    regime_conditioned : bool
        Whether to keep a separate calibration set and adaptive miscoverage per
        regime (default ``True``).
    """

    target_miscoverage: float = 0.1
    adaptation_rate: float = 0.02
    regime_conditioned: bool = True

    def __post_init__(self) -> None:
        alpha = _finite_real(self.target_miscoverage, name="target_miscoverage")
        if not 0.0 < alpha < 1.0:
            raise ValueError("target_miscoverage must be in (0, 1)")
        gamma = _finite_real(self.adaptation_rate, name="adaptation_rate")
        if not 0.0 < gamma <= 1.0:
            raise ValueError("adaptation_rate must be in (0, 1]")

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe audit mapping of the configuration.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe mapping of the configuration fields.
        """
        return {
            "target_miscoverage": self.target_miscoverage,
            "adaptation_rate": self.adaptation_rate,
            "regime_conditioned": self.regime_conditioned,
        }


@dataclass(frozen=True)
class ConformalDecision:
    """One conformal admission decision for a twin-confidence tick.

    Attributes
    ----------
    admitted : bool
        ``True`` when the nonconformity score is within the conformal band.
    nonconformity_score : float
        The scored tick's nonconformity value.
    threshold : float
        The conformal band upper bound used for this decision (may be infinite
        when the calibration set is too small to bound at the current level).
    effective_miscoverage : float
        The adaptive miscoverage ``alpha_t`` in force for this decision.
    empirical_coverage : float
        Running fraction of admitted ticks for this regime, including this tick.
    regime : str
        The regime key the decision was scored against.
    tick : int
        Per-regime decision index (1-based).
    decision_hash : str
        Deterministic SHA-256 over the audit record (excluding the hash).
    """

    admitted: bool
    nonconformity_score: float
    threshold: float
    effective_miscoverage: float
    empirical_coverage: float
    regime: str
    tick: int
    decision_hash: str

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe audit mapping of the decision.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe mapping of every decision field. An infinite
            threshold is serialised as ``None`` to stay JSON-safe.
        """
        return {
            "admitted": self.admitted,
            "nonconformity_score": self.nonconformity_score,
            "threshold": self.threshold if isfinite(self.threshold) else None,
            "effective_miscoverage": self.effective_miscoverage,
            "empirical_coverage": self.empirical_coverage,
            "regime": self.regime,
            "tick": self.tick,
            "decision_hash": self.decision_hash,
        }


@dataclass
class _RegimeState:
    """Immutable per-regime conformal-gate state (window and threshold)."""

    calibration: FloatArray
    alpha_t: float
    admitted: int = 0
    total: int = 0


@dataclass
class TwinConformalGate:
    """Adaptive conformal admission gate over twin nonconformity scores.

    Attributes
    ----------
    config : ConformalGateConfig
        The gate configuration.
    """

    config: ConformalGateConfig = field(default_factory=ConformalGateConfig)
    _regimes: dict[str, _RegimeState] = field(init=False, repr=False)
    _ticks: int = field(init=False, repr=False, default=0)

    def __post_init__(self) -> None:
        self._regimes = {}

    def calibrate(
        self,
        nominal_scores: Sequence[float],
        *,
        regime: str = _DEFAULT_REGIME,
    ) -> None:
        """Fit the conformal band for a regime from nominal nonconformity scores.

        Parameters
        ----------
        nominal_scores : Sequence[float]
            Nonconformity scores gathered during trusted nominal operation.
        regime : str, optional
            Regime key to calibrate (default ``"default"``).

        Raises
        ------
        ValueError
            If ``nominal_scores`` is empty, a score is non-finite, or ``regime``
            is not a non-empty string.
        """
        if not isinstance(regime, str) or not regime.strip():
            raise ValueError("regime must be a non-empty string")
        if len(nominal_scores) == 0:
            raise ValueError("calibration requires at least one nominal score")
        scores = np.sort(
            np.asarray(
                [_finite_real(value, name="nominal score") for value in nominal_scores],
                dtype=np.float64,
            )
        )
        self._regimes[regime] = _RegimeState(
            calibration=scores,
            alpha_t=self.config.target_miscoverage,
        )

    def update(
        self,
        nonconformity_score: float,
        *,
        regime: str | None = None,
    ) -> ConformalDecision:
        """Score one tick against the conformal band and adapt the threshold.

        Parameters
        ----------
        nonconformity_score : float
            The tick's nonconformity value (higher = more anomalous).
        regime : str or None, optional
            Detected regime; used when the gate is regime conditioned and the
            regime is calibrated, otherwise the ``"default"`` regime is used.

        Returns
        -------
        ConformalDecision
            The admission decision and the post-update running coverage.

        Raises
        ------
        ValueError
            If the score is non-finite or no applicable regime has been calibrated.
        """
        score = _finite_real(nonconformity_score, name="nonconformity_score")
        key = self._resolve_regime(regime)
        state = self._regimes[key]

        threshold = _conformal_threshold(state.calibration, state.alpha_t)
        admitted = score <= threshold
        alpha_used = state.alpha_t

        state.total += 1
        if admitted:
            state.admitted += 1
        self._ticks += 1

        # Adaptive Conformal Inference: alpha_{t+1} = alpha_t + gamma*(alpha - err)
        miscovered = 0.0 if admitted else 1.0
        state.alpha_t = float(
            np.clip(
                state.alpha_t
                + self.config.adaptation_rate
                * (self.config.target_miscoverage - miscovered),
                0.0,
                1.0,
            )
        )

        decision = ConformalDecision(
            admitted=bool(admitted),
            nonconformity_score=score,
            threshold=float(threshold),
            effective_miscoverage=alpha_used,
            empirical_coverage=state.admitted / state.total,
            regime=key,
            tick=state.total,
            decision_hash="",
        )
        return _with_decision_hash(decision)

    def empirical_coverage(self, *, regime: str | None = None) -> float:
        """Return the running admitted fraction for a regime.

        Parameters
        ----------
        regime : str or None, optional
            Regime key (default-resolved when ``None``).

        Returns
        -------
        float
            Admitted ticks over total ticks for the regime, or ``0.0`` before any
            tick has been scored.

        Raises
        ------
        ValueError
            If no applicable regime has been calibrated.
        """
        state = self._regimes[self._resolve_regime(regime)]
        if state.total == 0:
            return 0.0
        return state.admitted / state.total

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe audit mapping of the gate state.

        Returns
        -------
        dict[str, object]
            Configuration, total ticks scored, and per-regime calibration size,
            adaptive miscoverage, and coverage.
        """
        return {
            "config": self.config.to_audit_record(),
            "ticks": self._ticks,
            "regimes": {
                key: {
                    "calibration_size": int(state.calibration.size),
                    "effective_miscoverage": state.alpha_t,
                    "coverage": (state.admitted / state.total) if state.total else 0.0,
                    "ticks": state.total,
                }
                for key, state in sorted(self._regimes.items())
            },
        }

    def _resolve_regime(self, regime: str | None) -> str:
        """Return the regime label for a state under the conformal gate."""
        if self.config.regime_conditioned and regime is not None:
            if not isinstance(regime, str) or not regime.strip():
                raise ValueError("regime must be a non-empty string")
            if regime in self._regimes:
                return regime
        if _DEFAULT_REGIME in self._regimes:
            return _DEFAULT_REGIME
        if regime is not None and regime in self._regimes:
            return regime
        raise ValueError(
            "no calibrated regime is applicable; calibrate the regime "
            "or a 'default' regime first"
        )


def _conformal_threshold(sorted_scores: FloatArray, alpha_t: float) -> float:
    """Return the conformal admit/flag threshold for the regime."""
    n = int(sorted_scores.size)
    level = 1.0 - alpha_t
    rank = ceil(level * (n + 1))
    if rank <= 0:
        return float(sorted_scores[0])
    if rank > n:
        return float("inf")
    return float(sorted_scores[rank - 1])


def _with_decision_hash(decision: ConformalDecision) -> ConformalDecision:
    """Return the decision record augmented with its SHA-256 hash."""
    record = decision.to_audit_record()
    record.pop("decision_hash", None)
    serialised = json.dumps(record, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialised.encode("utf-8")).hexdigest()
    return ConformalDecision(
        admitted=decision.admitted,
        nonconformity_score=decision.nonconformity_score,
        threshold=decision.threshold,
        effective_miscoverage=decision.effective_miscoverage,
        empirical_coverage=decision.empirical_coverage,
        regime=decision.regime,
        tick=decision.tick,
        decision_hash=digest,
    )

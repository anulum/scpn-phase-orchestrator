# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Split-conformal alarm stream with coverage reporting

"""Split-conformal false-alarm control for early-warning detector streams.

An early-warning detector emits a stream of scores, higher meaning more evidence
of an approaching transition. Turning that stream into alarms needs a threshold
whose *false-alarm rate on nominal operation* is controlled, not guessed. This
module calibrates that threshold with the same finite-sample split-conformal
quantile the twin-confidence gate uses (Vovk et al.; Gibbs & Candès 2021), so on
exchangeable nominal scores the probability of an alarm is bounded by the target
false-alarm rate. It fires alarms on a live stream, reports the empirical
false-alarm rate over the nominal ticks it is told about, and can adapt the
threshold online with Adaptive Conformal Inference when the nominal distribution
drifts.

The coverage statement is exactly the split-conformal one — a bound on the
nominal false-alarm rate under exchangeability — and nothing more: an alarm on an
event tick is a detection, not a miscoverage, so online adaptation only ever
consumes ticks that are declared nominal. It makes no claim about detection power.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .twin_conformal_gate import _conformal_threshold

FloatArray = NDArray[np.float64]

_DEFAULT_REGIME = "default"

__all__ = [
    "ConformalAlarmConfig",
    "ConformalAlarmDecision",
    "ConformalAlarmStream",
]


def _finite_real(value: object, *, name: str) -> float:
    """Return *value* as a finite float, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True)
class ConformalAlarmConfig:
    """Configuration of a split-conformal alarm stream.

    Parameters
    ----------
    target_false_alarm : float
        Allowed long-run fraction of nominal ticks that raise an alarm (the
        conformal ``alpha``); in ``(0, 1)``, default ``0.1``.
    adaptation_rate : float
        Adaptive Conformal Inference step size; ``0.0`` (default) keeps the fixed
        split-conformal threshold, a positive value lets the threshold track a
        drifting nominal distribution over the ticks declared nominal.
    regime_conditioned : bool
        Whether to keep a separate calibration and adaptive rate per regime.

    Raises
    ------
    ValueError
        If ``target_false_alarm`` is not in ``(0, 1)`` or ``adaptation_rate`` is
        negative.
    """

    target_false_alarm: float = 0.1
    adaptation_rate: float = 0.0
    regime_conditioned: bool = False

    def __post_init__(self) -> None:
        alpha = _finite_real(self.target_false_alarm, name="target_false_alarm")
        if not 0.0 < alpha < 1.0:
            raise ValueError("target_false_alarm must be in (0, 1)")
        gamma = _finite_real(self.adaptation_rate, name="adaptation_rate")
        if gamma < 0.0:
            raise ValueError("adaptation_rate must be non-negative")

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe mapping of the configuration.

        Returns
        -------
        dict[str, object]
            The target false alarm, adaptation rate, and regime-conditioning flag.
        """
        return {
            "target_false_alarm": self.target_false_alarm,
            "adaptation_rate": self.adaptation_rate,
            "regime_conditioned": self.regime_conditioned,
        }


@dataclass(frozen=True)
class ConformalAlarmDecision:
    """The alarm decision for one tick and the running nominal coverage.

    Parameters
    ----------
    alarm : bool
        ``True`` when the score exceeds the conformal threshold.
    score : float
        The tick's detector score.
    threshold : float
        The conformal threshold used (may be ``+inf`` when the calibration is too
        small to place a finite bound at the target rate).
    effective_false_alarm : float
        The adaptive false-alarm target ``alpha_t`` in force for this decision.
    empirical_false_alarm : float
        Running fraction of nominal ticks that alarmed, this tick included when it
        is nominal.
    regime : str
        The regime the decision was scored under.
    nominal_ticks : int
        Number of nominal ticks scored in this regime so far.
    """

    alarm: bool
    score: float
    threshold: float
    effective_false_alarm: float
    empirical_false_alarm: float
    regime: str
    nominal_ticks: int

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe mapping of the decision.

        The threshold is serialised as the string ``"inf"`` when unbounded so the
        record stays strict JSON.

        Returns
        -------
        dict[str, object]
            The alarm flag, score, threshold, effective and empirical false-alarm
            rates, regime, and nominal tick count.
        """
        threshold: object = self.threshold
        if not np.isfinite(self.threshold):
            threshold = "inf"
        return {
            "alarm": self.alarm,
            "score": self.score,
            "threshold": threshold,
            "effective_false_alarm": self.effective_false_alarm,
            "empirical_false_alarm": self.empirical_false_alarm,
            "regime": self.regime,
            "nominal_ticks": self.nominal_ticks,
        }


@dataclass
class _RegimeState:
    """Per-regime calibration, adaptive rate, and nominal false-alarm counters."""

    calibration: FloatArray
    alpha_t: float
    alarmed_nominal: int = 0
    total_nominal: int = 0


@dataclass
class ConformalAlarmStream:
    """Adaptive split-conformal alarm stream over detector scores.

    Attributes
    ----------
    config : ConformalAlarmConfig
        The alarm-stream configuration.
    """

    config: ConformalAlarmConfig = field(default_factory=ConformalAlarmConfig)
    _regimes: dict[str, _RegimeState] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._regimes = {}

    def calibrate(
        self,
        nominal_scores: Sequence[float],
        *,
        regime: str = _DEFAULT_REGIME,
    ) -> None:
        """Fit the conformal threshold for a regime from nominal detector scores.

        Parameters
        ----------
        nominal_scores : Sequence[float]
            Detector scores gathered during trusted transition-free operation.
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
            alpha_t=self.config.target_false_alarm,
        )

    def update(
        self,
        score: float,
        *,
        is_nominal: bool | None = None,
        regime: str | None = None,
    ) -> ConformalAlarmDecision:
        """Score one tick against the conformal threshold and report coverage.

        Parameters
        ----------
        score : float
            The tick's detector score (higher = more anomalous).
        is_nominal : bool or None, optional
            Whether the tick is known to be transition-free. Only nominal ticks
            update the empirical false-alarm rate and the adaptive threshold; an
            alarm on an event tick is a detection, not a false alarm, and an
            unlabelled tick (``None``) is scored without touching the calibration.
        regime : str or None, optional
            Detected regime; used when the stream is regime conditioned and the
            regime is calibrated, otherwise the ``"default"`` regime is used.

        Returns
        -------
        ConformalAlarmDecision
            The alarm decision and the running nominal false-alarm rate.

        Raises
        ------
        ValueError
            If the score is non-finite or no applicable regime has been calibrated.
        """
        value = _finite_real(score, name="score")
        key = self._resolve_regime(regime)
        state = self._regimes[key]

        threshold = _conformal_threshold(state.calibration, state.alpha_t)
        alarm = value > threshold
        alpha_used = state.alpha_t

        if is_nominal:
            state.total_nominal += 1
            if alarm:
                state.alarmed_nominal += 1
            # Adaptive Conformal Inference over nominal ticks:
            # alpha_{t+1} = alpha_t + gamma * (target - realised false alarm).
            realised = 1.0 if alarm else 0.0
            state.alpha_t = float(
                np.clip(
                    state.alpha_t
                    + self.config.adaptation_rate
                    * (self.config.target_false_alarm - realised),
                    0.0,
                    1.0,
                )
            )

        return ConformalAlarmDecision(
            alarm=bool(alarm),
            score=value,
            threshold=float(threshold),
            effective_false_alarm=alpha_used,
            empirical_false_alarm=self._empirical_false_alarm(state),
            regime=key,
            nominal_ticks=state.total_nominal,
        )

    def empirical_false_alarm(self, *, regime: str | None = None) -> float:
        """Return the running nominal false-alarm rate for a regime.

        Parameters
        ----------
        regime : str or None, optional
            Regime key (default-resolved when ``None``).

        Returns
        -------
        float
            Alarmed nominal ticks over total nominal ticks, or ``0.0`` before any
            nominal tick has been scored.

        Raises
        ------
        ValueError
            If no applicable regime has been calibrated.
        """
        return self._empirical_false_alarm(self._regimes[self._resolve_regime(regime)])

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe audit mapping of the stream state.

        Returns
        -------
        dict[str, object]
            Configuration and, per regime, the calibration size, adaptive false
            alarm, nominal tick count, and empirical false-alarm rate.
        """
        return {
            "config": self.config.to_audit_record(),
            "regimes": {
                key: {
                    "calibration_size": int(state.calibration.size),
                    "effective_false_alarm": state.alpha_t,
                    "nominal_ticks": state.total_nominal,
                    "empirical_false_alarm": self._empirical_false_alarm(state),
                }
                for key, state in sorted(self._regimes.items())
            },
        }

    @staticmethod
    def _empirical_false_alarm(state: _RegimeState) -> float:
        """Return the nominal false-alarm rate recorded in *state*."""
        if state.total_nominal == 0:
            return 0.0
        return state.alarmed_nominal / state.total_nominal

    def _resolve_regime(self, regime: str | None) -> str:
        """Return the regime label to score a tick under."""
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

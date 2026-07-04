# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — sealed early-warning assurance evidence tests

"""Tests for the hash-sealed early-warning assurance evidence.

The suite exercises the neutral sealer (its provenance validation, the
alarm-flag consistency guard, the honest lead computation including a late
alarm, the verdict branches, and the content-hash determinism / mutation
detection), the pure helpers that select the reported window and build an
indicator contribution, and the three adapters that bridge each concrete suite
detector — critical slowing down and rising synchronisation driven end to end
through the real detectors, and the transition-entropy detector through a
constructed warning record.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash
from scpn_phase_orchestrator.assurance.early_warning_evidence import (
    DROP,
    EARLY_WARNING_DISCLAIMER,
    EARLY_WARNING_FLAGGED,
    EARLY_WARNING_FRAMEWORK,
    NO_EARLY_WARNING,
    RISE,
    EarlyWarningEvidence,
    EarlyWarningIndicator,
    _indicator_at,
    _lead,
    _report_window,
    seal_critical_slowing_down_alarm,
    seal_early_warning,
    seal_synchronisation_alarm,
    seal_transition_entropy_alarm,
)
from scpn_phase_orchestrator.monitor.critical_slowing_down import (
    critical_slowing_down_warning,
)
from scpn_phase_orchestrator.monitor.explosive_sync import ExplosiveSyncWarning
from scpn_phase_orchestrator.monitor.synchronisation import synchronisation_warning


def _indicator(
    *,
    name: str = "order_parameter",
    direction: str = RISE,
    robust_z: float = 5.0,
    baseline_median: float = 0.3,
    z_threshold: float = 3.0,
    breached: bool = True,
) -> EarlyWarningIndicator:
    """Return a valid indicator contribution for the core sealer tests."""
    return EarlyWarningIndicator(
        name=name,
        direction=direction,
        robust_z=robust_z,
        baseline_median=baseline_median,
        z_threshold=z_threshold,
        breached=breached,
    )


def _seal(**overrides: object) -> EarlyWarningEvidence:
    """Seal a triggered order-parameter alarm, overriding named fields."""
    payload: dict[str, object] = {
        "detector": "synchronisation",
        "observable": "cross_channel_order_parameter",
        "signal_source": "chb01_03",
        "captured_at": "2996s-onset-event",
        "sampling_rate_hz": 32.0,
        "window": 128,
        "step": 16,
        "persistence": 2,
        "n_baseline_windows": 4,
        "warning_triggered": True,
        "warning_window": 7,
        "warning_sample": 640,
        "indicators": (_indicator(),),
        "transition_onset_sample": 2048,
    }
    payload.update(overrides)
    return seal_early_warning(**payload)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Core sealer                                                                  #
# --------------------------------------------------------------------------- #


def test_seals_a_triggered_alarm_with_an_early_lead() -> None:
    evidence = _seal()
    assert evidence.verdict == EARLY_WARNING_FLAGGED
    assert evidence.warning_triggered is True
    assert evidence.lead_samples == 2048 - 640
    assert evidence.lead_seconds == pytest.approx((2048 - 640) / 32.0)
    assert evidence.lead_is_early is True
    assert evidence.framework == EARLY_WARNING_FRAMEWORK
    assert evidence.disclaimer == EARLY_WARNING_DISCLAIMER
    assert len(evidence.content_hash) == 64


def test_seals_a_negative_result_without_an_onset() -> None:
    evidence = _seal(
        warning_triggered=False,
        warning_window=None,
        warning_sample=None,
        transition_onset_sample=None,
    )
    assert evidence.verdict == NO_EARLY_WARNING
    assert evidence.lead_samples is None
    assert evidence.lead_seconds is None
    assert evidence.lead_is_early is False


def test_a_late_alarm_reports_a_non_positive_lead() -> None:
    # The alarm fired after the onset: the record must show the late lead, not
    # hide it — this is the falsification-honesty property of the moat.
    evidence = _seal(warning_sample=2600, transition_onset_sample=2048)
    assert evidence.lead_samples == 2048 - 2600
    assert evidence.lead_samples < 0
    assert evidence.lead_is_early is False


def test_a_coincident_alarm_is_not_early() -> None:
    evidence = _seal(warning_sample=2048, transition_onset_sample=2048)
    assert evidence.lead_samples == 0
    assert evidence.lead_is_early is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"warning_triggered": True, "warning_window": None, "warning_sample": 640},
        {"warning_triggered": True, "warning_window": 7, "warning_sample": None},
        {
            "warning_triggered": False,
            "warning_window": 7,
            "warning_sample": 640,
        },
    ],
)
def test_inconsistent_alarm_flags_are_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="warning_triggered"):
        _seal(**overrides)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"detector": " "}, "detector"),
        ({"detector": 7}, "detector"),
        ({"observable": ""}, "observable"),
        ({"signal_source": ""}, "signal_source"),
        ({"captured_at": ""}, "captured_at"),
        ({"sampling_rate_hz": 0.0}, "sampling_rate_hz"),
        ({"sampling_rate_hz": -1.0}, "sampling_rate_hz"),
        ({"sampling_rate_hz": float("inf")}, "sampling_rate_hz"),
        ({"sampling_rate_hz": True}, "sampling_rate_hz"),
        ({"window": 0}, "window"),
        ({"window": 1.5}, "window"),
        ({"window": True}, "window"),
        ({"step": 0}, "step"),
        ({"persistence": 0}, "persistence"),
        ({"n_baseline_windows": 0}, "n_baseline_windows"),
        ({"warning_triggered": "yes"}, "warning_triggered"),
        (
            {
                "warning_triggered": True,
                "warning_window": -1,
                "warning_sample": 640,
            },
            "warning_window",
        ),
        (
            {
                "warning_triggered": True,
                "warning_window": 7,
                "warning_sample": True,
            },
            "warning_sample",
        ),
        ({"transition_onset_sample": -5}, "transition_onset_sample"),
        ({"transition_onset_sample": 1.5}, "transition_onset_sample"),
    ],
)
def test_invalid_fields_are_rejected(overrides: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _seal(**overrides)


def test_empty_indicators_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least one contribution"):
        _seal(indicators=())


def test_non_indicator_elements_are_rejected() -> None:
    with pytest.raises(ValueError, match="must be an EarlyWarningIndicator"):
        _seal(indicators=({"name": "variance"},))


def test_unknown_indicator_direction_is_rejected() -> None:
    with pytest.raises(ValueError, match="direction must be"):
        _seal(indicators=(_indicator(direction="sideways"),))


@pytest.mark.parametrize(
    "bad_field",
    ["robust_z", "baseline_median", "z_threshold"],
)
def test_non_finite_indicator_fields_are_rejected(bad_field: str) -> None:
    with pytest.raises(ValueError, match=bad_field):
        _seal(indicators=(_indicator(**{bad_field: float("nan")}),))


def test_content_hash_is_deterministic_and_detects_mutation() -> None:
    first = _seal()
    second = _seal()
    assert first.content_hash == second.content_hash
    mutated = _seal(observable="a_different_observable")
    assert mutated.content_hash != first.content_hash


def test_audit_record_round_trips_and_reseals_to_the_same_hash() -> None:
    evidence = _seal()
    record = evidence.to_audit_record()
    assert record["content_hash"] == evidence.content_hash
    assert record["verdict"] == EARLY_WARNING_FLAGGED
    assert isinstance(record["indicators"], list)
    payload = {k: v for k, v in record.items() if k != "content_hash"}
    assert canonical_record_hash(payload) == evidence.content_hash


def test_indicator_audit_record_carries_every_field() -> None:
    record = _indicator(name="variance").to_audit_record()
    assert record == {
        "name": "variance",
        "direction": RISE,
        "robust_z": 5.0,
        "baseline_median": 0.3,
        "z_threshold": 3.0,
        "breached": True,
    }


# --------------------------------------------------------------------------- #
# Pure helpers (white-box edge branches)                                       #
# --------------------------------------------------------------------------- #


def test_report_window_returns_the_alarm_window_when_triggered() -> None:
    headline = [0.0, 1.0, 9.0, 2.0]
    assert _report_window(headline, RISE, 1, warning_window=3) == 3


def test_report_window_picks_the_rising_peak_of_the_tail() -> None:
    headline = [0.0, 0.0, 1.0, 7.0, 3.0]
    assert _report_window(headline, RISE, 2, warning_window=None) == 3


def test_report_window_picks_the_falling_trough_of_the_tail() -> None:
    headline = [0.0, 0.0, -1.0, -8.0, -2.0]
    assert _report_window(headline, DROP, 2, warning_window=None) == 3


def test_report_window_is_none_when_the_baseline_covers_every_window() -> None:
    headline = [0.0, 1.0, 2.0]
    assert _report_window(headline, RISE, 3, warning_window=None) is None


def test_indicator_at_none_window_is_a_non_breaching_zero() -> None:
    indicator = _indicator_at("variance", RISE, [9.0], 0.2, 3.0, None)
    assert indicator.robust_z == 0.0
    assert indicator.breached is False


@pytest.mark.parametrize(
    ("direction", "z", "expected"),
    [
        (RISE, 4.0, True),
        (RISE, 2.0, False),
        (DROP, -4.0, True),
        (DROP, -2.0, False),
    ],
)
def test_indicator_at_breach_logic(direction: str, z: float, expected: bool) -> None:
    indicator = _indicator_at("x", direction, [z], 0.1, 3.0, 0)
    assert indicator.breached is expected
    assert indicator.robust_z == z


@pytest.mark.parametrize(
    ("onset", "sample", "expected"),
    [
        (None, 100, (None, None, False)),
        (200, None, (None, None, False)),
        (200, 100, (100, 100 / 32.0, True)),
        (100, 200, (-100, -100 / 32.0, False)),
    ],
)
def test_lead_covers_defined_and_undefined_cases(
    onset: int | None,
    sample: int | None,
    expected: tuple[int | None, float | None, bool],
) -> None:
    assert _lead(onset, sample, 32.0) == expected


# --------------------------------------------------------------------------- #
# Adapters                                                                     #
# --------------------------------------------------------------------------- #


def _rising_variance_signals(
    n_nodes: int = 4, length: int = 2000, seed: int = 0
) -> np.ndarray:
    """Return noise whose amplitude ramps up, so the window variance rises."""
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    amplitude = 1.0 + 6.0 * (t / length) ** 2
    return rng.standard_normal((n_nodes, length)) * amplitude


def _rising_coherence_phases(
    n_nodes: int = 8, length: int = 2000, seed: int = 0
) -> np.ndarray:
    """Return phases that lock over the second half, so coherence climbs."""
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    lock = np.clip((t - length // 2) / (length // 2), 0.0, 1.0)
    common = 0.01 * t
    phases = np.empty((n_nodes, length), dtype=np.float64)
    for node in range(n_nodes):
        individual = (
            rng.uniform(0.0, 2.0 * np.pi) + 0.02 * t + rng.standard_normal(length) * 0.3
        )
        phases[node] = (1.0 - lock) * individual + lock * common
    return phases


def test_seal_critical_slowing_down_alarm_pins_both_indicators() -> None:
    warning = critical_slowing_down_warning(_rising_variance_signals())
    assert warning.warning_triggered is True
    evidence = seal_critical_slowing_down_alarm(
        warning,
        observable="node_amplitude",
        signal_source="rising_variance_fixture",
        captured_at="fixture-event",
        sampling_rate_hz=32.0,
        transition_onset_sample=1900,
    )
    assert evidence.detector == "critical_slowing_down"
    assert [indicator.name for indicator in evidence.indicators] == [
        "variance",
        "lag1_autocorrelation",
    ]
    assert evidence.verdict == EARLY_WARNING_FLAGGED
    assert evidence.lead_samples == 1900 - warning.warning_sample
    assert any(indicator.breached for indicator in evidence.indicators)


def test_seal_critical_slowing_down_alarm_seals_a_negative_result() -> None:
    # A very high z-threshold guarantees no trigger yet leaves a non-empty tail,
    # exercising the argmax report-window branch and the NO_EARLY_WARNING verdict.
    warning = critical_slowing_down_warning(
        _rising_variance_signals(), z_threshold=200.0
    )
    assert warning.warning_triggered is False
    evidence = seal_critical_slowing_down_alarm(
        warning,
        observable="node_amplitude",
        signal_source="rising_variance_fixture",
        captured_at="fixture-event",
        sampling_rate_hz=32.0,
    )
    assert evidence.verdict == NO_EARLY_WARNING
    assert evidence.warning_sample is None
    assert len(evidence.indicators) == 2


def test_seal_critical_slowing_down_alarm_rejects_a_foreign_object() -> None:
    with pytest.raises(ValueError, match="CriticalSlowingDownWarning"):
        seal_critical_slowing_down_alarm(
            object(),  # type: ignore[arg-type]
            observable="x",
            signal_source="y",
            captured_at="z",
            sampling_rate_hz=1.0,
        )


def test_seal_synchronisation_alarm_pins_the_order_parameter() -> None:
    warning = synchronisation_warning(_rising_coherence_phases())
    assert warning.warning_triggered is True
    evidence = seal_synchronisation_alarm(
        warning,
        observable="cross_node_order_parameter",
        signal_source="rising_coherence_fixture",
        captured_at="fixture-event",
        sampling_rate_hz=32.0,
        transition_onset_sample=1900,
    )
    assert evidence.detector == "synchronisation"
    assert [indicator.name for indicator in evidence.indicators] == ["order_parameter"]
    assert evidence.indicators[0].direction == RISE
    assert evidence.verdict == EARLY_WARNING_FLAGGED


def test_seal_synchronisation_alarm_seals_a_negative_result() -> None:
    warning = synchronisation_warning(_rising_coherence_phases(), z_threshold=200.0)
    assert warning.warning_triggered is False
    evidence = seal_synchronisation_alarm(
        warning,
        observable="cross_node_order_parameter",
        signal_source="rising_coherence_fixture",
        captured_at="fixture-event",
        sampling_rate_hz=32.0,
    )
    assert evidence.verdict == NO_EARLY_WARNING
    assert len(evidence.indicators) == 1


def test_seal_synchronisation_alarm_rejects_a_foreign_object() -> None:
    with pytest.raises(ValueError, match="SynchronisationWarning"):
        seal_synchronisation_alarm(
            object(),  # type: ignore[arg-type]
            observable="x",
            signal_source="y",
            captured_at="z",
            sampling_rate_hz=1.0,
        )


def _entropy_warning(*, triggered: bool) -> ExplosiveSyncWarning:
    """Return a transition-entropy warning with a clear regularisation drop."""
    window_starts = np.arange(8, dtype=np.int64) * 16
    robust_z = np.array([0.0, 0.1, -0.1, 0.0, -1.0, -8.0, -2.0, -1.0])
    entropy_index = np.linspace(0.9, 0.4, 8)
    return ExplosiveSyncWarning(
        window_starts=window_starts,
        entropy_index=entropy_index,
        per_node_entropy=np.zeros((8, 3), dtype=np.float64),
        robust_z=robust_z,
        relative_drop=np.zeros(8, dtype=np.float64),
        baseline_median=0.85,
        baseline_scale=0.05,
        n_baseline_windows=4,
        warning_triggered=triggered,
        warning_window=5 if triggered else None,
        warning_sample=int(window_starts[5]) if triggered else None,
        dimension=3,
        delay=1,
        window=128,
        step=16,
        z_threshold=3.0,
        drop_threshold=0.1,
        persistence=2,
    )


def test_seal_transition_entropy_alarm_pins_the_entropy_drop() -> None:
    warning = _entropy_warning(triggered=True)
    evidence = seal_transition_entropy_alarm(
        warning,
        observable="per_channel_sin_phase",
        signal_source="chb01_03",
        captured_at="2996s-onset-event",
        sampling_rate_hz=32.0,
        transition_onset_sample=2048,
    )
    assert evidence.detector == "transition_entropy"
    assert evidence.indicators[0].name == "transition_entropy"
    assert evidence.indicators[0].direction == DROP
    assert evidence.indicators[0].robust_z == -8.0
    assert evidence.indicators[0].breached is True
    assert evidence.verdict == EARLY_WARNING_FLAGGED


def test_seal_transition_entropy_alarm_uses_the_trough_when_silent() -> None:
    warning = _entropy_warning(triggered=False)
    evidence = seal_transition_entropy_alarm(
        warning,
        observable="per_channel_sin_phase",
        signal_source="chb01_03",
        captured_at="2996s-onset-event",
        sampling_rate_hz=32.0,
    )
    assert evidence.verdict == NO_EARLY_WARNING
    # The reported window is the deepest post-baseline trough (index 5, z = -8).
    assert evidence.indicators[0].robust_z == -8.0
    assert evidence.indicators[0].breached is True


def test_seal_transition_entropy_alarm_rejects_a_foreign_object() -> None:
    with pytest.raises(ValueError, match="ExplosiveSyncWarning"):
        seal_transition_entropy_alarm(
            object(),  # type: ignore[arg-type]
            observable="x",
            signal_source="y",
            captured_at="z",
            sampling_rate_hz=1.0,
        )

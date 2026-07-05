# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — ensemble early-warning fusion tests

"""Tests for the ensemble early-warning fusion over the detector suite.

The suite exercises both fusion rules (the weighted-mean oriented-z gate and the
vote gate) on directly-built member evidence, the grid-alignment and control
validation, the pure helpers (sustained-run detection, report-window selection,
member snapshot), the three detector adapters driven end to end on a shared
window grid, and the seal of a fused alarm into an ``EarlyWarningEvidence``
record.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.assurance.early_warning_evidence import (
    EarlyWarningEvidence,
    seal_ensemble_alarm,
)
from scpn_phase_orchestrator.monitor.critical_slowing_down import (
    critical_slowing_down_warning,
)
from scpn_phase_orchestrator.monitor.ensemble_warning import (
    DROP,
    RISE,
    VOTE_RULE,
    WEIGHTED_RULE,
    EnsembleWarning,
    MemberContribution,
    MemberEvidence,
    _first_sustained_breach,
    _report_window,
    ensemble_warning,
    member_from_critical_slowing_down,
    member_from_synchronisation,
    member_from_transition_entropy,
)
from scpn_phase_orchestrator.monitor.explosive_sync import explosive_sync_warning
from scpn_phase_orchestrator.monitor.synchronisation import synchronisation_warning

_GRID = np.arange(8, dtype=np.int64) * 16


def _member(
    *,
    name: str = "member",
    direction: str = RISE,
    window_starts: np.ndarray | None = None,
    oriented_z: list[float],
    native_robust_z: list[float] | None = None,
    breaches: list[bool] | None = None,
    baseline_median: float = 0.2,
    z_threshold: float = 3.0,
    n_baseline_windows: int = 2,
) -> MemberEvidence:
    """Build member evidence directly for precise fusion-logic tests."""
    oriented = np.asarray(oriented_z, dtype=np.float64)
    native = (
        oriented if native_robust_z is None else np.asarray(native_robust_z, np.float64)
    )
    gate = (
        np.zeros(oriented.shape[0], dtype=bool)
        if breaches is None
        else np.asarray(breaches, dtype=bool)
    )
    return MemberEvidence(
        name=name,
        native_direction=direction,
        window_starts=_GRID if window_starts is None else window_starts,
        oriented_z=oriented,
        native_robust_z=native,
        breaches=gate,
        baseline_median=baseline_median,
        z_threshold=z_threshold,
        n_baseline_windows=n_baseline_windows,
    )


# --------------------------------------------------------------------------- #
# Fusion rules                                                                 #
# --------------------------------------------------------------------------- #


def test_weighted_rule_alarms_on_the_combined_evidence() -> None:
    a = _member(name="a", oriented_z=[0, 0, 0, 4, 5, 6, 5, 4])
    b = _member(name="b", oriented_z=[0, 0, 0, 2, 2, 2, 2, 2])
    result = ensemble_warning([a, b], rule=WEIGHTED_RULE, fused_threshold=3.0)
    assert result.rule == WEIGHTED_RULE
    assert result.warning_triggered is True
    assert result.warning_window == 3
    assert result.warning_sample == int(_GRID[3])
    # fused[3] = mean(4, 2) = 3.0, exactly on the gate.
    assert result.fused_score[3] == pytest.approx(3.0)
    assert result.member_names == ("a", "b")


def test_weighted_rule_honours_unequal_weights() -> None:
    a = _member(name="a", oriented_z=[0, 0, 0, 4, 4, 4, 4, 4])
    b = _member(name="b", oriented_z=[0, 0, 0, 0, 0, 0, 0, 0])
    # Equal weights give fused 2.0 (< 3) and no alarm; weighting a up alarms.
    quiet = ensemble_warning([a, b], fused_threshold=3.0)
    assert quiet.warning_triggered is False
    loud = ensemble_warning([a, b], weights=[3.0, 1.0], fused_threshold=3.0)
    assert loud.warning_triggered is True
    assert loud.fused_score[3] == pytest.approx(3.0)


def test_vote_rule_requires_a_quorum_of_members() -> None:
    a = _member(
        name="a",
        oriented_z=[0.0] * 8,
        breaches=[False, False, False, True, True, True, True, True],
    )
    b = _member(
        name="b",
        oriented_z=[0.0] * 8,
        breaches=[False, False, False, False, True, True, False, False],
    )
    result = ensemble_warning([a, b], rule=VOTE_RULE, min_votes=2, persistence=2)
    assert result.rule == VOTE_RULE
    assert list(result.vote_count) == [0, 0, 0, 1, 2, 2, 1, 1]
    assert result.warning_triggered is True
    assert result.warning_window == 4


def test_vote_rule_is_silent_below_quorum() -> None:
    a = _member(name="a", oriented_z=[0.0] * 8, breaches=[False] * 3 + [True] * 5)
    b = _member(name="b", oriented_z=[0.0] * 8, breaches=[False] * 8)
    result = ensemble_warning([a, b], rule=VOTE_RULE, min_votes=2)
    assert result.warning_triggered is False
    assert result.warning_sample is None


def test_untriggered_report_window_is_the_strongest_fused_approach() -> None:
    a = _member(
        name="a",
        oriented_z=[0, 0, 0, 1, 2, 1, 0, 0],
        native_robust_z=[0, 0, 0, 1, 9, 1, 0, 0],
    )
    b = _member(name="b", oriented_z=[0, 0, 0, 1, 1, 1, 0, 0])
    result = ensemble_warning([a, b], fused_threshold=50.0)
    assert result.warning_triggered is False
    # Strongest fused window is index 4; member a's native z there is pinned.
    contribution = {c.name: c for c in result.contributions}["a"]
    assert contribution.robust_z == 9.0


def test_baseline_covering_every_window_pins_zero_contributions() -> None:
    a = _member(name="a", oriented_z=[5.0] * 8, n_baseline_windows=8)
    b = _member(name="b", oriented_z=[5.0] * 8, n_baseline_windows=8)
    result = ensemble_warning([a, b], fused_threshold=1.0)
    assert result.warning_triggered is False
    assert all(c.robust_z == 0.0 and c.breached is False for c in result.contributions)


def test_fused_baseline_is_the_widest_member_baseline() -> None:
    a = _member(name="a", oriented_z=[9.0] * 8, n_baseline_windows=2)
    b = _member(name="b", oriented_z=[9.0] * 8, n_baseline_windows=5)
    result = ensemble_warning([a, b], fused_threshold=1.0, persistence=2)
    assert result.n_baseline_windows == 5
    # No window before index 5 may alarm despite the high fused score.
    assert result.warning_window == 5


# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("members", ["not-a-list", 7])
def test_non_sequence_members_are_rejected(members: object) -> None:
    with pytest.raises(ValueError, match="list or tuple"):
        ensemble_warning(members)  # type: ignore[arg-type]


def test_empty_members_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ensemble_warning([])


def test_non_member_elements_are_rejected() -> None:
    with pytest.raises(ValueError, match="must be a MemberEvidence"):
        ensemble_warning([{"name": "a"}])  # type: ignore[list-item]


def test_misaligned_window_grids_are_rejected() -> None:
    a = _member(name="a", oriented_z=[0.0] * 8)
    b = _member(
        name="b", oriented_z=[0.0] * 6, window_starts=np.arange(6, dtype=np.int64)
    )
    with pytest.raises(ValueError, match="window grid does not match"):
        ensemble_warning([a, b])


def test_unknown_rule_is_rejected() -> None:
    with pytest.raises(ValueError, match="rule must be"):
        ensemble_warning([_member(oriented_z=[0.0] * 8)], rule="mean")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"fused_threshold": -1.0}, "fused_threshold"),
        ({"fused_threshold": "x"}, "fused_threshold"),
        ({"persistence": 0}, "persistence"),
        ({"persistence": True}, "persistence"),
        ({"min_votes": 0}, "min_votes"),
        ({"min_votes": 3}, "exceeds the member count"),
    ],
)
def test_out_of_range_controls_are_rejected(
    kwargs: dict[str, object], match: str
) -> None:
    members = [
        _member(name="a", oriented_z=[0.0] * 8),
        _member(name="b", oriented_z=[0.0] * 8),
    ]
    with pytest.raises(ValueError, match=match):
        ensemble_warning(members, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("weights", "match"),
    [
        ("heavy", "list or tuple"),
        ([1.0], "one entry per member"),
        ([1.0, 0.0], r"weights\[1\]"),
        ([1.0, -2.0], r"weights\[1\]"),
        ([1.0, "x"], r"weights\[1\]"),
    ],
)
def test_malformed_weights_are_rejected(weights: object, match: str) -> None:
    members = [
        _member(name="a", oriented_z=[0.0] * 8),
        _member(name="b", oriented_z=[0.0] * 8),
    ]
    with pytest.raises(ValueError, match=match):
        ensemble_warning(members, weights=weights)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Pure helpers                                                                 #
# --------------------------------------------------------------------------- #


def test_first_sustained_breach_skips_isolated_breaches() -> None:
    breaches = np.array([False, True, False, True, True], dtype=bool)
    assert _first_sustained_breach(breaches, 2) == 3


def test_first_sustained_breach_returns_none_without_a_run() -> None:
    assert (
        _first_sustained_breach(np.array([False, True, False], dtype=bool), 2) is None
    )


def test_report_window_returns_none_when_baseline_covers_all() -> None:
    assert _report_window(np.array([1.0, 2.0, 3.0]), 3, warning_window=None) is None


def test_report_window_prefers_the_alarm_window() -> None:
    assert _report_window(np.array([0.0, 9.0, 1.0]), 0, warning_window=1) == 1


# --------------------------------------------------------------------------- #
# Adapters + seal (end to end on a shared grid)                                #
# --------------------------------------------------------------------------- #


def _rising_variance_signals(
    n_nodes: int = 4, length: int = 2000, seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    amplitude = 1.0 + 6.0 * (t / length) ** 2
    return rng.standard_normal((n_nodes, length)) * amplitude


def _rising_coherence_phases(
    n_nodes: int = 8, length: int = 2000, seed: int = 1
) -> np.ndarray:
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


def _suite_members(length: int = 640) -> list[MemberEvidence]:
    """Build the three adapter members on one window=128, step=16 grid."""
    csd = member_from_critical_slowing_down(
        critical_slowing_down_warning(
            _rising_variance_signals(length=length), window=128, step=16
        )
    )
    sync = member_from_synchronisation(
        synchronisation_warning(
            _rising_coherence_phases(length=length), window=128, step=16
        )
    )
    entropy = member_from_transition_entropy(
        explosive_sync_warning(
            _rising_variance_signals(length=length), window=128, step=16
        )
    )
    return [csd, sync, entropy]


def test_adapters_orient_each_member_and_align_on_one_grid() -> None:
    csd, sync, entropy = _suite_members()
    assert csd.name == "critical_slowing_down" and csd.native_direction == RISE
    assert sync.name == "synchronisation" and sync.native_direction == RISE
    assert entropy.name == "transition_entropy" and entropy.native_direction == DROP
    # The entropy member warns on a drop, so its oriented z negates the native z.
    assert np.allclose(entropy.oriented_z, -entropy.native_robust_z)
    assert np.array_equal(csd.window_starts, sync.window_starts)
    assert np.array_equal(csd.window_starts, entropy.window_starts)


def test_transition_entropy_member_reconstructs_the_detector_gate() -> None:
    warning = explosive_sync_warning(
        _rising_variance_signals(length=640), window=128, step=16
    )
    member = member_from_transition_entropy(warning)
    n = warning.window_starts.shape[0]
    expected = (
        (np.arange(n) >= warning.n_baseline_windows)
        & (warning.robust_z <= -warning.z_threshold)
        & (warning.relative_drop >= warning.drop_threshold)
    )
    assert np.array_equal(member.breaches, expected)


def test_fused_suite_alarm_seals_into_evidence() -> None:
    members = _suite_members()
    ensemble = ensemble_warning(members, rule=WEIGHTED_RULE, fused_threshold=3.0)
    evidence = seal_ensemble_alarm(
        ensemble,
        observable="suite_multichannel",
        signal_source="rising_fixture",
        captured_at="fixture-event",
        sampling_rate_hz=32.0,
        window=128,
        step=16,
        transition_onset_sample=1900,
    )
    assert isinstance(evidence, EarlyWarningEvidence)
    assert evidence.detector == "ensemble_weighted"
    assert {i.name for i in evidence.indicators} == {
        "critical_slowing_down",
        "synchronisation",
        "transition_entropy",
    }
    assert len(evidence.content_hash) == 64


@pytest.mark.parametrize(
    ("adapter", "match"),
    [
        (member_from_critical_slowing_down, "CriticalSlowingDownWarning"),
        (member_from_synchronisation, "SynchronisationWarning"),
        (member_from_transition_entropy, "ExplosiveSyncWarning"),
    ],
)
def test_adapters_reject_a_foreign_object(adapter: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        adapter(object())  # type: ignore[operator]


def test_seal_ensemble_alarm_rejects_a_foreign_object() -> None:
    with pytest.raises(ValueError, match="EnsembleWarning"):
        seal_ensemble_alarm(
            object(),  # type: ignore[arg-type]
            observable="x",
            signal_source="y",
            captured_at="z",
            sampling_rate_hz=1.0,
            window=128,
            step=16,
        )


# --------------------------------------------------------------------------- #
# Summary                                                                      #
# --------------------------------------------------------------------------- #


def test_summary_reports_the_fusion_verdict() -> None:
    result = ensemble_warning(
        [
            _member(name="a", oriented_z=[0, 0, 0, 4, 5, 6, 5, 4]),
            _member(name="b", oriented_z=[0, 0, 0, 2, 2, 2, 2, 2]),
        ],
        fused_threshold=3.0,
    )
    summary = result.summary()
    assert summary["rule"] == WEIGHTED_RULE
    assert summary["n_windows"] == 8
    assert summary["warning_triggered"] is True
    assert summary["max_fused_score"] == pytest.approx(float(result.fused_score.max()))


def test_summary_handles_an_empty_result() -> None:
    empty = EnsembleWarning(
        window_starts=np.empty(0, dtype=np.int64),
        fused_score=np.empty(0, dtype=np.float64),
        vote_count=np.empty(0, dtype=np.int64),
        rule=WEIGHTED_RULE,
        fused_threshold=3.0,
        min_votes=2,
        persistence=2,
        n_baseline_windows=0,
        member_names=(),
        contributions=(),
        warning_triggered=False,
        warning_window=None,
        warning_sample=None,
    )
    summary = empty.summary()
    assert summary["max_fused_score"] == 0.0
    assert summary["max_vote_count"] == 0


def test_member_contribution_is_frozen() -> None:
    contribution = MemberContribution(
        name="a",
        direction=RISE,
        robust_z=1.0,
        baseline_median=0.2,
        z_threshold=3.0,
        breached=False,
    )
    with pytest.raises((AttributeError, TypeError)):
        contribution.robust_z = 2.0  # type: ignore[misc]

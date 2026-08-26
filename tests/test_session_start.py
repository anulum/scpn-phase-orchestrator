# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Session-start coherence gate tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.imprint.state import ImprintState
from scpn_phase_orchestrator.monitor.session_start import (
    SessionCoherenceReport,
    check_session_start,
)
from scpn_phase_orchestrator.oscillators.base import PhaseState

TWO_PI = 2.0 * np.pi


def _make_states(n: int, quality: float = 0.8, channel: str = "P") -> list[PhaseState]:
    return [
        PhaseState(
            theta=float(i) * TWO_PI / n,
            omega=1.0,
            amplitude=1.0,
            quality=quality,
            channel=channel,
            node_id=f"{channel}_{i}",
        )
        for i in range(n)
    ]


def test_healthy_session_passes():
    n = 10
    states = _make_states(n, quality=0.8, channel="P")
    states += _make_states(n, quality=0.7, channel="I")
    states += _make_states(n, quality=0.6, channel="S")
    phases = np.linspace(0, 0.1, n)  # near-synchronised
    imprint = ImprintState(m_k=np.full(n, 0.2), last_update=100.0)

    report = check_session_start(states, phases, imprint, n)
    assert report.passed
    assert not report.errors
    assert report.quality_scores["P"] > 0.5
    assert report.initial_r > 0.9
    assert abs(report.imprint_level - 0.2) < 1e-6


def test_low_quality_warns():
    n = 10
    states = _make_states(n, quality=0.1, channel="P")
    phases = np.linspace(0, TWO_PI, n)
    imprint = ImprintState(m_k=np.zeros(n), last_update=0.0)

    report = check_session_start(states, phases, imprint, n)
    assert report.passed  # low quality is warning, not error
    assert any("low quality" in w for w in report.warnings)


def test_signal_collapse_fails():
    n = 10
    states = _make_states(n, quality=0.05, channel="P")
    phases = np.zeros(n)
    imprint = ImprintState(m_k=np.zeros(n), last_update=0.0)

    report = check_session_start(states, phases, imprint, n)
    assert not report.passed
    assert any("collapse" in e.lower() for e in report.errors)


def test_imprint_size_mismatch_fails():
    n = 10
    states = _make_states(n, quality=0.8)
    phases = np.zeros(n)
    bad_imprint = ImprintState(m_k=np.zeros(5), last_update=0.0)

    report = check_session_start(states, phases, bad_imprint, n)
    assert not report.passed
    assert any("mismatch" in e.lower() for e in report.errors)


def test_fresh_imprint_passes():
    n = 10
    states = _make_states(n, quality=0.8)
    phases = np.linspace(0, 0.1, n)
    fresh = ImprintState(m_k=np.zeros(n), last_update=0.0)

    report = check_session_start(states, phases, fresh, n)
    assert report.passed
    assert report.imprint_level == 0.0


def test_report_dataclass_defaults():
    r = SessionCoherenceReport()
    assert r.passed is True
    assert r.errors == []
    assert r.warnings == []
    assert r.quality_scores == {}


def test_multi_channel_quality_scores():
    n = 5
    p_states = _make_states(n, quality=0.9, channel="P")
    i_states = _make_states(n, quality=0.5, channel="I")
    s_states = _make_states(n, quality=0.3, channel="S")
    phases = np.zeros(n)
    imprint = ImprintState(m_k=np.zeros(n), last_update=0.0)

    report = check_session_start(p_states + i_states + s_states, phases, imprint, n)
    assert "P" in report.quality_scores
    assert "I" in report.quality_scores
    assert "S" in report.quality_scores
    assert report.quality_scores["P"] > report.quality_scores["S"]


def test_named_extension_channel_quality_score():
    n = 5
    states = _make_states(n, quality=0.8, channel="Q")
    phases = np.zeros(n)
    imprint = ImprintState(m_k=np.zeros(n), last_update=0.0)

    report = check_session_start(states, phases, imprint, n)
    assert report.passed
    assert report.quality_scores["Q"] == 0.8


def test_low_initial_coherence_warns():
    n = 10
    states = _make_states(n, quality=0.8)
    random_phases = np.linspace(0.0, TWO_PI, n, endpoint=False)
    imprint = ImprintState(m_k=np.zeros(n), last_update=0.0)

    report = check_session_start(states, random_phases, imprint, n)
    assert report.initial_r < 0.05
    assert any("Low initial coherence" in warning for warning in report.warnings)


def test_initial_phase_shape_mismatch_fails():
    """A wrong-sized engine seed must fail the gate, mirroring the imprint check."""
    n = 6
    states = _make_states(n, quality=0.8)
    imprint = ImprintState(m_k=np.full(n, 0.3), last_update=10.0)

    report = check_session_start(states, np.zeros(n - 1), imprint, n)

    assert not report.passed
    assert report.initial_r == 0.0
    assert report.imprint_level == 0.3
    assert any("Initial phase size mismatch" in e for e in report.errors)


def test_initial_phases_non_ndarray_fails():
    n = 4
    states = _make_states(n, quality=0.8)
    imprint = ImprintState(m_k=np.zeros(n), last_update=0.0)

    report = check_session_start(states, [0.0, 0.1, 0.2, 0.3], imprint, n)  # type: ignore[arg-type]

    assert not report.passed
    assert report.initial_r == 0.0
    assert any("initial_phases" in e and "numpy array" in e for e in report.errors)


def test_initial_phases_wrong_ndim_fails():
    n = 4
    states = _make_states(n, quality=0.8)
    imprint = ImprintState(m_k=np.zeros(n), last_update=0.0)

    report = check_session_start(states, np.zeros((n, 2)), imprint, n)

    assert not report.passed
    assert report.initial_r == 0.0
    assert any("one-dimensional" in e for e in report.errors)


def test_initial_phases_rejected_dtypes_fail():
    n = 4
    states = _make_states(n, quality=0.8)
    imprint = ImprintState(m_k=np.zeros(n), last_update=0.0)

    for bad in (
        np.zeros(n, dtype=bool),
        np.zeros(n, dtype=complex),
        np.array(["0.0", "0.1", "0.2", "0.3"]),
        np.array([0.0, 0.1, 0.2, object()], dtype=object),
    ):
        report = check_session_start(states, bad, imprint, n)
        assert not report.passed
        assert report.initial_r == 0.0
        assert any("dtype" in e for e in report.errors)


def test_initial_phases_non_finite_fails():
    n = 4
    states = _make_states(n, quality=0.8)
    imprint = ImprintState(m_k=np.zeros(n), last_update=0.0)

    for poison in (np.nan, np.inf, -np.inf):
        phases = np.array([0.0, 0.1, poison, 0.3])
        report = check_session_start(states, phases, imprint, n)
        assert not report.passed
        assert report.initial_r == 0.0
        assert any("finite" in e for e in report.errors)


def test_initial_phases_integer_dtype_accepted():
    """Exact integer radians are valid evidence; conversion must be lossless."""
    n = 4
    states = _make_states(n, quality=0.8)
    imprint = ImprintState(m_k=np.zeros(n), last_update=0.0)

    report = check_session_start(states, np.zeros(n, dtype=np.int64), imprint, n)

    assert report.passed
    assert report.initial_r > 0.99


def test_imprint_non_finite_fails():
    n = 4
    states = _make_states(n, quality=0.8)
    phases = np.zeros(n)
    imprint = ImprintState(m_k=np.array([0.1, np.nan, 0.2, 0.3]), last_update=0.0)

    report = check_session_start(states, phases, imprint, n)

    assert not report.passed
    assert report.imprint_level == 0.0
    assert any("imprint" in e.lower() and "finite" in e for e in report.errors)


def test_imprint_wrong_ndim_fails():
    n = 4
    states = _make_states(n, quality=0.8)
    phases = np.zeros(n)
    imprint = ImprintState(m_k=np.zeros((n, 2)), last_update=0.0)

    report = check_session_start(states, phases, imprint, n)

    assert not report.passed
    assert report.imprint_level == 0.0
    assert any("one-dimensional" in e for e in report.errors)


def test_imprint_rejected_dtype_fails():
    n = 4
    states = _make_states(n, quality=0.8)
    phases = np.zeros(n)
    imprint = ImprintState(m_k=np.zeros(n, dtype=bool), last_update=0.0)

    report = check_session_start(states, phases, imprint, n)

    assert not report.passed
    assert report.imprint_level == 0.0
    assert any("dtype" in e for e in report.errors)


def test_quality_non_finite_fails():
    n = 4
    states = _make_states(n, quality=0.8)
    states[1].quality = float("nan")
    phases = np.zeros(n)
    imprint = ImprintState(m_k=np.zeros(n), last_update=0.0)

    report = check_session_start(states, phases, imprint, n)

    assert not report.passed
    assert report.quality_scores == {}
    assert any("quality" in e for e in report.errors)


def test_quality_out_of_range_fails():
    n = 4
    phases = np.zeros(n)
    imprint = ImprintState(m_k=np.zeros(n), last_update=0.0)

    for bad in (1.5, -0.1):
        states = _make_states(n, quality=0.8)
        states[0].quality = bad
        report = check_session_start(states, phases, imprint, n)
        assert not report.passed
        assert report.quality_scores == {}
        assert any("quality" in e for e in report.errors)


def test_quality_non_float_fails():
    n = 4
    phases = np.zeros(n)
    imprint = ImprintState(m_k=np.zeros(n), last_update=0.0)

    for bad in (True, "0.8", None):
        states = _make_states(n, quality=0.8)
        states[2].quality = bad  # type: ignore[assignment]
        report = check_session_start(states, phases, imprint, n)
        assert not report.passed
        assert report.quality_scores == {}
        assert any("quality" in e for e in report.errors)


def test_n_osc_bool_raises():
    import pytest

    imprint = ImprintState(m_k=np.zeros(1), last_update=0.0)
    with pytest.raises(TypeError):
        check_session_start([], np.zeros(1), imprint, True)  # type: ignore[arg-type]


def test_n_osc_non_int_raises():
    import pytest

    imprint = ImprintState(m_k=np.zeros(4), last_update=0.0)
    with pytest.raises(TypeError):
        check_session_start([], np.zeros(4), imprint, 4.0)  # type: ignore[arg-type]


def test_n_osc_non_positive_raises():
    import pytest

    imprint = ImprintState(m_k=np.zeros(4), last_update=0.0)
    for bad in (0, -3):
        with pytest.raises(ValueError):
            check_session_start([], np.zeros(4), imprint, bad)


def test_empty_phase_states_does_not_block_shape_validation():
    n = 4
    phases = np.arange(n, dtype=float)
    imprint = ImprintState(m_k=np.zeros(n), last_update=0.0)

    report = check_session_start([], phases, imprint, n)

    assert not report.passed
    assert report.quality_scores == {}
    assert any("Signal collapse" in message for message in report.errors)


class TestSessionStartPipelineWiring:
    """Pipeline: extraction → session check → engine initialisation."""

    def test_session_start_gates_engine_run(self):
        """check_session_start → report.passed gates whether engine runs.
        Proves session start is wired as a pipeline gate."""
        n = 6
        states = _make_states(n, quality=0.9, channel="P")
        phases = np.linspace(0, 0.1, n)
        imprint = ImprintState(m_k=np.zeros(n), last_update=0.0)

        report = check_session_start(states, phases, imprint, n)
        assert report.passed, "Good extraction should pass session gate"
        assert report.initial_r > 0.9, "Clustered phases → high R"

        # Only if passed: run engine
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import (
            compute_order_parameter,
        )

        eng = UPDEEngine(n, dt=0.01)
        omegas = np.ones(n)
        knm = 0.5 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        for _ in range(100):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, np.zeros((n, n)))
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0

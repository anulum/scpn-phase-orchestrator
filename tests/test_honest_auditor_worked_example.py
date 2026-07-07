# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — worked auditor example tests

from __future__ import annotations

import numpy as np

from bench.honest_auditor_worked_example import (
    EVENT_AR1,
    WINDOW,
    audit_both,
    build_corpus,
    lag1_autocorrelation_score,
    main,
    window_mean_score,
)


class TestBuildCorpus:
    def test_arm_sizes_and_zero_mean(self):
        events, nulls = build_corpus(seed=0)
        assert len(events) == 40
        assert len(nulls) == 200
        # Both arms are constructed zero-mean per window.
        assert all(abs(float(np.mean(window))) < 1e-9 for window in events)
        assert all(window.shape == (WINDOW,) for window in events)


class TestDetectorScores:
    def test_autocorrelation_separates_event_from_null(self):
        events, nulls = build_corpus(seed=0)
        event_ac = np.mean([lag1_autocorrelation_score(w) for w in events])
        null_ac = np.mean([lag1_autocorrelation_score(w) for w in nulls])
        # AR(1) events carry the rising autocorrelation the nulls lack.
        assert event_ac > null_ac
        assert event_ac > 0.3

    def test_constant_window_autocorrelation_is_zero(self):
        # Guards the zero-variance branch of the score.
        assert lag1_autocorrelation_score(np.zeros(WINDOW)) == 0.0

    def test_window_mean_score_matches_numpy(self):
        window = np.array([1.0, 2.0, 3.0])
        assert window_mean_score(window) == 2.0


class TestAuditBoth:
    def test_skilful_beats_chance_control_does_not(self):
        audits = audit_both(seed=0)
        skilful = audits["lag1-autocorrelation"]
        control = audits["window-mean-control"]
        assert skilful.beats_chance is True
        assert skilful.p_value < 0.01
        assert control.beats_chance is False
        assert control.p_value > 0.05
        # The event AR(1) coefficient drives the separation the control cannot see.
        assert EVENT_AR1 > 0.0


class TestMain:
    def test_main_runs_and_prints(self, capsys):
        main()
        captured = capsys.readouterr().out
        assert "lag1-autocorrelation" in captured
        assert "window-mean-control" in captured
        assert "beats_chance=True" in captured
        assert "beats_chance=False" in captured

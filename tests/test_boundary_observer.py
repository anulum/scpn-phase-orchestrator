# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Boundary observer tests

from __future__ import annotations

from scpn_phase_orchestrator.binding.types import BoundaryDef
from scpn_phase_orchestrator.monitor.boundaries import BoundaryObserver


def _defs():
    return [
        BoundaryDef(
            name="R_floor",
            variable="R",
            lower=0.2,
            upper=None,
            severity="soft",
        ),
        BoundaryDef(
            name="T_cap",
            variable="T",
            lower=None,
            upper=100.0,
            severity="hard",
        ),
        BoundaryDef(
            name="P_band",
            variable="P",
            lower=1.0,
            upper=10.0,
            severity="soft",
        ),
    ]


def test_lower_bound_soft_violation():
    obs = BoundaryObserver(_defs())
    state = obs.observe({"R": 0.1, "T": 50.0, "P": 5.0})
    assert len(state.violations) == 1
    assert len(state.soft_violations) == 1
    assert state.hard_violations == []


def test_upper_bound_hard_violation():
    obs = BoundaryObserver(_defs())
    state = obs.observe({"R": 0.5, "T": 200.0, "P": 5.0})
    assert len(state.violations) == 1
    assert len(state.hard_violations) == 1
    assert state.soft_violations == []


def test_missing_variable_skipped():
    obs = BoundaryObserver(_defs())
    state = obs.observe({"R": 0.5})
    assert state.violations == []


def test_no_violation():
    obs = BoundaryObserver(_defs())
    state = obs.observe({"R": 0.5, "T": 50.0, "P": 5.0})
    assert state.violations == []
    assert state.soft_violations == []
    assert state.hard_violations == []


def test_unknown_severity_defaults_to_hard(caplog):
    defs = [
        BoundaryDef(
            name="X_check",
            variable="X",
            lower=0.0,
            upper=None,
            severity="medium",
        ),
    ]
    obs = BoundaryObserver(defs)
    import logging

    log_name = "scpn_phase_orchestrator.monitor.boundaries"
    with caplog.at_level(logging.WARNING, logger=log_name):
        state = obs.observe({"X": -1.0})
    assert len(state.hard_violations) == 1
    assert "unknown severity" in caplog.text

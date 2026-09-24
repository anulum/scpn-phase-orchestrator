# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — deterministic loop budget and GC-state contracts

"""Budget flags are typed, periods resolvable, and caller GC state survives."""

from __future__ import annotations

import gc

import numpy as np
import pytest

from scpn_phase_orchestrator.runtime.deterministic import (
    DeadlineBudget,
    run_deterministic_loop,
)


@pytest.mark.parametrize("flag", ["False", "", 0, 1, None])
def test_freeze_gc_must_be_a_bool(flag: object) -> None:
    with pytest.raises(ValueError, match="freeze_gc must be a bool"):
        DeadlineBudget(period_s=1e-3, freeze_gc=flag)  # type: ignore[arg-type]


@pytest.mark.parametrize("flag", [False, np.bool_(False)])
def test_false_flag_leaves_gc_running_and_reports_a_bool(flag: object) -> None:
    budget = DeadlineBudget(
        period_s=1e-3,
        freeze_gc=flag,  # type: ignore[arg-type]
        miss_policy="observe",
    )
    enabled_during_steps: list[bool] = []
    was_enabled = gc.isenabled()
    report = run_deterministic_loop(
        lambda _i: enabled_during_steps.append(gc.isenabled()),
        steps=2,
        budget=budget,
    )
    assert enabled_during_steps == [was_enabled, was_enabled]
    assert report.summary()["gc_frozen"] is False


@pytest.mark.parametrize("period_s", [1e-10, 4.9e-10])
def test_sub_nanosecond_period_is_refused(period_s: float) -> None:
    with pytest.raises(ValueError, match="at least 1 ns"):
        DeadlineBudget(period_s=period_s)


def test_one_nanosecond_period_is_accepted() -> None:
    assert DeadlineBudget(period_s=1e-9).period_s == 1e-9


def test_loop_keeps_the_callers_frozen_generation_frozen() -> None:
    held = [[index] for index in range(2_000)]
    gc.collect()
    gc.freeze()
    try:
        frozen_before = gc.get_freeze_count()
        assert frozen_before > 0
        run_deterministic_loop(
            lambda _i: None,
            steps=2,
            budget=DeadlineBudget(period_s=1e-3, miss_policy="observe"),
        )
        assert gc.get_freeze_count() >= frozen_before
    finally:
        gc.unfreeze()
    assert len(held) == 2_000


def test_loop_thaws_what_it_froze_when_nothing_was_frozen() -> None:
    gc.unfreeze()
    assert gc.get_freeze_count() == 0
    frozen_during: list[int] = []
    run_deterministic_loop(
        lambda _i: frozen_during.append(gc.get_freeze_count()),
        steps=1,
        budget=DeadlineBudget(period_s=1e-3, miss_policy="observe"),
    )
    assert frozen_during[0] > 0
    assert gc.get_freeze_count() == 0

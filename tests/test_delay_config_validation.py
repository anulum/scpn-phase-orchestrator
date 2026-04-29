# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — delay constructor validation tests

from __future__ import annotations

import pytest

from scpn_phase_orchestrator.upde.delay import DelayBuffer, DelayedEngine


@pytest.mark.parametrize("n_oscillators", [False, 0, -1, 1.5, "4"])
def test_delay_buffer_rejects_invalid_oscillator_count(n_oscillators: object) -> None:
    with pytest.raises(ValueError, match="n_oscillators"):
        DelayBuffer(n_oscillators=n_oscillators, max_delay_steps=3)


@pytest.mark.parametrize("max_delay_steps", [True, 0, -1, 1.5, "3"])
def test_delay_buffer_rejects_invalid_max_delay(max_delay_steps: object) -> None:
    with pytest.raises(ValueError, match="max_delay_steps"):
        DelayBuffer(n_oscillators=4, max_delay_steps=max_delay_steps)


@pytest.mark.parametrize("n_oscillators", [True, 0, -1, 1.5, "4"])
def test_delayed_engine_rejects_invalid_oscillator_count(n_oscillators: object) -> None:
    with pytest.raises(ValueError, match="n_oscillators"):
        DelayedEngine(n_oscillators=n_oscillators, dt=0.01)


@pytest.mark.parametrize("dt", [False, 0.0, -0.01, float("nan"), float("inf"), "0.01"])
def test_delayed_engine_rejects_invalid_dt(dt: object) -> None:
    with pytest.raises(ValueError, match="dt"):
        DelayedEngine(n_oscillators=4, dt=dt)


@pytest.mark.parametrize("delay_steps", [False, 0, -1, 1.5, "3"])
def test_delayed_engine_rejects_invalid_delay_steps(delay_steps: object) -> None:
    with pytest.raises(ValueError, match="delay_steps"):
        DelayedEngine(n_oscillators=4, dt=0.01, delay_steps=delay_steps)

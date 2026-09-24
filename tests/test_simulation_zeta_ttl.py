# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — zeta actions last their TTL, then the baseline returns

"""A zeta control action is an offset that expires back to the spec's drive.

At expiry ``simulate`` set ``zeta = 0.0``, dropping the spec's non-zero
baseline drive for the rest of the run; ``int(ttl / dt)`` made a TTL shorter
than one step (or zero) never expire and applied every other TTL one step
short.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from scpn_phase_orchestrator.binding import load_binding_spec
from scpn_phase_orchestrator.runtime.simulation import _ttl_steps, simulate

_MINIMAL = (
    Path(__file__).resolve().parents[1]
    / "domainpacks"
    / "minimal_domain"
    / "binding_spec.yaml"
)
_POLICY = """
rules:
  - name: one_shot_drive
    regime: [NOMINAL, DEGRADED, RECOVERY, CRITICAL]
    condition:
      metric: R_good
      layer: 0
      op: ">"
      threshold: -1.0
    max_fires: 1
    action:
      knob: zeta
      scope: global
      value: 0.1
      ttl_s: {ttl}
      justification: one-shot drive probe
"""


def _zeta_trajectory(tmp_path: Path, ttl_s: float, steps: int = 20) -> list[float]:
    spec_path = tmp_path / "binding_spec.yaml"
    shutil.copyfile(_MINIMAL, spec_path)
    (tmp_path / "policy.yaml").write_text(_POLICY.format(ttl=ttl_s), encoding="utf-8")
    seen: list[float] = []
    simulate(
        load_binding_spec(spec_path),
        steps=steps,
        seed=0,
        policy_enabled=True,
        binding_spec_path=spec_path,
        scenario_hook=lambda context: seen.append(round(float(context.zeta), 6)),
    )
    return seen


def test_expired_drive_returns_to_the_spec_baseline(tmp_path: Path) -> None:
    zetas = _zeta_trajectory(tmp_path, 0.05)
    baseline = zetas[0]
    assert baseline == pytest.approx(0.02)
    # fired after step 0, so steps 1..5 integrate with the offset, then baseline
    assert zetas[1:6] == pytest.approx([baseline + 0.1] * 5)
    assert zetas[6:] == pytest.approx([baseline] * (len(zetas) - 6))


def test_zero_ttl_action_does_not_persist(tmp_path: Path) -> None:
    zetas = _zeta_trajectory(tmp_path, 0.0)
    assert zetas == pytest.approx([zetas[0]] * len(zetas))


@pytest.mark.parametrize(
    ("ttl_s", "dt", "steps"),
    [
        (0.3, 0.1, 3),
        (0.05, 0.01, 5),
        (0.005, 0.01, 1),
        (0.0, 0.01, 0),
        (10.0, 0.001, 10000),
    ],
)
def test_ttl_step_count_rounds_a_partial_step_up(
    ttl_s: float, dt: float, steps: int
) -> None:
    assert _ttl_steps(ttl_s, dt) == steps

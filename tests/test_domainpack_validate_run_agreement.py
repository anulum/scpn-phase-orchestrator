# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — a spec that validates is a spec simulate accepts

"""``validate_binding_spec`` must refuse what the runtime refuses.

Two shipped domainpacks validated cleanly and then failed on the first
``simulate`` call: ``agent_coordination`` (string Petri arcs, place-count
guards, an empty ``place_regime``) and ``identity_coherence`` (two ``K``
actuators with different bounds, which the knob-indexed projector refuses).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scpn_phase_orchestrator.binding import load_binding_spec
from scpn_phase_orchestrator.binding.validator import validate_binding_spec
from scpn_phase_orchestrator.runtime.simulation import simulate

_REPO = Path(__file__).resolve().parents[1]
_PACKS = sorted((_REPO / "domainpacks").glob("*/binding_spec.yaml"))
_MINIMAL = _REPO / "domainpacks" / "minimal_domain" / "binding_spec.yaml"


@pytest.mark.parametrize("path", _PACKS, ids=[p.parent.name for p in _PACKS])
def test_every_shipped_pack_validates_and_runs(path: Path) -> None:
    spec = load_binding_spec(path)
    assert validate_binding_spec(spec) == []
    result = simulate(spec, steps=3, seed=0, policy_enabled=True)
    assert result.steps == 3


def _spec_with(tmp_path: Path, **overrides: object) -> Path:
    data = yaml.safe_load(_MINIMAL.read_text(encoding="utf-8"))
    data.update(overrides)
    path = tmp_path / "binding_spec.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _net(**transition: object) -> dict[str, object]:
    base = {
        "name": "advance",
        "inputs": [{"place": "a", "weight": 1}],
        "outputs": [{"place": "b", "weight": 1}],
    }
    base.update(transition)
    return {
        "places": ["a", "b"],
        "initial": {"a": 1},
        "place_regime": {"a": "NOMINAL", "b": "DEGRADED"},
        "transitions": [base],
    }


def _errors(tmp_path: Path, **overrides: object) -> list[str]:
    return validate_binding_spec(load_binding_spec(_spec_with(tmp_path, **overrides)))


def _runs(tmp_path: Path, **overrides: object) -> bool:
    spec = load_binding_spec(_spec_with(tmp_path, **overrides))
    try:
        simulate(spec, steps=2, seed=0, policy_enabled=True)
    except Exception:
        return False
    return True


def test_well_formed_protocol_net_validates_and_runs(tmp_path: Path) -> None:
    net = _net(guard="stability_proxy > 0.6")
    assert _errors(tmp_path, protocol_net=net) == []
    assert _runs(tmp_path, protocol_net=net)


@pytest.mark.parametrize(
    ("net", "fragment"),
    [
        pytest.param(_net(inputs=["a"]), "must be a mapping", id="string-arc"),
        pytest.param(
            _net(outputs=[{"place": "zz", "weight": 1}]),
            "unknown place",
            id="bad-place",
        ),
        pytest.param(
            _net(inputs=[{"place": "a", "weight": 0}]), "weight", id="zero-weight"
        ),
        pytest.param(
            _net(guard="R_0 > 0.1 and R_1 < 0.2"), "metric op threshold", id="compound"
        ),
        pytest.param(
            {**_net(), "place_regime": {}}, "refused by the runtime", id="empty-regimes"
        ),
        pytest.param(
            {**_net(), "place_regime": {"a": "PANIC"}},
            "refused by the runtime",
            id="unknown-regime",
        ),
    ],
)
def test_protocol_net_the_runtime_refuses_does_not_validate(
    tmp_path: Path, net: dict[str, object], fragment: str
) -> None:
    errors = _errors(tmp_path, protocol_net=net)
    assert any(fragment in error for error in errors), errors
    assert not _runs(tmp_path, protocol_net=net)


def test_place_count_guard_does_not_validate(tmp_path: Path) -> None:
    # The runtime accepts it, but guards read context metrics: "a" is never in
    # the context, the guard is always false, and the transition never fires.
    errors = _errors(tmp_path, protocol_net=_net(guard="a > 0"))
    assert any("is a place name" in error for error in errors), errors


def _actuators(*records: dict[str, object]) -> list[dict[str, object]]:
    return [
        {"name": f"act_{i}", "knob": "K", "scope": "global", "limits": [0.0, 2.0]}
        | record
        for i, record in enumerate(records)
    ]


def test_one_knob_with_two_bounds_does_not_validate(tmp_path: Path) -> None:
    actuators = _actuators({}, {"scope": "layer_0", "limits": [0.0, 3.0]})
    errors = _errors(tmp_path, actuators=actuators)
    assert any("one bound per knob" in error for error in errors), errors
    assert not _runs(tmp_path, actuators=actuators)


def test_one_knob_with_two_rate_limits_does_not_validate(tmp_path: Path) -> None:
    actuators = _actuators(
        {"rate_limit_per_step": 0.1},
        {"scope": "layer_0", "rate_limit_per_step": 0.2},
    )
    errors = _errors(tmp_path, actuators=actuators)
    assert any("one rate limit per knob" in error for error in errors), errors
    assert not _runs(tmp_path, actuators=actuators)


def test_unset_rate_limit_does_not_conflict(tmp_path: Path) -> None:
    actuators = _actuators({"rate_limit_per_step": 0.1}, {"scope": "layer_0"})
    assert _errors(tmp_path, actuators=actuators) == []
    assert _runs(tmp_path, actuators=actuators)

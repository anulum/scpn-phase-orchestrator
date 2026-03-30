# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Petri net tests

from __future__ import annotations

import pytest

from scpn_phase_orchestrator.supervisor.petri_net import (
    Arc,
    Guard,
    Marking,
    PetriNet,
    Place,
    Transition,
    parse_guard,
)


def _simple_net():
    """warmup -> nominal -> cooldown -> done"""
    places = [Place("warmup"), Place("nominal"), Place("cooldown"), Place("done")]
    transitions = [
        Transition(
            name="start",
            inputs=[Arc("warmup")],
            outputs=[Arc("nominal")],
            guard=Guard("stability_proxy", ">", 0.6),
        ),
        Transition(
            name="wind_down",
            inputs=[Arc("nominal")],
            outputs=[Arc("cooldown")],
            guard=Guard("R_0", "<", 0.3),
        ),
        Transition(
            name="finish",
            inputs=[Arc("cooldown")],
            outputs=[Arc("done")],
        ),
    ]
    return PetriNet(places, transitions)


def test_initial_marking():
    m = Marking(tokens={"warmup": 1})
    assert m["warmup"] == 1
    assert m["nominal"] == 0


def test_marking_active_places():
    m = Marking(tokens={"warmup": 1, "cooldown": 0})
    assert m.active_places() == ["warmup"]


def test_marking_copy_independent():
    m = Marking(tokens={"a": 1})
    m2 = m.copy()
    m2["a"] = 2
    assert m["a"] == 1


def test_marking_negative_rejected():
    m = Marking()
    with pytest.raises(ValueError, match="negative"):
        m["x"] = -1


def test_marking_zero_removes_key():
    m = Marking(tokens={"a": 1})
    m["a"] = 0
    assert "a" not in m.tokens


def test_parse_guard():
    g = parse_guard("stability_proxy > 0.6")
    assert g.metric == "stability_proxy"
    assert g.op == ">"
    assert g.threshold == 0.6


def test_parse_guard_bad_format():
    with pytest.raises(ValueError, match="guard must be"):
        parse_guard("bad")


def test_guard_evaluate_true():
    g = Guard("R", ">", 0.5)
    assert g.evaluate({"R": 0.7})


def test_guard_evaluate_false():
    g = Guard("R", ">", 0.5)
    assert not g.evaluate({"R": 0.3})


def test_guard_missing_metric():
    g = Guard("missing", ">", 0.5)
    assert not g.evaluate({"R": 0.7})


def test_enabled_transitions():
    net = _simple_net()
    m = Marking(tokens={"warmup": 1})
    ctx = {"stability_proxy": 0.8, "R_0": 0.5}
    enabled = net.enabled(m, ctx)
    assert len(enabled) == 1
    assert enabled[0].name == "start"


def test_no_enabled_guard_fails():
    net = _simple_net()
    m = Marking(tokens={"warmup": 1})
    ctx = {"stability_proxy": 0.3}
    enabled = net.enabled(m, ctx)
    assert len(enabled) == 0


def test_fire_moves_tokens():
    net = _simple_net()
    m = Marking(tokens={"warmup": 1})
    t = net.transitions[0]
    m2 = net.fire(m, t)
    assert m2["warmup"] == 0
    assert m2["nominal"] == 1
    assert m["warmup"] == 1  # original unchanged


def test_step_fires_first_enabled():
    net = _simple_net()
    m = Marking(tokens={"warmup": 1})
    ctx = {"stability_proxy": 0.8, "R_0": 0.5}
    m2, fired = net.step(m, ctx)
    assert fired is not None
    assert fired.name == "start"
    assert m2["nominal"] == 1


def test_step_no_enabled():
    net = _simple_net()
    m = Marking(tokens={"warmup": 1})
    ctx = {"stability_proxy": 0.1}
    m2, fired = net.step(m, ctx)
    assert fired is None
    assert m2["warmup"] == 1


def test_full_protocol_sequence():
    net = _simple_net()
    m = Marking(tokens={"warmup": 1})

    m, t = net.step(m, {"stability_proxy": 0.8, "R_0": 0.5})
    assert t.name == "start"

    m, t = net.step(m, {"stability_proxy": 0.8, "R_0": 0.2})
    assert t.name == "wind_down"

    m, t = net.step(m, {})
    assert t.name == "finish"
    assert m["done"] == 1


def test_guardless_transition_fires_when_tokens_present():
    net = _simple_net()
    m = Marking(tokens={"cooldown": 1})
    m2, t = net.step(m, {})
    assert t.name == "finish"
    assert m2["done"] == 1


def test_invalid_place_reference():
    with pytest.raises(ValueError, match="unknown place"):
        PetriNet(
            [Place("a")],
            [Transition("t", inputs=[Arc("b")], outputs=[Arc("a")])],
        )


def test_weighted_arcs():
    net = PetriNet(
        [Place("a"), Place("b")],
        [Transition("t", inputs=[Arc("a", weight=2)], outputs=[Arc("b", weight=3)])],
    )
    m = Marking(tokens={"a": 2})
    m2, t = net.step(m, {})
    assert t is not None
    assert m2["a"] == 0
    assert m2["b"] == 3


def test_insufficient_tokens_blocks():
    net = PetriNet(
        [Place("a"), Place("b")],
        [Transition("t", inputs=[Arc("a", weight=2)], outputs=[Arc("b")])],
    )
    m = Marking(tokens={"a": 1})
    m2, t = net.step(m, {})
    assert t is None


def test_guard_all_ops():
    for op_str, val, threshold, expected in [
        (">", 1.0, 0.5, True),
        (">=", 0.5, 0.5, True),
        ("<", 0.3, 0.5, True),
        ("<=", 0.5, 0.5, True),
        ("==", 0.5, 0.5, True),
        (">", 0.3, 0.5, False),
    ]:
        g = Guard("x", op_str, threshold)
        assert g.evaluate({"x": val}) == expected, f"{op_str} failed"


def test_token_conservation_unweighted():
    """Firing an unweighted transition conserves total tokens:
    consumed from input places, produced at output places."""
    net = _simple_net()
    m = Marking(tokens={"warmup": 1})
    total_before = sum(m.tokens.values())
    m2 = net.fire(m, net.transitions[0])  # start: warmup→nominal
    total_after = sum(m2.tokens.values())
    assert total_after == total_before, (
        f"Unweighted 1:1 arc must conserve tokens: {total_before} → {total_after}"
    )


def test_weighted_token_transformation():
    """Weighted arcs: 2 tokens consumed, 3 produced → net +1."""
    net = PetriNet(
        [Place("a"), Place("b")],
        [Transition("t", inputs=[Arc("a", weight=2)], outputs=[Arc("b", weight=3)])],
    )
    m = Marking(tokens={"a": 5})
    m2, t = net.step(m, {})
    assert t is not None
    assert m2["a"] == 3, "Should consume 2 from a (5-2=3)"
    assert m2["b"] == 3, "Should produce 3 at b"


def test_priority_deterministic():
    """When multiple transitions are enabled, step fires the first one
    (deterministic ordering). This is critical for reproducibility."""
    net = PetriNet(
        [Place("p"), Place("q1"), Place("q2")],
        [
            Transition("t1", inputs=[Arc("p")], outputs=[Arc("q1")]),
            Transition("t2", inputs=[Arc("p")], outputs=[Arc("q2")]),
        ],
    )
    m = Marking(tokens={"p": 1})
    m2, fired = net.step(m, {})
    assert fired.name == "t1", "First-declared transition must fire first"
    assert m2["q1"] == 1


def test_multi_step_protocol_token_tracking():
    """Run the full 3-step protocol and verify token accounting at each stage."""
    net = _simple_net()
    m = Marking(tokens={"warmup": 1})

    # Step 1: warmup → nominal
    m, t = net.step(m, {"stability_proxy": 0.8, "R_0": 0.5})
    assert m.active_places() == ["nominal"]

    # Step 2: nominal → cooldown
    m, t = net.step(m, {"stability_proxy": 0.8, "R_0": 0.2})
    assert m.active_places() == ["cooldown"]

    # Step 3: cooldown → done
    m, t = net.step(m, {})
    assert m.active_places() == ["done"]

    # No more transitions possible
    m, t = net.step(m, {"stability_proxy": 0.9, "R_0": 0.9})
    assert t is None, "All transitions exhausted"


def test_parse_guard_all_operators():
    """parse_guard must handle all 5 operators: > >= < <= =="""
    for op in [">", ">=", "<", "<=", "=="]:
        g = parse_guard(f"metric {op} 0.5")
        assert g.op == op
        assert g.metric == "metric"
        assert g.threshold == 0.5

# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Petri net FFI parity tests

"""Cross-validate Python PetriNet against Rust PyPetriNet."""

from __future__ import annotations

import pytest

from scpn_phase_orchestrator._compat import HAS_RUST
from scpn_phase_orchestrator.supervisor.petri_net import (
    Arc,
    Guard,
    Marking,
    PetriNet,
    Place,
    Transition,
)

_HAS_PETRI = HAS_RUST and hasattr(
    __import__("spo_kernel") if HAS_RUST else None, "PyPetriNet"
)
pytestmark = pytest.mark.skipif(
    not _HAS_PETRI, reason="spo_kernel.PyPetriNet not available"
)


@pytest.fixture()
def spo():
    import spo_kernel

    return spo_kernel


def _py_simple_net():
    places = [Place("idle"), Place("active"), Place("done")]
    transitions = [
        Transition(
            name="start",
            inputs=[Arc("idle")],
            outputs=[Arc("active")],
        ),
        Transition(
            name="finish",
            inputs=[Arc("active")],
            outputs=[Arc("done")],
        ),
    ]
    return PetriNet(places, transitions)


def _rust_simple_net(spo):
    return spo.PyPetriNet(
        ["idle", "active", "done"],
        [
            ("start", [("idle", 1)], [("active", 1)], None),
            ("finish", [("active", 1)], [("done", 1)], None),
        ],
    )


def test_step_parity(spo):
    py_net = _py_simple_net()
    rust_net = _rust_simple_net(spo)
    py_marking = Marking(tokens={"idle": 1})

    py_new, py_fired = py_net.step(py_marking, {})
    rust_tokens, rust_fired = rust_net.step({"idle": 1}, {})

    assert py_fired.name == rust_fired
    assert py_new["active"] == rust_tokens["active"]
    assert py_new["idle"] == rust_tokens.get("idle", 0)


def test_sequential_step_parity(spo):
    py_net = _py_simple_net()
    rust_net = _rust_simple_net(spo)
    py_marking = Marking(tokens={"idle": 1})
    rust_tokens = {"idle": 1}

    # Step 1: idle → active
    py_marking, py_fired = py_net.step(py_marking, {})
    rust_tokens, rust_fired = rust_net.step(rust_tokens, {})
    assert py_fired.name == rust_fired

    # Step 2: active → done
    py_marking, py_fired = py_net.step(py_marking, {})
    rust_tokens_dict = dict(rust_tokens)
    rust_tokens, rust_fired = rust_net.step(rust_tokens_dict, {})
    assert py_fired.name == rust_fired
    assert py_marking["done"] == 1


def test_no_enabled_parity(spo):
    py_net = _py_simple_net()
    rust_net = _rust_simple_net(spo)

    py_marking, py_fired = py_net.step(Marking(), {})
    rust_tokens, rust_fired = rust_net.step({}, {})

    assert py_fired is None
    assert rust_fired is None


def test_guard_parity(spo):
    py_net = PetriNet(
        [Place("a"), Place("b")],
        [
            Transition(
                name="guarded",
                inputs=[Arc("a")],
                outputs=[Arc("b")],
                guard=Guard("x", ">", 0.5),
            )
        ],
    )
    rust_net = spo.PyPetriNet(
        ["a", "b"],
        [("guarded", [("a", 1)], [("b", 1)], "x > 0.5")],
    )

    # Guard blocks (x=0.3)
    py_m, py_f = py_net.step(Marking(tokens={"a": 1}), {"x": 0.3})
    rust_t, rust_f = rust_net.step({"a": 1}, {"x": 0.3})
    assert py_f is None
    assert rust_f is None

    # Guard passes (x=0.8)
    py_m, py_f = py_net.step(Marking(tokens={"a": 1}), {"x": 0.8})
    rust_t, rust_f = rust_net.step({"a": 1}, {"x": 0.8})
    assert py_f.name == rust_f == "guarded"


def test_enabled_parity(spo):
    py_net = PetriNet(
        [Place("a"), Place("b"), Place("c")],
        [
            Transition(name="t1", inputs=[Arc("a")], outputs=[Arc("b")]),
            Transition(name="t2", inputs=[Arc("a")], outputs=[Arc("c")]),
        ],
    )
    rust_net = spo.PyPetriNet(
        ["a", "b", "c"],
        [
            ("t1", [("a", 1)], [("b", 1)], None),
            ("t2", [("a", 1)], [("c", 1)], None),
        ],
    )

    py_enabled = [t.name for t in py_net.enabled(Marking(tokens={"a": 2}), {})]
    rust_enabled = rust_net.enabled({"a": 2}, {})

    assert set(py_enabled) == set(rust_enabled)
    assert len(py_enabled) == 2


def test_weighted_arcs_parity(spo):
    py_net = PetriNet(
        [Place("pool"), Place("out")],
        [
            Transition(
                name="consume",
                inputs=[Arc("pool", weight=3)],
                outputs=[Arc("out", weight=1)],
            )
        ],
    )
    rust_net = spo.PyPetriNet(
        ["pool", "out"],
        [("consume", [("pool", 3)], [("out", 1)], None)],
    )

    # Not enough tokens
    _, py_f = py_net.step(Marking(tokens={"pool": 2}), {})
    _, rust_f = rust_net.step({"pool": 2}, {})
    assert py_f is None
    assert rust_f is None

    # Enough tokens
    py_m, py_f = py_net.step(Marking(tokens={"pool": 3}), {})
    rust_t, rust_f = rust_net.step({"pool": 3}, {})
    assert py_f.name == rust_f == "consume"
    assert py_m["pool"] == rust_t.get("pool", 0) == 0
    assert py_m["out"] == rust_t["out"] == 1


def test_place_names_parity(spo):
    py_net = _py_simple_net()
    rust_net = _rust_simple_net(spo)
    assert set(py_net.place_names) == set(rust_net.place_names)


class TestPipelineWiring:
    """Pipeline wiring: proves this module is not decorative."""

    def test_wires_into_pipeline(self):
        import numpy as np

        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        n = 8
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        for _ in range(100):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0

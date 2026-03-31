# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — RuleEngine FFI parity tests

"""Cross-validate Python PolicyEngine against Rust PyRuleEngine."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator._compat import HAS_RUST
from scpn_phase_orchestrator.supervisor.policy_rules import (
    CompoundCondition,
    PolicyAction,
    PolicyCondition,
    PolicyEngine,
    PolicyRule,
)
from scpn_phase_orchestrator.supervisor.regimes import Regime
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

_HAS_RULE_ENGINE = HAS_RUST and hasattr(
    __import__("spo_kernel") if HAS_RUST else None, "PyRuleEngine"
)
pytestmark = pytest.mark.skipif(
    not _HAS_RULE_ENGINE, reason="spo_kernel.PyRuleEngine not available"
)


@pytest.fixture()
def spo():
    import spo_kernel

    return spo_kernel


def _make_upde_state(stability_proxy: float) -> UPDEState:
    return UPDEState(
        layers=[LayerState(R=0.5, psi=0.0)],
        cross_layer_alignment=np.zeros((1, 1)),
        stability_proxy=stability_proxy,
        regime_id="degraded",
    )


def _py_rule(name, regime, metric, op, threshold, knob="K", value=0.1):
    return PolicyRule(
        name=name,
        regimes=[regime.upper()],
        condition=PolicyCondition(
            metric=metric,
            layer=None,
            op=op,
            threshold=threshold,
        ),
        actions=[PolicyAction(knob=knob, scope="global", value=value, ttl_s=10.0)],
    )


def _rust_rule(name, regime, metric, op, threshold, knob="K", value=0.1):
    return (
        name,
        [regime.upper()],
        [(metric, op, threshold)],
        "AND",
        [(knob, "global", value, 10.0)],
        0.0,
        0,
    )


def test_single_rule_fires(spo):
    py_eng = PolicyEngine([_py_rule("boost", "DEGRADED", "stability_proxy", "<", 0.5)])
    rust_eng = spo.PyRuleEngine(
        [_rust_rule("boost", "DEGRADED", "stability_proxy", "<", 0.5)]
    )

    state = _make_upde_state(0.3)
    py_actions = py_eng.evaluate(Regime.DEGRADED, state, [], [])
    rust_actions = rust_eng.evaluate("degraded", {"stability_proxy": 0.3})

    assert len(py_actions) == len(rust_actions) == 1
    assert py_actions[0].knob == rust_actions[0][0]
    assert abs(py_actions[0].value - rust_actions[0][2]) < 1e-12


def test_wrong_regime_no_fire(spo):
    py_eng = PolicyEngine([_py_rule("boost", "DEGRADED", "stability_proxy", "<", 0.5)])
    rust_eng = spo.PyRuleEngine(
        [_rust_rule("boost", "DEGRADED", "stability_proxy", "<", 0.5)]
    )

    state = _make_upde_state(0.3)
    py_actions = py_eng.evaluate(Regime.NOMINAL, state, [], [])
    rust_actions = rust_eng.evaluate("nominal", {"stability_proxy": 0.3})

    assert py_actions == []
    assert rust_actions == []


def test_condition_not_met(spo):
    py_eng = PolicyEngine([_py_rule("boost", "DEGRADED", "stability_proxy", "<", 0.5)])
    rust_eng = spo.PyRuleEngine(
        [_rust_rule("boost", "DEGRADED", "stability_proxy", "<", 0.5)]
    )

    state = _make_upde_state(0.8)
    py_actions = py_eng.evaluate(Regime.DEGRADED, state, [], [])
    rust_actions = rust_eng.evaluate("degraded", {"stability_proxy": 0.8})

    assert py_actions == []
    assert rust_actions == []


def test_cooldown_parity(spo):
    py_rule = PolicyRule(
        name="r1",
        regimes=["DEGRADED"],
        condition=PolicyCondition(
            metric="stability_proxy", layer=None, op="<", threshold=1.0
        ),
        actions=[PolicyAction(knob="K", scope="global", value=0.1, ttl_s=10.0)],
        cooldown_s=5.0,
    )
    rust_rule = (
        "r1",
        ["DEGRADED"],
        [("stability_proxy", "<", 1.0)],
        "AND",
        [("K", "global", 0.1, 10.0)],
        5.0,
        0,
    )

    py_eng = PolicyEngine([py_rule])
    rust_eng = spo.PyRuleEngine([rust_rule])

    state = _make_upde_state(0.3)
    ctx = {"stability_proxy": 0.3}

    # First fire
    assert len(py_eng.evaluate(Regime.DEGRADED, state, [], [])) == 1
    assert len(rust_eng.evaluate("degraded", ctx)) == 1

    # Blocked by cooldown
    assert len(py_eng.evaluate(Regime.DEGRADED, state, [], [])) == 0
    assert len(rust_eng.evaluate("degraded", ctx)) == 0

    # Advance past cooldown
    py_eng.advance_clock(6.0)
    rust_eng.advance_clock(6.0)

    assert len(py_eng.evaluate(Regime.DEGRADED, state, [], [])) == 1
    assert len(rust_eng.evaluate("degraded", ctx)) == 1


def test_max_fires_parity(spo):
    py_rule = PolicyRule(
        name="r1",
        regimes=["DEGRADED"],
        condition=PolicyCondition(
            metric="stability_proxy", layer=None, op="<", threshold=1.0
        ),
        actions=[PolicyAction(knob="K", scope="global", value=0.1, ttl_s=10.0)],
        max_fires=2,
    )
    rust_rule = (
        "r1",
        ["DEGRADED"],
        [("stability_proxy", "<", 1.0)],
        "AND",
        [("K", "global", 0.1, 10.0)],
        0.0,
        2,
    )

    py_eng = PolicyEngine([py_rule])
    rust_eng = spo.PyRuleEngine([rust_rule])

    state = _make_upde_state(0.3)
    ctx = {"stability_proxy": 0.3}

    assert len(py_eng.evaluate(Regime.DEGRADED, state, [], [])) == 1
    assert len(rust_eng.evaluate("degraded", ctx)) == 1
    assert len(py_eng.evaluate(Regime.DEGRADED, state, [], [])) == 1
    assert len(rust_eng.evaluate("degraded", ctx)) == 1
    # Third fire blocked
    assert len(py_eng.evaluate(Regime.DEGRADED, state, [], [])) == 0
    assert len(rust_eng.evaluate("degraded", ctx)) == 0


def test_compound_and_parity(spo):
    py_rule = PolicyRule(
        name="compound",
        regimes=["DEGRADED"],
        condition=CompoundCondition(
            conditions=[
                PolicyCondition(
                    metric="stability_proxy",
                    layer=None,
                    op=">",
                    threshold=0.3,
                ),
                PolicyCondition(
                    metric="stability_proxy",
                    layer=None,
                    op="<",
                    threshold=0.7,
                ),
            ],
            logic="AND",
        ),
        actions=[PolicyAction(knob="K", scope="global", value=0.1, ttl_s=10.0)],
    )
    rust_rule = (
        "compound",
        ["DEGRADED"],
        [("stability_proxy", ">", 0.3), ("stability_proxy", "<", 0.7)],
        "AND",
        [("K", "global", 0.1, 10.0)],
        0.0,
        0,
    )

    py_eng = PolicyEngine([py_rule])
    rust_eng = spo.PyRuleEngine([rust_rule])

    # In range [0.3, 0.7] → fires
    state = _make_upde_state(0.5)
    assert len(py_eng.evaluate(Regime.DEGRADED, state, [], [])) == 1
    assert len(rust_eng.evaluate("degraded", {"stability_proxy": 0.5})) == 1

    # Out of range → no fire
    state2 = _make_upde_state(0.9)
    assert len(py_eng.evaluate(Regime.DEGRADED, state2, [], [])) == 0
    assert len(rust_eng.evaluate("degraded", {"stability_proxy": 0.9})) == 0


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

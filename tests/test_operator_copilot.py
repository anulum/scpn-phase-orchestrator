# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — operator copilot tests

"""Tests for the grounded LLM operator copilot.

A deterministic echo backend lets the suite inspect the grounded prompt the
copilot would send, and a fixed-answer backend checks the completion is returned.
A fully-populated report and an empty report exercise every grounding branch, and
the question-validation path is checked.
"""

from __future__ import annotations

import pytest

from scpn_phase_orchestrator.reporting.explainability import (
    ActionExplanation,
    ExplainabilityReport,
)
from scpn_phase_orchestrator.reporting.operator_copilot import OperatorCopilot


class _Echo:
    def complete(self, prompt: str) -> str:
        return prompt


class _Fixed:
    def __init__(self, answer: str) -> None:
        self._answer = answer

    def complete(self, prompt: str) -> str:
        return self._answer


def _full_report() -> ExplainabilityReport:
    return ExplainabilityReport(
        steps=250,
        layers=5,
        hash_chain_ok=True,
        hash_chain_verified=251,
        final_regime="NOMINAL",
        final_stability=0.8712,
        regime_counts={"NOMINAL": 200, "DEGRADED": 50},
        regime_transitions=("NOMINAL", "DEGRADED", "NOMINAL"),
        action_explanations=(
            ActionExplanation(
                step=42,
                regime="DEGRADED",
                knob="alpha",
                scope="layer_3",
                value=0.4,
                ttl_s=5.0,
                reason="suppress frequency oscillation",
                evidence=("R_bad[1]=0.74 > 0.70",),
            ),
        ),
        events=("boundary frequency_dev near limit at step 40",),
        metric_summary=("mean R_good=0.81", "mean R_bad=0.32"),
    )


def _empty_report() -> ExplainabilityReport:
    return ExplainabilityReport(
        steps=10,
        layers=2,
        hash_chain_ok=False,
        hash_chain_verified=0,
        final_regime="NOMINAL",
        final_stability=0.5,
        regime_counts={},
        regime_transitions=(),
        action_explanations=(),
        events=(),
        metric_summary=(),
    )


def test_answer_returns_the_model_completion() -> None:
    copilot = OperatorCopilot(_Fixed("The mode was poorly damped."), _full_report())
    assert copilot.answer("What happened?") == "The mode was poorly damped."


def test_prompt_grounds_in_the_full_evidence() -> None:
    copilot = OperatorCopilot(_Echo(), _full_report())
    prompt = copilot.answer("Why did the controller act at step 42?")
    assert "ONLY the audit evidence" in prompt
    assert "review-only surface" in prompt
    assert "hash chain verified (251 records)" in prompt
    assert "Final regime NOMINAL; final stability 0.8712" in prompt
    assert "DEGRADED=50, NOMINAL=200" in prompt
    assert "NOMINAL -> DEGRADED -> NOMINAL" in prompt
    assert "step 42 [DEGRADED]: alpha@layer_3=0.4000 (ttl 5.0s)" in prompt
    assert "suppress frequency oscillation" in prompt
    assert "evidence: R_bad[1]=0.74 > 0.70" in prompt
    assert "Metrics: mean R_good=0.81; mean R_bad=0.32" in prompt
    assert "Events: boundary frequency_dev near limit at step 40" in prompt
    assert "OPERATOR QUESTION: Why did the controller act at step 42?" in prompt


def test_prompt_handles_an_empty_report() -> None:
    copilot = OperatorCopilot(_Echo(), _empty_report())
    prompt = copilot.answer("Anything to report?")
    assert "hash chain NOT verified" in prompt
    assert "Regime distribution: none recorded" in prompt
    assert "Control actions: none recorded" in prompt
    assert "Regime transitions:" not in prompt
    assert "Metrics:" not in prompt
    assert "Events:" not in prompt


def test_action_without_evidence_is_rendered() -> None:
    report = _full_report()
    bare = ActionExplanation(
        step=7,
        regime="NOMINAL",
        knob="K",
        scope="global",
        value=0.2,
        ttl_s=1.0,
        reason="restore synchronism",
        evidence=(),
    )
    grounded = OperatorCopilot(
        _Echo(),
        ExplainabilityReport(
            steps=report.steps,
            layers=report.layers,
            hash_chain_ok=report.hash_chain_ok,
            hash_chain_verified=report.hash_chain_verified,
            final_regime=report.final_regime,
            final_stability=report.final_stability,
            regime_counts=report.regime_counts,
            regime_transitions=report.regime_transitions,
            action_explanations=(bare,),
            events=report.events,
            metric_summary=report.metric_summary,
        ),
    ).build_prompt("?")
    assert "evidence: none" in grounded


def test_max_actions_truncates_the_context() -> None:
    actions = tuple(
        ActionExplanation(
            step=index,
            regime="DEGRADED",
            knob="alpha",
            scope="layer_3",
            value=0.1 * index,
            ttl_s=1.0,
            reason=f"action {index}",
            evidence=(),
        )
        for index in range(5)
    )
    report = _empty_report()
    populated = ExplainabilityReport(
        steps=report.steps,
        layers=report.layers,
        hash_chain_ok=True,
        hash_chain_verified=1,
        final_regime=report.final_regime,
        final_stability=report.final_stability,
        regime_counts={"DEGRADED": 5},
        regime_transitions=(),
        action_explanations=actions,
        events=(),
        metric_summary=(),
    )
    prompt = OperatorCopilot(_Echo(), populated, max_actions=2).build_prompt("?")
    assert "action 0" in prompt
    assert "action 1" in prompt
    assert "action 4" not in prompt


@pytest.mark.parametrize("question", ["", "   "])
def test_answer_rejects_an_empty_question(question: str) -> None:
    copilot = OperatorCopilot(_Fixed("x"), _full_report())
    with pytest.raises(ValueError, match="must not be empty"):
        copilot.answer(question)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — grounded LLM operator copilot

"""A grounded LLM copilot that answers operator questions from audit evidence.

The copilot turns a hash-verified :class:`~scpn_phase_orchestrator.reporting.
explainability.ExplainabilityReport` into a question-answering surface for a
control-room operator. It does not let the language model speak freely: it
renders the report — regime distribution, transitions, every control action with
its recorded reason and evidence, the metric summary — into the prompt, and
instructs the model to answer *only* from that evidence and to decline when the
evidence is silent. The model therefore explains and locates what the audit
already records; it does not invent control history or recommend actuation.

The language model is any provider with a ``complete(prompt) -> str`` method
(:class:`OperatorLLM`) — a local HTTP model, or a deterministic stub for tests —
so no network backend is required and the prompt construction is fully testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from scpn_phase_orchestrator.reporting.explainability import ExplainabilityReport

__all__ = ["OperatorCopilot", "OperatorLLM"]


class OperatorLLM(Protocol):
    """A language-model backend that completes a prompt to an answer."""

    def complete(self, prompt: str) -> str:
        """Return the model completion for ``prompt``.

        Parameters
        ----------
        prompt : str
            The grounded operator prompt.

        Returns
        -------
        str
            The model's answer.
        """
        ...


_GROUNDING_INSTRUCTION = (
    "You are an operator assistant for the SCPN Phase Orchestrator. Answer the "
    "operator's question using ONLY the audit evidence below. If the evidence "
    "does not cover the question, say so plainly. Do not speculate, invent "
    "control history, or recommend actuation — this is a review-only surface."
)


@dataclass(frozen=True)
class OperatorCopilot:
    """A grounded operator copilot over one explainability report.

    Parameters
    ----------
    llm : OperatorLLM
        The language-model backend.
    report : ExplainabilityReport
        The hash-verified audit evidence the answers are grounded in.
    max_actions : int
        The most control actions to include in the grounding context.
    """

    llm: OperatorLLM
    report: ExplainabilityReport
    max_actions: int = 12

    def answer(self, question: str) -> str:
        """Answer an operator question grounded in the audit evidence.

        Parameters
        ----------
        question : str
            The operator's question.

        Returns
        -------
        str
            The grounded model answer.

        Raises
        ------
        ValueError
            If ``question`` is empty.
        """
        if not question.strip():
            raise ValueError("question must not be empty")
        return self.llm.complete(self.build_prompt(question))

    def build_prompt(self, question: str) -> str:
        """Render the grounded prompt for an operator question.

        Parameters
        ----------
        question : str
            The operator's question.

        Returns
        -------
        str
            The prompt: the grounding instruction, the rendered evidence, and the
            question.
        """
        lines = [_GROUNDING_INSTRUCTION, "", "AUDIT EVIDENCE:"]
        report = self.report
        chain = "verified" if report.hash_chain_ok else "NOT verified"
        lines.append(
            f"- {report.steps} steps over {report.layers} layers; "
            f"hash chain {chain} ({report.hash_chain_verified} records)."
        )
        lines.append(
            f"- Final regime {report.final_regime}; "
            f"final stability {report.final_stability:.4f}."
        )
        distribution = ", ".join(
            f"{regime}={count}"
            for regime, count in sorted(report.regime_counts.items())
        )
        lines.append(f"- Regime distribution: {distribution or 'none recorded'}.")
        if report.regime_transitions:
            lines.append(
                f"- Regime transitions: {' -> '.join(report.regime_transitions)}."
            )
        if report.action_explanations:
            lines.append("- Control actions:")
            for action in report.action_explanations[: self.max_actions]:
                evidence = "; ".join(action.evidence) if action.evidence else "none"
                lines.append(
                    f"  - step {action.step} [{action.regime}]: "
                    f"{action.knob}@{action.scope}={action.value:.4f} "
                    f"(ttl {action.ttl_s:.1f}s) — {action.reason} "
                    f"(evidence: {evidence})"
                )
        else:
            lines.append("- Control actions: none recorded.")
        if report.metric_summary:
            lines.append(f"- Metrics: {'; '.join(report.metric_summary)}.")
        if report.events:
            lines.append(f"- Events: {'; '.join(report.events)}.")
        lines.extend(["", f"OPERATOR QUESTION: {question}", "", "ANSWER:"])
        return "\n".join(lines)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — nn/supervisor package wiring tests

"""Wiring contract for the ``nn.supervisor`` package.

Asserts that every responsibility submodule imports under its dotted path and
that the package ``__init__`` re-exports each public symbol as the *same object*
defined in its owning submodule, so the split preserves the original flat
``nn.supervisor`` import surface with zero consumer churn.
"""

from __future__ import annotations

import importlib

import pytest

import scpn_phase_orchestrator.nn.supervisor as supervisor

PACKAGE = "scpn_phase_orchestrator.nn.supervisor"

# Full dotted paths spelled out so the module-linkage guard
# (tools/check_test_module_linkage.py) finds each submodule's import path.
SUBMODULES = (
    "scpn_phase_orchestrator.nn.supervisor._shared",
    "scpn_phase_orchestrator.nn.supervisor._types",
    "scpn_phase_orchestrator.nn.supervisor.checkpoint",
    "scpn_phase_orchestrator.nn.supervisor.comparison",
    "scpn_phase_orchestrator.nn.supervisor.policy",
    "scpn_phase_orchestrator.nn.supervisor.ppo",
    "scpn_phase_orchestrator.nn.supervisor.projection",
    "scpn_phase_orchestrator.nn.supervisor.replay",
    "scpn_phase_orchestrator.nn.supervisor.rollouts",
)

# Owning submodule for each public symbol re-exported by the package __init__.
PUBLIC_SYMBOL_MODULE = {
    "DifferentiableSupervisorConfig": "_types",
    "DifferentiableSupervisorPolicy": "policy",
    "KuramotoSupervisorScenario": "_types",
    "SupervisorAction": "_types",
    "SupervisorActionProjection": "_types",
    "SupervisorBaselineReport": "_types",
    "SupervisorCorpusReplayProposals": "_types",
    "SupervisorExperimentManifest": "_types",
    "SupervisorHandTunedBaselineComparison": "_types",
    "SupervisorLearnerProposalComparison": "_types",
    "SupervisorLossAux": "_types",
    "SupervisorPPOAux": "_types",
    "SupervisorPPOBatch": "_types",
    "SupervisorPPOCheckpoint": "_types",
    "SupervisorPPOCorpusRollout": "_types",
    "SupervisorPPORollout": "_types",
    "SupervisorPPOTrainResult": "_types",
    "SupervisorRandomBaselineComparison": "_types",
    "SupervisorReplayComparison": "_types",
    "SupervisorReplayProposal": "_types",
    "SupervisorScenarioCorpus": "_types",
    "SupervisorStaticBaselineComparison": "_types",
    "apply_supervisor_action": "policy",
    "build_supervisor_baseline_report": "replay",
    "build_supervisor_corpus_replay_proposals": "replay",
    "build_supervisor_experiment_manifest": "replay",
    "build_supervisor_replay_proposal": "replay",
    "build_supervisor_scenario_corpus": "rollouts",
    "closed_loop_supervisor_loss": "policy",
    "collect_supervisor_corpus_rollouts": "rollouts",
    "collect_supervisor_rollouts": "rollouts",
    "compare_supervisor_hand_tuned_baseline": "comparison",
    "compare_supervisor_learner_proposals": "comparison",
    "compare_supervisor_random_baseline": "comparison",
    "compare_supervisor_replay_proposal": "comparison",
    "compare_supervisor_static_baseline": "comparison",
    "control_actions_from_supervisor": "policy",
    "load_supervisor_ppo_checkpoint": "checkpoint",
    "masked_order_parameter": "_shared",
    "pack_supervisor_action": "policy",
    "ppo_supervisor_loss": "ppo",
    "ppo_supervisor_train_epochs": "ppo",
    "ppo_supervisor_train_step": "ppo",
    "ppo_supervisor_train_with_checkpoint": "ppo",
    "project_supervisor_action_for_audit": "projection",
    "sample_supervisor_action": "policy",
    "save_supervisor_ppo_checkpoint": "checkpoint",
    "supervisor_action_bound_penalty": "policy",
    "supervisor_action_log_prob": "policy",
    "supervisor_train_step": "policy",
    "unpack_supervisor_action": "policy",
}


@pytest.mark.parametrize("dotted_path", SUBMODULES)
def test_submodule_imports_under_dotted_path(dotted_path: str) -> None:
    assert dotted_path.startswith(f"{PACKAGE}.")
    module = importlib.import_module(dotted_path)
    assert module.__name__ == dotted_path
    assert (module.__doc__ or "").strip(), f"{dotted_path} has no module docstring"


def test_public_symbol_module_table_covers_dunder_all() -> None:
    assert set(PUBLIC_SYMBOL_MODULE) == set(supervisor.__all__)
    assert len(supervisor.__all__) == len(set(supervisor.__all__))


@pytest.mark.parametrize("symbol,module_name", sorted(PUBLIC_SYMBOL_MODULE.items()))
def test_public_symbol_reexport_is_owning_module_object(
    symbol: str, module_name: str
) -> None:
    owner = importlib.import_module(f"{PACKAGE}.{module_name}")
    assert getattr(supervisor, symbol) is getattr(owner, symbol)


def test_every_reexport_owner_is_a_real_submodule() -> None:
    owners = set(PUBLIC_SYMBOL_MODULE.values())
    submodule_names = {path.rsplit(".", 1)[1] for path in SUBMODULES}
    assert owners <= submodule_names

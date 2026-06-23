# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Differentiable supervisor policy for nn/

"""Differentiable supervisor policies for closed-loop Kuramoto control.

This package is the JAX/equinox counterpart to
``supervisor.policy.SupervisorPolicy``. It keeps the learning surface fully
differentiable and array-native, then exposes a small adapter for the existing
``ControlAction`` actuation path. Safety projection, rate limits, and live
interlocks remain outside the gradient path.
"""

from __future__ import annotations

from ._shared import (
    masked_order_parameter,
)
from ._types import (
    DifferentiableSupervisorConfig,
    KuramotoSupervisorScenario,
    SupervisorAction,
    SupervisorActionProjection,
    SupervisorBaselineReport,
    SupervisorCorpusReplayProposals,
    SupervisorExperimentManifest,
    SupervisorHandTunedBaselineComparison,
    SupervisorLearnerProposalComparison,
    SupervisorLossAux,
    SupervisorPPOAux,
    SupervisorPPOBatch,
    SupervisorPPOCheckpoint,
    SupervisorPPOCorpusRollout,
    SupervisorPPORollout,
    SupervisorPPOTrainResult,
    SupervisorRandomBaselineComparison,
    SupervisorReplayComparison,
    SupervisorReplayProposal,
    SupervisorScenarioCorpus,
    SupervisorStaticBaselineComparison,
)
from .candidate_bridge import (
    supervisor_action_to_candidate,
    supervisor_policy_to_candidate,
)
from .checkpoint import (
    load_supervisor_ppo_checkpoint,
    save_supervisor_ppo_checkpoint,
)
from .comparison import (
    compare_supervisor_hand_tuned_baseline,
    compare_supervisor_learner_proposals,
    compare_supervisor_random_baseline,
    compare_supervisor_replay_proposal,
    compare_supervisor_static_baseline,
)
from .policy import (
    DifferentiableSupervisorPolicy,
    apply_supervisor_action,
    closed_loop_supervisor_loss,
    control_actions_from_supervisor,
    pack_supervisor_action,
    sample_supervisor_action,
    supervisor_action_bound_penalty,
    supervisor_action_log_prob,
    supervisor_train_step,
    unpack_supervisor_action,
)
from .ppo import (
    ppo_supervisor_loss,
    ppo_supervisor_train_epochs,
    ppo_supervisor_train_step,
    ppo_supervisor_train_with_checkpoint,
)
from .projection import (
    project_supervisor_action_for_audit,
)
from .replay import (
    build_supervisor_baseline_report,
    build_supervisor_corpus_replay_proposals,
    build_supervisor_experiment_manifest,
    build_supervisor_replay_proposal,
)
from .rollouts import (
    build_supervisor_scenario_corpus,
    collect_supervisor_corpus_rollouts,
    collect_supervisor_rollouts,
)

__all__ = [
    "DifferentiableSupervisorConfig",
    "DifferentiableSupervisorPolicy",
    "SupervisorAction",
    "SupervisorActionProjection",
    "SupervisorBaselineReport",
    "SupervisorCorpusReplayProposals",
    "SupervisorExperimentManifest",
    "KuramotoSupervisorScenario",
    "SupervisorLossAux",
    "SupervisorPPOBatch",
    "SupervisorPPOCorpusRollout",
    "SupervisorPPORollout",
    "SupervisorPPOAux",
    "SupervisorPPOCheckpoint",
    "SupervisorPPOTrainResult",
    "SupervisorHandTunedBaselineComparison",
    "SupervisorLearnerProposalComparison",
    "SupervisorRandomBaselineComparison",
    "SupervisorReplayComparison",
    "SupervisorReplayProposal",
    "SupervisorScenarioCorpus",
    "SupervisorStaticBaselineComparison",
    "masked_order_parameter",
    "apply_supervisor_action",
    "closed_loop_supervisor_loss",
    "supervisor_train_step",
    "pack_supervisor_action",
    "unpack_supervisor_action",
    "sample_supervisor_action",
    "supervisor_action_bound_penalty",
    "supervisor_action_to_candidate",
    "supervisor_policy_to_candidate",
    "supervisor_action_log_prob",
    "project_supervisor_action_for_audit",
    "build_supervisor_baseline_report",
    "build_supervisor_experiment_manifest",
    "build_supervisor_corpus_replay_proposals",
    "build_supervisor_replay_proposal",
    "compare_supervisor_hand_tuned_baseline",
    "compare_supervisor_learner_proposals",
    "compare_supervisor_random_baseline",
    "compare_supervisor_replay_proposal",
    "compare_supervisor_static_baseline",
    "ppo_supervisor_loss",
    "ppo_supervisor_train_step",
    "ppo_supervisor_train_epochs",
    "ppo_supervisor_train_with_checkpoint",
    "collect_supervisor_corpus_rollouts",
    "collect_supervisor_rollouts",
    "build_supervisor_scenario_corpus",
    "save_supervisor_ppo_checkpoint",
    "load_supervisor_ppo_checkpoint",
    "control_actions_from_supervisor",
]

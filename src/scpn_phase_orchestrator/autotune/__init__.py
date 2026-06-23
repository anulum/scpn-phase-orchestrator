# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Auto-tune pipeline

"""Public auto-tune facade for reviewable binding and parameter proposals.

The package gathers time-series discovery, phase extraction, frequency
identification, coupling estimation, symbolic SINDy discovery, replay-policy
search, learner proposals, and binding-spec proposal helpers. Public exports
produce review artifacts and inferred parameters only; validation, provenance,
and mutation boundaries are owned by the concrete modules rather than hidden in
the package initializer.
"""

from __future__ import annotations

from scpn_phase_orchestrator.autotune.binding_proposal import (
    propose_binding_from_event_log,
    propose_binding_from_graph,
    propose_binding_from_time_series_csv,
)
from scpn_phase_orchestrator.autotune.candidate_safety_certificate import (
    CandidateSafetyCertificate,
    certify_candidate_safety,
)
from scpn_phase_orchestrator.autotune.coupling_est import estimate_coupling
from scpn_phase_orchestrator.autotune.discovery import (
    TimeSeriesDiscoveryConfig,
    TimeSeriesDiscoveryReport,
    discover_time_series_structure,
    infer_sample_rate_from_time_column,
)
from scpn_phase_orchestrator.autotune.freq_id import (
    FrequencyResult,
    identify_frequencies,
)
from scpn_phase_orchestrator.autotune.knob_attribution import (
    KnobAttribution,
    KnobAttributionConfig,
    KnobAttributionReport,
    attribute_knob_policy,
)
from scpn_phase_orchestrator.autotune.learners import (
    LearnerPolicyProposal,
    generate_hybrid_physics_proposal,
    generate_ppo_like_proposal,
    generate_sac_like_proposal,
)
from scpn_phase_orchestrator.autotune.phase_extract import PhaseResult, extract_phases
from scpn_phase_orchestrator.autotune.pipeline import (
    AutoTuneResult,
    identify_binding_spec,
)
from scpn_phase_orchestrator.autotune.policy_search import (
    AdaptiveReplayPolicySearchConfig,
    AdaptiveReplayPolicySearchResult,
    ReplayPolicyEvaluator,
    ReplayPolicySearchResult,
    search_adaptive_replay_policy,
    search_replay_policy,
)
from scpn_phase_orchestrator.autotune.reward import (
    AutotunePolicyProposal,
    AutotuneRewardReport,
    KnobPolicyCandidate,
    OfflinePolicySearchConfig,
    PolicyProposalConfig,
    RewardConfig,
    RewardObservation,
    SafetyConstraintConfig,
    evaluate_knob_policy,
    generate_offline_policy_candidates,
    propose_replay_policy,
    rank_replay_candidates,
)
from scpn_phase_orchestrator.autotune.sindy import PhaseSINDy

__all__ = [
    "AutoTuneResult",
    "AdaptiveReplayPolicySearchConfig",
    "AdaptiveReplayPolicySearchResult",
    "AutotunePolicyProposal",
    "AutotuneRewardReport",
    "CandidateSafetyCertificate",
    "FrequencyResult",
    "KnobAttribution",
    "KnobAttributionConfig",
    "KnobAttributionReport",
    "KnobPolicyCandidate",
    "LearnerPolicyProposal",
    "OfflinePolicySearchConfig",
    "PolicyProposalConfig",
    "PhaseResult",
    "PhaseSINDy",
    "RewardConfig",
    "RewardObservation",
    "ReplayPolicyEvaluator",
    "ReplayPolicySearchResult",
    "SafetyConstraintConfig",
    "TimeSeriesDiscoveryConfig",
    "TimeSeriesDiscoveryReport",
    "attribute_knob_policy",
    "certify_candidate_safety",
    "discover_time_series_structure",
    "evaluate_knob_policy",
    "estimate_coupling",
    "extract_phases",
    "generate_offline_policy_candidates",
    "generate_hybrid_physics_proposal",
    "generate_ppo_like_proposal",
    "generate_sac_like_proposal",
    "identify_binding_spec",
    "identify_frequencies",
    "infer_sample_rate_from_time_column",
    "propose_binding_from_event_log",
    "propose_binding_from_graph",
    "propose_binding_from_time_series_csv",
    "propose_replay_policy",
    "rank_replay_candidates",
    "search_adaptive_replay_policy",
    "search_replay_policy",
]

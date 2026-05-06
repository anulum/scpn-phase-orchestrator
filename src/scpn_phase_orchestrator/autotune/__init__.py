# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Auto-tune pipeline

from __future__ import annotations

from scpn_phase_orchestrator.autotune.coupling_est import estimate_coupling
from scpn_phase_orchestrator.autotune.freq_id import (
    FrequencyResult,
    identify_frequencies,
)
from scpn_phase_orchestrator.autotune.phase_extract import PhaseResult, extract_phases
from scpn_phase_orchestrator.autotune.pipeline import (
    AutoTuneResult,
    identify_binding_spec,
)
from scpn_phase_orchestrator.autotune.reward import (
    AutotunePolicyProposal,
    AutotuneRewardReport,
    KnobPolicyCandidate,
    OfflinePolicySearchConfig,
    PolicyProposalConfig,
    RewardConfig,
    RewardObservation,
    evaluate_knob_policy,
    generate_offline_policy_candidates,
    propose_replay_policy,
    rank_replay_candidates,
)
from scpn_phase_orchestrator.autotune.sindy import PhaseSINDy

__all__ = [
    "AutoTuneResult",
    "AutotunePolicyProposal",
    "AutotuneRewardReport",
    "FrequencyResult",
    "KnobPolicyCandidate",
    "OfflinePolicySearchConfig",
    "PolicyProposalConfig",
    "PhaseResult",
    "PhaseSINDy",
    "RewardConfig",
    "RewardObservation",
    "evaluate_knob_policy",
    "estimate_coupling",
    "extract_phases",
    "generate_offline_policy_candidates",
    "identify_binding_spec",
    "identify_frequencies",
    "propose_replay_policy",
    "rank_replay_candidates",
]

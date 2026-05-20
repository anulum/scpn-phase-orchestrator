# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Supervisor subsystem

"""Public supervisor facade for policy, topology, hierarchy, and audit helpers.

The supervisor package gathers regime management, event buses, causal and
predictive rollouts, Petri nets, formal exporters, hierarchy transports,
morphogenetic topology diagnostics, value-alignment guards, sheaf/topology
supervisors, and policy-rule engines. Importing the facade exposes types and
functions only; concrete modules own validation, state mutation, export text,
runtime transport boundaries, and any emitted control proposals.
"""

from __future__ import annotations

from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.supervisor.alignment import (
    ValueAlignmentDecision,
    ValueAlignmentGuard,
    ValueAlignmentPolicy,
    ValueConstraint,
    ValueScoreCounterfactual,
    ValueViolation,
    calibrate_value_alignment_replay_evidence,
    value_alignment_policy_from_binding_spec,
    value_alignment_policy_from_template,
)
from scpn_phase_orchestrator.supervisor.byzantine import (
    build_bft_meta_orchestrator_manifest,
    sign_policy_proposal,
)
from scpn_phase_orchestrator.supervisor.causal import (
    CausalGraphEstimate,
    CausalInfluenceEdge,
    CausalInterventionEngine,
    CounterfactualRollout,
    InterventionParameters,
    learn_causal_graph,
)
from scpn_phase_orchestrator.supervisor.events import EventBus, RegimeEvent
from scpn_phase_orchestrator.supervisor.formal_export import (
    FormalCheckerAvailability,
    FormalCheckerCommand,
    FormalSafetyProperty,
    FormalVerificationPackage,
    PrismExport,
    TLAExport,
    audit_formal_checker_availability,
    build_formal_verification_package,
    export_petri_net_prism,
    export_petri_net_tla,
    export_policy_rules_prism,
    export_policy_rules_tla,
    export_stl_specs_prism,
)
from scpn_phase_orchestrator.supervisor.hierarchy import (
    ChildSupervisorSummary,
    HierarchicalOrchestrationPlan,
    HierarchyConsensusRound,
    HierarchyConsensusState,
    HierarchyEscalation,
    HierarchySyncEnvelope,
    HierarchySyncLedger,
    HierarchyTransportRuntime,
    build_hierarchical_orchestration_plan,
    build_hierarchy_sync_envelope,
    ingest_hierarchy_sync_envelopes,
    load_hierarchy_sync_envelope,
    simulate_hierarchy_gossip_consensus,
)
from scpn_phase_orchestrator.supervisor.hierarchy_adapters import (
    HierarchyAdapterResult,
    handle_hierarchy_frame,
    handle_hierarchy_rest_payload,
    replay_hierarchy_jsonl,
)
from scpn_phase_orchestrator.supervisor.lineage import (
    build_autopoietic_lineage_sandbox,
    build_intergenerational_policy_inheritance,
)
from scpn_phase_orchestrator.supervisor.morphogenetic import (
    MorphogeneticFieldPolicy,
    MorphogeneticFieldResult,
    MorphogeneticFieldSnapshot,
    MorphogeneticFieldState,
    MorphogeneticFieldSVG,
    MorphogeneticTopologySupervisor,
    build_morphogenetic_field_snapshot,
    render_morphogenetic_field_svg,
)
from scpn_phase_orchestrator.supervisor.petri_adapter import PetriNetAdapter
from scpn_phase_orchestrator.supervisor.petri_net import (
    Arc,
    Marking,
    PetriNet,
    Place,
    Transition,
)
from scpn_phase_orchestrator.supervisor.policy import SupervisorPolicy
from scpn_phase_orchestrator.supervisor.policy_diagnostics import (
    PolicyDryRunReport,
    PolicyDryRunStep,
    dry_run_policy_rules,
)
from scpn_phase_orchestrator.supervisor.policy_rules import (
    CompoundCondition,
    PolicyAction,
    PolicyCondition,
    PolicyEngine,
    PolicyRule,
    PolicySTLAutomaton,
    PolicySTLResult,
    PolicySTLSpec,
    evaluate_policy_stl_specs,
    load_policy_rules,
    load_policy_stl_specs,
    synthesise_policy_stl_automata,
    synthesize_policy_stl_automata,
)
from scpn_phase_orchestrator.supervisor.predictive import (
    FEPHierarchyAssessment,
    FEPHierarchyChildAssessment,
    FEPPredictionAssessment,
    FEPPredictiveSupervisor,
    Prediction,
    PredictiveSupervisor,
    assess_fep_hierarchy,
)
from scpn_phase_orchestrator.supervisor.regimes import Regime, RegimeManager
from scpn_phase_orchestrator.supervisor.sheaf import (
    SheafCoherenceResult,
    SheafCoherenceSupervisor,
    SheafObstructionSummary,
    build_sheaf_obstruction_summary,
    sheaf_coherence,
    sheaf_laplacian,
)
from scpn_phase_orchestrator.supervisor.strange_loop import (
    StrangeLoopAssessment,
    StrangeLoopSupervisor,
)
from scpn_phase_orchestrator.supervisor.topology import (
    HigherOrderTopologySupervisor,
    TopologyMutationPolicy,
    TopologyMutationResult,
)

__all__ = [
    "Arc",
    "CausalInterventionEngine",
    "CausalGraphEstimate",
    "CausalInfluenceEdge",
    "ChildSupervisorSummary",
    "CompoundCondition",
    "ControlAction",
    "CounterfactualRollout",
    "EventBus",
    "FEPHierarchyAssessment",
    "FEPHierarchyChildAssessment",
    "FEPPredictionAssessment",
    "FEPPredictiveSupervisor",
    "FormalCheckerAvailability",
    "FormalCheckerCommand",
    "FormalSafetyProperty",
    "FormalVerificationPackage",
    "HierarchicalOrchestrationPlan",
    "HierarchyAdapterResult",
    "HierarchyConsensusRound",
    "HierarchyConsensusState",
    "HierarchyEscalation",
    "HierarchySyncEnvelope",
    "HierarchySyncLedger",
    "HierarchyTransportRuntime",
    "HigherOrderTopologySupervisor",
    "InterventionParameters",
    "Marking",
    "MorphogeneticFieldPolicy",
    "MorphogeneticFieldResult",
    "MorphogeneticFieldSnapshot",
    "MorphogeneticFieldSVG",
    "MorphogeneticFieldState",
    "MorphogeneticTopologySupervisor",
    "PetriNet",
    "PetriNetAdapter",
    "Place",
    "PolicyAction",
    "PolicyDryRunReport",
    "PolicyDryRunStep",
    "PolicyCondition",
    "PolicyEngine",
    "PolicyRule",
    "PolicySTLAutomaton",
    "PolicySTLResult",
    "PolicySTLSpec",
    "PrismExport",
    "Prediction",
    "PredictiveSupervisor",
    "Regime",
    "RegimeEvent",
    "RegimeManager",
    "SheafCoherenceResult",
    "SheafCoherenceSupervisor",
    "SheafObstructionSummary",
    "StrangeLoopAssessment",
    "StrangeLoopSupervisor",
    "SupervisorPolicy",
    "Transition",
    "TopologyMutationPolicy",
    "TopologyMutationResult",
    "TLAExport",
    "ValueAlignmentDecision",
    "ValueAlignmentGuard",
    "ValueAlignmentPolicy",
    "ValueScoreCounterfactual",
    "ValueConstraint",
    "ValueViolation",
    "calibrate_value_alignment_replay_evidence",
    "assess_fep_hierarchy",
    "audit_formal_checker_availability",
    "build_bft_meta_orchestrator_manifest",
    "build_autopoietic_lineage_sandbox",
    "build_formal_verification_package",
    "build_hierarchical_orchestration_plan",
    "build_hierarchy_sync_envelope",
    "build_intergenerational_policy_inheritance",
    "build_morphogenetic_field_snapshot",
    "build_sheaf_obstruction_summary",
    "render_morphogenetic_field_svg",
    "dry_run_policy_rules",
    "export_petri_net_prism",
    "export_petri_net_tla",
    "export_policy_rules_prism",
    "export_policy_rules_tla",
    "export_stl_specs_prism",
    "evaluate_policy_stl_specs",
    "handle_hierarchy_frame",
    "handle_hierarchy_rest_payload",
    "load_policy_rules",
    "load_policy_stl_specs",
    "ingest_hierarchy_sync_envelopes",
    "load_hierarchy_sync_envelope",
    "learn_causal_graph",
    "sheaf_coherence",
    "sheaf_laplacian",
    "replay_hierarchy_jsonl",
    "simulate_hierarchy_gossip_consensus",
    "sign_policy_proposal",
    "synthesise_policy_stl_automata",
    "synthesize_policy_stl_automata",
    "value_alignment_policy_from_binding_spec",
    "value_alignment_policy_from_template",
]

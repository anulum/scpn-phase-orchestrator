# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Supervisor subsystem

from __future__ import annotations

from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.supervisor.alignment import (
    ValueAlignmentDecision,
    ValueAlignmentGuard,
    ValueAlignmentPolicy,
    ValueConstraint,
    ValueScoreCounterfactual,
    ValueViolation,
    value_alignment_policy_from_binding_spec,
    value_alignment_policy_from_template,
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
    PrismExport,
    TLAExport,
    export_petri_net_prism,
    export_petri_net_tla,
    export_policy_rules_prism,
    export_policy_rules_tla,
    export_stl_specs_prism,
)
from scpn_phase_orchestrator.supervisor.hierarchy import (
    ChildSupervisorSummary,
    HierarchicalOrchestrationPlan,
    HierarchyEscalation,
    HierarchySyncEnvelope,
    HierarchySyncLedger,
    build_hierarchical_orchestration_plan,
    build_hierarchy_sync_envelope,
    ingest_hierarchy_sync_envelopes,
)
from scpn_phase_orchestrator.supervisor.morphogenetic import (
    MorphogeneticFieldPolicy,
    MorphogeneticFieldResult,
    MorphogeneticFieldState,
    MorphogeneticTopologySupervisor,
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
    "HierarchicalOrchestrationPlan",
    "HierarchyEscalation",
    "HierarchySyncEnvelope",
    "HierarchySyncLedger",
    "HigherOrderTopologySupervisor",
    "InterventionParameters",
    "Marking",
    "MorphogeneticFieldPolicy",
    "MorphogeneticFieldResult",
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
    "assess_fep_hierarchy",
    "build_hierarchical_orchestration_plan",
    "build_hierarchy_sync_envelope",
    "dry_run_policy_rules",
    "export_petri_net_prism",
    "export_petri_net_tla",
    "export_policy_rules_prism",
    "export_policy_rules_tla",
    "export_stl_specs_prism",
    "evaluate_policy_stl_specs",
    "load_policy_rules",
    "load_policy_stl_specs",
    "ingest_hierarchy_sync_envelopes",
    "learn_causal_graph",
    "sheaf_coherence",
    "sheaf_laplacian",
    "synthesise_policy_stl_automata",
    "synthesize_policy_stl_automata",
    "value_alignment_policy_from_binding_spec",
    "value_alignment_policy_from_template",
]

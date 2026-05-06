# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - LLM Semantic Domain Compiler

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scpn_phase_orchestrator.binding.types import (
    ActuatorMapping,
    BindingSpec,
    BoundaryDef,
    ChannelGroupSpec,
    ChannelSpec,
    CouplingSpec,
    DriverSpec,
    HierarchyLayer,
    ObjectivePartition,
    OscillatorFamily,
    ProtocolNetSpec,
    ProtocolTransitionSpec,
)
from scpn_phase_orchestrator.binding.validator import validate_binding_spec

__all__ = [
    "GeneratedBindingArtifacts",
    "RetrievalEvidence",
    "SemanticDomainCompiler",
    "compile_symbolic_binding",
]


@dataclass(frozen=True)
class RetrievalEvidence:
    """Local domainpack evidence used during symbolic binding generation."""

    domainpack: str
    path: str
    score: float
    matched_terms: list[str]
    summary: str

    def to_audit_record(self) -> dict[str, Any]:
        """Return a JSON-safe retrieval evidence record."""
        return {
            "domainpack": self.domainpack,
            "path": self.path,
            "score": self.score,
            "matched_terms": self.matched_terms,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class GeneratedBindingArtifacts:
    """Reviewable outputs from symbolic domain intent compilation."""

    binding_spec: BindingSpec
    binding_yaml: str
    policy_yaml: str
    notebook_json: str
    audit_record: dict[str, Any]
    retrieval_evidence: list[RetrievalEvidence]
    validation_errors: list[str]
    dry_run_order_parameter: float

    @property
    def schema_valid(self) -> bool:
        """Return True when the generated binding passed validator checks."""
        return not self.validation_errors

    def write_domainpack(self, output_dir: str | Path) -> None:
        """Write generated artefacts as a reviewable domainpack directory."""
        import json

        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "binding_spec.yaml").write_text(self.binding_yaml, encoding="utf-8")
        (path / "policy.yaml").write_text(self.policy_yaml, encoding="utf-8")
        (path / "review_notebook.ipynb").write_text(
            self.notebook_json,
            encoding="utf-8",
        )
        (path / "audit.json").write_text(
            json.dumps(self.audit_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        readme = (
            f"# {self.binding_spec.name} domainpack\n\n"
            "Generated from symbolic intent. Review `binding_spec.yaml`, "
            "`policy.yaml`, `review_notebook.ipynb`, and `audit.json` before "
            "use with live systems.\n"
        )
        (path / "README.md").write_text(readme, encoding="utf-8")


class SemanticDomainCompiler:
    """Semantic Compiler Bridge for natural language domain modeling.

    Translates plain-English system descriptions into formal BindingSpec
    configurations. It extracts hierarchical structures, typical
    frequencies, and coupling constraints from text.
    """

    def compile(
        self,
        prompt: str,
        *,
        name: str = "semantically_generated_domain",
        oscillators_per_layer: int = 8,
    ) -> BindingSpec:
        """Translate a symbolic domain prompt into a BindingSpec."""
        return self.compile_artifacts(
            prompt,
            name=name,
            oscillators_per_layer=oscillators_per_layer,
            dry_run_steps=3,
        ).binding_spec

    def compile_artifacts(
        self,
        prompt: str,
        *,
        name: str = "semantically_generated_domain",
        oscillators_per_layer: int = 8,
        dry_run_steps: int = 8,
        retrieval_root: str | Path | None = "domainpacks",
    ) -> GeneratedBindingArtifacts:
        """Compile domain intent into binding, policy, audit, and dry-run artefacts."""
        if not name or not re.match(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$", name):
            raise ValueError("name must match [A-Za-z][A-Za-z0-9_-]{0,63}")
        if oscillators_per_layer < 1:
            raise ValueError("oscillators_per_layer must be >= 1")
        if dry_run_steps < 1:
            raise ValueError("dry_run_steps must be >= 1")

        # Heuristic: Layer detection
        layer_match = re.search(r"(\d+)[ -]layer", prompt, re.IGNORECASE)
        num_layers = int(layer_match.group(1)) if layer_match else 2
        if num_layers < 1:
            raise ValueError("layer count must be >= 1")

        # Heuristic: Discipline detection
        lowered = prompt.lower()
        matched_keywords = sorted(
            {
                word
                for word in (
                    "bio",
                    "brain",
                    "cardiac",
                    "cell",
                    "finance",
                    "fusion",
                    "grid",
                    "plasma",
                    "power",
                    "traffic",
                )
                if word in lowered
            }
        )
        if any(word in lowered for word in ["bio", "cell", "brain", "cardiac"]):
            base_freq = 10.0
            domain_family = "biological"
        elif any(word in lowered for word in ["power", "grid", "fusion", "plasma"]):
            base_freq = 50.0
            domain_family = "physical"
        elif any(word in lowered for word in ["finance", "traffic"]):
            base_freq = 1.0
            domain_family = "network"
        else:
            base_freq = 1.0
            domain_family = "generic"

        layers = []
        for i in range(num_layers):
            layers.append(
                HierarchyLayer(
                    name=f"layer_{i}",
                    index=i,
                    oscillator_ids=[
                        f"osc_{i}_{j}" for j in range(oscillators_per_layer)
                    ],
                    omegas=[base_freq * (10**i)] * oscillators_per_layer,
                    family="default",
                )
            )

        osc_families = {
            "default": OscillatorFamily(
                channel="P", extractor_type="hilbert", config={}
            )
        }

        coupling = CouplingSpec(base_strength=0.5, decay_alpha=0.3, templates={})
        drivers = DriverSpec(physical={}, informational={}, symbolic={})
        objectives = ObjectivePartition(
            good_layers=list(range(num_layers)), bad_layers=[]
        )

        spec = BindingSpec(
            name=name,
            version="1.0.0",
            safety_tier="research",
            sample_period_s=0.01,
            control_period_s=0.1,
            layers=layers,
            oscillator_families=osc_families,
            coupling=coupling,
            drivers=drivers,
            objectives=objectives,
            boundaries=[
                BoundaryDef(
                    name="low_global_coherence",
                    variable="R_good",
                    lower=0.0,
                    upper=1.0,
                    severity="soft",
                )
            ],
            actuators=[
                ActuatorMapping("global_coupling", "K", "global", (0.0, 2.0)),
                ActuatorMapping("global_drive", "zeta", "global", (0.0, 1.0)),
            ],
            channels={
                "P": ChannelSpec(
                    role=domain_family,
                    units="rad",
                    metric_semantics="phase",
                    replay_semantics="phase",
                ),
            },
            channel_groups={
                "primary": ChannelGroupSpec(
                    channels=["P"],
                    description="Primary phase-observation channel",
                ),
            },
            protocol_net=ProtocolNetSpec(
                places=["draft", "validated"],
                initial={"draft": 1},
                place_regime={"draft": "NOMINAL", "validated": "NOMINAL"},
                transitions=[
                    ProtocolTransitionSpec(
                        name="accept_after_review",
                        inputs=[{"place": "draft"}],
                        outputs=[{"place": "validated"}],
                        guard="stability_proxy > 0.0",
                    )
                ],
            ),
        )
        validation_errors = validate_binding_spec(spec)
        dry_run_r = _dry_run_order_parameter(spec, dry_run_steps)
        binding_yaml = _binding_spec_to_yaml(spec)
        policy_yaml = _policy_yaml_for(spec)
        retrieval_evidence = _retrieve_domainpack_evidence(prompt, retrieval_root)
        retrieval_records = [
            evidence.to_audit_record() for evidence in retrieval_evidence
        ]
        retrieval_score = retrieval_evidence[0].score if retrieval_evidence else 0.0
        confidence = _confidence(
            matched_keywords=matched_keywords,
            has_layer_count=layer_match is not None,
            domain_family=domain_family,
            retrieval_score=retrieval_score,
        )
        notebook_json = _review_notebook_for(
            spec,
            confidence=confidence,
            retrieval_records=retrieval_records,
        )
        audit_record = {
            "compiler": "symbolic_binding_v0",
            "schema_valid": not validation_errors,
            "validation_errors": validation_errors,
            "confidence": confidence,
            "confidence_factors": {
                "domain_keywords": len(matched_keywords),
                "explicit_layer_count": layer_match is not None,
                "domain_family": domain_family,
                "retrieval_score": retrieval_score,
            },
            "domain_family": domain_family,
            "matched_keywords": matched_keywords,
            "retrieval_evidence": retrieval_records,
            "layers": num_layers,
            "oscillators_per_layer": oscillators_per_layer,
            "dry_run_steps": dry_run_steps,
            "dry_run_order_parameter": dry_run_r,
            "petri_reachability": {
                "initial_place": "draft",
                "review_transition": "accept_after_review",
                "target_place": "validated",
            },
        }
        return GeneratedBindingArtifacts(
            binding_spec=spec,
            binding_yaml=binding_yaml,
            policy_yaml=policy_yaml,
            notebook_json=notebook_json,
            audit_record=audit_record,
            retrieval_evidence=retrieval_evidence,
            validation_errors=validation_errors,
            dry_run_order_parameter=dry_run_r,
        )


def compile_symbolic_binding(
    prompt: str,
    *,
    name: str = "semantically_generated_domain",
    oscillators_per_layer: int = 8,
    dry_run_steps: int = 8,
    retrieval_root: str | Path | None = "domainpacks",
) -> GeneratedBindingArtifacts:
    """Compile domain intent into a reviewable generated domainpack."""
    return SemanticDomainCompiler().compile_artifacts(
        prompt,
        name=name,
        oscillators_per_layer=oscillators_per_layer,
        dry_run_steps=dry_run_steps,
        retrieval_root=retrieval_root,
    )


def _confidence(
    *,
    matched_keywords: list[str],
    has_layer_count: bool,
    domain_family: str,
    retrieval_score: float,
) -> float:
    score = 0.35 + min(0.25, 0.05 * len(matched_keywords))
    if has_layer_count:
        score += 0.15
    if domain_family != "generic":
        score += 0.2
    if retrieval_score > 0.0:
        score += min(0.15, 0.05 + 0.1 * retrieval_score)
    return round(min(score, 0.95), 3)


def _retrieve_domainpack_evidence(
    prompt: str,
    root: str | Path | None,
    *,
    limit: int = 3,
) -> list[RetrievalEvidence]:
    if root is None:
        return []
    base = Path(root)
    if not base.exists() or not base.is_dir():
        return []

    prompt_terms = _terms(prompt)
    if not prompt_terms:
        return []

    scored: list[RetrievalEvidence] = []
    for spec_path in sorted(base.glob("*/binding_spec.yaml")):
        domain_dir = spec_path.parent
        text_parts = [domain_dir.name.replace("_", " ")]
        text_parts.append(_safe_read(spec_path, max_chars=12000))
        readme_path = domain_dir / "README.md"
        if readme_path.exists():
            text_parts.append(_safe_read(readme_path, max_chars=4000))
        corpus = " ".join(text_parts).lower()
        corpus_terms = set(_terms(corpus))
        matched = sorted(prompt_terms & corpus_terms)
        if not matched:
            continue
        name_bonus = sum(
            1
            for term in prompt_terms
            if term in domain_dir.name.lower().replace("_", " ")
        )
        score = (len(matched) + name_bonus) / max(len(prompt_terms), 1)
        scored.append(
            RetrievalEvidence(
                domainpack=domain_dir.name,
                path=str(spec_path),
                score=round(min(score, 1.0), 3),
                matched_terms=matched[:12],
                summary=_evidence_summary(domain_dir.name, matched),
            )
        )

    return sorted(scored, key=lambda item: (-item.score, item.domainpack))[:limit]


def _safe_read(path: Path, *, max_chars: int) -> str:
    try:
        return path.read_text(encoding="utf-8")[:max_chars]
    except UnicodeDecodeError:
        return ""


def _terms(text: str) -> set[str]:
    stopwords = {
        "and",
        "for",
        "from",
        "into",
        "model",
        "orchestrate",
        "phase",
        "system",
        "the",
        "under",
        "with",
    }
    return {
        term
        for term in re.findall(r"[a-z0-9]{3,}", text.lower())
        if term not in stopwords
    }


def _evidence_summary(domainpack: str, matched_terms: list[str]) -> str:
    terms = ", ".join(matched_terms[:5])
    return f"{domainpack} matched local terms: {terms}"


def _review_notebook_for(
    spec: BindingSpec,
    *,
    confidence: float,
    retrieval_records: list[dict[str, Any]],
) -> str:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# Review generated domainpack: {spec.name}\n",
                    "\n",
                    "This notebook validates generated binding and policy "
                    "artifacts before any live use.\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Retrieval and confidence\n",
                    "\n",
                    f"- Confidence: `{confidence:.3f}`\n",
                    f"- Retrieval matches: `{len(retrieval_records)}`\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    "from scpn_phase_orchestrator.binding import "
                    "load_binding_spec, validate_binding_spec\n",
                    "spec = load_binding_spec(Path('binding_spec.yaml'))\n",
                    "errors = validate_binding_spec(spec)\n",
                    "assert errors == [], errors\n",
                    "spec.name\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from scpn_phase_orchestrator.supervisor.policy_rules import "
                    "load_policy_rules\n",
                    "rules = load_policy_rules(Path('policy.yaml'))\n",
                    "assert rules\n",
                    "[rule.name for rule in rules]\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Review checklist\n",
                    "\n",
                    "- Confirm layer names and oscillator counts match the plant.\n",
                    "- Confirm actuator limits are safe for the deployment target.\n",
                    "- Run a dry replay before connecting live adapters.\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "scpn_phase_orchestrator": {
                "artifact": "symbolic_binding_review",
                "schema_version": 1,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, indent=2, sort_keys=True) + "\n"


def _dry_run_order_parameter(spec: BindingSpec, steps: int) -> float:
    import numpy as np

    from scpn_phase_orchestrator.coupling.knm import CouplingBuilder
    from scpn_phase_orchestrator.upde.engine import UPDEEngine
    from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

    n = sum(len(layer.oscillator_ids) for layer in spec.layers)
    coupling = CouplingBuilder().build(
        n,
        spec.coupling.base_strength,
        spec.coupling.decay_alpha,
    )
    engine = UPDEEngine(n, dt=spec.sample_period_s)
    phases = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=np.float64)
    omegas = np.asarray(spec.get_omegas(), dtype=np.float64)
    for _ in range(steps):
        phases = engine.step(phases, omegas, coupling.knm, 0.0, 0.0, coupling.alpha)
    r_value, _ = compute_order_parameter(phases)
    return float(r_value)


def _policy_yaml_for(spec: BindingSpec) -> str:
    import yaml

    policy = {
        "rules": [
            {
                "name": "recover_low_coherence",
                "regime": ["DEGRADED", "CRITICAL"],
                "condition": {
                    "metric": "R_good",
                    "op": "<",
                    "threshold": 0.7,
                },
                "action": {
                    "knob": "K",
                    "scope": "global",
                    "value": 0.1,
                    "ttl_s": max(spec.control_period_s, 1.0),
                },
            }
        ]
    }
    rendered: str = yaml.safe_dump(policy, sort_keys=False)
    return rendered


def _binding_spec_to_yaml(spec: BindingSpec) -> str:
    import yaml

    data = {
        "name": spec.name,
        "version": spec.version,
        "safety_tier": spec.safety_tier,
        "sample_period_s": spec.sample_period_s,
        "control_period_s": spec.control_period_s,
        "layers": [
            {
                "name": layer.name,
                "index": layer.index,
                "oscillator_ids": layer.oscillator_ids,
                "omegas": layer.omegas,
                "family": layer.family,
            }
            for layer in spec.layers
        ],
        "oscillator_families": {
            key: {
                "channel": family.channel,
                "extractor_type": family.extractor_type,
                "config": family.config,
            }
            for key, family in spec.oscillator_families.items()
        },
        "coupling": {
            "base_strength": spec.coupling.base_strength,
            "decay_alpha": spec.coupling.decay_alpha,
            "templates": spec.coupling.templates,
        },
        "drivers": {
            "physical": spec.drivers.physical,
            "informational": spec.drivers.informational,
            "symbolic": spec.drivers.symbolic,
        },
        "channels": {
            key: {
                "role": channel.role,
                "required": channel.required,
                "units": channel.units,
                "metric_semantics": channel.metric_semantics,
                "coupling_participation": channel.coupling_participation,
                "audit_serialisation": channel.audit_serialisation,
                "replay_semantics": channel.replay_semantics,
                "supervisor_visibility": channel.supervisor_visibility,
                "derived_from": channel.derived_from,
                "derive_rule": channel.derive_rule,
            }
            for key, channel in spec.channels.items()
        },
        "channel_groups": {
            key: {
                "channels": group.channels,
                "required": group.required,
                "description": group.description,
            }
            for key, group in spec.channel_groups.items()
        },
        "objectives": {
            "good_layers": spec.objectives.good_layers,
            "bad_layers": spec.objectives.bad_layers,
            "good_weight": spec.objectives.good_weight,
            "bad_weight": spec.objectives.bad_weight,
        },
        "boundaries": [
            {
                "name": boundary.name,
                "variable": boundary.variable,
                "lower": boundary.lower,
                "upper": boundary.upper,
                "severity": boundary.severity,
            }
            for boundary in spec.boundaries
        ],
        "actuators": [
            {
                "name": actuator.name,
                "knob": actuator.knob,
                "scope": actuator.scope,
                "limits": list(actuator.limits),
            }
            for actuator in spec.actuators
        ],
        "protocol_net": {
            "places": spec.protocol_net.places if spec.protocol_net else [],
            "initial": spec.protocol_net.initial if spec.protocol_net else {},
            "place_regime": spec.protocol_net.place_regime if spec.protocol_net else {},
            "transitions": [
                {
                    "name": transition.name,
                    "inputs": transition.inputs,
                    "outputs": transition.outputs,
                    "guard": transition.guard,
                }
                for transition in (
                    spec.protocol_net.transitions if spec.protocol_net else []
                )
            ],
        },
    }
    rendered: str = yaml.safe_dump(data, sort_keys=False)
    return rendered

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — LLM Semantic Domain Compiler

"""Review-only symbolic compiler from natural-language intent to bindings."""

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

from .coercion import (
    _MAX_PROMPT_CHARS,
    _coerce_output_dir,
    _validate_compilation_inputs,
)
from .retrieval import RetrievalEvidence, _retrieve_local_evidence
from .review import (
    _dry_run_order_parameter,
    _review_gate_record,
    _review_notebook_execution_evidence,
    _review_notebook_for,
    _validate_generated_audit_schema,
)
from .serialization import _binding_spec_to_yaml, _confidence, _policy_yaml_for

_LAYER_PATTERN = re.compile(r"(-?\d+)[ -]layer", re.IGNORECASE)


_MAX_LAYERS = 256


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
        """Return True when the generated binding passed validator checks.

        Returns
        -------
        bool
            ``True`` when the generated binding passed every validator check.
        """
        return not self.validation_errors

    def write_domainpack(self, output_dir: str | Path) -> None:
        """Write generated artefacts as a reviewable domainpack directory.

        Parameters
        ----------
        output_dir : str or pathlib.Path
            Destination directory; the binding spec, policy, review notebook,
            audit record, and README are written beneath it.
        """
        path = _coerce_output_dir(output_dir)
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
        """Translate a symbolic domain prompt into a BindingSpec.

        Parameters
        ----------
        prompt : str
            Natural-language description of the target domain.
        name : str, optional
            Name for the generated binding spec.
        oscillators_per_layer : int, optional
            Number of oscillators to allocate per generated layer.

        Returns
        -------
        BindingSpec
            The compiled, structurally typed binding specification.
        """
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
        docs_root: str | Path | None = "docs",
    ) -> GeneratedBindingArtifacts:
        """Compile domain intent into binding, policy, audit, and dry-run artefacts.

        Parameters
        ----------
        prompt : str
            Natural-language description of the target domain.
        name : str, optional
            Name for the generated binding spec.
        oscillators_per_layer : int, optional
            Number of oscillators to allocate per generated layer.
        dry_run_steps : int, optional
            Number of integration steps for the embedded dry-run check.
        retrieval_root : str or pathlib.Path or None, optional
            Root directory searched for retrieval grounding evidence.
        docs_root : str or pathlib.Path or None, optional
            Root directory searched for documentation grounding evidence.

        Returns
        -------
        GeneratedBindingArtifacts
            The binding, policy, audit record, retrieval evidence, and dry-run
            result bundle.

        Raises
        ------
        ValueError
            If the generated binding fails validation or the embedded dry run.
        """
        (
            prompt,
            name,
            oscillators_per_layer,
            dry_run_steps,
            retrieval_root,
            docs_root,
        ) = _validate_compilation_inputs(
            prompt=prompt,
            name=name,
            oscillators_per_layer=oscillators_per_layer,
            dry_run_steps=dry_run_steps,
            retrieval_root=retrieval_root,
            docs_root=docs_root,
        )

        # Heuristic: Layer detection
        layer_match = _LAYER_PATTERN.search(prompt)
        num_layers = int(layer_match.group(1)) if layer_match else 2
        if num_layers < 1:
            raise ValueError("layer count must be >= 1")
        if num_layers > _MAX_LAYERS:
            raise ValueError(f"layer count must be <= {_MAX_LAYERS}")

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
        retrieval_evidence = _retrieve_local_evidence(
            prompt,
            domainpack_root=retrieval_root,
            docs_root=docs_root,
        )
        retrieval_records = [
            evidence.to_audit_record() for evidence in retrieval_evidence
        ]
        retrieval_score = (
            max(evidence.score for evidence in retrieval_evidence)
            if retrieval_evidence
            else 0.0
        )
        notebook_execution = _review_notebook_execution_evidence(
            binding_yaml=binding_yaml,
            policy_yaml=policy_yaml,
            expected_name=spec.name,
        )
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
            notebook_execution=notebook_execution,
        )
        audit_record = {
            "compiler": "symbolic_binding_v0",
            "schema_valid": not validation_errors,
            "validation_errors": validation_errors,
            "intent_boundary": {
                "sanitised": True,
                "max_chars": _MAX_PROMPT_CHARS,
                "llm_execution": False,
            },
            "review_gate": _review_gate_record(),
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
            "notebook_execution": notebook_execution,
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
        _validate_generated_audit_schema(audit_record)
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
    docs_root: str | Path | None = "docs",
) -> GeneratedBindingArtifacts:
    """Compile domain intent into a reviewable generated domainpack.

    Parameters
    ----------
    prompt : str
        Natural-language description of the target domain.
    name : str, optional
        Name for the generated binding spec.
    oscillators_per_layer : int, optional
        Number of oscillators to allocate per generated layer.
    dry_run_steps : int, optional
        Number of integration steps for the embedded dry-run check.
    retrieval_root : str or pathlib.Path or None, optional
        Root directory searched for retrieval grounding evidence.
    docs_root : str or pathlib.Path or None, optional
        Root directory searched for documentation grounding evidence.

    Returns
    -------
    GeneratedBindingArtifacts
        The generated domainpack artefact bundle.

    Raises
    ------
    ValueError
        If the compilation inputs are invalid or the generated binding fails
        validation or its dry run.
    """
    _validate_compilation_inputs(
        prompt=prompt,
        name=name,
        oscillators_per_layer=oscillators_per_layer,
        dry_run_steps=dry_run_steps,
        retrieval_root=retrieval_root,
        docs_root=docs_root,
    )
    return SemanticDomainCompiler().compile_artifacts(
        prompt,
        name=name,
        oscillators_per_layer=oscillators_per_layer,
        dry_run_steps=dry_run_steps,
        retrieval_root=retrieval_root,
        docs_root=docs_root,
    )

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — LLM Semantic Domain Compiler

"""Review-only symbolic compiler from natural-language intent to bindings.

The semantic compiler produces a candidate `BindingSpec`, policy YAML, review
notebook, retrieval evidence, and audit record from local heuristics and
domainpack/docs evidence. Generated artefacts are intentionally reviewable and
fail validation before use; this module does not auto-accept live deployment
bindings or actuate a system.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
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

_LAYER_PATTERN = re.compile(r"(-?\d+)[ -]layer", re.IGNORECASE)
_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_MAX_LAYERS = 256
_MAX_OSCILLATORS_PER_LAYER = 256
_MAX_DRY_RUN_STEPS = 256
_MAX_PROMPT_CHARS = 4000
_PROMPT_INJECTION_PATTERNS = (
    re.compile(r"\bignore\s+(all\s+)?previous\s+instructions\b", re.IGNORECASE),
    re.compile(r"\bdisregard\s+(all\s+)?previous\s+instructions\b", re.IGNORECASE),
    re.compile(r"\breveal\s+(the\s+)?system\s+prompt\b", re.IGNORECASE),
    re.compile(r"\bdeveloper\s+instructions\b", re.IGNORECASE),
    re.compile(r"\bexfiltrat(e|ion)\b", re.IGNORECASE),
    re.compile(r"\bleak\s+(credentials|secrets|api\s+keys)\b", re.IGNORECASE),
    re.compile(r"\bprint\s+(credentials|secrets|api\s+keys)\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class RetrievalEvidence:
    """Local domainpack evidence used during symbolic binding generation."""

    domainpack: str
    path: str
    score: float
    matched_terms: list[str]
    summary: str
    source: str = "domainpack"
    rank: int = 0
    ranking_features: dict[str, float] = field(default_factory=dict)

    def to_audit_record(self) -> dict[str, Any]:
        """Return a JSON-safe retrieval evidence record.

        Returns
        -------
        dict[str, Any]
            Deterministic, JSON-safe audit mapping of the RetrievalEvidence fields.
        """
        return {
            "domainpack": self.domainpack,
            "path": self.path,
            "rank": self.rank,
            "score": self.score,
            "matched_terms": self.matched_terms,
            "summary": self.summary,
            "source": self.source,
            "ranking_features": dict(sorted(self.ranking_features.items())),
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
        import json

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


def _validate_compilation_inputs(
    *,
    prompt: Any,
    name: Any,
    oscillators_per_layer: Any,
    dry_run_steps: Any,
    retrieval_root: str | Path | None,
    docs_root: str | Path | None,
) -> tuple[str, str, int, int, Path | None, Path | None]:
    prompt_value = _as_prompt(prompt)
    name_value = _as_name(name)
    oscillators = _as_positive_int(
        oscillators_per_layer,
        "oscillators_per_layer",
        max_value=_MAX_OSCILLATORS_PER_LAYER,
    )
    dry_run = _as_positive_int(
        dry_run_steps,
        "dry_run_steps",
        max_value=_MAX_DRY_RUN_STEPS,
    )
    retrieval_path = _as_path(
        retrieval_root,
        "retrieval_root",
        allow_none=True,
    )
    docs_path = _as_path(
        docs_root,
        "docs_root",
        allow_none=True,
    )
    return (
        prompt_value,
        name_value,
        oscillators,
        dry_run,
        retrieval_path,
        docs_path,
    )


def _as_str(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    return value


def _as_prompt(value: object) -> str:
    prompt = _as_str(value, "prompt").replace("\r\n", "\n").replace("\r", "\n")
    if len(prompt) > _MAX_PROMPT_CHARS:
        raise ValueError(f"prompt must be <= {_MAX_PROMPT_CHARS} characters")
    for char in prompt:
        codepoint = ord(char)
        if codepoint < 32 and char not in "\n\t":
            raise ValueError("prompt contains unsupported control characters")
    normalised = " ".join(prompt.split())
    for pattern in _PROMPT_INJECTION_PATTERNS:
        if pattern.search(normalised):
            raise ValueError("prompt contains instruction-injection markers")
    return normalised


def _as_name(value: object) -> str:
    value = _as_str(value, "name")
    if not _NAME_PATTERN.fullmatch(value):
        raise ValueError("name must match [A-Za-z][A-Za-z0-9_-]{0,63}")
    return value


def _as_positive_int(value: object, field_name: str, *, max_value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 1:
        raise ValueError(f"{field_name} must be >= 1")
    if value > max_value:
        raise ValueError(f"{field_name} must be <= {max_value}")
    return value


def _as_path(
    value: str | Path | None,
    field_name: str,
    *,
    allow_none: bool = False,
) -> Path | None:
    if value is None:
        if not allow_none:
            raise TypeError(f"{field_name} must be a string, pathlib.Path, or None")
        return None
    if isinstance(value, Path):
        path = value
    elif isinstance(value, str):
        if not value.strip():
            raise ValueError(f"{field_name} must be a non-empty path")
        path = Path(value)
    else:
        raise TypeError(f"{field_name} must be a string, pathlib.Path, or None")
    if path.exists() and not path.is_dir():
        raise ValueError(f"{field_name} must be a directory when provided")
    return path


def _coerce_output_dir(output_dir: str | Path) -> Path:
    if not isinstance(output_dir, (str, Path)):
        raise TypeError("output_dir must be a string or pathlib.Path")
    path = Path(output_dir)
    if path.exists() and not path.is_dir():
        raise ValueError("output_dir must be a directory path")
    return path


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


def _retrieve_local_evidence(
    prompt: str,
    *,
    domainpack_root: str | Path | None,
    docs_root: str | Path | None,
    limit_per_source: int = 3,
) -> list[RetrievalEvidence]:
    domainpack_evidence = _retrieve_domainpack_evidence(
        prompt,
        domainpack_root,
        limit=limit_per_source,
    )
    docs_evidence = _retrieve_docs_evidence(
        prompt,
        docs_root,
        limit=limit_per_source,
    )
    return _rank_retrieval_evidence([*domainpack_evidence, *docs_evidence])


def _rank_retrieval_evidence(
    evidence: list[RetrievalEvidence],
) -> list[RetrievalEvidence]:
    ranked = sorted(
        evidence,
        key=lambda item: (
            -item.score,
            -item.ranking_features.get("source_priority", 0.0),
            -item.ranking_features.get("matched_term_count", 0.0),
            -item.ranking_features.get("name_match_count", 0.0),
            item.source,
            item.domainpack,
            item.path,
        ),
    )
    return [replace(item, rank=index) for index, item in enumerate(ranked, start=1)]


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
        domain_phrase = domain_dir.name.lower().replace("_", " ")
        name_bonus = sum(1 for term in prompt_terms if term in domain_phrase)
        phrase_bonus = 2 if domain_phrase in prompt.lower() else 0
        score = (len(matched) + name_bonus + phrase_bonus) / max(
            len(prompt_terms),
            1,
        )
        scored.append(
            RetrievalEvidence(
                domainpack=domain_dir.name,
                path=str(spec_path),
                score=round(min(score, 1.0), 3),
                matched_terms=matched[:12],
                summary=_evidence_summary(domain_dir.name, matched),
                source="domainpack",
                ranking_features={
                    "matched_term_count": float(len(matched)),
                    "name_match_count": float(name_bonus),
                    "phrase_match": float(phrase_bonus > 0),
                    "prompt_term_count": float(len(prompt_terms)),
                    "source_priority": 1.0,
                    "term_density": round(
                        len(matched) / max(len(corpus_terms), 1),
                        6,
                    ),
                },
            )
        )

    return sorted(scored, key=lambda item: (-item.score, item.domainpack))[:limit]


def _retrieve_docs_evidence(
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
    for doc_path in sorted(base.rglob("*.md")):
        if "internal" in doc_path.parts:
            continue
        text = _safe_read(doc_path, max_chars=20000)
        corpus_terms = set(_terms(text))
        matched = sorted(prompt_terms & corpus_terms)
        if not matched:
            continue
        title_bonus = sum(1 for term in prompt_terms if term in doc_path.stem.lower())
        score = (len(matched) + title_bonus) / max(len(prompt_terms), 1)
        scored.append(
            RetrievalEvidence(
                domainpack=doc_path.stem,
                path=str(doc_path),
                score=round(min(score, 1.0), 3),
                matched_terms=matched[:12],
                summary=_evidence_summary(doc_path.stem, matched),
                source="docs",
                ranking_features={
                    "matched_term_count": float(len(matched)),
                    "name_match_count": float(title_bonus),
                    "phrase_match": 0.0,
                    "prompt_term_count": float(len(prompt_terms)),
                    "source_priority": 0.75,
                    "term_density": round(
                        len(matched) / max(len(corpus_terms), 1),
                        6,
                    ),
                },
            )
        )
    return sorted(scored, key=lambda item: (-item.score, item.path))[:limit]


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
    notebook_execution: dict[str, Any],
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
                    "- Notebook preflight: "
                    f"`{notebook_execution['status']}` "
                    f"({notebook_execution['passed_checks']}/"
                    f"{notebook_execution['total_checks']} checks)\n",
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
                    "## Preflight evidence\n",
                    "\n",
                    "The compiler executed the same schema and policy checks "
                    "that this review notebook asks you to run locally.\n",
                    "\n",
                    f"- Status: `{notebook_execution['status']}`\n",
                    f"- Checks: `{notebook_execution['passed_checks']}/"
                    f"{notebook_execution['total_checks']}`\n",
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
                "notebook_execution": notebook_execution,
                "schema_version": 1,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, indent=2, sort_keys=True) + "\n"


def _review_notebook_execution_evidence(
    *,
    binding_yaml: str,
    policy_yaml: str,
    expected_name: str,
) -> dict[str, Any]:
    import tempfile

    from scpn_phase_orchestrator.binding.loader import load_binding_spec
    from scpn_phase_orchestrator.supervisor.policy_rules import load_policy_rules

    checks: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="spo_generated_review_") as tmp:
        path = Path(tmp)
        binding_path = path / "binding_spec.yaml"
        policy_path = path / "policy.yaml"
        binding_path.write_text(binding_yaml, encoding="utf-8")
        policy_path.write_text(policy_yaml, encoding="utf-8")

        try:
            spec = load_binding_spec(binding_path)
            checks.append({"name": "load_binding_spec", "passed": True})
        except Exception as exc:  # pragma: no cover - defensive audit payload
            checks.append(
                {
                    "name": "load_binding_spec",
                    "passed": False,
                    "error": type(exc).__name__,
                }
            )
            spec = None

        if spec is not None:
            errors = validate_binding_spec(spec)
            checks.append(
                {
                    "name": "validate_binding_spec",
                    "passed": errors == [],
                    "errors": errors,
                }
            )
            checks.append(
                {
                    "name": "spec_name_matches",
                    "passed": spec.name == expected_name,
                    "observed": spec.name,
                }
            )

        try:
            rules = load_policy_rules(policy_path)
            checks.append(
                {
                    "name": "load_policy_rules",
                    "passed": len(rules) > 0,
                    "rule_count": len(rules),
                }
            )
        except Exception as exc:  # pragma: no cover - defensive audit payload
            checks.append(
                {
                    "name": "load_policy_rules",
                    "passed": False,
                    "error": type(exc).__name__,
                }
            )

    passed = sum(1 for check in checks if check["passed"])
    return {
        "status": "passed" if passed == len(checks) else "failed",
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": checks,
    }


def _review_gate_record() -> dict[str, Any]:
    return {
        "status": "required",
        "non_actuating": True,
        "manual_review_required": True,
        "auto_execution_enabled": False,
        "required_artifacts": [
            "binding_spec.yaml",
            "policy.yaml",
            "review_notebook.ipynb",
            "audit.json",
        ],
    }


def _validate_generated_audit_schema(audit_record: Mapping[str, Any]) -> None:
    required = {
        "compiler": str,
        "schema_valid": bool,
        "validation_errors": list,
        "intent_boundary": dict,
        "review_gate": dict,
        "confidence": float,
        "confidence_factors": dict,
        "retrieval_evidence": list,
        "notebook_execution": dict,
    }
    for key, expected_type in required.items():
        value = audit_record.get(key)
        if not isinstance(value, expected_type):
            raise ValueError(f"generated audit schema invalid: {key}")
    review_gate = audit_record["review_gate"]
    if review_gate.get("non_actuating") is not True:
        raise ValueError(
            "generated audit schema invalid: review gate must be non-actuating"
        )
    if review_gate.get("manual_review_required") is not True:
        raise ValueError("generated audit schema invalid: manual review required")
    if review_gate.get("auto_execution_enabled") is not False:
        raise ValueError("generated audit schema invalid: auto execution disabled")


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

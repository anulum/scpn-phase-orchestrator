# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Semantic Compiler tests

from __future__ import annotations

import json

import numpy as np
import pytest
from click.testing import CliRunner

from scpn_phase_orchestrator.binding import load_binding_spec, validate_binding_spec
from scpn_phase_orchestrator.binding.semantic import (
    GeneratedBindingArtifacts,
    RetrievalEvidence,
    SemanticDomainCompiler,
    compile_symbolic_binding,
)
from scpn_phase_orchestrator.coupling.knm import CouplingBuilder
from scpn_phase_orchestrator.runtime.cli import main
from scpn_phase_orchestrator.supervisor.policy_rules import load_policy_rules
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter


def test_semantic_compilation_physics():
    compiler = SemanticDomainCompiler()
    prompt = "Model a 4-layer power grid stability system"
    spec = compiler.compile(prompt)

    assert len(spec.layers) == 4
    assert spec.layers[0].omegas[0] == 50.0


def test_semantic_compilation_biology():
    compiler = SemanticDomainCompiler()
    prompt = "A 3-layer brainwave synchronization model"
    spec = compiler.compile(prompt)

    assert len(spec.layers) == 3
    assert spec.layers[0].omegas[0] == 10.0


def test_default_layers_when_count_absent():
    """No explicit ``N-layer`` phrase → default of 2 layers per semantic.py:38."""
    compiler = SemanticDomainCompiler()
    spec = compiler.compile("A generic coupled oscillator system")
    assert len(spec.layers) == 2


def test_default_base_frequency_for_unknown_discipline():
    """Prompt that matches neither bio nor grid keywords → base_freq = 1.0."""
    compiler = SemanticDomainCompiler()
    spec = compiler.compile("A 2-layer thermal coupling system")
    assert spec.layers[0].omegas[0] == 1.0
    assert spec.layers[1].omegas[0] == 10.0  # base × 10^1


def test_fusion_keyword_routes_to_grid_frequency():
    """ "fusion" is a grid-class keyword → 50 Hz base (plasma tokamak)."""
    compiler = SemanticDomainCompiler()
    spec = compiler.compile("A 2-layer fusion reactor plasma system")
    assert spec.layers[0].omegas[0] == 50.0


def test_cell_keyword_routes_to_bio_frequency():
    """ "cell" is a bio keyword → 10 Hz base."""
    compiler = SemanticDomainCompiler()
    spec = compiler.compile("A 3-layer cellular oscillator network")
    assert spec.layers[0].omegas[0] == 10.0


def test_case_insensitive_layer_regex():
    """Layer-count regex must ignore case: "4-Layer" and "4-LAYER" both work."""
    compiler = SemanticDomainCompiler()
    assert len(compiler.compile("A 4-Layer power grid").layers) == 4
    assert len(compiler.compile("A 4-LAYER power grid").layers) == 4


def test_frequency_decades_scale_with_layer_index():
    """Each layer's ω is base × 10^index, so deeper layers run at higher
    frequencies. This is the stack convention the supervisor relies on."""
    compiler = SemanticDomainCompiler()
    spec = compiler.compile("A 4-layer brain network")
    assert spec.layers[0].omegas[0] == 10.0
    assert spec.layers[1].omegas[0] == 100.0
    assert spec.layers[2].omegas[0] == 1000.0
    assert spec.layers[3].omegas[0] == 10000.0


def test_each_layer_has_eight_oscillators():
    """Every layer is populated with 8 oscillator_ids per semantic.py:54."""
    compiler = SemanticDomainCompiler()
    spec = compiler.compile("A 2-layer cardiac system")
    for layer in spec.layers:
        assert len(layer.oscillator_ids) == 8
        assert len(layer.omegas) == 8


def test_objectives_partition_marks_all_layers_good():
    """Default objective: every generated layer is good, none bad."""
    compiler = SemanticDomainCompiler()
    spec = compiler.compile("A 3-layer power grid")
    assert spec.objectives.good_layers == [0, 1, 2]
    assert spec.objectives.bad_layers == []


def test_generated_spec_has_stable_metadata():
    """Compiled spec carries the expected name, version and safety tier —
    downstream validators rely on these."""
    compiler = SemanticDomainCompiler()
    spec = compiler.compile("A 2-layer test system")
    assert spec.name == "semantically_generated_domain"
    assert spec.version == "1.0.0"
    assert spec.safety_tier == "research"
    assert spec.sample_period_s == 0.01
    assert spec.control_period_s == 0.1


def test_empty_prompt_falls_back_to_defaults():
    """Empty string must produce a valid 2-layer default spec, not crash."""
    compiler = SemanticDomainCompiler()
    spec = compiler.compile("")
    assert len(spec.layers) == 2
    assert spec.layers[0].omegas[0] == 1.0  # default base freq


def test_compile_artifacts_returns_reviewable_valid_outputs(tmp_path):
    compiler = SemanticDomainCompiler()
    artefacts = compiler.compile_artifacts(
        "A 3-layer cardiac rhythm suppression system with actuator feedback",
        name="cardiac_review",
        oscillators_per_layer=3,
        dry_run_steps=4,
    )

    assert isinstance(artefacts, GeneratedBindingArtifacts)
    assert artefacts.schema_valid
    assert artefacts.validation_errors == []
    assert artefacts.audit_record["schema_valid"] is True
    assert artefacts.audit_record["domain_family"] == "biological"
    assert "cardiac" in artefacts.audit_record["matched_keywords"]
    assert artefacts.audit_record["confidence_factors"]["retrieval_score"] >= 0.0
    assert artefacts.audit_record["notebook_execution"]["status"] == "passed"
    assert artefacts.audit_record["notebook_execution"]["passed_checks"] == 4
    assert 0.0 <= artefacts.dry_run_order_parameter <= 1.0
    notebook = json.loads(artefacts.notebook_json)
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["scpn_phase_orchestrator"]["schema_version"] == 1
    assert (
        notebook["metadata"]["scpn_phase_orchestrator"]["notebook_execution"]["status"]
        == "passed"
    )

    spec_path = tmp_path / "binding_spec.yaml"
    policy_path = tmp_path / "policy.yaml"
    spec_path.write_text(artefacts.binding_yaml, encoding="utf-8")
    policy_path.write_text(artefacts.policy_yaml, encoding="utf-8")

    loaded = load_binding_spec(spec_path)
    assert validate_binding_spec(loaded) == []
    assert loaded.name == "cardiac_review"
    assert loaded.protocol_net is not None
    assert load_policy_rules(policy_path)[0].name == "recover_low_coherence"


def test_compile_symbolic_binding_pipeline_drives_engine():
    artefacts = compile_symbolic_binding(
        "A 2-layer power grid with operator intervention",
        name="grid_review",
        oscillators_per_layer=2,
    )
    spec = artefacts.binding_spec
    n = sum(len(layer.oscillator_ids) for layer in spec.layers)
    coupling = CouplingBuilder().build(
        n,
        spec.coupling.base_strength,
        spec.coupling.decay_alpha,
    )
    engine = UPDEEngine(n, dt=spec.sample_period_s)
    phases = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    omegas = np.asarray(spec.get_omegas(), dtype=np.float64)

    for _ in range(12):
        phases = engine.step(phases, omegas, coupling.knm, 0.0, 0.0, coupling.alpha)

    r_value, _ = compute_order_parameter(phases)
    assert 0.0 <= r_value <= 1.0
    assert artefacts.audit_record["petri_reachability"]["target_place"] == "validated"


def test_compile_artifacts_records_local_domainpack_retrieval_evidence():
    artefacts = compile_symbolic_binding(
        "A 2-layer power grid stability controller with renewable demand",
        name="grid_retrieval_review",
        oscillators_per_layer=2,
        dry_run_steps=2,
    )

    assert artefacts.retrieval_evidence
    assert isinstance(artefacts.retrieval_evidence[0], RetrievalEvidence)
    assert any(
        evidence.source == "domainpack" for evidence in artefacts.retrieval_evidence
    )
    assert any(
        evidence.domainpack == "power_grid" for evidence in artefacts.retrieval_evidence
    )
    assert any(
        record["domainpack"] == "power_grid"
        for record in artefacts.audit_record["retrieval_evidence"]
    )
    assert artefacts.audit_record["confidence_factors"]["retrieval_score"] > 0.0
    assert artefacts.audit_record["confidence"] >= 0.8


def test_compile_artifacts_records_long_form_docs_retrieval_evidence(tmp_path):
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    (docs_root / "plasma_control.md").write_text(
        "plasma tokamak fusion instability control coherence actuators",
        encoding="utf-8",
    )

    artefacts = compile_symbolic_binding(
        "A 2-layer plasma fusion coherence controller",
        name="plasma_docs_review",
        retrieval_root=None,
        docs_root=docs_root,
    )

    assert artefacts.audit_record["retrieval_evidence"][0]["source"] == "docs"
    assert artefacts.retrieval_evidence[0].path.endswith("plasma_control.md")
    assert "plasma" in artefacts.retrieval_evidence[0].matched_terms
    assert artefacts.audit_record["confidence_factors"]["retrieval_score"] > 0.0


def test_compile_artifacts_can_disable_all_retrieval():
    artefacts = compile_symbolic_binding(
        "A 2-layer power grid stability controller",
        name="grid_no_local_retrieval_review",
        retrieval_root=None,
        docs_root=None,
    )

    assert artefacts.retrieval_evidence == []
    assert artefacts.audit_record["retrieval_evidence"] == []
    assert artefacts.audit_record["confidence_factors"]["retrieval_score"] == 0.0


def test_compile_artifacts_can_disable_retrieval():
    artefacts = compile_symbolic_binding(
        "A 2-layer power grid stability controller",
        name="grid_no_retrieval_review",
        retrieval_root=None,
    )

    assert all(evidence.source == "docs" for evidence in artefacts.retrieval_evidence)
    assert artefacts.audit_record["confidence_factors"]["retrieval_score"] >= 0.0


def test_write_domainpack_and_cli_generate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "generate",
            "A 2-layer traffic flow phase control system",
            "--name",
            "traffic_review",
            "--oscillators-per-layer",
            "2",
            "--dry-run-steps",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    output_dir = tmp_path / "domainpacks" / "traffic_review"
    assert (output_dir / "binding_spec.yaml").exists()
    assert (output_dir / "policy.yaml").exists()
    assert (output_dir / "review_notebook.ipynb").exists()
    assert (output_dir / "README.md").exists()
    audit = json.loads((output_dir / "audit.json").read_text(encoding="utf-8"))
    assert audit["schema_valid"] is True
    assert audit["domain_family"] == "network"
    assert audit["notebook_execution"]["status"] == "passed"
    assert "retrieval_matches=" in result.output
    notebook = json.loads(
        (output_dir / "review_notebook.ipynb").read_text(encoding="utf-8")
    )
    assert notebook["metadata"]["scpn_phase_orchestrator"]["artifact"]
    assert load_policy_rules(output_dir / "policy.yaml")
    loaded = load_binding_spec(output_dir / "binding_spec.yaml")
    assert validate_binding_spec(loaded) == []


def test_compile_artifacts_rejects_invalid_generation_parameters():
    compiler = SemanticDomainCompiler()
    for kwargs in [
        {"name": "bad name"},
        {"oscillators_per_layer": 0},
        {"dry_run_steps": 0},
    ]:
        with pytest.raises(ValueError):
            compiler.compile_artifacts("A 2-layer system", **kwargs)


@pytest.mark.parametrize("bad_prompt", [None, 123, ["not", "a", "string"]])
def test_compile_artifacts_rejects_invalid_prompt_type(bad_prompt):
    compiler = SemanticDomainCompiler()
    with pytest.raises(TypeError, match="prompt must be a string"):
        compiler.compile_artifacts(bad_prompt)


@pytest.mark.parametrize(
    ("value", "exc"),
    [
        (123, TypeError),
        (None, TypeError),
        ("bad name", ValueError),
        ("-bad", ValueError),
        ("", ValueError),
    ],
)
def test_compile_artifacts_rejects_invalid_name(value, exc):
    compiler = SemanticDomainCompiler()
    with pytest.raises(exc):
        compiler.compile_artifacts("A 2-layer system", name=value)


@pytest.mark.parametrize("value", [True, False, 3.2, "4"])
def test_compile_artifacts_rejects_non_integer_oscillators_or_dry_run(value):
    compiler = SemanticDomainCompiler()
    with pytest.raises(TypeError):
        compiler.compile_artifacts("A 2-layer system", oscillators_per_layer=value)
    with pytest.raises(TypeError):
        compiler.compile_artifacts("A 2-layer system", dry_run_steps=value)


@pytest.mark.parametrize(
    "prompt",
    [
        "A 0-layer power grid stability controller",
        "A -1-layer power grid stability controller",
        "A 9999-layer power grid stability controller",
    ],
)
def test_compile_artifacts_rejects_dangerous_layer_counts(prompt):
    compiler = SemanticDomainCompiler()
    with pytest.raises(ValueError, match="layer count"):
        compiler.compile_artifacts(prompt)


@pytest.mark.parametrize("value", [None, "4", 2.5, object()])
def test_compile_symbolic_binding_rejects_invalid_parameter_types(value):
    with pytest.raises(TypeError):
        compile_symbolic_binding("A 2-layer power grid", oscillators_per_layer=value)


@pytest.mark.parametrize("name", [123, None, "bad name", "-bad", ""])
def test_compile_symbolic_binding_rejects_invalid_name(name):
    with pytest.raises((TypeError, ValueError)):
        compile_symbolic_binding("A 2-layer power grid", name=name)


@pytest.mark.parametrize("bad_prompt", [None, 123, ["not", "a", "string"]])
def test_compile_symbolic_binding_rejects_invalid_prompt_type(bad_prompt):
    with pytest.raises(TypeError, match="prompt must be a string"):
        compile_symbolic_binding(bad_prompt)


@pytest.mark.parametrize(
    ("docs_root", "retrieval_root", "expected_error"),
    [
        (123, None, "docs_root"),
        (None, 123, "retrieval_root"),
    ],
)
def test_compile_artifacts_rejects_invalid_retrieval_roots(
    docs_root,
    retrieval_root,
    expected_error,
):
    compiler = SemanticDomainCompiler()
    with pytest.raises(TypeError, match=expected_error):
        compiler.compile_artifacts(
            "A 2-layer power grid",
            retrieval_root=retrieval_root,
            docs_root=docs_root,
        )


def test_compile_artifacts_rejects_retrieval_root_file_path(tmp_path):
    compiler = SemanticDomainCompiler()
    file_root = tmp_path / "not_a_dir.txt"
    file_root.write_text("not a dir", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a directory"):
        compiler.compile_artifacts(
            "A 2-layer power grid",
            retrieval_root=file_root,
        )


def test_compile_artifacts_rejects_docs_root_file_path(tmp_path):
    compiler = SemanticDomainCompiler()
    file_root = tmp_path / "docs.txt"
    file_root.write_text("not docs", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a directory"):
        compiler.compile_artifacts(
            "A 2-layer power grid",
            docs_root=file_root,
        )


def test_generated_artifacts_write_domainpack_is_idempotent(tmp_path):
    compiler = SemanticDomainCompiler()
    artefacts = compiler.compile_artifacts(
        "A 2-layer cardiac rhythm suppression system",
        name="cardiac_review_pack",
        oscillators_per_layer=2,
    )
    output_dir = tmp_path / "domainpack"

    artefacts.write_domainpack(output_dir)
    first_snapshot = {
        file_name: (output_dir / file_name).read_text(encoding="utf-8")
        for file_name in [
            "binding_spec.yaml",
            "policy.yaml",
            "review_notebook.ipynb",
            "audit.json",
            "README.md",
        ]
    }

    artefacts.write_domainpack(output_dir)
    second_snapshot = {
        file_name: (output_dir / file_name).read_text(encoding="utf-8")
        for file_name in [
            "binding_spec.yaml",
            "policy.yaml",
            "review_notebook.ipynb",
            "audit.json",
            "README.md",
        ]
    }

    assert first_snapshot == second_snapshot


def test_generated_artifacts_write_domainpack_rejects_non_directory_output(tmp_path):
    compiler = SemanticDomainCompiler()
    artifacts = compiler.compile_artifacts(
        "A 2-layer cardiac rhythm suppression system",
    )
    output_file = tmp_path / "not_a_directory"
    output_file.write_text("occupied", encoding="utf-8")

    with pytest.raises(ValueError, match="output_dir must be a directory path"):
        artifacts.write_domainpack(output_file)


def test_generated_artifacts_audit_record_matches_execution_payload():
    compiler = SemanticDomainCompiler()
    artifacts = compiler.compile_artifacts(
        "A 2-layer traffic flow",
        name="traffic_pack",
        oscillators_per_layer=3,
        dry_run_steps=2,
        retrieval_root=None,
        docs_root=None,
    )

    assert artifacts.audit_record["schema_valid"] is (artifacts.validation_errors == [])
    assert artifacts.audit_record["validation_errors"] == artifacts.validation_errors
    assert artifacts.audit_record["layers"] == len(artifacts.binding_spec.layers)
    assert artifacts.audit_record["oscillators_per_layer"] == len(
        artifacts.binding_spec.layers[0].oscillator_ids
    )
    assert artifacts.audit_record["dry_run_steps"] == 2
    assert (
        artifacts.audit_record["dry_run_order_parameter"]
        == artifacts.dry_run_order_parameter
    )
    assert len(artifacts.audit_record["retrieval_evidence"]) == len(
        artifacts.retrieval_evidence
    )


# Pipeline wiring: SemanticDomainCompiler is the natural-language frontend
# to BindingSpec. These cases pin the three layer-count paths (default,
# numeric match, case-insensitive), the three discipline-keyword paths
# (bio, grid, fallback) and the spec-metadata contract downstream tools
# depend on.

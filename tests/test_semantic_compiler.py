# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - Semantic Compiler tests

from scpn_phase_orchestrator.binding.semantic import SemanticDomainCompiler


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
    """"fusion" is a grid-class keyword → 50 Hz base (plasma tokamak)."""
    compiler = SemanticDomainCompiler()
    spec = compiler.compile("A 2-layer fusion reactor plasma system")
    assert spec.layers[0].omegas[0] == 50.0


def test_cell_keyword_routes_to_bio_frequency():
    """"cell" is a bio keyword → 10 Hz base."""
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


# Pipeline wiring: SemanticDomainCompiler is the natural-language frontend
# to BindingSpec. These cases pin the three layer-count paths (default,
# numeric match, case-insensitive), the three discipline-keyword paths
# (bio, grid, fallback) and the spec-metadata contract downstream tools
# depend on.

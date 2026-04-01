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

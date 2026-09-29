# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Semantic retrieval document visibility tests

"""Exercise document visibility through generated symbolic binding artefacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_phase_orchestrator.binding.semantic import compile_symbolic_binding
from scpn_phase_orchestrator.binding.semantic.retrieval import RetrievalEvidence


@pytest.mark.parametrize(
    "private_path", ["internal", "reference/internal", "internal/reviews"]
)
@pytest.mark.parametrize("with_public_document", [False, True])
def test_internal_documents_do_not_enter_generated_evidence(
    tmp_path: Path, private_path: str, with_public_document: bool
) -> None:
    """Keep internal matches out of ranking, audit and notebook artefacts."""
    source = Path(__file__).resolve().parents[1] / "domainpacks/power_grid/README.md"
    content = source.read_bytes()
    docs_root = tmp_path / "docs"
    private_root = docs_root / private_path
    private_root.mkdir(parents=True)
    private_documents = [private_root / f"power_grid_{index}.md" for index in range(4)]
    for document in private_documents:
        document.write_bytes(content)
    public_document = docs_root / "internal_notes" / "power_grid.md"
    if with_public_document:
        public_document.parent.mkdir()
        public_document.write_bytes(content)

    artefacts = compile_symbolic_binding(
        "A 2-layer power grid stability controller",
        name="grid_document_review",
        oscillators_per_layer=2,
        dry_run_steps=2,
        retrieval_root=None,
        docs_root=docs_root,
    )

    evidence: list[RetrievalEvidence] = artefacts.retrieval_evidence
    expected_paths = [str(public_document)] if with_public_document else []
    assert [item.path for item in evidence] == expected_paths
    records = [item.to_audit_record() for item in evidence]
    assert artefacts.audit_record["retrieval_evidence"] == records
    notebook = json.loads(artefacts.notebook_json)
    markdown = [
        line
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
        for line in cell["source"]
    ]
    assert f"- Retrieval matches: `{len(records)}`\n" in markdown
    confidence = artefacts.audit_record["confidence"]
    assert f"- Confidence: `{confidence:.3f}`\n" in markdown
    assert (
        notebook["metadata"]["scpn_phase_orchestrator"]["notebook_execution"]
        == artefacts.audit_record["notebook_execution"]
    )
    assert artefacts.validation_errors == []
    assert 0.0 <= artefacts.dry_run_order_parameter <= 1.0
    if with_public_document:
        assert evidence[0].rank == 1
        assert evidence[0].source == "docs"
        assert {"power", "grid"} <= set(evidence[0].matched_terms)
        assert artefacts.audit_record["confidence_factors"]["retrieval_score"] > 0.0
        assert public_document.read_bytes() == content
    else:
        assert artefacts.audit_record["confidence_factors"]["retrieval_score"] == 0.0
    for document in private_documents:
        assert str(document) not in artefacts.notebook_json
        assert document.read_bytes() == content
    assert source.read_bytes() == content

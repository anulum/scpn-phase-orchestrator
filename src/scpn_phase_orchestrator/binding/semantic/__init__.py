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
domainpack/docs evidence. The implementation is split into responsibility
modules (input coercion, retrieval evidence, review notebook, YAML serialisation,
and the orchestrating compiler) behind a stable re-export surface. Generated
artefacts are intentionally reviewable and fail validation before use; this
package does not auto-accept live deployment bindings or actuate a system.
"""

from __future__ import annotations

from .compiler import (
    GeneratedBindingArtifacts,
    SemanticDomainCompiler,
    compile_symbolic_binding,
)
from .retrieval import (
    RetrievalEvidence,
)
from .retrieval import (
    _safe_read as _safe_read,
)

__all__ = [
    "GeneratedBindingArtifacts",
    "RetrievalEvidence",
    "SemanticDomainCompiler",
    "compile_symbolic_binding",
]

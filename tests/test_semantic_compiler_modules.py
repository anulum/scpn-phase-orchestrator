# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — binding/semantic package wiring tests

"""Wiring contract for the ``binding.semantic`` package.

Asserts that every responsibility submodule imports under its dotted path and
that the package ``__init__`` re-exports each public symbol — and the one private
helper that an existing caller imports by name — as the *same object* defined in
its owning submodule, so the split preserves the original flat ``binding.semantic``
import surface with zero consumer churn.
"""

from __future__ import annotations

import importlib

import pytest

import scpn_phase_orchestrator.binding.semantic as semantic

PACKAGE = "scpn_phase_orchestrator.binding.semantic"

# Full dotted paths spelled out so the module-linkage guard
# (tools/check_test_module_linkage.py) finds each submodule's import path.
SUBMODULES = (
    "scpn_phase_orchestrator.binding.semantic.coercion",
    "scpn_phase_orchestrator.binding.semantic.retrieval",
    "scpn_phase_orchestrator.binding.semantic.review",
    "scpn_phase_orchestrator.binding.semantic.serialization",
    "scpn_phase_orchestrator.binding.semantic.compiler",
)

# Owning submodule for each public symbol re-exported by the package __init__.
PUBLIC_SYMBOL_MODULE = {
    "GeneratedBindingArtifacts": "compiler",
    "RetrievalEvidence": "retrieval",
    "SemanticDomainCompiler": "compiler",
    "compile_symbolic_binding": "compiler",
}

# Private helper imported by name by an existing caller (binding.topos_examples
# and the studio facade reach the compiler; the retrieval reader is reused).
PRIVATE_REEXPORT_MODULE = {
    "_safe_read": "retrieval",
}


@pytest.mark.parametrize("dotted_path", SUBMODULES)
def test_submodule_imports_under_dotted_path(dotted_path: str) -> None:
    assert dotted_path.startswith(f"{PACKAGE}.")
    module = importlib.import_module(dotted_path)
    assert module.__name__ == dotted_path
    assert (module.__doc__ or "").strip(), f"{dotted_path} has no module docstring"


def test_public_symbol_module_table_covers_dunder_all() -> None:
    assert set(PUBLIC_SYMBOL_MODULE) == set(semantic.__all__)
    assert len(semantic.__all__) == len(set(semantic.__all__))


@pytest.mark.parametrize(
    "symbol,module_name",
    sorted({**PUBLIC_SYMBOL_MODULE, **PRIVATE_REEXPORT_MODULE}.items()),
)
def test_symbol_reexport_is_owning_module_object(symbol: str, module_name: str) -> None:
    owner = importlib.import_module(f"{PACKAGE}.{module_name}")
    assert getattr(semantic, symbol) is getattr(owner, symbol)


def test_every_reexport_owner_is_a_real_submodule() -> None:
    owners = set(PUBLIC_SYMBOL_MODULE.values()) | set(PRIVATE_REEXPORT_MODULE.values())
    submodule_names = {path.rsplit(".", 1)[1] for path in SUBMODULES}
    assert owners <= submodule_names

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — supervisor/formal_export package wiring tests

"""Wiring contract for the ``supervisor.formal_export`` package.

Asserts that every responsibility submodule imports under its dotted path and
that the package ``__init__`` re-exports each public symbol as the *same object*
defined in its owning submodule, so the split preserves the original flat
``supervisor.formal_export`` import surface with zero consumer churn. ``shutil``
stays resolvable on the package namespace so the checker-availability monkeypatch
contract keeps working.
"""

from __future__ import annotations

import importlib
import shutil

import pytest

import scpn_phase_orchestrator.supervisor.formal_export as formal_export

PACKAGE = "scpn_phase_orchestrator.supervisor.formal_export"

# Full dotted paths spelled out so the module-linkage guard
# (tools/check_test_module_linkage.py) finds each submodule's import path.
SUBMODULES = (
    "scpn_phase_orchestrator.supervisor.formal_export._shared",
    "scpn_phase_orchestrator.supervisor.formal_export.verification_package",
    "scpn_phase_orchestrator.supervisor.formal_export.runtime_certificate",
    "scpn_phase_orchestrator.supervisor.formal_export.petri_export",
    "scpn_phase_orchestrator.supervisor.formal_export.policy_export",
    "scpn_phase_orchestrator.supervisor.formal_export.smt_export",
    "scpn_phase_orchestrator.supervisor.formal_export.stl_export",
)

# Owning submodule for each public symbol re-exported by the package __init__.
PUBLIC_SYMBOL_MODULE = {
    "FormalCheckerAvailability": "runtime_certificate",
    "FormalCheckerCommand": "verification_package",
    "FormalCheckerResult": "runtime_certificate",
    "FormalRuntimeCertificate": "runtime_certificate",
    "FormalSafetyProperty": "verification_package",
    "FormalTextArtifact": "verification_package",
    "FormalVerificationPackage": "verification_package",
    "PrismExport": "_shared",
    "TLAExport": "_shared",
    "audit_formal_checker_availability": "runtime_certificate",
    "build_formal_verification_package": "verification_package",
    "build_runtime_control_certificate": "runtime_certificate",
    "export_petri_net_prism": "petri_export",
    "export_petri_net_tla": "petri_export",
    "export_policy_rules_prism": "policy_export",
    "export_policy_rules_smt": "smt_export",
    "export_policy_rules_tla": "policy_export",
    "export_stl_specs_prism": "stl_export",
}


@pytest.mark.parametrize("dotted_path", SUBMODULES)
def test_submodule_imports_under_dotted_path(dotted_path: str) -> None:
    assert dotted_path.startswith(f"{PACKAGE}.")
    module = importlib.import_module(dotted_path)
    assert module.__name__ == dotted_path
    assert (module.__doc__ or "").strip(), f"{dotted_path} has no module docstring"


def test_public_symbol_module_table_covers_dunder_all() -> None:
    assert set(PUBLIC_SYMBOL_MODULE) == set(formal_export.__all__)
    assert len(formal_export.__all__) == len(set(formal_export.__all__))


@pytest.mark.parametrize("symbol,module_name", sorted(PUBLIC_SYMBOL_MODULE.items()))
def test_public_symbol_reexport_is_owning_module_object(
    symbol: str, module_name: str
) -> None:
    owner = importlib.import_module(f"{PACKAGE}.{module_name}")
    assert getattr(formal_export, symbol) is getattr(owner, symbol)


def test_every_reexport_owner_is_a_real_submodule() -> None:
    owners = set(PUBLIC_SYMBOL_MODULE.values())
    submodule_names = {path.rsplit(".", 1)[1] for path in SUBMODULES}
    assert owners <= submodule_names


def test_shutil_resolves_on_package_for_checker_monkeypatch() -> None:
    # tests patch ``...formal_export.shutil.which`` to stub checker discovery.
    assert formal_export.shutil is shutil

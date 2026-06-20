# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Formal supervisor exporters

"""Formal-model exporters for Petri nets, policy rules, and STL monitors.

The exporter functions convert already-validated supervisor structures into
PRISM or TLA+ text plus identifier maps, sanitising names and preserving metric,
transition, rule, action, and STL mappings for auditability, split into
responsibility modules (shared identifiers, verification package, runtime
certificate, and per-formalism exporters) behind a stable re-export surface.
Export routines are pure text generation; they do not invoke model checkers,
write files, or change the source policy/Petri structures. ``shutil`` is
re-exported so checker-availability tests resolve it on this package namespace.
"""

from __future__ import annotations

import shutil as shutil

from ._shared import (
    PrismExport,
    TLAExport,
)
from .petri_export import (
    export_petri_net_prism,
    export_petri_net_tla,
)
from .policy_export import (
    export_policy_rules_prism,
    export_policy_rules_tla,
)
from .runtime_certificate import (
    FormalCheckerAvailability,
    FormalCheckerResult,
    FormalRuntimeCertificate,
    audit_formal_checker_availability,
    build_runtime_control_certificate,
)
from .stl_export import export_stl_specs_prism
from .verification_package import (
    FormalCheckerCommand,
    FormalSafetyProperty,
    FormalTextArtifact,
    FormalVerificationPackage,
    build_formal_verification_package,
)

__all__ = [
    "FormalCheckerAvailability",
    "FormalCheckerCommand",
    "FormalCheckerResult",
    "FormalRuntimeCertificate",
    "FormalSafetyProperty",
    "FormalTextArtifact",
    "FormalVerificationPackage",
    "PrismExport",
    "TLAExport",
    "audit_formal_checker_availability",
    "build_formal_verification_package",
    "build_runtime_control_certificate",
    "export_petri_net_prism",
    "export_petri_net_tla",
    "export_policy_rules_prism",
    "export_policy_rules_tla",
    "export_stl_specs_prism",
]

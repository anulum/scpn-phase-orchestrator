# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — gRPC generated-stub compatibility alias

"""Public compatibility alias for runtime fallback gRPC service stubs.

This module keeps the historical import path stable:

- ``scpn_phase_orchestrator.grpc_gen._spo_pb2_grpc_fallback``

while delegating implementation ownership to:

- ``scpn_phase_orchestrator.runtime.grpc_gen._spo_pb2_grpc_fallback``

Operational contract:

1. Importing this module reuses runtime service stub definitions.
2. Alias behaviour does not fork or mutate runtime handler contracts.
3. Public callers receive the same symbol surface as runtime fallback stubs.
"""

from __future__ import annotations

import sys
from importlib import import_module

_module = import_module(
    "scpn_phase_orchestrator.runtime.grpc_gen._spo_pb2_grpc_fallback"
)
sys.modules[__name__] = _module

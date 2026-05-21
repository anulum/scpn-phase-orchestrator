# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — gRPC generated-stub compatibility alias

"""Public compatibility alias for runtime generated gRPC service bindings.

This module preserves the historical public import path:

- ``scpn_phase_orchestrator.grpc_gen.spo_pb2_grpc``

and delegates to the canonical runtime implementation:

- ``scpn_phase_orchestrator.runtime.grpc_gen.spo_pb2_grpc``

Operational contract:

1. gRPC service interfaces are owned by runtime generated bindings.
2. The alias module must not alter method signatures or handler names.
3. Existing callers importing from the public path remain source-compatible.
"""

from __future__ import annotations

import sys
from importlib import import_module

_module = import_module("scpn_phase_orchestrator.runtime.grpc_gen.spo_pb2_grpc")
sys.modules[__name__] = _module

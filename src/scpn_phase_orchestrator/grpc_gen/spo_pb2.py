# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — gRPC generated-stub compatibility alias

"""Public compatibility alias for runtime protobuf message bindings.

This module preserves the historical import path:

- ``scpn_phase_orchestrator.grpc_gen.spo_pb2``

and delegates to the canonical runtime module:

- ``scpn_phase_orchestrator.runtime.grpc_gen.spo_pb2``

Operational contract:

1. No generated-message duplication in public alias layer.
2. Runtime message descriptors remain the single source of truth.
3. Alias import path remains stable for existing external clients.
"""

from __future__ import annotations

import sys
from importlib import import_module

_module = import_module("scpn_phase_orchestrator.runtime.grpc_gen.spo_pb2")
sys.modules[__name__] = _module

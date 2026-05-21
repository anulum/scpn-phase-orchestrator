# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — gRPC generated-stub compatibility alias

"""Public compatibility alias for runtime fallback protobuf message classes.

This module exists to preserve historical import paths:

- ``scpn_phase_orchestrator.grpc_gen._spo_pb2_fallback``

The canonical implementation lives under:

- ``scpn_phase_orchestrator.runtime.grpc_gen._spo_pb2_fallback``

Operational contract:

1. Importing this module must not redefine fallback message types.
2. The module object in ``sys.modules`` is replaced with the runtime module.
3. Downstream callers receive the same ``__all__`` export surface and class
   semantics as the runtime implementation.
"""

from __future__ import annotations

import sys
from importlib import import_module

_module = import_module("scpn_phase_orchestrator.runtime.grpc_gen._spo_pb2_fallback")
sys.modules[__name__] = _module

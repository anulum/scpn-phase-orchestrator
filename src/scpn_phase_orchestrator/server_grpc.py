# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - Legacy runtime import wrapper

"""Compatibility alias for ``scpn_phase_orchestrator.runtime.server_grpc``."""

from __future__ import annotations

import sys
from importlib import import_module

_module = import_module("scpn_phase_orchestrator.runtime.server_grpc")
sys.modules[__name__] = _module

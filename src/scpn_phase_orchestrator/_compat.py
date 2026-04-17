# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Rust/Python compatibility shim

from __future__ import annotations

import importlib.util

import numpy as np

__all__ = ["TWO_PI", "HAS_RUST"]

TWO_PI = 2.0 * np.pi
HAS_RUST = importlib.util.find_spec("spo_kernel") is not None

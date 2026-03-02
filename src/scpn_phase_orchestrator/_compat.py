# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

import importlib.util

import numpy as np

__all__ = ["TWO_PI", "HAS_RUST"]

TWO_PI = 2.0 * np.pi
HAS_RUST = importlib.util.find_spec("spo_kernel") is not None

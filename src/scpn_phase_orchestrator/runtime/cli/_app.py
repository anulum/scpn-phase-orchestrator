# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI application group and shared constants

"""Command-line entry point for validation, replay, export, and review workflows.

The CLI wraps public SPO APIs behind explicit commands for binding validation,
inspection, auto-binding proposals, coupling estimation, formal export, replay,
plugin catalogs, scaffolding, and selected runtime utilities. Commands validate
local inputs and emit text or JSON review artifacts; they do not push commits,
start network services, or perform live actuation unless an explicit subcommand
is invoked for that runtime path.
"""

from __future__ import annotations

from typing import TypeAlias

import click
import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
_PHYSIONET_HEARTBEAT_URL = (
    "https://physionet.org/files/respiratory-heartrate-dataset/1.0.0/"
    "HRM_rawData/HRB/3.txt"
)
_PHYSIONET_HEARTBEAT_CITATION = (
    "Guy et al. (2024), Respiratory and heart rate monitoring dataset from "
    "aeration study, PhysioNet, doi:10.13026/e4dt-f689"
)


@click.group()
@click.version_option(
    package_name="scpn-phase-orchestrator", prog_name="scpn-phase-orchestrator"
)
def main() -> None:
    """SCPN Phase Orchestrator CLI."""

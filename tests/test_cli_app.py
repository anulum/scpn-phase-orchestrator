# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI application group tests

"""Tests for the CLI application group and shared constants.

``runtime.cli._app`` holds the click ``main`` group every command module attaches
to, plus the ``FloatArray`` alias and the PhysioNet heartbeat-dataset constants the
demo data path uses. These assertions pin the group identity and the constants so
the command modules have a stable anchor to register against.
"""

from __future__ import annotations

from importlib.metadata import version

import click
import numpy as np
from click.testing import CliRunner

from scpn_phase_orchestrator.runtime.cli._app import (
    _PHYSIONET_HEARTBEAT_CITATION,
    _PHYSIONET_HEARTBEAT_URL,
    FloatArray,
    main,
)


def test_main_is_a_click_group() -> None:
    assert isinstance(main, click.Group)
    assert main.name == "main"


def test_float_array_alias_is_float64() -> None:
    array: FloatArray = np.zeros(3, dtype=np.float64)
    assert array.dtype == np.float64


def test_physionet_constants_are_well_formed() -> None:
    assert _PHYSIONET_HEARTBEAT_URL.startswith("https://physionet.org/")
    assert "doi:" in _PHYSIONET_HEARTBEAT_CITATION


def test_main_reports_the_installed_version() -> None:
    result = CliRunner().invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "scpn-phase-orchestrator" in result.output
    assert version("scpn-phase-orchestrator") in result.output

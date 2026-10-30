# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-module pytest duration recorder

"""Record measured setup, call and teardown seconds per test module.

Load with ``-p tools.pytest_timing_plugin --module-timings PATH``. The
manifest it writes is what :mod:`tools.pytest_shards` schedules with. Without
the option the plugin records nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

SCHEMA = 1
"""Manifest version, equal to :data:`tools.pytest_shards.SCHEMA`."""


class ModuleTimings:
    """Accumulate report durations by the module part of each node id."""

    def __init__(self, output: Path) -> None:
        """Remember where the manifest goes."""
        self.output = output
        self.durations: dict[str, float] = {}

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        """Add one phase's measured seconds to its module."""
        module = report.nodeid.split("::", 1)[0]
        self.durations[module] = self.durations.get(module, 0.0) + report.duration

    def pytest_sessionfinish(self) -> None:
        """Write the manifest; every timed module gets at least a millisecond."""
        durations = {
            name: max(seconds, 0.001) for name, seconds in self.durations.items()
        }
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(
            json.dumps({"schema": SCHEMA, "durations": durations}, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )


def pytest_addoption(parser: pytest.Parser) -> None:
    """Declare the manifest path option."""
    parser.addoption(
        "--module-timings", type=Path, help="write per-module seconds here"
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the recorder when a manifest path was given."""
    output = config.getoption("--module-timings")
    if isinstance(output, Path):
        config.pluginmanager.register(ModuleTimings(output), "spo-module-timings")

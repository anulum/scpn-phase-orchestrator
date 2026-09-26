# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI startup integration

"""Exercise CLI help in a fresh process without optional backend probes."""

from __future__ import annotations

import json
import subprocess
import sys


def test_cli_help_keeps_numeric_backend_probes_deferred() -> None:
    script = """
import json
import sys
from click.testing import CliRunner
from scpn_phase_orchestrator.runtime.cli import main

result = CliRunner().invoke(main, ["--help"])
modules = (
    "scpn_phase_orchestrator.upde._run",
    "scpn_phase_orchestrator.upde.engine",
    "scpn_phase_orchestrator.upde.order_params",
    "scpn_phase_orchestrator.upde.pac",
    "scpn_phase_orchestrator.upde.reduction",
    "scpn_phase_orchestrator.upde.hypergraph",
    "scpn_phase_orchestrator.monitor.koopman_edmd",
    "scpn_phase_orchestrator.monitor.twin_confidence",
)
print(json.dumps({
    "exit_code": result.exit_code,
    "help_has_run": "run" in result.output,
    "help_has_twin_confidence": "twin-confidence" in result.output,
    "probed": {
        name: name in sys.modules and (
            bool(vars(sys.modules[name]).get("_BACKEND_CACHE"))
            or "ACTIVE_BACKEND" in vars(sys.modules[name])
        )
        for name in modules
    },
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=45,
        check=True,
    )
    result = json.loads(completed.stdout)
    assert result["exit_code"] == 0
    assert result["help_has_run"] is True
    assert result["help_has_twin_confidence"] is True
    assert result["probed"] == dict.fromkeys(result["probed"], False)

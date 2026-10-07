# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Actual CLI consumer observation

"""Invoke the public CLI and report installed SINDy owner calls on stderr.

The CLI receives normal arguments and retains its stdout, errors and exit code.
Only the Python profiling hook is installed; no computation is substituted.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import FrameType, FunctionType

from scpn_phase_orchestrator.autotune.sindy import PhaseSINDy
from scpn_phase_orchestrator.runtime.cli import main as cli_main


def main() -> None:
    """Run the installed CLI with an observer that preserves its terminal outcome.

    Raises
    ------
    SystemExit
        With the genuine CLI status for valid or refused operator inputs.
    """
    present = importlib.util.find_spec("spo_kernel") is not None
    owner: object = None
    binary: dict[str, str] | None = None
    if present:
        kernel = importlib.import_module("spo_kernel")
        owner = kernel.sindy_fit_rust
        native = importlib.import_module("spo_kernel.spo_kernel")
        path = Path(inspect.getfile(native))
        binary = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "version": importlib.metadata.version("spo-kernel"),
        }
    scipy_owner: object = importlib.import_module("scipy.linalg").lstsq
    calls: list[str] = []

    def observe(frame: FrameType, event: str, argument: object) -> None:
        """Record actual installed owners without intercepting their results."""
        if event == "c_call" and argument is owner:
            calls.append("spo_kernel.spo_kernel.sindy_fit_rust")
        if (
            event == "call"
            and isinstance(scipy_owner, FunctionType)
            and frame.f_code is scipy_owner.__code__
        ):
            calls.append("scipy.linalg.lstsq")

    source = Path(inspect.getfile(PhaseSINDy))
    previous_profile = sys.getprofile()
    try:
        sys.setprofile(observe)
        cli_main(prog_name="spo")
    finally:
        sys.setprofile(previous_profile)
        print(
            "SINDY_CLI_TRACE="
            + json.dumps(
                {
                    "interpreter": sys.executable,
                    "kernel_present": present,
                    "native_artifact": binary,
                    "estimator_path": str(source),
                    "estimator_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "observed_calls": calls,
                },
                sort_keys=True,
                allow_nan=False,
            ),
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()

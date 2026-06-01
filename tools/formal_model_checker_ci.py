# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CI-only external formal checker smoke gate

"""CI-only smoke execution for external SPIN and Z3 formal checkers.

This tool materialises reviewed Promela and SMT-LIB artefacts through the
formal package API, checks deterministic package/readiness metadata, and runs
the external tools only in GitHub Actions with an explicit execution flag. It is
not a runtime execution path and does not weaken the formal package
``execution_permitted=False`` contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Final

from scpn_phase_orchestrator.exceptions import PolicyError
from scpn_phase_orchestrator.supervisor import (
    FormalSafetyProperty,
    FormalTextArtifact,
    audit_formal_checker_availability,
    build_formal_verification_package,
)

PROMELA_ARTIFACT: Final[str] = """\
active proctype supervisor_smoke()
{
  byte tokens = 1;
  assert(tokens <= 1);
}
"""

SMT_ARTIFACT: Final[str] = """\
(set-logic QF_LRA)
(declare-const R Real)
(assert (>= R 0.0))
(assert (<= R 1.0))
(check-sat)
"""


def build_smoke_package():
    """Return the deterministic SPIN/Z3 smoke package."""

    return build_formal_verification_package(
        {
            "protocol_spin": FormalTextArtifact(
                artifact_type="promela",
                text=PROMELA_ARTIFACT,
            ),
            "barrier_smt": FormalTextArtifact(
                artifact_type="smt2",
                text=SMT_ARTIFACT,
            ),
        },
        (
            FormalSafetyProperty(
                name="spin_token_bound",
                artifact_name="protocol_spin",
                checker="spin",
                expression="assert_tokens_bounded",
                description="SPIN smoke proof checks bounded supervisor tokens.",
            ),
            FormalSafetyProperty(
                name="smt_unit_interval_feasible",
                artifact_name="barrier_smt",
                checker="smt",
                expression="check-sat",
                description="Z3 smoke proof checks feasible unit interval guard.",
            ),
        ),
        package_name="spo-formal-checker-ci",
    )


def _require_ci_execution(execute: bool) -> None:
    if not execute:
        return
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise PolicyError("external formal checker execution is CI-only")


def _run_checked(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def run_ci_smoke(*, execute: bool) -> dict[str, object]:
    """Build readiness metadata and optionally run external checkers in CI."""

    _require_ci_execution(execute)
    package = build_smoke_package()
    readiness = audit_formal_checker_availability(package)
    readiness_records = [record.to_audit_record() for record in readiness]
    missing = [
        record["executable"]
        for record in readiness_records
        if record["status"] != "ready_not_executed"
    ]
    if execute and missing:
        raise PolicyError(
            "missing external formal checker executable(s): " + ", ".join(missing)
        )

    execution_records: list[dict[str, object]] = []
    if execute:
        with tempfile.TemporaryDirectory(prefix="spo-formal-ci-") as temp_dir:
            workdir = Path(temp_dir)
            (workdir / "protocol_spin.pml").write_text(
                PROMELA_ARTIFACT,
                encoding="utf-8",
            )
            (workdir / "barrier_smt.smt2").write_text(
                SMT_ARTIFACT,
                encoding="utf-8",
            )
            for command in package.checker_commands:
                completed = _run_checked(list(command.command), cwd=workdir)
                execution_records.append(
                    {
                        "property_name": command.property_name,
                        "checker": command.checker,
                        "artifact_name": command.artifact_name,
                        "returncode": completed.returncode,
                        "stdout_sha256": hashlib.sha256(
                            completed.stdout.encode("utf-8")
                        ).hexdigest(),
                        "stderr_sha256": hashlib.sha256(
                            completed.stderr.encode("utf-8")
                        ).hexdigest(),
                    }
                )

    return {
        "package": package.to_audit_record(),
        "checker_availability": readiness_records,
        "executed": execute,
        "execution_records": execution_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CI-only external formal checker smoke gate.",
    )
    parser.add_argument(
        "--execute-ci-smoke",
        action="store_true",
        help="Execute external checkers; requires GITHUB_ACTIONS=true.",
    )
    args = parser.parse_args()
    record = run_ci_smoke(execute=args.execute_ci_smoke)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

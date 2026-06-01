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
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from scpn_phase_orchestrator.exceptions import PolicyError
from scpn_phase_orchestrator.supervisor import (
    FormalSafetyProperty,
    FormalTextArtifact,
    FormalVerificationPackage,
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

DOMAINPACK_SAFETY_CASES: Final[tuple[dict[str, object], ...]] = (
    {
        "domainpack": "cardiac_rhythm",
        "safety_tier": "clinical",
        "smt_artifact": "cardiac_rhythm_hard_bounds",
        "spin_artifact": "cardiac_rhythm_operator_gate",
        "smt_text": """\
(set-logic QF_LRA)
(declare-const heart_rate_bpm Real)
(declare-const qt_ms Real)
(assert (>= heart_rate_bpm 40.0))
(assert (<= heart_rate_bpm 180.0))
(assert (<= qt_ms 500.0))
(check-sat)
""",
        "smt_property": "cardiac_hard_bounds_feasible",
        "spin_property": "cardiac_operator_gate_safe",
        "smt_description": (
            "Cardiac hard-bound envelope remains feasible for heart-rate and QT "
            "safety limits."
        ),
    },
    {
        "domainpack": "chemical_reactor",
        "safety_tier": "production",
        "smt_artifact": "chemical_reactor_hard_bounds",
        "spin_artifact": "chemical_reactor_operator_gate",
        "smt_text": """\
(set-logic QF_LRA)
(declare-const temperature_c Real)
(declare-const pressure_bar Real)
(assert (<= temperature_c 450.0))
(assert (<= pressure_bar 15.0))
(check-sat)
""",
        "smt_property": "chemical_reactor_hard_bounds_feasible",
        "spin_property": "chemical_reactor_operator_gate_safe",
        "smt_description": (
            "Chemical-reactor hard-bound envelope remains feasible for thermal "
            "and pressure safety limits."
        ),
    },
    {
        "domainpack": "power_grid",
        "safety_tier": "production",
        "smt_artifact": "power_grid_hard_bounds",
        "spin_artifact": "power_grid_operator_gate",
        "smt_text": """\
(set-logic QF_LRA)
(declare-const frequency_dev Real)
(declare-const voltage_pu Real)
(declare-const rotor_angle_deg Real)
(assert (>= frequency_dev (- 0.5)))
(assert (<= frequency_dev 0.5))
(assert (>= voltage_pu 0.95))
(assert (<= voltage_pu 1.05))
(assert (<= rotor_angle_deg 90.0))
(check-sat)
""",
        "smt_property": "power_grid_hard_bounds_feasible",
        "spin_property": "power_grid_operator_gate_safe",
        "smt_description": (
            "Power-grid hard-bound envelope remains feasible for frequency, "
            "voltage, and rotor-angle limits."
        ),
    },
    {
        "domainpack": "pll_clock",
        "safety_tier": "production",
        "smt_artifact": "pll_clock_hard_bounds",
        "spin_artifact": "pll_clock_operator_gate",
        "smt_text": """\
(set-logic QF_LRA)
(declare-const phase_error_ns Real)
(declare-const freq_drift_ppm Real)
(assert (<= phase_error_ns 100.0))
(assert (<= freq_drift_ppm 10.0))
(check-sat)
""",
        "smt_property": "pll_clock_hard_bounds_feasible",
        "spin_property": "pll_clock_operator_gate_safe",
        "smt_description": (
            "PLL clock hard-bound envelope remains feasible for phase-error and "
            "frequency-drift limits."
        ),
    },
    {
        "domainpack": "autonomous_vehicles",
        "safety_tier": "research",
        "smt_artifact": "autonomous_vehicles_hard_bounds",
        "spin_artifact": "autonomous_vehicles_operator_gate",
        "smt_text": """\
(set-logic QF_LRA)
(declare-const gap_distance Real)
(declare-const brake_reaction Real)
(assert (>= gap_distance 0.2))
(assert (<= brake_reaction 0.3))
(check-sat)
""",
        "smt_property": "autonomous_vehicles_hard_bounds_feasible",
        "spin_property": "autonomous_vehicles_operator_gate_safe",
        "smt_description": (
            "Autonomous-vehicle boundary envelope remains feasible for minimum "
            "gap distance and brake-reaction limits."
        ),
    },
    {
        "domainpack": "satellite_constellation",
        "safety_tier": "research",
        "smt_artifact": "satellite_constellation_hard_bounds",
        "spin_artifact": "satellite_constellation_operator_gate",
        "smt_text": """\
(set-logic QF_LRA)
(declare-const link_budget Real)
(assert (>= link_budget 0.3))
(check-sat)
""",
        "smt_property": "satellite_constellation_hard_bounds_feasible",
        "spin_property": "satellite_constellation_operator_gate_safe",
        "smt_description": (
            "Satellite-constellation boundary envelope remains feasible for the "
            "minimum link-budget limit."
        ),
    },
    {
        "domainpack": "power_safety_nchannel",
        "safety_tier": "research",
        "smt_artifact": "power_safety_nchannel_hard_bounds",
        "spin_artifact": "power_safety_nchannel_operator_gate",
        "smt_text": """\
(set-logic QF_LRA)
(declare-const R_2 Real)
(assert (>= R_2 0.52))
(check-sat)
""",
        "smt_property": "power_safety_nchannel_hard_bounds_feasible",
        "spin_property": "power_safety_nchannel_operator_gate_safe",
        "smt_description": (
            "Power-safety N-channel boundary envelope remains feasible for the "
            "dispatch-lock coherence floor."
        ),
    },
    {
        "domainpack": "traffic_flow",
        "safety_tier": "consumer",
        "smt_artifact": "traffic_flow_hard_bounds",
        "spin_artifact": "traffic_flow_operator_gate",
        "smt_text": """\
(set-logic QF_LRA)
(declare-const queue_vehicles Real)
(assert (<= queue_vehicles 50.0))
(check-sat)
""",
        "smt_property": "traffic_flow_hard_bounds_feasible",
        "spin_property": "traffic_flow_operator_gate_safe",
        "smt_description": (
            "Traffic-flow boundary envelope remains feasible for the hard queue "
            "overflow limit."
        ),
    },
    {
        "domainpack": "swarm_robotics",
        "safety_tier": "consumer",
        "smt_artifact": "swarm_robotics_hard_bounds",
        "spin_artifact": "swarm_robotics_operator_gate",
        "smt_text": """\
(set-logic QF_LRA)
(declare-const formation_error_m Real)
(declare-const min_dist_m Real)
(assert (<= formation_error_m 2.0))
(assert (>= min_dist_m 0.5))
(check-sat)
""",
        "smt_property": "swarm_robotics_hard_bounds_feasible",
        "spin_property": "swarm_robotics_operator_gate_safe",
        "smt_description": (
            "Swarm-robotics boundary envelope remains feasible for formation "
            "error and minimum-distance limits."
        ),
    },
    {
        "domainpack": "manufacturing_spc",
        "safety_tier": "consumer",
        "smt_artifact": "manufacturing_spc_hard_bounds",
        "spin_artifact": "manufacturing_spc_operator_gate",
        "smt_text": """\
(set-logic QF_LRA)
(declare-const temperature Real)
(declare-const pressure Real)
(assert (<= temperature 85.0))
(assert (>= pressure 2.0))
(check-sat)
""",
        "smt_property": "manufacturing_spc_hard_bounds_feasible",
        "spin_property": "manufacturing_spc_operator_gate_safe",
        "smt_description": (
            "Manufacturing SPC boundary envelope remains feasible for hard "
            "temperature and pressure limits."
        ),
    },
    {
        "domainpack": "robotic_cpg",
        "safety_tier": "consumer",
        "smt_artifact": "robotic_cpg_hard_bounds",
        "spin_artifact": "robotic_cpg_operator_gate",
        "smt_text": """\
(set-logic QF_LRA)
(declare-const joint_angle_rad Real)
(declare-const joint_torque_nm Real)
(assert (>= joint_angle_rad (- 2.0)))
(assert (<= joint_angle_rad 2.0))
(assert (<= joint_torque_nm 50.0))
(check-sat)
""",
        "smt_property": "robotic_cpg_hard_bounds_feasible",
        "spin_property": "robotic_cpg_operator_gate_safe",
        "smt_description": (
            "Robotic CPG boundary envelope remains feasible for joint-angle and "
            "torque limits."
        ),
    },
)


@dataclass(frozen=True)
class FormalCheckerCiBundle:
    """Materialisable formal package plus reviewed artefact text."""

    name: str
    package: FormalVerificationPackage
    artifacts: Mapping[str, str]
    domainpack: str | None = None
    invariant_summary: str = ""

    def to_audit_record(self) -> dict[str, object]:
        package_record = self.package.to_audit_record()
        return {
            "name": self.name,
            "domainpack": self.domainpack,
            "artifact_count": len(self.artifacts),
            "invariant_summary": self.invariant_summary,
            "package": package_record,
        }


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


def build_smoke_bundle() -> FormalCheckerCiBundle:
    """Return the materialisable smoke package bundle."""

    return FormalCheckerCiBundle(
        name="smoke",
        package=build_smoke_package(),
        artifacts={
            "protocol_spin": PROMELA_ARTIFACT,
            "barrier_smt": SMT_ARTIFACT,
        },
        invariant_summary="SPIN token bound and Z3 unit-interval feasibility.",
    )


def _operator_gate_promela(domainpack: str, safety_tier: str) -> str:
    proctype_name = f"{domainpack}_operator_gate"
    return f"""\
active proctype {proctype_name}()
{{
  byte proposed_actions = 0;
  byte operator_approved = 0;

  if
  :: proposed_actions = 0
  :: proposed_actions = 1; operator_approved = 1
  fi;

  assert(proposed_actions <= 1);
  assert((proposed_actions == 0) || (operator_approved == 1));
  /* safety_tier: {safety_tier} */
}}
"""


def build_domainpack_formal_packages() -> tuple[FormalCheckerCiBundle, ...]:
    """Return materialisable SPIN/Z3 packages for safety-critical domainpacks."""

    bundles: list[FormalCheckerCiBundle] = []
    for case in DOMAINPACK_SAFETY_CASES:
        domainpack = str(case["domainpack"])
        safety_tier = str(case["safety_tier"])
        spin_artifact = str(case["spin_artifact"])
        smt_artifact = str(case["smt_artifact"])
        spin_text = _operator_gate_promela(domainpack, safety_tier)
        smt_text = str(case["smt_text"])
        package = build_formal_verification_package(
            {
                spin_artifact: FormalTextArtifact(
                    artifact_type="promela",
                    text=spin_text,
                ),
                smt_artifact: FormalTextArtifact(
                    artifact_type="smt2",
                    text=smt_text,
                ),
            },
            (
                FormalSafetyProperty(
                    name=str(case["spin_property"]),
                    artifact_name=spin_artifact,
                    checker="spin",
                    expression="operator_approval_required",
                    description=(
                        f"{domainpack} formal gate permits proposed actions only "
                        "with operator approval."
                    ),
                ),
                FormalSafetyProperty(
                    name=str(case["smt_property"]),
                    artifact_name=smt_artifact,
                    checker="smt",
                    expression="check-sat",
                    description=str(case["smt_description"]),
                ),
            ),
            package_name=f"spo-{domainpack}-formal-ci",
        )
        bundles.append(
            FormalCheckerCiBundle(
                name=domainpack,
                package=package,
                artifacts={
                    spin_artifact: spin_text,
                    smt_artifact: smt_text,
                },
                domainpack=domainpack,
                invariant_summary=(
                    f"{safety_tier} operator approval gate plus hard-bound SMT "
                    "feasibility."
                ),
            )
        )
    return tuple(bundles)


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
        capture_output=True,
    )


def _artifact_filename(artifact_name: str, artifact_type: str) -> str:
    if artifact_type == "promela":
        return f"{artifact_name}.pml"
    if artifact_type == "smt2":
        return f"{artifact_name}.smt2"
    raise PolicyError(f"unsupported CI formal artifact type: {artifact_type}")


def _run_bundle(bundle: FormalCheckerCiBundle, *, execute: bool) -> dict[str, object]:
    package = bundle.package
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
            for artifact_name, text in bundle.artifacts.items():
                artifact_type = package.artifact_types[artifact_name]
                (workdir / _artifact_filename(artifact_name, artifact_type)).write_text(
                    text,
                    encoding="utf-8",
                )
            for command in package.checker_commands:
                completed = _run_checked(list(command.command), cwd=workdir)
                if command.checker == "smt":
                    first_line = next(iter(completed.stdout.splitlines()), "").strip()
                    if first_line != "sat":
                        raise PolicyError(
                            f"SMT checker for {command.property_name} returned "
                            f"{first_line!r}, expected 'sat'"
                        )
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
        **bundle.to_audit_record(),
        "checker_availability": readiness_records,
        "executed": execute,
        "execution_records": execution_records,
    }


def run_ci_smoke(*, execute: bool) -> dict[str, object]:
    """Build readiness metadata and optionally run smoke checkers in CI."""

    _require_ci_execution(execute)
    bundle = build_smoke_bundle()
    bundle_record = _run_bundle(bundle, execute=execute)
    return {
        "package": bundle.package.to_audit_record(),
        "checker_availability": bundle_record["checker_availability"],
        "executed": execute,
        "execution_records": bundle_record["execution_records"],
    }


def run_formal_checker_ci(*, execute: bool) -> dict[str, object]:
    """Build and optionally execute all formal checker CI packages."""

    _require_ci_execution(execute)
    bundles = (build_smoke_bundle(), *build_domainpack_formal_packages())
    records = [_run_bundle(bundle, execute=execute) for bundle in bundles]
    domain_records = [record for record in records if record["domainpack"] is not None]
    return {
        "executed": execute,
        "bundle_count": len(records),
        "domainpack_bundle_count": len(domain_records),
        "checker_command_count": sum(
            len(record["package"]["checker_commands"]) for record in records
        ),
        "records": records,
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
    record = run_formal_checker_ci(execute=args.execute_ci_smoke)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

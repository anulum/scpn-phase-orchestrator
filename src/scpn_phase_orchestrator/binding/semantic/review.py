# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Semantic compiler review notebook generation

"""Review notebook, gate record, and dry-run evidence for generated bindings."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scpn_phase_orchestrator.binding.types import (
    BindingSpec,
)
from scpn_phase_orchestrator.binding.validator import validate_binding_spec


def _review_notebook_for(
    spec: BindingSpec,
    *,
    confidence: float,
    retrieval_records: list[dict[str, Any]],
    notebook_execution: dict[str, Any],
) -> str:
    """Return the review record for a notebook."""
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# Review generated domainpack: {spec.name}\n",
                    "\n",
                    "This notebook validates generated binding and policy "
                    "artifacts before any live use.\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Retrieval and confidence\n",
                    "\n",
                    f"- Confidence: `{confidence:.3f}`\n",
                    f"- Retrieval matches: `{len(retrieval_records)}`\n",
                    "- Notebook preflight: "
                    f"`{notebook_execution['status']}` "
                    f"({notebook_execution['passed_checks']}/"
                    f"{notebook_execution['total_checks']} checks)\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    "from scpn_phase_orchestrator.binding import "
                    "load_binding_spec, validate_binding_spec\n",
                    "spec = load_binding_spec(Path('binding_spec.yaml'))\n",
                    "errors = validate_binding_spec(spec)\n",
                    "assert errors == [], errors\n",
                    "spec.name\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from scpn_phase_orchestrator.supervisor.policy_rules import "
                    "load_policy_rules\n",
                    "rules = load_policy_rules(Path('policy.yaml'))\n",
                    "assert rules\n",
                    "[rule.name for rule in rules]\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Preflight evidence\n",
                    "\n",
                    "The compiler executed the same schema and policy checks "
                    "that this review notebook asks you to run locally.\n",
                    "\n",
                    f"- Status: `{notebook_execution['status']}`\n",
                    f"- Checks: `{notebook_execution['passed_checks']}/"
                    f"{notebook_execution['total_checks']}`\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Review checklist\n",
                    "\n",
                    "- Confirm layer names and oscillator counts match the plant.\n",
                    "- Confirm actuator limits are safe for the deployment target.\n",
                    "- Run a dry replay before connecting live adapters.\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "scpn_phase_orchestrator": {
                "artifact": "symbolic_binding_review",
                "notebook_execution": notebook_execution,
                "schema_version": 1,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, indent=2, sort_keys=True) + "\n"


def _review_notebook_execution_evidence(
    *,
    binding_yaml: str,
    policy_yaml: str,
    expected_name: str,
) -> dict[str, Any]:
    """Return the execution evidence from a reviewed notebook."""
    import tempfile

    from scpn_phase_orchestrator.binding.loader import load_binding_spec
    from scpn_phase_orchestrator.supervisor.policy_rules import load_policy_rules

    checks: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="spo_generated_review_") as tmp:
        path = Path(tmp)
        binding_path = path / "binding_spec.yaml"
        policy_path = path / "policy.yaml"
        binding_path.write_text(binding_yaml, encoding="utf-8")
        policy_path.write_text(policy_yaml, encoding="utf-8")

        try:
            spec = load_binding_spec(binding_path)
            checks.append({"name": "load_binding_spec", "passed": True})
        except Exception as exc:  # pragma: no cover - defensive audit payload
            checks.append(
                {
                    "name": "load_binding_spec",
                    "passed": False,
                    "error": type(exc).__name__,
                }
            )
            spec = None

        if spec is not None:
            errors = validate_binding_spec(spec)
            checks.append(
                {
                    "name": "validate_binding_spec",
                    "passed": errors == [],
                    "errors": errors,
                }
            )
            checks.append(
                {
                    "name": "spec_name_matches",
                    "passed": spec.name == expected_name,
                    "observed": spec.name,
                }
            )

        try:
            rules = load_policy_rules(policy_path)
            checks.append(
                {
                    "name": "load_policy_rules",
                    "passed": len(rules) > 0,
                    "rule_count": len(rules),
                }
            )
        except Exception as exc:  # pragma: no cover - defensive audit payload
            checks.append(
                {
                    "name": "load_policy_rules",
                    "passed": False,
                    "error": type(exc).__name__,
                }
            )

    passed = sum(1 for check in checks if check["passed"])
    return {
        "status": "passed" if passed == len(checks) else "failed",
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": checks,
    }


def _review_gate_record() -> dict[str, Any]:
    """Build the review-gate record."""
    return {
        "status": "required",
        "non_actuating": True,
        "manual_review_required": True,
        "auto_execution_enabled": False,
        "required_artifacts": [
            "binding_spec.yaml",
            "policy.yaml",
            "review_notebook.ipynb",
            "audit.json",
        ],
    }


def _validate_generated_audit_schema(audit_record: Mapping[str, Any]) -> None:
    """Validate the generated audit schema, else raise."""
    required = {
        "compiler": str,
        "schema_valid": bool,
        "validation_errors": list,
        "intent_boundary": dict,
        "review_gate": dict,
        "confidence": float,
        "confidence_factors": dict,
        "retrieval_evidence": list,
        "notebook_execution": dict,
    }
    for key, expected_type in required.items():
        value = audit_record.get(key)
        if not isinstance(value, expected_type):
            raise ValueError(f"generated audit schema invalid: {key}")
    review_gate = audit_record["review_gate"]
    if review_gate.get("non_actuating") is not True:
        raise ValueError(
            "generated audit schema invalid: review gate must be non-actuating"
        )
    if review_gate.get("manual_review_required") is not True:
        raise ValueError("generated audit schema invalid: manual review required")
    if review_gate.get("auto_execution_enabled") is not False:
        raise ValueError("generated audit schema invalid: auto execution disabled")


def _dry_run_order_parameter(spec: BindingSpec, steps: int) -> float:
    """Return the dry-run order parameter for the review."""
    import numpy as np

    from scpn_phase_orchestrator.coupling.knm import CouplingBuilder
    from scpn_phase_orchestrator.upde.engine import UPDEEngine
    from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

    n = sum(len(layer.oscillator_ids) for layer in spec.layers)
    coupling = CouplingBuilder().build(
        n,
        spec.coupling.base_strength,
        spec.coupling.decay_alpha,
    )
    engine = UPDEEngine(n, dt=spec.sample_period_s)
    phases = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=np.float64)
    omegas = np.asarray(spec.get_omegas(), dtype=np.float64)
    for _ in range(steps):
        phases = engine.step(phases, omegas, coupling.knm, 0.0, 0.0, coupling.alpha)
    r_value, _ = compute_order_parameter(phases)
    return float(r_value)

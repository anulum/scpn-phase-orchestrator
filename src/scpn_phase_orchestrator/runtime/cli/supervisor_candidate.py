# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI supervisor-candidate bundle command

"""Command-line generation of the auditable supervisor-candidate bundle.

The command reads a scenario file describing a candidate, the baseline to
attribute it against, the incumbent it would replace, the replay observations,
the safety constraints, the safety tier, and the numeric provenance. It scores
the candidate with the model-free reward over the representative observation,
assembles the sealed ``studio.supervisor_candidate.v1`` bundle, prints a summary,
and optionally writes the full record as JSON. It performs no control actuation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import click
import numpy as np

from scpn_phase_orchestrator.autotune import (
    AutotuneRewardReport,
    KnobPolicyCandidate,
    NumericProvenance,
    RewardObservation,
    SafetyConstraintConfig,
    build_supervisor_candidate_bundle,
    evaluate_knob_policy,
)
from scpn_phase_orchestrator.runtime.cli._app import main


def _require_number(value: object) -> float:
    """Return ``value`` as a float, rejecting booleans and non-numbers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"expected a number, got {value!r}")
    return float(value)


def _scalar_or_array(value: object) -> float | np.ndarray:
    """Coerce a JSON number or list into a float or a float array."""
    if isinstance(value, list):
        return np.array(value, dtype=float)
    return _require_number(value)


def _float_tuple(payload: Mapping[str, object], key: str) -> tuple[float, ...]:
    """Return a JSON list field of numbers as a float tuple, defaulting empty."""
    value = payload.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return tuple(_require_number(item) for item in value)


def _candidate(payload: Mapping[str, object]) -> KnobPolicyCandidate:
    """Build a candidate from a scenario knob mapping."""
    return KnobPolicyCandidate(
        alpha=_scalar_or_array(payload.get("alpha", 0.0)),
        zeta=_scalar_or_array(payload.get("zeta", 0.0)),
        channel_weights=_float_tuple(payload, "channel_weights"),
        cross_channel_gains=_float_tuple(payload, "cross_channel_gains"),
    )


def _optional_float(payload: Mapping[str, object], key: str) -> float | None:
    """Return a float field or ``None`` when it is absent or null."""
    value = payload.get(key)
    return None if value is None else float(value)  # type: ignore[arg-type]


def _observation(payload: Mapping[str, object]) -> RewardObservation:
    """Build a reward observation from a scenario observation mapping."""
    return RewardObservation(
        coherence=float(payload["coherence"]),  # type: ignore[arg-type]
        previous_coherence=_optional_float(payload, "previous_coherence"),
        unsafe=bool(payload.get("unsafe", False)),
        regime_changed=bool(payload.get("regime_changed", False)),
        lyapunov_exponent=_optional_float(payload, "lyapunov_exponent"),
        stl_robustness=_optional_float(payload, "stl_robustness"),
        safety_cost=float(payload.get("safety_cost", 0.0)),  # type: ignore[arg-type]
    )


def _constraints(payload: Mapping[str, object]) -> SafetyConstraintConfig:
    """Build the safety-constraint config from a scenario mapping."""
    return SafetyConstraintConfig(
        max_lyapunov_exponent=_optional_float(payload, "max_lyapunov_exponent"),
        min_stl_robustness=_optional_float(payload, "min_stl_robustness"),
        max_safety_cost=_optional_float(payload, "max_safety_cost"),
        require_lyapunov=bool(payload.get("require_lyapunov", False)),
        require_stl=bool(payload.get("require_stl", False)),
        require_safety_cost=bool(payload.get("require_safety_cost", False)),
    )


def _provenance(payload: Mapping[str, object]) -> NumericProvenance:
    """Build the numeric provenance from a scenario mapping."""
    return NumericProvenance(
        active_backend=str(payload["active_backend"]),
        parity_tolerance=float(payload["parity_tolerance"]),  # type: ignore[arg-type]
    )


@main.command("supervisor-candidate")
@click.argument("scenario", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--output",
    default=None,
    type=click.Path(),
    help="Write the sealed supervisor-candidate bundle here as JSON.",
)
def supervisor_candidate(scenario: str, output: str | None) -> None:
    """Assemble the review-only supervisor-candidate bundle from a scenario file.

    Parameters
    ----------
    scenario : str
        Path to a JSON scenario with ``candidate``, ``baseline``, ``incumbent``,
        ``observations``, ``constraints``, ``safety_tier``, and
        ``numeric_provenance`` fields.
    output : str | None
        Optional path for the full sealed bundle JSON.

    Raises
    ------
    ClickException
        If the scenario is missing, malformed, or describes invalid inputs.
    """
    try:
        payload = json.loads(Path(scenario).read_text(encoding="utf-8"))
        candidate = _candidate(payload["candidate"])
        baseline = _candidate(payload["baseline"])
        incumbent = _candidate(payload["incumbent"])
        observations = [_observation(item) for item in payload["observations"]]
        constraints = _constraints(payload.get("constraints", {}))
        provenance = _provenance(payload["numeric_provenance"])
        safety_tier = str(payload["safety_tier"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise click.ClickException(str(exc)) from exc

    if not observations:
        raise click.ClickException("scenario must list at least one observation")

    representative = observations[0]

    def evaluate(policy: KnobPolicyCandidate) -> AutotuneRewardReport:
        return evaluate_knob_policy(policy, representative)

    bundle = build_supervisor_candidate_bundle(
        candidate,
        baseline,
        incumbent,
        evaluate,
        observations=observations,
        constraints=constraints,
        safety_tier=safety_tier,
        numeric_provenance=provenance,
    )

    comparison = bundle.comparison
    click.echo("=== supervisor candidate ===")
    click.echo(
        f"reward: candidate={comparison.candidate_reward:.4f}  "
        f"incumbent={comparison.incumbent_reward:.4f}  "
        f"delta={comparison.reward_delta:+.4f}"
    )
    click.echo(
        f"safe={bundle.safety.safe}  improved={comparison.improved}  "
        f"safe_and_improved={bundle.safe_and_improved}"
    )
    click.echo(f"evidence: {bundle.evidence_kind}  digest={bundle.digest[:12]}")
    if bundle.attribution.attributions:
        top = bundle.attribution.attributions[0]
        click.echo(f"top knob: {top.knob} (shapley={top.shapley_total:+.4f})")

    if output is not None:
        Path(output).write_text(
            json.dumps(bundle.to_audit_record(), indent=2), encoding="utf-8"
        )
        click.echo(f"\nbundle written to {output}")

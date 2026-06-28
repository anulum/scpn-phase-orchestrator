# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Evolutionary supervisor grammar CLIs

"""CLI commands for the review-only evolutionary supervisor grammar family.

Three operator commands wrap the deterministic offline mutation-search grammars
in ``supervisor.evolutionary_policy_dsl``,
``supervisor.evolutionary_petri_grammar``, and
``supervisor.evolutionary_topology_grammar``. Each reads a local source artefact,
runs the offline search, and emits one deterministic review bundle. None of the
commands actuate, merge, hot-patch, or execute any mutated candidate; they only
generate operator-review evidence.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import click

from scpn_phase_orchestrator.runtime.cli._app import main
from scpn_phase_orchestrator.runtime.cli._payloads import _load_json_file, _record_hash
from scpn_phase_orchestrator.supervisor.evolutionary_petri_grammar import (
    run_offline_evolutionary_petri_mutation_grammar,
)
from scpn_phase_orchestrator.supervisor.evolutionary_policy_dsl import (
    run_offline_evolutionary_policy_dsl_search,
)
from scpn_phase_orchestrator.supervisor.evolutionary_topology_grammar import (
    run_offline_evolutionary_topology_mutation_search,
)

_OUTPUT_OPTION = click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path for the deterministic review bundle JSON.",
)


@main.command("evolutionary-policy-dsl-search")
@click.argument(
    "policy_dsl_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--generations", type=int, default=2, help="Number of search generations."
)
@click.option("--population", type=int, default=6, help="Candidates per generation.")
@click.option(
    "--mutation-step",
    type=float,
    default=0.05,
    help="Mutation step size per generation.",
)
@_OUTPUT_OPTION
def evolutionary_policy_dsl_search(
    policy_dsl_file: Path,
    generations: int,
    population: int,
    mutation_step: float,
    output: Path | None,
) -> None:
    """Emit review evidence for offline policy-DSL mutation candidates.

    Parameters
    ----------
    policy_dsl_file : Path
        Text file containing the compact policy-DSL source.
    generations : int
        Number of search generations.
    population : int
        Number of candidates per generation.
    mutation_step : float
        Mutation step size applied per generation.
    output : Path | None
        Optional path to write the emitted bundle JSON.

    Raises
    ------
    ClickException
        If the DSL source or search parameters are invalid.
    """
    policy_dsl = _read_text(policy_dsl_file, artifact="policy DSL")
    try:
        report = run_offline_evolutionary_policy_dsl_search(
            policy_dsl,
            generation_count=generations,
            population_size=population,
            mutation_step=mutation_step,
        )
    except ValueError as exc:
        raise click.ClickException(
            f"evolutionary policy-DSL search failed: {exc}"
        ) from exc
    _emit(_grammar_bundle("policy-dsl", report.to_audit_record()), output)


@main.command("evolutionary-petri-mutation")
@click.argument(
    "net_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--generations", type=int, default=2, help="Number of search generations."
)
@click.option(
    "--candidates-per-generation",
    type=int,
    default=6,
    help="Candidates evaluated per generation.",
)
@click.option(
    "--mutation-step",
    type=float,
    default=0.1,
    help="Mutation step size per generation.",
)
@click.option(
    "--max-arc-weight", type=int, default=4, help="Maximum mutated arc weight."
)
@click.option(
    "--max-token-bound", type=int, default=128, help="Maximum token count per place."
)
@_OUTPUT_OPTION
def evolutionary_petri_mutation(
    net_json: Path,
    generations: int,
    candidates_per_generation: int,
    mutation_step: float,
    max_arc_weight: int,
    max_token_bound: int,
    output: Path | None,
) -> None:
    """Emit review evidence for offline Petri-net mutation candidates.

    Parameters
    ----------
    net_json : Path
        JSON file with a net-like payload of places, transitions, and arcs.
    generations : int
        Number of search generations.
    candidates_per_generation : int
        Candidates evaluated per generation.
    mutation_step : float
        Mutation step size applied per generation.
    max_arc_weight : int
        Maximum arc weight allowed in a mutated net.
    max_token_bound : int
        Maximum token count allowed per place.
    output : Path | None
        Optional path to write the emitted bundle JSON.

    Raises
    ------
    ClickException
        If the net payload or bounds are invalid.
    """
    net_like = _load_net_like(net_json)
    try:
        plan = run_offline_evolutionary_petri_mutation_grammar(
            net_like,
            generation_count=generations,
            candidates_per_generation=candidates_per_generation,
            mutation_step=mutation_step,
            max_arc_weight=max_arc_weight,
            max_token_bound=max_token_bound,
        )
    except ValueError as exc:
        raise click.ClickException(
            f"evolutionary Petri mutation failed: {exc}"
        ) from exc
    _emit(_grammar_bundle("petri", plan.to_audit_record()), output)


@main.command("evolutionary-topology-mutation")
@click.argument(
    "topology_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--generations", type=int, default=2, help="Number of search generations."
)
@click.option("--population", type=int, default=8, help="Candidates per generation.")
@click.option(
    "--mutation-step",
    type=float,
    default=0.05,
    help="Mutation step size per generation.",
)
@click.option(
    "--min-edge-weight", type=float, default=0.0, help="Minimum retained edge weight."
)
@click.option(
    "--max-edge-weight", type=float, default=10.0, help="Maximum allowed edge weight."
)
@click.option(
    "--edge-add-base-weight",
    type=float,
    default=0.4,
    help="Base weight assigned to added edges.",
)
@click.option(
    "--max-add-candidates",
    type=int,
    default=16,
    help="Maximum number of edge-addition candidates.",
)
@_OUTPUT_OPTION
def evolutionary_topology_mutation(
    topology_json: Path,
    generations: int,
    population: int,
    mutation_step: float,
    min_edge_weight: float,
    max_edge_weight: float,
    edge_add_base_weight: float,
    max_add_candidates: int,
    output: Path | None,
) -> None:
    """Emit review evidence for offline topology mutation candidates.

    Parameters
    ----------
    topology_json : Path
        JSON file with a ``nodes`` array and an ``edges`` array.
    generations : int
        Number of search generations.
    population : int
        Number of candidates per generation.
    mutation_step : float
        Mutation step size applied per generation.
    min_edge_weight : float
        Minimum retained edge weight.
    max_edge_weight : float
        Maximum allowed edge weight.
    edge_add_base_weight : float
        Base weight assigned to newly added edges.
    max_add_candidates : int
        Maximum number of edge-addition candidates.
    output : Path | None
        Optional path to write the emitted bundle JSON.

    Raises
    ------
    ClickException
        If the topology records or bounds are invalid.
    """
    payload = _load_json_file(topology_json, artifact="topology")
    nodes = _json_array_of_objects(payload, "nodes")
    edges = _json_array_of_objects(payload, "edges")
    try:
        report = run_offline_evolutionary_topology_mutation_search(
            nodes,
            edges,
            generation_count=generations,
            population_size=population,
            mutation_step=mutation_step,
            min_edge_weight=min_edge_weight,
            max_edge_weight=max_edge_weight,
            edge_add_base_weight=edge_add_base_weight,
            max_add_candidates=max_add_candidates,
        )
    except ValueError as exc:
        raise click.ClickException(
            f"evolutionary topology mutation failed: {exc}"
        ) from exc
    _emit(_grammar_bundle("topology", report.to_audit_record()), output)


def _grammar_bundle(grammar: str, report: dict[str, object]) -> dict[str, object]:
    """Return the deterministic review bundle for one grammar report.

    Parameters
    ----------
    grammar : str
        Grammar identifier (``policy-dsl``, ``petri``, or ``topology``).
    report : dict[str, object]
        The grammar report or plan audit record.

    Returns
    -------
    dict[str, object]
        JSON-safe review bundle with a deterministic ``bundle_hash``.
    """
    bundle: dict[str, object] = {
        "schema": "scpn_evolutionary_grammar_review_bundle_v1",
        "version": "1.0.0",
        "grammar": grammar,
        "execution_disabled": True,
        "actuation_permitted": False,
        "live_merge_permitted": False,
        "hot_patch_permitted": False,
        "operator_review_required": True,
        "non_actuating": True,
        "report": report,
    }
    bundle["bundle_hash"] = _record_hash(bundle)
    return bundle


def _emit(bundle: dict[str, object], output: Path | None) -> None:
    """Render ``bundle`` to stdout and optionally to ``output``.

    Parameters
    ----------
    bundle : dict[str, object]
        The review bundle to render.
    output : Path | None
        Optional path to write the rendered JSON.

    Raises
    ------
    ClickException
        If the output file cannot be written.
    """
    rendered = json.dumps(bundle, indent=2, sort_keys=True)
    if output is not None:
        try:
            output.write_text(rendered + "\n", encoding="utf-8")
        except OSError as exc:
            raise click.ClickException(
                f"cannot write evolutionary grammar review bundle {output!s}: {exc}"
            ) from exc
    click.echo(rendered)


def _read_text(path: Path, *, artifact: str) -> str:
    """Return the UTF-8 text of ``path`` or raise a ClickException."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise click.ClickException(
            f"cannot read {artifact} file {path!s}: {exc}"
        ) from exc


def _load_net_like(path: Path) -> Mapping[str, object] | Sequence[object]:
    """Load a net-like JSON payload (object or array) or raise.

    Parameters
    ----------
    path : Path
        Path to the JSON net payload.

    Returns
    -------
    Mapping[str, object] | Sequence[object]
        The parsed net-like payload.

    Raises
    ------
    ClickException
        If the file is unreadable, malformed, or not a JSON object/array.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise click.ClickException(f"cannot read net file {path!s}: {exc}") from exc
    try:
        payload: object = json.loads(text)
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"malformed net JSON: {exc}") from exc
    if isinstance(payload, Mapping):
        return payload
    if isinstance(payload, Sequence) and not isinstance(
        payload, (str, bytes, bytearray)
    ):
        return payload
    raise click.ClickException("net payload must be a JSON object or array")


def _json_array_of_objects(
    payload: Mapping[str, object],
    field: str,
) -> tuple[Mapping[str, object], ...]:
    """Return ``payload[field]`` as a tuple of JSON-object mappings.

    Parameters
    ----------
    payload : Mapping[str, object]
        Source mapping.
    field : str
        Field expected to hold a JSON array of objects.

    Returns
    -------
    tuple[Mapping[str, object], ...]
        The parsed records.

    Raises
    ------
    ClickException
        If the field is absent, not an array, or contains a non-object item.
    """
    value = payload.get(field)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise click.ClickException(f"{field} must be a JSON array")
    records: list[Mapping[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise click.ClickException(f"{field}[{index}] must be a JSON object")
        records.append(item)
    return tuple(records)

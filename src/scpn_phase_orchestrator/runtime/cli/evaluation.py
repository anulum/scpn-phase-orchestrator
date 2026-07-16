# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI honest early-warning auditor command

"""Command-line entry point for the detector-agnostic early-warning auditor.

``spo audit-detector`` audits any early-warning detector's event-vs-null skill
from a JSON file of per-segment scores, without writing Python. The file holds the
detector's score on each genuine pre-transition event segment and on each
transition-free null segment; the command calibrates a matched-false-alarm
threshold, runs the label-permutation significance test, and emits the verdict as
JSON. Given a corpus identifier and a capture timestamp it seals the verdict into
a hash-addressed record. The command reads a local file and prints JSON; it never
actuates, signs, or reaches the network.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import click

from scpn_phase_orchestrator.evaluation import (
    DEFAULT_ALPHA,
    DEFAULT_PERMUTATION_SEED,
    DEFAULT_PERMUTATIONS,
    DEFAULT_TARGET_FALSE_ALARM,
    audit_detector,
    seal_detector_audit,
)
from scpn_phase_orchestrator.runtime.cli._app import main


def _load_scores_spec(path: Path) -> dict[str, Any]:
    """Return the parsed scores spec, or raise a ``ClickException``."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise click.ClickException(f"{path} must hold a JSON object")
    return payload


def _require_scores(spec: dict[str, Any], key: str) -> list[float]:
    """Return ``spec[key]`` as a list of finite floats, else raise.

    Strict on purpose: a missing key, a non-list value, an empty list, or a
    non-numeric or non-finite entry is an error, never a silently dropped score.
    """
    value = spec.get(key)
    if not isinstance(value, list):
        raise click.ClickException(f"'{key}' must be a JSON array of numbers")
    if not value:
        raise click.ClickException(f"'{key}' must not be empty")
    scores: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise click.ClickException(
                f"'{key}[{index}]' must be a number, got {item!r}"
            )
        number = float(item)
        if number != number or number in (float("inf"), float("-inf")):
            raise click.ClickException(f"'{key}[{index}]' must be finite, got {item!r}")
        scores.append(number)
    return scores


@main.command(name="audit-detector")
@click.argument(
    "scores_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--target-false-alarm",
    default=DEFAULT_TARGET_FALSE_ALARM,
    type=float,
    help="False-alarm rate the alarm threshold is held at or below (0–1).",
)
@click.option(
    "--n-permutations",
    default=DEFAULT_PERMUTATIONS,
    type=int,
    help="Random relabellings drawn for the permutation p-value.",
)
@click.option(
    "--seed",
    default=DEFAULT_PERMUTATION_SEED,
    type=int,
    help="Seed of the permutation resampling, so the p-value is reproducible.",
)
@click.option(
    "--alpha",
    default=DEFAULT_ALPHA,
    type=float,
    help="Significance level for the convenience beats-chance flag (0–1).",
)
@click.option(
    "--corpus-id",
    default=None,
    help="Corpus provenance identifier; seals the verdict when given with "
    "--captured-at.",
)
@click.option(
    "--captured-at",
    default=None,
    help="Caller-supplied audit timestamp; required to seal the verdict.",
)
@click.option(
    "--sign",
    is_flag=True,
    default=False,
    help="HMAC-sign the sealed record with the key in SPO_AUDIT_KEY; requires "
    "--corpus-id and --captured-at.",
)
@click.option(
    "--output",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write the JSON record to this path in addition to stdout.",
)
def audit_detector_command(
    scores_json: Path,
    target_false_alarm: float,
    n_permutations: int,
    seed: int,
    alpha: float,
    corpus_id: str | None,
    captured_at: str | None,
    sign: bool,
    output: Path | None,
) -> None:
    """Audit a detector's event-vs-null skill from a JSON scores file.

    The file is a JSON object with ``event_scores`` and ``null_scores`` arrays of
    per-segment scores (higher means more evidence of a transition) and an optional
    ``detector_name``. The verdict — matched threshold, achieved false alarm,
    detection rate, and permutation p-value — is printed as JSON. Passing both
    ``--corpus-id`` and ``--captured-at`` seals it into a hash-addressed record.

    Parameters
    ----------
    scores_json : Path
        Path to the JSON scores file.
    target_false_alarm : float
        False-alarm rate the threshold is calibrated to hold at or below.
    n_permutations : int
        Random relabellings drawn for the permutation p-value.
    seed : int
        Seed of the permutation resampling.
    alpha : float
        Significance level for the convenience beats-chance flag.
    corpus_id : str | None
        Corpus provenance identifier; with ``captured_at`` seals the verdict.
    captured_at : str | None
        Caller-supplied audit timestamp; required to seal the verdict.
    sign : bool
        When set, HMAC-sign the sealed record with the ``SPO_AUDIT_KEY`` key.
    output : Path | None
        Optional path to also write the JSON record to.

    Raises
    ------
    ClickException
        If the file is invalid, the scores are malformed, only one of
        ``--corpus-id`` / ``--captured-at`` is given, ``--sign`` is used without
        sealing or without a ``SPO_AUDIT_KEY``, or an audit parameter is out of
        range.
    """
    spec = _load_scores_spec(scores_json)
    event_scores = _require_scores(spec, "event_scores")
    null_scores = _require_scores(spec, "null_scores")
    detector_name = str(spec.get("detector_name", "detector"))
    if (corpus_id is None) != (captured_at is None):
        raise click.ClickException(
            "--corpus-id and --captured-at must be given together to seal a verdict"
        )
    if sign and corpus_id is None:
        raise click.ClickException(
            "--sign requires --corpus-id and --captured-at to seal a verdict first"
        )
    try:
        audit = audit_detector(
            event_scores=event_scores,
            null_scores=null_scores,
            detector_name=detector_name,
            target_false_alarm=target_false_alarm,
            n_permutations=n_permutations,
            seed=seed,
            alpha=alpha,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if corpus_id is not None and captured_at is not None:
        key: str | None = None
        if sign:
            key = os.environ.get("SPO_AUDIT_KEY")
            if not key:
                raise click.ClickException(
                    "--sign requires a non-empty SPO_AUDIT_KEY environment variable"
                )
        record = seal_detector_audit(
            audit, corpus_id=corpus_id, captured_at=captured_at, key=key
        )
        payload: dict[str, object] = record.to_record()
    else:
        payload = audit.to_record()

    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if output is not None:
        output.write_text(rendered + "\n", encoding="utf-8")
    click.echo(rendered)

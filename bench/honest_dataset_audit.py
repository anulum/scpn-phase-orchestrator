# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — reusable honest dataset audit harness

"""Reusable harness for honest detector audits across datasets and domains.

This module generalises the pattern used in ``bench/cap_multichannel_n3_vs_wake.py``
and ``bench/sleep_staging_sleepedf.py``:

1. Load one or more recordings via a domain-specific ``loader``.
2. Extract per-recording labels via a domain-specific ``label_extractor``.
3. Score each recording with every detector in a named registry.
4. Audit each detector at a matched false-alarm rate and seal the verdict.
5. Aggregate per-recording results into a cross-subject comparison JSON.
6. Optionally emit a study markdown page.

The harness is intentionally agnostic to the file format, the number of channels,
and the semantics of the event/null labels. It only requires:

* a ``RecordingSpec`` that identifies a recording and points to its raw files,
* a ``loader`` that turns the spec into a domain-specific object,
* detector callables that map that object to a 1-D score array,
* a ``label_extractor`` that maps that object to a parallel list of labels,
* a pair of canonical labels that define the event and null classes.

Example::

    from bench.honest_dataset_audit import (
        AuditConfig,
        RecordingSpec,
        run_honest_audit,
    )

    def load_edf(spec):
        ...
        return {"eeg": eeg, "fs": fs}

    def labels(obj):
        ...
        return per_epoch_labels

    def delta_envelope(obj):
        ...
        return per_epoch_scores

    manifest = [
        RecordingSpec("n1", {"edf": Path("n1.edf"), "txt": Path("n1.txt")}),
    ]

    aggregate = run_honest_audit(
        manifest=manifest,
        loader=load_edf,
        detectors={"delta_envelope": delta_envelope},
        label_extractor=labels,
        output_dir=Path("out"),
        config=AuditConfig(),
        benchmark="my_dataset",
        corpus="My public corpus",
    )
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.evaluation.auditor import audit_detector
from scpn_phase_orchestrator.evaluation.record import seal_detector_audit

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class AuditConfig:
    """Shared protocol parameters for an honest dataset audit."""

    target_false_alarm: float = 0.10
    n_permutations: int = 10_000
    seed: int = 42
    alpha: float = 0.05
    score_precision: int = 6
    captured_at: str = "2026-07-09T00:00:00Z"


@dataclass(frozen=True)
class RecordingSpec:
    """One recording in a dataset manifest.

    Parameters
    ----------
    recording_id : str
        Stable identifier used in filenames and the aggregate JSON.
    paths : dict[str, Path]
        Domain-specific raw-file paths, e.g. ``{"edf": ..., "txt": ...}``.
    metadata : dict[str, Any] | None
        Optional provenance (condition, age group, etc.) copied into the
        aggregate JSON unchanged.
    """

    recording_id: str
    paths: dict[str, Path]
    metadata: dict[str, Any] | None = None


# Domain-specific callables used by ``run_honest_audit``.
Loader = Callable[[RecordingSpec], Any]
Detector = Callable[[Any], FloatArray]
LabelExtractor = Callable[[Any], list[str]]
RecommendationFn = Callable[[list[dict[str, Any]], dict[str, Any]], dict[str, Any]]


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of ``path``."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _round_scores(scores: FloatArray, precision: int) -> FloatArray:
    """Round scores to ``precision`` decimal places for reproducibility."""
    return np.round(scores, precision)


def run_audit(
    scores: FloatArray,
    labels: Sequence[str],
    detector_name: str,
    corpus_id: str,
    config: AuditConfig,
    *,
    event_label: str = "event",
    null_label: str = "null",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Audit one detector on a single recording and return sealed record + summary.

    Parameters
    ----------
    scores : FloatArray
        Per-epoch detector scores, shape ``(n_epochs,)``.
    labels : sequence of str
        One canonical label per epoch.
    detector_name : str
        Name of the detector being audited.
    corpus_id : str
        Corpus identifier recorded in the sealed audit.
    config : AuditConfig
        Protocol parameters.
    event_label : str
        Label that identifies the event class.
    null_label : str
        Label that identifies the null class.

    Returns
    -------
    audit_record : dict
        Content-addressed sealed audit record.
    summary : dict
        Human-readable summary fragment (also used by the aggregate JSON).
    """
    event_scores = [float(scores[i]) for i, s in enumerate(labels) if s == event_label]
    null_scores = [float(scores[i]) for i, s in enumerate(labels) if s == null_label]

    if not event_scores:
        raise ValueError(f"no event epochs found for {detector_name}")
    if not null_scores:
        raise ValueError(f"no null epochs found for {detector_name}")

    audit = audit_detector(
        event_scores=event_scores,
        null_scores=null_scores,
        detector_name=detector_name,
        target_false_alarm=config.target_false_alarm,
        n_permutations=config.n_permutations,
        seed=config.seed,
        alpha=config.alpha,
    )
    sealed = seal_detector_audit(
        audit,
        corpus_id=corpus_id,
        captured_at=config.captured_at,
    )
    audit_record = sealed.to_record()
    summary: dict[str, Any] = {
        "detector_name": audit.detector_name,
        "n_events": audit.n_events,
        "n_nulls": audit.n_nulls,
        "score_mean_events": round(float(np.mean(event_scores)), 6),
        "score_mean_nulls": round(float(np.mean(null_scores)), 6),
        "target_false_alarm": audit.target_false_alarm,
        "matched_threshold": audit.matched_threshold,
        "achieved_false_alarm": audit.achieved_false_alarm,
        "detection_rate": audit.detection_rate,
        "n_events_alarmed": audit.n_events_alarmed,
        "p_value": audit.p_value,
        "beats_chance": audit.beats_chance,
        "permutation_seed": config.seed,
        "n_permutations": config.n_permutations,
        "corpus_id": corpus_id,
        "captured_at": config.captured_at,
        "audit_content_hash": audit_record["content_hash"],
    }
    return audit_record, summary


def write_audit_files(
    output_dir: Path,
    prefix: str,
    detector_name: str,
    audit_record: dict[str, Any],
    summary: dict[str, Any],
    *,
    audit_suffix: str = "audit",
    summary_suffix: str = "summary",
) -> None:
    """Write a sealed audit record and its summary to ``output_dir``.

    Filenames are ``{prefix}_{detector_name}_{audit_suffix}.json`` and
    ``{prefix}_{detector_name}_{summary_suffix}.json``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_detector_name = detector_name.replace(" ", "_").replace("-", "_")
    audit_path = output_dir / f"{prefix}_{safe_detector_name}_{audit_suffix}.json"
    summary_path = output_dir / f"{prefix}_{safe_detector_name}_{summary_suffix}.json"
    audit_path.write_text(json.dumps(audit_record, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _detector_stats(
    records: list[dict[str, Any]],
    detector_name: str,
) -> dict[str, Any]:
    """Compute cross-recording statistics for one detector."""
    drs = [r["detectors"][detector_name]["detection_rate"] for r in records]
    fas = [r["detectors"][detector_name]["achieved_false_alarm"] for r in records]
    pvals = [r["detectors"][detector_name]["p_value"] for r in records]
    beats = [r["detectors"][detector_name]["beats_chance"] for r in records]
    return {
        "mean_detection_rate": round(float(np.mean(drs)), 6),
        "std_detection_rate": round(float(np.std(drs)), 6),
        "mean_achieved_false_alarm": round(float(np.mean(fas)), 6),
        "geometric_mean_p_value": round(
            float(np.exp(np.mean(np.log(np.maximum(pvals, 1e-300))))), 6
        ),
        "fraction_beats_chance": round(float(np.mean(beats)), 6),
    }


def default_recommendation(
    records: list[dict[str, Any]],
    stats_by_detector: dict[str, Any],
) -> dict[str, Any]:
    """Return a data-driven recommendation based on mean detection rate.

    The best detector by mean detection rate is preferred. If the best detector
    does not beat chance on any recording, the recommendation is to stop
    refining.
    """
    best_detector = max(
        stats_by_detector,
        key=lambda name: stats_by_detector[name]["mean_detection_rate"],
    )
    best_stats = stats_by_detector[best_detector]
    any_beats = any(r["detectors"][best_detector]["beats_chance"] for r in records)

    if not any_beats:
        return {
            "refine": False,
            "preferred_variant": best_detector,
            "rationale": (
                f"{best_detector} has the highest mean detection rate on the panel, "
                "but it does not beat chance on any recording; further refinement is "
                "not supported by the data."
            ),
        }

    return {
        "refine": True,
        "preferred_variant": best_detector,
        "rationale": (
            f"{best_detector} has the highest mean detection rate on the panel "
            f"({best_stats['mean_detection_rate']:.3f}) and is the preferred "
            "variant for further refinement."
        ),
    }


def compute_aggregate(
    records: list[dict[str, Any]],
    benchmark: str,
    corpus: str,
    detector_names: Sequence[str],
    *,
    recommendation_fn: RecommendationFn | None = None,
) -> dict[str, Any]:
    """Compute cross-recording aggregate statistics and recommendation.

    Parameters
    ----------
    records : list[dict]
        Per-recording fragments produced by ``run_honest_audit``.
    benchmark : str
        Benchmark identifier.
    corpus : str
        Corpus name.
    detector_names : sequence of str
        Detectors to include in the aggregate.
    recommendation_fn : callable, optional
        Function ``(records, stats_by_detector) -> recommendation_dict``. Uses
        :func:`default_recommendation` when omitted.

    Returns
    -------
    dict
        Aggregate comparison JSON.
    """
    stats_by_detector = {
        name: _detector_stats(records, name) for name in detector_names
    }
    recommendation = (recommendation_fn or default_recommendation)(
        records, stats_by_detector
    )

    aggregate: dict[str, Any] = {
        "benchmark": benchmark,
        "corpus": corpus,
        "target_false_alarm": records[0]["detectors"][detector_names[0]][
            "target_false_alarm"
        ],
        "n_recordings": len(records),
        "recording_ids": [r["recording_id"] for r in records],
        "per_recording": records,
    }
    aggregate.update(stats_by_detector)
    aggregate["recommendation"] = recommendation
    return aggregate


def write_aggregate(output_dir: Path, aggregate: dict[str, Any]) -> None:
    """Write ``aggregate`` to ``output_dir/{benchmark}.json``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{aggregate['benchmark']}.json"
    path.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")


def run_honest_audit(
    manifest: Sequence[RecordingSpec],
    loader: Loader,
    detectors: dict[str, Detector],
    label_extractor: LabelExtractor,
    output_dir: Path,
    config: AuditConfig,
    *,
    benchmark: str = "honest_dataset_audit",
    corpus: str = "",
    event_label: str = "event",
    null_label: str = "null",
    corpus_id_fn: Callable[[RecordingSpec], str] = lambda spec: spec.recording_id,
    filename_prefix: str | None = None,
    recommendation_fn: RecommendationFn | None = None,
) -> dict[str, Any]:
    """Run a complete honest audit across a dataset manifest.

    Parameters
    ----------
    manifest : sequence of RecordingSpec
        Recordings to audit.
    loader : callable
        ``loader(spec) -> domain_object``.
    detectors : dict[str, callable]
        ``detector_name -> detector(domain_object) -> scores``.
    label_extractor : callable
        ``label_extractor(domain_object) -> list[str]``.
    output_dir : Path
        Root directory for sealed audit output.
    config : AuditConfig
        Protocol parameters.
    benchmark : str
        Benchmark identifier used in the aggregate filename.
    corpus : str
        Corpus name recorded in the aggregate JSON.
    event_label : str
        Label that identifies the event class.
    null_label : str
        Label that identifies the null class.
    corpus_id_fn : callable
        Function ``spec -> corpus_id`` used for sealed records.
    filename_prefix : str | None
        Prefix for per-recording output files. Defaults to ``recording_id``.
    recommendation_fn : callable, optional
        Custom recommendation logic.

    Returns
    -------
    dict
        Aggregate comparison JSON.
    """
    if not detectors:
        raise ValueError("at least one detector is required")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for spec in manifest:
        domain_object = loader(spec)
        labels = label_extractor(domain_object)
        corpus_id = corpus_id_fn(spec)

        detector_summaries: dict[str, Any] = {}
        for detector_name, detector_fn in detectors.items():
            scores = _round_scores(
                np.asarray(detector_fn(domain_object), dtype=np.float64),
                config.score_precision,
            )
            audit_record, summary = run_audit(
                scores=scores,
                labels=labels,
                detector_name=detector_name,
                corpus_id=corpus_id,
                config=config,
                event_label=event_label,
                null_label=null_label,
            )
            detector_summaries[detector_name] = summary

            rec_out = output_dir / spec.recording_id
            rec_out.mkdir(parents=True, exist_ok=True)
            prefix = filename_prefix or spec.recording_id
            write_audit_files(
                output_dir=rec_out,
                prefix=prefix,
                detector_name=detector_name,
                audit_record=audit_record,
                summary=summary,
            )

        records.append(
            {
                "recording_id": spec.recording_id,
                "metadata": spec.metadata,
                "detectors": detector_summaries,
            }
        )

    aggregate = compute_aggregate(
        records=records,
        benchmark=benchmark,
        corpus=corpus,
        detector_names=list(detectors.keys()),
        recommendation_fn=recommendation_fn,
    )
    write_aggregate(output_dir, aggregate)
    return aggregate

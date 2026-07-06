# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — bulk DNB early-warning capstone (GSE2565 phosgene lung)

"""Bulk DNB early-warning capstone: a selection-controlled surrogate test on GSE2565.

The molecular companion to the single-cell Mojtahedi capstone
(:mod:`bench.early_warning_dnb`), and the case the dynamical-network-biomarker method
was first demonstrated on: Chen, Liu, Liu & Aihara 2012 (*Sci Rep* 2:342) read a rising
DNB composite index on GSE2565, a phosgene acute-lung-injury time course, peaking at the
~8 h critical transition. Here that benchmark is run through the programme's lens.

The catch a bulk DNB analysis must confront is **selection freedom**. The composite
index (:func:`bench.dnb_detector.dnb_index`) is evaluated on a *module* of genes, and
the module is *chosen to peak at the transition*
(:func:`bench.dnb_detector.select_dnb_module`). So a rise-to-the-transition is partly
guaranteed by construction: given enough genes and a target timepoint, some module will
always appear to rise. The only honest significance test re-runs the *entire* module
selection on each surrogate. This capstone does exactly that: the null shuffles the
timepoint labels across the arm's samples and **re-selects the module from scratch** on
the shuffled data, so a surrogate has the same selection freedom as the real analysis.
The observed rise is significant only if it beats surrogates that were *also* allowed to
cherry-pick a rising module on scrambled time.

Corpus and design
-----------------
GSE2565 (raw Affymetrix intensities, **citation-only**, never redistributed here): 104
mouse-lung samples, an air-control (``Air``) and a phosgene-exposed (``CG``) arm, each
at eight times (0.5–72 h) with ~6 replicates, plus an untreated baseline. Sample group
and timepoint are recovered from the GEO SOFT family file's per-sample description. The
rising limb up to the ~8 h transition (0.5, 1, 4, 8 h) is the analysis window; on the
arm's most-variable probes the DNB module is selected at 8 h, its composite-index
trajectory is reduced to a least-squares rising slope, and that slope is ranked against
``N`` selection-controlled surrogates for a one-sided p-value and a matched-false-alarm
alarm. The exposed (``CG``) arm is the transition; ``Air`` is the control context.
Only the derived, hash-sealed artefact is written; no raw data is.

An honest limitation belongs on this proof: there is **one exposed arm**, so this is a
single-transition surrogate test, not a corpus — it bounds demonstrated skill on this
record, not a population claim. Its value is the *selection-controlled* null, which the
usual "the DNB index rose" reading omits.

References
----------
* Chen, Liu, Liu & Aihara 2012, *Sci Rep* 2:342 — the dynamical network biomarker,
  demonstrated on GSE2565.
* Sciuto, Phillips, Orzolek, Hensley, Chang & Nadadur 2005, *Chem Res Toxicol* — the
  GSE2565 phosgene acute-lung-injury expression time course.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from bench.dnb_detector import dnb_index, dnb_regression_slope, select_dnb_module
from bench.early_warning_domain import (
    DEFAULT_TARGET_FALSE_ALARM,
    calibrate_score_threshold,
)
from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash

if TYPE_CHECKING:  # pragma: no cover - import only for static typing
    from collections.abc import Sequence

FloatArray = NDArray[np.float64]

#: Rising-limb timepoints (hours) up to the ~8 h critical transition.
RISING_LIMB_HOURS: tuple[str, ...] = ("0.5", "1", "4", "8")
#: Index of the transition timepoint (8 h) within the rising limb.
TRANSITION_INDEX = 3
#: Number of most-variable probes the module is selected from.
N_TOP_PROBES = 150
#: Fraction of those probes taken as rising-variability candidates.
CANDIDATE_FRACTION = 0.1
#: Minimum candidate-pool size for the module search.
MIN_MODULE = 5
#: Selection-controlled surrogates drawn for the null distribution.
N_SURROGATES = 1000
#: Target false-alarm rate the slope threshold is held at or below.
TARGET_FALSE_ALARM = DEFAULT_TARGET_FALSE_ALARM
#: Seed of the surrogate resampling, so the sealed artefact is byte-reproducible.
DEFAULT_SEED = 0
#: The exposed (transition) and control arm labels in the SOFT metadata.
EXPOSED_GROUP = "CG"
CONTROL_GROUP = "Air"

__all__ = [
    "CANDIDATE_FRACTION",
    "CONTROL_GROUP",
    "EXPOSED_GROUP",
    "MIN_MODULE",
    "N_SURROGATES",
    "N_TOP_PROBES",
    "RISING_LIMB_HOURS",
    "TARGET_FALSE_ALARM",
    "TRANSITION_INDEX",
    "ArmResult",
    "arm_rising_frames",
    "dnb_rise_score",
    "evaluate_arm",
    "evaluate_gse2565",
    "load_expression_matrix",
    "main",
    "parse_sample_groups",
    "surrogate_rank_pvalue",
    "surrogate_rise_scores",
]


# --------------------------------------------------------------------------- #
# Ingestion (thin — touches the citation-only corpus, tested on synthetic text) #
# --------------------------------------------------------------------------- #


def _open_maybe_gzip(path: str | Path) -> IO[str]:
    """Open a text file, transparently handling a ``.gz`` suffix."""
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", errors="ignore")
    return path.open("rt", encoding="utf-8", errors="ignore")


def parse_sample_groups(path: str | Path) -> dict[str, tuple[str, str]]:
    """Return each sample's ``(group, timepoint)`` from a GEO SOFT family file.

    Each ``^SAMPLE`` block carries ordered ``!Sample_description`` lines whose second is
    the exposure group (``Air`` or ``CG``) and whose fourth is the timepoint in hours;
    this recovers the group and timepoint the series matrix omits.

    Parameters
    ----------
    path : str or Path
        Path to the SOFT family file (``.soft`` or ``.soft.gz``).

    Returns
    -------
    dict[str, tuple[str, str]]
        Sample accession → ``(group, timepoint)``.
    """
    groups: dict[str, tuple[str, str]] = {}
    current: str | None = None
    descriptions: list[str] = []
    with _open_maybe_gzip(path) as handle:
        for line in handle:
            if line.startswith("^SAMPLE"):
                _record_sample(groups, current, descriptions)
                current = line.split("=", 1)[1].strip()
                descriptions = []
            elif line.startswith("!Sample_description ="):
                descriptions.append(line.split("=", 1)[1].strip())
        _record_sample(groups, current, descriptions)
    return groups


def _record_sample(
    groups: dict[str, tuple[str, str]],
    accession: str | None,
    descriptions: list[str],
) -> None:
    """Store one sample's group and timepoint when its block is complete."""
    if accession is not None and len(descriptions) >= 4:
        groups[accession] = (descriptions[1], descriptions[3])


def load_expression_matrix(path: str | Path) -> tuple[list[str], list[str], FloatArray]:
    """Return the samples, probe ids, and log2 matrix from a GEO series matrix.

    Parameters
    ----------
    path : str or Path
        Path to the series-matrix file (``.txt`` or ``.txt.gz``).

    Returns
    -------
    tuple[list[str], list[str], FloatArray]
        The sample accessions in column order, the probe identifiers in row order, and
        the ``log2(intensity + 1)`` matrix, shape ``(probes, samples)``.

    Raises
    ------
    ValueError
        If the file carries no ``!series_matrix_table_begin`` block.
    """
    with _open_maybe_gzip(path) as handle:
        lines = handle.readlines()
    begin = _table_begin(lines)
    header = [
        cell.strip().strip('"') for cell in lines[begin + 1].rstrip("\n").split("\t")
    ]
    accessions = header[1:]
    probe_ids: list[str] = []
    rows: list[list[float]] = []
    for line in lines[begin + 2 :]:
        if line.startswith("!series_matrix_table_end"):
            break
        cells = line.rstrip("\n").split("\t")
        probe_ids.append(cells[0].strip().strip('"'))
        rows.append([float(cell) for cell in cells[1:]])
    matrix = np.log2(np.asarray(rows, dtype=np.float64) + 1.0)
    return accessions, probe_ids, matrix


def _table_begin(lines: Sequence[str]) -> int:
    """Return the index of the series-matrix table header, else raise ``ValueError``."""
    for index, line in enumerate(lines):
        if line.startswith("!series_matrix_table_begin"):
            return index
    raise ValueError("no !series_matrix_table_begin block in the series matrix")


def arm_rising_frames(
    matrix: FloatArray,
    accessions: Sequence[str],
    sample_groups: dict[str, tuple[str, str]],
    *,
    group: str,
    timepoints: Sequence[str] = RISING_LIMB_HOURS,
    n_top_probes: int = N_TOP_PROBES,
) -> tuple[list[FloatArray], list[int]]:
    """Return one arm's rising-limb frames on its most-variable probes.

    Parameters
    ----------
    matrix : FloatArray
        The log2 expression matrix, shape ``(probes, samples)``.
    accessions : sequence of str
        The sample accessions in column order.
    sample_groups : dict[str, tuple[str, str]]
        Sample accession → ``(group, timepoint)``.
    group : str
        The arm to build (``CG`` exposed or ``Air`` control).
    timepoints : sequence of str
        The rising-limb timepoints, in order.
    n_top_probes : int
        Number of most-variable probes (across the arm) to keep.

    Returns
    -------
    tuple[list[FloatArray], list[int]]
        One ``(samples, probes)`` frame per timepoint on the top probes, and the kept
        probe row indices (for provenance).

    Raises
    ------
    ValueError
        If any timepoint has no samples for the arm.
    """
    frames: list[FloatArray] = []
    for timepoint in timepoints:
        columns = [
            index
            for index, accession in enumerate(accessions)
            if sample_groups.get(accession) == (group, timepoint)
        ]
        if not columns:
            raise ValueError(f"no {group} samples at timepoint {timepoint}")
        frames.append(np.ascontiguousarray(matrix[:, columns].T))
    pooled = np.vstack(frames)
    top = np.argsort(pooled.var(axis=0))[::-1][:n_top_probes]
    top_probes = [int(probe) for probe in top]
    return [frame[:, top_probes] for frame in frames], top_probes


# --------------------------------------------------------------------------- #
# Selection-controlled surrogate test (pure — exercised on synthetic frames)   #
# --------------------------------------------------------------------------- #


def dnb_rise_score(
    frames: Sequence[FloatArray],
    *,
    transition_index: int = TRANSITION_INDEX,
    candidate_fraction: float = CANDIDATE_FRACTION,
    min_module: int = MIN_MODULE,
) -> tuple[float, tuple[int, ...]]:
    """Return the rising-limb DNB slope and the module selected at the transition.

    Selects the DNB module at the transition timepoint, evaluates the composite index at
    every rising-limb timepoint, and reduces that trajectory to its least-squares slope.

    Parameters
    ----------
    frames : sequence of FloatArray
        The rising-limb ``(samples, probes)`` frames, in time order.
    transition_index : int
        Index of the transition timepoint the module is selected at.
    candidate_fraction, min_module : float, int
        The module-search parameters.

    Returns
    -------
    tuple[float, tuple[int, ...]]
        The rising slope and the selected module's probe indices.
    """
    module = select_dnb_module(
        frames,
        transition_index=transition_index,
        candidate_fraction=candidate_fraction,
        min_module=min_module,
    )
    trajectory = [dnb_index(frame, list(module)) for frame in frames]
    return dnb_regression_slope(trajectory), module


def surrogate_rise_scores(
    frames: Sequence[FloatArray],
    *,
    n_surrogates: int = N_SURROGATES,
    seed: int = DEFAULT_SEED,
    transition_index: int = TRANSITION_INDEX,
    candidate_fraction: float = CANDIDATE_FRACTION,
    min_module: int = MIN_MODULE,
) -> list[float]:
    """Return selection-controlled surrogate rising slopes.

    Each surrogate shuffles the timepoint labels across the arm's pooled samples and
    **re-selects the module from scratch** on the shuffled frames, so it enjoys the same
    selection freedom as the real analysis. A surrogate whose module selection fails on
    the degenerate shuffled data (no gene rises) contributes a zero slope — no trend.

    Parameters
    ----------
    frames : sequence of FloatArray
        The rising-limb ``(samples, probes)`` frames, in time order.
    n_surrogates : int
        Number of surrogate resamples.
    seed : int
        Seed of the resampling, so the null is reproducible.
    transition_index, candidate_fraction, min_module :
        The module-search parameters, matched to the real analysis.

    Returns
    -------
    list[float]
        The ``n_surrogates`` surrogate rising slopes.
    """
    pooled = np.vstack(list(frames))
    labels = np.concatenate(
        [np.full(frame.shape[0], index) for index, frame in enumerate(frames)]
    )
    n_timepoints = len(frames)
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    for _ in range(n_surrogates):
        permuted = rng.permutation(labels)
        shuffled = [pooled[permuted == index] for index in range(n_timepoints)]
        try:
            slope, _ = dnb_rise_score(
                shuffled,
                transition_index=transition_index,
                candidate_fraction=candidate_fraction,
                min_module=min_module,
            )
        except ValueError:
            slope = 0.0
        scores.append(slope)
    return scores


def surrogate_rank_pvalue(observed: float, surrogates: Sequence[float]) -> float:
    """Return the one-sided surrogate-rank p-value with the add-one correction.

    Parameters
    ----------
    observed : float
        The observed rising slope.
    surrogates : sequence of float
        The selection-controlled surrogate slopes.

    Returns
    -------
    float
        ``(1 + #{surrogate >= observed}) / (1 + n_surrogates)`` — never zero.

    Raises
    ------
    ValueError
        If ``surrogates`` is empty.
    """
    if not surrogates:
        raise ValueError("surrogates must not be empty")
    reached = int(sum(1 for score in surrogates if score >= observed))
    return (1 + reached) / (1 + len(surrogates))


@dataclass(frozen=True)
class ArmResult:
    """One arm's selection-controlled DNB surrogate test.

    Attributes
    ----------
    group : str
        The arm label (``CG`` exposed or ``Air`` control).
    observed_slope : float
        The observed rising-limb DNB slope.
    module_probes : tuple[str, ...]
        The probe identifiers of the genes in the selected module.
    surrogate_mean : float
        Mean of the selection-controlled surrogate slopes.
    surrogate_p90 : float
        The 90th percentile of the surrogate slopes.
    p_value : float
        The one-sided surrogate-rank p-value.
    matched_false_alarm_threshold : float
        The slope threshold holding the surrogate false alarm at the target.
    alarmed : bool
        Whether the observed slope met the matched-false-alarm threshold.
    """

    group: str
    observed_slope: float
    module_probes: tuple[str, ...]
    surrogate_mean: float
    surrogate_p90: float
    p_value: float
    matched_false_alarm_threshold: float
    alarmed: bool

    def to_record(self) -> dict[str, object]:
        """Return a JSON-safe mapping of the arm result."""
        return {
            "group": self.group,
            "observed_slope": self.observed_slope,
            "module_probes": list(self.module_probes),
            "surrogate_mean": self.surrogate_mean,
            "surrogate_p90": self.surrogate_p90,
            "p_value": self.p_value,
            "matched_false_alarm_threshold": self.matched_false_alarm_threshold,
            "alarmed": self.alarmed,
        }


def evaluate_arm(
    frames: Sequence[FloatArray],
    probe_ids: Sequence[str],
    top_probes: Sequence[int],
    *,
    group: str,
    n_surrogates: int = N_SURROGATES,
    seed: int = DEFAULT_SEED,
    target_fa: float = TARGET_FALSE_ALARM,
    transition_index: int = TRANSITION_INDEX,
    candidate_fraction: float = CANDIDATE_FRACTION,
    min_module: int = MIN_MODULE,
) -> ArmResult:
    """Score one arm through the selection-controlled surrogate test.

    Parameters
    ----------
    frames : sequence of FloatArray
        The arm's rising-limb frames on the top probes.
    probe_ids : sequence of str
        All probe identifiers, indexed by matrix row.
    top_probes : sequence of int
        The matrix row indices of the frames' probes, so a selected module index maps
        back through the top-probe subset to a probe identifier.
    group : str
        The arm label.
    n_surrogates, seed, target_fa, transition_index, candidate_fraction, min_module :
        The test parameters.

    Returns
    -------
    ArmResult
        The observed slope, surrogate summary, p-value, and matched-false-alarm alarm.
    """
    observed, module = dnb_rise_score(
        frames,
        transition_index=transition_index,
        candidate_fraction=candidate_fraction,
        min_module=min_module,
    )
    surrogates = surrogate_rise_scores(
        frames,
        n_surrogates=n_surrogates,
        seed=seed,
        transition_index=transition_index,
        candidate_fraction=candidate_fraction,
        min_module=min_module,
    )
    threshold = calibrate_score_threshold(surrogates, target_fa=target_fa)
    module_probe_ids = tuple(str(probe_ids[top_probes[index]]) for index in module)
    return ArmResult(
        group=group,
        observed_slope=float(observed),
        module_probes=module_probe_ids,
        surrogate_mean=float(np.mean(surrogates)),
        surrogate_p90=float(np.quantile(surrogates, 0.9)),
        p_value=surrogate_rank_pvalue(observed, surrogates),
        matched_false_alarm_threshold=threshold,
        alarmed=bool(observed >= threshold),
    )


# --------------------------------------------------------------------------- #
# Orchestration (I/O shell over the tested logic)                             #
# --------------------------------------------------------------------------- #


def evaluate_gse2565(
    matrix_path: str | Path,
    soft_path: str | Path,
    *,
    n_surrogates: int = N_SURROGATES,
    seed: int = DEFAULT_SEED,
    n_top_probes: int = N_TOP_PROBES,
) -> dict[str, object]:
    """Read GSE2565 and return the hash-sealed selection-controlled result payload.

    Parameters
    ----------
    matrix_path : str or Path
        Path to the GEO series matrix (citation-only, not redistributed).
    soft_path : str or Path
        Path to the GEO SOFT family file (for the group/timepoint map).
    n_surrogates : int
        Selection-controlled surrogates per arm.
    seed : int
        Seed of the surrogate resampling.
    n_top_probes : int
        Number of most-variable probes the module is selected from.

    Returns
    -------
    dict[str, object]
        The JSON-safe, hash-sealed result payload (with a ``content_hash`` field).
    """
    accessions, probe_ids, matrix = load_expression_matrix(matrix_path)
    sample_groups = parse_sample_groups(soft_path)
    arms: list[dict[str, object]] = []
    for group in (EXPOSED_GROUP, CONTROL_GROUP):
        frames, top = arm_rising_frames(
            matrix,
            accessions,
            sample_groups,
            group=group,
            n_top_probes=n_top_probes,
        )
        result = evaluate_arm(
            frames,
            probe_ids,
            top,
            group=group,
            n_surrogates=n_surrogates,
            seed=seed,
        )
        arms.append(result.to_record())
    exposed = arms[0]
    payload: dict[str, object] = {
        "benchmark": "early_warning_dnb_gse2565",
        "corpus": (
            "GSE2565 phosgene acute lung injury (Chen/Aihara 2012 DNB benchmark)"
        ),
        "modality": "bulk dynamical network biomarker (selection-controlled surrogate)",
        "rising_limb_hours": list(RISING_LIMB_HOURS),
        "n_top_probes": n_top_probes,
        "n_surrogates_per_arm": n_surrogates,
        "seed": seed,
        "target_false_alarm": TARGET_FALSE_ALARM,
        "arms": arms,
        "verdict": _gse2565_verdict(exposed),
        "note": (
            "single exposed arm — a selection-controlled surrogate test, not a corpus; "
            "bounds demonstrated skill on this record, not a population claim"
        ),
    }
    payload["content_hash"] = canonical_record_hash(payload)
    return payload


def _gse2565_verdict(exposed: dict[str, object]) -> str:
    """Return a one-sentence honest reading of the exposed arm's result."""
    p_value = float(exposed["p_value"])  # type: ignore[arg-type]
    reach = "beats chance" if p_value <= 0.05 else "does not reach significance"
    return (
        "The phosgene-exposed arm's DNB rise, tested against selection-controlled "
        f"surrogates that re-select the module on shuffled time, {reach} "
        f"(surrogate-rank p={p_value:.3f}): the apparent module rise is largely a "
        "selection artefact once the null is allowed the same freedom."
    )


def main(data_dir: str | Path, output_dir: str | Path) -> None:
    """Run the bulk DNB capstone over GSE2565 and write the sealed artefact.

    Reads the citation-only series matrix and SOFT file from ``data_dir`` and writes one
    hash-sealed aggregate JSON to ``output_dir``. Only the derived artefact is written.

    Parameters
    ----------
    data_dir : str or Path
        Directory holding ``GSE2565_series_matrix.txt.gz`` and
        ``GSE2565_family.soft.gz``.
    output_dir : str or Path
        Directory the sealed derived artefact is written to.
    """
    data = Path(data_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = evaluate_gse2565(
        data / "GSE2565_series_matrix.txt.gz",
        data / "GSE2565_family.soft.gz",
    )
    (out / "early_warning_dnb_gse2565.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(payload["verdict"])
    print(f"content hash {payload['content_hash']}")
    print(f"results written to {out}")


if __name__ == "__main__":  # pragma: no cover - CLI shell over the tested logic
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_dir", help="directory holding the citation-only GSE2565 files"
    )
    parser.add_argument("output_dir", help="directory for the sealed derived output")
    arguments = parser.parse_args()
    main(arguments.data_dir, arguments.output_dir)

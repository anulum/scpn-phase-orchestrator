# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — cross-domain detector meta-analysis

"""Cross-domain meta-analysis of committed detector-evidence aggregates.

This module reads aggregate JSONs produced by the honest-audit and
early-warning lead-time pipelines, normalises each detector's performance
into a common table, ranks detectors per domain and overall, and emits a
ranked refinement backlog that can be written to ``docs/studies/``.

Supported schema families:

* **Honest-audit aggregates** — top-level detector summary objects that
  contain ``mean_detection_rate``, ``geometric_mean_p_value`` and
  ``fraction_beats_chance`` (e.g. ``cap_multichannel_aggregate.json``,
  ``synthetic_honest_audit_demo.json``).
* **Early-warning lead-time results** — top-level
  ``permutation_significance`` mapping detectors to ``observed_led``,
  ``n_transitions`` and ``p_value`` (e.g. EEG/cardiac/climate/grid
  ``early_warning_leadtime_*_results.json``).

Other JSON artefacts are discovered but reported as unsupported rather than
raising an error, so the tool stays safe to run as the corpus evolves.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

KNOWN_AGGREGATE_NAMES: frozenset[str] = frozenset(
    {
        "cap_multichannel_aggregate.json",
        "cap_kuramoto_variants.json",
        "sleepedf_kuramoto_variants.json",
        "regime_adaptive_ensemble.json",
        "synthetic_honest_audit_demo.json",
        "csd_variant_synthetic_results.json",
        "early_warning_leadtime_eeg_results.json",
        "early_warning_leadtime_eeg_multiscale_results.json",
        "early_warning_leadtime_cardiac_results.json",
        "early_warning_leadtime_cardiac_multiscale_results.json",
        "early_warning_leadtime_climate_results.json",
        "early_warning_leadtime_climate_multiscale_results.json",
        "early_warning_leadtime_grid_results.json",
        "early_warning_leadtime_grid_multiscale_results.json",
    }
)
AGGREGATE_SUFFIXES: tuple[str, ...] = (
    "_aggregate.json",
    "_results.json",
    "_demo.json",
)


@dataclass(frozen=True)
class EvidenceRow:
    """Normalised detector-performance row used for ranking."""

    domain: str
    detector: str
    detection_rate: float
    p_value: float
    beats_chance: bool
    source_file: str


def discover_aggregate_jsons(root: Path) -> list[Path]:
    """Return every aggregate JSON found directly under ``root/*``.

    A file is treated as an aggregate if it is a direct child of a domain
    directory and either matches a known committed-aggregate filename or ends
    with one of the recognised aggregate suffixes.

    Parameters
    ----------
    root : Path
        Directory whose immediate subdirectories are domain folders; each is
        scanned one level deep for aggregate JSON files.

    Returns
    -------
    list of Path
        Sorted paths of the discovered aggregate JSONs. Empty when ``root`` is
        not a directory or holds no matching files.
    """
    paths: list[Path] = []
    if not root.is_dir():
        return paths
    for domain_dir in sorted(root.iterdir()):
        if not domain_dir.is_dir():
            continue
        for candidate in domain_dir.iterdir():
            if not candidate.is_file() or candidate.suffix != ".json":
                continue
            if candidate.name in KNOWN_AGGREGATE_NAMES or candidate.name.endswith(
                AGGREGATE_SUFFIXES
            ):
                paths.append(candidate)
    return sorted(paths)


def _extract_honest_audit(
    path: Path, data: dict[str, Any], domain: str
) -> list[EvidenceRow]:
    """Extract rows from an honest-audit aggregate schema."""
    rows: list[EvidenceRow] = []
    for key, value in data.items():
        if key in {"per_recording", "recommendation"}:
            continue
        if not isinstance(value, dict) or "mean_detection_rate" not in value:
            continue
        p_value = float(value.get("geometric_mean_p_value", 1.0))
        beats_chance = value.get("fraction_beats_chance")
        if beats_chance is None:
            beats_chance = p_value < 0.05
        rows.append(
            EvidenceRow(
                domain=domain,
                detector=key,
                detection_rate=float(value["mean_detection_rate"]),
                p_value=p_value,
                beats_chance=bool(beats_chance),
                source_file=path.name,
            )
        )
    return rows


def _extract_leadtime(
    path: Path, data: dict[str, Any], domain: str
) -> list[EvidenceRow]:
    """Extract rows from an early-warning lead-time results schema."""
    perm = data.get("permutation_significance")
    if not isinstance(perm, dict):
        return []

    # Single-series climate aggregate stores a single flat significance record.
    if "p_value" in perm and "observed_led" in perm:
        observed = int(perm["observed_led"])
        n_transitions = int(perm["n_transitions"])
        rate = observed / n_transitions if n_transitions else 0.0
        p_value = float(perm["p_value"])
        detector = (
            "critical_slowing_down_multiscale"
            if data.get("multiscale")
            else "critical_slowing_down"
        )
        return [
            EvidenceRow(
                domain=domain,
                detector=detector,
                detection_rate=rate,
                p_value=p_value,
                beats_chance=p_value < 0.05,
                source_file=path.name,
            )
        ]

    rows: list[EvidenceRow] = []
    for detector, stats in perm.items():
        if not isinstance(stats, dict):
            continue
        observed = int(stats.get("observed_led", 0))
        n_transitions = int(stats.get("n_transitions", 0))
        rate = observed / n_transitions if n_transitions else 0.0
        p_value = float(stats.get("p_value", 1.0))
        rows.append(
            EvidenceRow(
                domain=domain,
                detector=detector,
                detection_rate=rate,
                p_value=p_value,
                beats_chance=p_value < 0.05,
                source_file=path.name,
            )
        )
    return rows


def extract_evidence(path: Path) -> list[EvidenceRow]:
    """Parse a single aggregate JSON into normalised evidence rows.

    The schema family is inferred from the payload: a ``per_recording`` key
    selects the honest-audit extractor, a ``permutation_significance`` key
    selects the early-warning lead-time extractor.

    Parameters
    ----------
    path : Path
        Aggregate JSON file to parse. Its parent directory name becomes the
        domain label.

    Returns
    -------
    list of EvidenceRow
        Normalised rows, one per detector. Empty when the payload matches no
        supported schema.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    domain = path.parent.name

    if "per_recording" in data:
        honest = _extract_honest_audit(path, data, domain)
        if honest:
            return honest

    if "permutation_significance" in data:
        return _extract_leadtime(path, data, domain)

    return []


def rank_per_domain(
    rows: list[EvidenceRow],
) -> dict[str, list[tuple[int, EvidenceRow]]]:
    """Return detectors ranked within each domain.

    Ranking uses competition ranking (1, 2, 2, 4, …) on detection rate
    descending, with p-value ascending as a tie-breaker and detector name as a
    final stable tie-breaker.

    Parameters
    ----------
    rows : list of EvidenceRow
        Normalised detector rows spanning all domains.

    Returns
    -------
    dict
        Mapping of domain name to a list of ``(rank, EvidenceRow)`` pairs,
        ordered by rank within that domain.
    """
    by_domain: dict[str, list[EvidenceRow]] = {}
    for row in rows:
        by_domain.setdefault(row.domain, []).append(row)

    rankings: dict[str, list[tuple[int, EvidenceRow]]] = {}
    for domain, drows in sorted(by_domain.items()):
        sorted_rows = sorted(
            drows,
            key=lambda r: (-r.detection_rate, r.p_value, r.detector),
        )
        ranked: list[tuple[int, EvidenceRow]] = []
        current_rank = 0
        previous_key: tuple[float, float] | None = None
        for position, row in enumerate(sorted_rows, start=1):
            key = (-row.detection_rate, row.p_value)
            if key != previous_key:
                current_rank = position
                previous_key = key
            ranked.append((current_rank, row))
        rankings[domain] = ranked
    return rankings


def rank_overall(
    rankings: dict[str, list[tuple[int, EvidenceRow]]],
) -> list[dict[str, Any]]:
    """Aggregate per-domain ranks into an overall detector ranking.

    The overall ordering prioritises lower mean rank, then more outright
    domain wins, then broader cross-domain presence, and finally alphabetical
    detector name for stability.

    Parameters
    ----------
    rankings : dict
        Per-domain rankings as returned by :func:`rank_per_domain`.

    Returns
    -------
    list of dict
        One entry per detector with keys ``detector``, ``mean_rank``,
        ``domains_present``, ``wins`` and ``domain_ranks``, sorted best-first.
    """
    detector_ranks: dict[str, list[tuple[str, int]]] = {}
    detector_wins: dict[str, int] = {}

    for domain, ranked in rankings.items():
        for rank, row in ranked:
            detector_ranks.setdefault(row.detector, []).append((domain, rank))
            if rank == 1:
                detector_wins[row.detector] = detector_wins.get(row.detector, 0) + 1

    overall: list[dict[str, Any]] = []
    for detector, ranks in detector_ranks.items():
        overall.append(
            {
                "detector": detector,
                "mean_rank": statistics.mean(rank for _, rank in ranks),
                "domains_present": len(ranks),
                "wins": detector_wins.get(detector, 0),
                "domain_ranks": ranks,
            }
        )

    overall.sort(
        key=lambda x: (
            x["mean_rank"],
            -x["wins"],
            -x["domains_present"],
            x["detector"],
        )
    )
    return overall


def _format_fraction(value: float) -> str:
    """Format a fraction as a percentage with one decimal place."""
    return f"{value * 100.0:.1f}%"


def _format_p(value: float) -> str:
    """Format a p-value in scientific notation with three significant digits."""
    return f"{value:.3e}"


def build_report(
    rows: list[EvidenceRow],
    rankings: dict[str, list[tuple[int, EvidenceRow]]],
    overall: list[dict[str, Any]],
    source_paths: list[Path],
    unsupported_paths: list[Path],
) -> str:
    """Build the Markdown cross-domain detector ranking report.

    Parameters
    ----------
    rows : list of EvidenceRow
        Normalised evidence rows, used to label each domain's source schema.
    rankings : dict
        Per-domain rankings from :func:`rank_per_domain`.
    overall : list of dict
        Overall detector ranking from :func:`rank_overall`.
    source_paths : list of Path
        Every aggregate JSON that was discovered.
    unsupported_paths : list of Path
        Discovered files whose schema was not recognised.

    Returns
    -------
    str
        The full report body rendered as Markdown.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = [
        "# Cross-Domain Detector Meta-Analysis Report",
        "",
        "**Generated:** " + now,
        "",
        "This report is produced automatically from the committed detector-"
        "evidence aggregates under ``examples/real_data/*/``. It normalises "
        "each detector's performance, ranks detectors within every domain, "
        "and derives a ranked backlog of refinement candidates.",
        "",
    ]

    lines.extend(
        [
            "## Data sources",
            "",
            "| Domain | Source file | Schema |",
            "| --- | --- | --- |",
        ]
    )
    schema_names: dict[str, str] = {}
    for row in rows:
        schema_names[row.domain] = (
            "honest-audit aggregate"
            if "_demo" in row.source_file or "_aggregate" in row.source_file
            else "early-warning lead-time"
        )
    for path in source_paths:
        domain = path.parent.name
        schema = schema_names.get(domain, "unsupported / unknown")
        lines.append(f"| {domain} | `{path.name}` | {schema} |")
    lines.append("")

    if unsupported_paths:
        lines.extend(
            [
                "### Unsupported artefacts",
                "",
                "The following files were discovered but do not match a known "
                "aggregate schema, so they were not included in the ranking:",
                "",
            ]
        )
        for path in unsupported_paths:
            lines.append(f"* `{path.parent.name}/{path.name}`")
        lines.append("")

    lines.extend(["## Per-domain rankings", ""])
    for domain in sorted(rankings):
        lines.extend(
            [
                f"### {domain}",
                "",
                "| Rank | Detector | Detection rate | p-value | Beats chance |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for rank, row in rankings[domain]:
            rate_str = _format_fraction(row.detection_rate)
            p_str = _format_p(row.p_value)
            lines.append(
                f"| {rank} | `{row.detector}` | {rate_str} | {p_str} | "
                f"{row.beats_chance} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Cross-domain overall ranking",
            "",
            "| Rank | Detector | Mean rank | Domains present | Wins | Domain wins |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    overall_rank = 0
    previous_sort_key: tuple[float, int, int] | None = None
    for position, entry in enumerate(overall, start=1):
        sort_key = (entry["mean_rank"], -entry["wins"], -entry["domains_present"])
        if sort_key != previous_sort_key:
            overall_rank = position
            previous_sort_key = sort_key
        win_list = (
            ", ".join(
                f"{domain} ({rank})"
                for domain, rank in entry["domain_ranks"]
                if rank == 1
            )
            or "—"
        )
        lines.append(
            f"| {overall_rank} | `{entry['detector']}` | "
            f"{entry['mean_rank']:.2f} | {entry['domains_present']} | "
            f"{entry['wins']} | {win_list} |"
        )
    lines.append("")

    lines.extend(["## Cross-domain patterns", ""])
    multi_domain = [e for e in overall if e["domains_present"] > 1]
    if multi_domain:
        lines.append(
            "Detectors that appear in more than one domain, sorted by mean rank:"
        )
        lines.append("")
        for entry in multi_domain:
            rank_summary = ", ".join(
                f"{domain} ({rank})" for domain, rank in entry["domain_ranks"]
            )
            lines.append(
                f"* **`{entry['detector']}`** — mean rank "
                f"{entry['mean_rank']:.2f}, present in "
                f"{entry['domains_present']} domain(s), wins "
                f"{entry['wins']}: {rank_summary}."
            )
    else:
        lines.append("No detector currently appears in more than one domain.")
    lines.append("")

    lines.extend(["## Ranked refinement backlog", ""])
    backlog = _build_backlog(overall, rankings)
    for idx, item in enumerate(backlog, start=1):
        lines.append(f"{idx}. {item}")
    lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "* Detection rate for early-warning aggregates is approximated by "
            "``observed_led / n_transitions`` — the fraction of transitions "
            "for which the detector produced a statistically meaningful lead.",
            "* A detector is marked as *beating chance* when its reported "
            "p-value is below 0.05; honest-audit aggregates additionally "
            "report the committed ``fraction_beats_chance`` value.",
            "* The CAP multichannel finding that **SNR-weighted Kuramoto did "
            "not improve** over the simple mean-R Kuramoto detector is "
            "carried forward explicitly; further investment in that exact "
            "spatial-R feature is not supported by the current evidence.",
            "",
        ]
    )

    return "\n".join(lines)


def _build_backlog(
    overall: list[dict[str, Any]],
    rankings: dict[str, list[tuple[int, EvidenceRow]]],
) -> list[str]:
    """Return concrete refinement recommendations as Markdown strings."""
    backlog: list[str] = []

    # Generalisable early-warning winner.
    csd_ms = next(
        (e for e in overall if e["detector"] == "critical_slowing_down_multiscale"),
        None,
    )
    csd = next((e for e in overall if e["detector"] == "critical_slowing_down"), None)
    if csd_ms and csd_ms["wins"] >= 2:
        domains = ", ".join(
            domain for domain, rank in csd_ms["domain_ranks"] if rank == 1
        )
        backlog.append(
            f"**Advance the `critical_slowing_down_multiscale` variant** — it wins "
            f"in {csd_ms['wins']} early-warning domain(s) ({domains}) and has the "
            f"best mean rank ({csd_ms['mean_rank']:.2f}) among detectors present "
            f"in multiple domains. Extend it to the remaining real-data domains "
            f"(EEG, cardiac) and compare it head-to-head with the baseline CSD "
            f"on every corpus."
        )
    elif csd and csd["wins"] >= 2:
        domains = ", ".join(domain for domain, rank in csd["domain_ranks"] if rank == 1)
        backlog.append(
            f"**Refine the `critical_slowing_down` family** — it wins in "
            f"{csd['wins']} early-warning domain(s) ({domains}) and has the "
            f"best mean rank ({csd['mean_rank']:.2f}) among detectors present "
            f"in multiple domains. Prioritise adaptive bandwidths, "
            f"surrogate-aware thresholds, and multi-scale aggregation to "
            f"turn sparse wins into robust precursors."
        )

    # Domain-specific stars.
    for detector, label in (
        ("normalized_delta_envelope", "CAP sleep staging"),
        ("lag1_autocorrelation", "synthetic critical-slowing-down corpus"),
    ):
        entry = next((e for e in overall if e["detector"] == detector), None)
        if entry and entry["wins"] >= 1:
            backlog.append(
                f"**Protect and productise `{detector}` for {label}** — "
                f"it dominates its domain (mean rank {entry['mean_rank']:.2f}) "
                f"and should become the default reference detector there."
            )

    # Loser / do-not-invest flag.
    snr = next(
        (e for e in overall if e["detector"] == "snr_weighted_delta_kuramoto"),
        None,
    )
    kuramoto = next(
        (e for e in overall if e["detector"] == "multi_channel_delta_kuramoto"),
        None,
    )
    if snr and kuramoto and snr["mean_rank"] >= kuramoto["mean_rank"]:
        backlog.append(
            "**Deprioritise SNR-weighted Kuramoto** — in the CAP multichannel "
            "panel it does not outperform the unweighted multi-channel "
            "Kuramoto detector. Reallocate effort toward channel-selection or "
            "coupling-structure variants rather than a raw SNR weighting."
        )

    # Fusion / ensemble candidates.
    ensemble = next((e for e in overall if e["detector"] == "ensemble_weighted"), None)
    if ensemble and ensemble["domains_present"] >= 2:
        backlog.append(
            f"**Audit the `ensemble_weighted` fusion rule** — it is present "
            f"in {ensemble['domains_present']} early-warning domains but "
            f"rarely wins (mean rank {ensemble['mean_rank']:.2f}). Investigate "
            f"whether the current weighting is dominated by a single "
            f"indicator and whether a learned combination would help."
        )

    # Data-availability blocker.
    if csd_ms and csd_ms["domains_present"] < 4:
        backlog.append(
            "**Complete the PhysioNet corpora** — the multi-scale CSD variant "
            "is ready to run on CHB-MIT scalp EEG and MIT-BIH AFDB via "
            "`tools/fetch_real_corpora.py` and the `--multiscale` flag on each "
            "early-warning capstone. Download requires a free credentialed "
            "PhysioNet account (PHYSIONET_USER/PHYSIONET_PASSWORD)."
        )

    # Catch-all for remaining detectors.
    mentioned = {
        "critical_slowing_down",
        "critical_slowing_down_multiscale",
        "normalized_delta_envelope",
        "lag1_autocorrelation",
        "snr_weighted_delta_kuramoto",
        "ensemble_weighted",
    }
    for entry in overall:
        if entry["detector"] in mentioned:
            continue
        if entry["domains_present"] <= 1 and entry["wins"] == 0:
            backlog.append(
                f"**Re-evaluate `{entry['detector']}`** — it appears in only "
                f"one domain and never wins; consider whether the feature is "
                f"under-powered or simply unsuited to that data regime."
            )
        elif entry["domains_present"] <= 1 and entry["wins"] >= 1:
            backlog.append(
                f"**Study `{entry['detector']}` transferability** — it wins "
                f"in its single domain but has not been tested elsewhere; run "
                f"the same pipeline on a second domain to judge generality."
            )

    return backlog


def run_analysis(
    root: Path,
) -> tuple[
    list[EvidenceRow],
    dict[str, list[tuple[int, EvidenceRow]]],
    list[dict[str, Any]],
    list[Path],
    list[Path],
]:
    """Discover, extract, rank and return all meta-analysis artefacts.

    Parameters
    ----------
    root : Path
        Directory containing the per-domain aggregate JSON subdirectories.

    Returns
    -------
    tuple
        ``(rows, rankings, overall, source_paths, unsupported)`` — the
        normalised rows, per-domain rankings, overall ranking, every discovered
        source path, and the subset whose schema was unsupported.
    """
    source_paths = discover_aggregate_jsons(root)
    rows: list[EvidenceRow] = []
    unsupported: list[Path] = []
    for path in source_paths:
        extracted = extract_evidence(path)
        if extracted:
            rows.extend(extracted)
        else:
            unsupported.append(path)
    rankings = rank_per_domain(rows)
    overall = rank_overall(rankings)
    return rows, rankings, overall, source_paths, unsupported


def generate_report(root: Path) -> str:
    """Run the full meta-analysis and return the Markdown report body.

    Parameters
    ----------
    root : Path
        Directory containing the per-domain aggregate JSON subdirectories.

    Returns
    -------
    str
        The rendered Markdown report.
    """
    rows, rankings, overall, source_paths, unsupported = run_analysis(root)
    return build_report(rows, rankings, overall, source_paths, unsupported)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for the cross-domain meta-analysis tool.

    Parameters
    ----------
    argv : list of str or None
        Command-line arguments excluding the program name. When ``None``, the
        arguments are read from :data:`sys.argv`.

    Returns
    -------
    int
        Process exit code: ``0`` on success, ``1`` when the root directory does
        not exist.
    """
    parser = argparse.ArgumentParser(
        description="Cross-domain detector ranking and refinement backlog."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("examples/real_data"),
        help="Directory containing domain subdirectories of aggregate JSONs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/studies/detector_ranking_report.md"),
        help="Path where the Markdown report will be written.",
    )
    args = parser.parse_args(argv)

    if not args.root.is_dir():
        print(
            f"ERROR: root directory not found: {args.root}",
            file=__import__("sys").stderr,
        )
        return 1

    report = generate_report(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

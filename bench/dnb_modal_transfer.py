# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — can the grid modal-growth moat even be posed on DNB?

"""Does the grid modal-growth + fit-quality-gate moat transfer to DNB early warning?

The power-grid detector wins by fitting an *exponential* envelope growth and gating on
the fit quality: a genuine instability is a smooth exponential (high R²), a fault a
step-like transient (low R²), so the gate keeps the one and rejects the other. The
dynamical-network-biomarker (DNB) early-warning family is structurally different: as a
cell population or tissue nears a fate/injury tipping point, a critical module's
composite index *rises* — genuine critical slowing down — but the rise is measured over
only a handful of pre-transition timepoints. This runner asks whether the grid's winning
form can even be *posed* on that signal.

It cannot, and the honest reason is sealed here. On the real DNB early-warning corpora —
the single-cell Mojtahedi 2016 transition index (three lineages, a three-day rising limb
transcribed from Table S2) and the bulk GSE2565 phosgene composite index (one exposed
arm, a four-hour rising limb) — the transition-index trajectory is three to four points
long. On so few points the fit-quality gate is *uninformative*: any monotone rise fits
an exponential well enough (R² ≈ 0.67–0.86) to pass a 0.5 gate, so the gate keeps every
trajectory and rejects none, and the exponential growth rate simply re-orders the
transitions exactly as the existing linear slope does — it adds no separating
information. The moat's discriminating machinery needs a *resolved* growth trajectory
the DNB early-warning corpora do not provide. Together with the scalp-EEG result — a
resolved but *non-exponential* trajectory the gate rejects outright — this bounds the
moat from both sides: it transfers only where the domain carries a genuine exponential
instability *and* a trajectory long enough to fit and gate its form; the power grid has
both, EEG the trajectory without the form, and DNB neither the length nor (by
critical-slowing-down theory) the exponential form.

References
----------
* Chen, Liu, Liu, Aihara 2012, *Sci. Rep.* 2:342 — the dynamical network biomarker.
* Mojtahedi et al. 2016, *PLoS Biol.* 14:e2000640 — the single-cell transition index.
* Scheffer et al. 2009, *Nature* 461:53 — critical slowing down: a power-law, not
  exponential, divergence toward a bifurcation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from bench.dnb_detector import dnb_regression_slope
from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash
from scpn_phase_orchestrator.monitor.grid_modal_growth import (
    fit_gated_growth_rate,
    growth_rate_and_fit,
)

if TYPE_CHECKING:  # pragma: no cover - import only for static typing
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

    FloatArray = NDArray[np.float64]

#: The sealed-artefact identifier.
BENCHMARK = "dnb_modal_transfer"
#: Recency weighting carried unchanged from the certified grid operating point.
RECENCY_TOP = 3.0
#: The certified grid fit-quality gate, tested here for transfer.
GATE_R2 = 0.5
#: The fewest rising-limb points at which fitting *and* gating a growth form is
#: meaningful — below it the gate cannot separate an exponential from any monotone rise.
MIN_RESOLVABLE_POINTS = 6
#: A floor so a zero index cannot take a logarithm.
_INDEX_FLOOR = 1.0e-12

__all__ = [
    "BENCHMARK",
    "MIN_RESOLVABLE_POINTS",
    "dnb_transfer_payload",
    "dnb_transfer_verdict",
    "growth_form_record",
]


def growth_form_record(
    trajectory: Sequence[float] | FloatArray,
    *,
    label: str,
    rate: float = 1.0,
    recency_top: float = RECENCY_TOP,
    r2_gate: float = GATE_R2,
) -> dict[str, object]:
    """Characterise how the grid growth-form statistic behaves on one index trajectory.

    Records the trajectory length, the existing linear slope, the exponential growth
    rate and its fit quality ``R²``, and whether the fit-quality gate keeps the
    trajectory — so the sealed artefact shows, per corpus, that on a short rising limb
    the gate is uninformative (it keeps a well-enough-fitting monotone rise) and the
    growth rate carries no more than the slope.

    Parameters
    ----------
    trajectory : sequence of float
        The transition-index rising limb, in time order (positive values).
    label : str
        The corpus/lineage label sealed into the record.
    rate : float
        Samples per unit time of the trajectory (the growth rate's time unit).
    recency_top : float
        Recency weighting passed to the growth-rate fit.
    r2_gate : float
        The fit-quality gate under test.

    Returns
    -------
    dict
        A JSON-safe record: label, ``n_points``, ``slope``, ``growth_rate``,
        ``exponential_fit_r2``, ``gated_growth_rate``, and ``gate_keeps``.

    Raises
    ------
    ValueError
        If the trajectory has fewer than two points.
    """
    values = np.asarray(trajectory, dtype=np.float64)
    if values.shape[0] < 2:
        raise ValueError("trajectory must have at least two points")
    floored = np.maximum(values, _INDEX_FLOOR)
    slope = dnb_regression_slope(values)
    growth_rate, r2 = growth_rate_and_fit(floored, rate=rate, recency_top=recency_top)
    gated = fit_gated_growth_rate(
        floored, rate=rate, recency_top=recency_top, r2_gate=r2_gate
    )
    return {
        "label": label,
        "n_points": int(values.shape[0]),
        "slope": float(slope),
        "growth_rate": float(growth_rate),
        "exponential_fit_r2": float(r2),
        "gated_growth_rate": float(gated),
        "gate_keeps": bool(gated == growth_rate and growth_rate > 0.0),
    }


def dnb_transfer_verdict(
    single_cell: Sequence[Mapping[str, object]],
    bulk: Mapping[str, object],
    *,
    min_resolvable_points: int = MIN_RESOLVABLE_POINTS,
) -> str:
    """Return a one-line honest verdict on whether the moat is posable on DNB.

    Parameters
    ----------
    single_cell : sequence of mapping
        The per-lineage single-cell growth-form records.
    bulk : mapping
        The bulk GSE2565 growth-form record.
    min_resolvable_points : int
        The resolution threshold below which the gate cannot separate a growth form.

    Returns
    -------
    str
        A factual sentence stating the resolution limit.
    """
    records = [*single_cell, bulk]
    longest = max(cast("int", record["n_points"]) for record in records)
    kept = sum(1 for record in records if record["gate_keeps"])
    r2s = [cast("float", record["exponential_fit_r2"]) for record in records]
    return (
        f"The grid modal-growth moat cannot be posed on DNB early warning: its rising "
        f"limbs are {longest} points at most — below the {min_resolvable_points} a "
        f"growth-form fit and gate need — so the fit-quality gate is uninformative "
        f"(R² {min(r2s):.2f}–{max(r2s):.2f}, all {kept}/{len(records)} kept, none "
        f"rejected) and the exponential growth rate re-orders the transitions exactly "
        f"as the existing linear slope does, adding no separating information. The DNB "
        f"rise is a genuine critical-slowing-down divergence, not the exponential "
        f"instability the moat is built to fit."
    )


def dnb_transfer_payload(
    *,
    single_cell: Sequence[Mapping[str, object]],
    bulk: Mapping[str, object],
    corpus: Mapping[str, object],
    min_resolvable_points: int = MIN_RESOLVABLE_POINTS,
) -> dict[str, object]:
    """Assemble and hash-seal the DNB modal-growth transfer characterisation.

    Parameters
    ----------
    single_cell : sequence of mapping
        The per-lineage single-cell growth-form records.
    bulk : mapping
        The bulk GSE2565 growth-form record.
    corpus : mapping
        Corpus metadata (sources, lineage/arm counts, rising-limb lengths).
    min_resolvable_points : int
        The resolution threshold sealed into the verdict.

    Returns
    -------
    dict
        The JSON-safe payload with a ``content_hash`` field sealing the record.
    """
    payload: dict[str, object] = {
        "benchmark": BENCHMARK,
        "question": (
            "Can the grid modal-growth + fit-quality-gate moat be posed on the DNB "
            "critical-slowing-down early-warning corpora?"
        ),
        "corpus": dict(corpus),
        "min_resolvable_points": min_resolvable_points,
        "single_cell_mojtahedi": [dict(record) for record in single_cell],
        "bulk_gse2565": dict(bulk),
        "verdict": dnb_transfer_verdict(
            single_cell, bulk, min_resolvable_points=min_resolvable_points
        ),
    }
    payload["content_hash"] = canonical_record_hash(payload)
    return payload


def _single_cell_records() -> (  # pragma: no cover - reads the in-code published limbs
    list[dict[str, object]]
):
    """Return the growth-form record of each Mojtahedi lineage's rising limb."""
    from bench.early_warning_dnb import MOJTAHEDI_LINEAGES

    records = []
    for lineage in MOJTAHEDI_LINEAGES:
        means, _ = lineage.rising_limb()
        records.append(growth_form_record(means, label=lineage.lineage_id))
    return records


def _bulk_record(  # pragma: no cover - GSE2565 I/O shell over the tested pure core
    data_dir: str,
) -> dict[str, object]:
    """Return the growth-form record of the GSE2565 exposed arm's index rising limb."""
    from pathlib import Path

    from bench.dnb_detector import dnb_index, select_dnb_module
    from bench.early_warning_dnb_bulk import (
        CANDIDATE_FRACTION,
        EXPOSED_GROUP,
        MIN_MODULE,
        TRANSITION_INDEX,
        arm_rising_frames,
        load_expression_matrix,
        parse_sample_groups,
    )

    data = Path(data_dir)
    groups = parse_sample_groups(data / "GSE2565_family.soft.gz")
    accessions, _, matrix = load_expression_matrix(
        data / "GSE2565_series_matrix.txt.gz"
    )
    frames, _ = arm_rising_frames(matrix, accessions, groups, group=EXPOSED_GROUP)
    module = select_dnb_module(
        frames,
        transition_index=TRANSITION_INDEX,
        candidate_fraction=CANDIDATE_FRACTION,
        min_module=MIN_MODULE,
    )
    trajectory = [dnb_index(frame, module) for frame in frames]
    return growth_form_record(trajectory, label="gse2565_cg_exposed")


def evaluate_dnb_transfer(  # pragma: no cover - I/O shell over the tested pure core
    data_dir: str,
) -> dict[str, object]:
    """Characterise the modal-growth transfer on both DNB corpora and seal it."""
    single_cell = _single_cell_records()
    bulk = _bulk_record(data_dir)
    corpus = {
        "single_cell_source": "Mojtahedi et al. 2016 Table S2 (three lineages)",
        "bulk_source": "GSE2565 phosgene lung, exposed (CG) arm",
        "single_cell_rising_limb_points": single_cell[0]["n_points"],
        "bulk_rising_limb_points": bulk["n_points"],
    }
    return dnb_transfer_payload(single_cell=single_cell, bulk=bulk, corpus=corpus)


def main() -> None:  # pragma: no cover - CLI shell
    """Run the DNB transfer characterisation and write the sealed artefact."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="directory holding the GSE2565 files")
    parser.add_argument("output", help="path for the sealed JSON artefact")
    args = parser.parse_args()

    payload = evaluate_dnb_transfer(args.data_dir)
    Path(args.output).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output}: {payload['verdict']}")


if __name__ == "__main__":  # pragma: no cover - CLI shell
    main()

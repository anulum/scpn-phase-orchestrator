# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — grid modal-growth vs generic-suite head-to-head runner

"""Head-to-head: the domain-specific grid modal-growth detector vs the generic suite.

This is the flagship comparison. The programme's four-domain result put every generic
early-warning member (critical slowing down, synchronisation, ordinal entropy, their
fusion) at chance at a matched false-alarm rate. The domain-specific modal envelope-
growth detector (:mod:`bench.grid_oscillation_detector`), at its validated operating
point — the ``"focal"`` most-unstable-bus aggregation with a recency-weighted growth
rate — is the counterpoint: on a real power-grid corpus it leads far more instability
transitions than any generic member, and the shared moat certifies it.

The comparison is deliberately fair and non-circular:

* **Identical split.** Both detectors read the *same* two-second pre-onset voltage
  segments of the *same* PSML scenarios, labelled the same way, and are calibrated to
  the *same* matched false-alarm rate on the *same* damped null segments. Only the
  detector differs — the modal detector reads the raw cross-bus voltage deviation, the
  generic suite reads the cross-bus Kuramoto observables — never the data or the
  operating point.
* **Non-circular labels.** A transition is any *generator-trip* scenario and a null is
  any *damped bus-fault or branch-trip* scenario — the label is the **disturbance
  type**, a physical annotation independent of the growth statistic the modal detector
  measures, so a growing-oscillation detector cannot be scored against a growth label.
* **Disclosed data-quality gate.** A handful of scenarios carry a non-monotonic time
  column (a non-physical, negative inferred sampling rate); they are dropped and their
  count recorded in the sealed payload, never silently.
* **Pre-registered operating point.** The ``"focal"`` aggregation and recency weighting
  were selected on a development half of the corpus and validated on a held-out half
  before this full-corpus comparison; the held-out lead count is sealed alongside as the
  unbiased estimate.

The runner is an I/O shell over already-tested logic: the modal detector, the generic
suite, and the label-permutation core are unit-tested; the payload assembly and the
verdict are unit-tested; and the sealed artefact's integrity is checked by recomputing
its hash from the committed payload alone (no raw re-run, so no cross-platform float
drift). Only the derived, sealed result is written — the raw PSML data is never copied.

References
----------
* Zheng et al. 2021 — the PSML power-system dataset (23-bus millisecond-level PMU
  measurements) with disturbance-type annotations.
* Kundur 1994, *Power System Stability and Control* — small-signal (modal) stability.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from bench.early_warning_domain import (
    DEFAULT_PERMUTATION_SEED,
    DEFAULT_PERMUTATIONS,
    DEFAULT_TARGET_FALSE_ALARM,
)
from bench.grid_oscillation_detector import DEFAULT_AGGREGATION, DEFAULT_RECENCY_TOP
from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash

if TYPE_CHECKING:  # pragma: no cover - import only for static typing
    from collections.abc import Mapping

    import numpy as np
    from numpy.typing import NDArray

    from scpn_phase_orchestrator.monitor.early_warning_suite import SuiteObservables

    FloatArray = NDArray[np.float64]

__all__ = [
    "BENCHMARK",
    "evaluate_head_to_head",
    "head_to_head_payload",
    "main",
    "modal_versus_generic_verdict",
]

#: The sealed artefact's benchmark identifier.
BENCHMARK = "grid_modal_head_to_head"

#: Physical PMU sampling-rate band in hertz; a scenario whose rate is outside it has a
#: non-monotonic time column and is dropped as data-corrupt (the count is disclosed).
RATE_BAND_HZ = (100.0, 400.0)


@dataclass(frozen=True)
class _BuiltSegments:
    """The identical transition and null segments both detectors read."""

    transition_raw: list[FloatArray]
    null_raw: list[FloatArray]
    transition_obs: list[SuiteObservables]
    null_obs: list[SuiteObservables]
    rate: float
    dropped: int


def modal_versus_generic_verdict(
    modal_significance: Mapping[str, object],
    generic_records: Mapping[str, Mapping[str, object]],
) -> str:
    """Return a one-line verdict of the modal detector against the best generic member.

    Parameters
    ----------
    modal_significance : mapping
        The modal detector's permutation-significance record (``observed_led``,
        ``n_transitions``, ``p_value``).
    generic_records : mapping
        Each generic detector label mapped to its permutation-significance record.

    Returns
    -------
    str
        A factual sentence stating whether the modal detector leads more transitions
        than every generic member at a significant matched-false-alarm operating point.
    """
    best_name, best = max(
        generic_records.items(),
        key=lambda item: (
            cast("int", item[1]["observed_led"]),
            -cast("float", item[1]["p_value"]),
        ),
    )
    modal_led = cast("int", modal_significance["observed_led"])
    total = cast("int", modal_significance["n_transitions"])
    modal_p = cast("float", modal_significance["p_value"])
    best_led = cast("int", best["observed_led"])
    best_p = cast("float", best["p_value"])
    lead = (
        f"The domain-specific modal envelope-growth detector leads {modal_led}/{total} "
        f"transitions (p={modal_p:.4f})"
    )
    generic = (
        f"the best generic member ({best_name}, {best_led}/{total}, p={best_p:.4f})"
    )
    if modal_led > best_led and modal_p < 0.05:
        return (
            f"{lead}, beating {generic} and every generic detector at a matched false "
            f"alarm on the identical disturbance-type split."
        )
    return (
        f"{lead}; it does not beat {generic} on the identical disturbance-type split."
    )


def head_to_head_payload(
    *,
    modal_record: Mapping[str, object],
    generic_records: Mapping[str, Mapping[str, object]],
    corpus: Mapping[str, object],
    operating_point: Mapping[str, object],
    held_out_validation: Mapping[str, object],
    target_fa: float,
    n_permutations: int,
    seed: int,
) -> dict[str, object]:
    """Assemble and hash-seal the head-to-head result payload.

    Parameters
    ----------
    modal_record : mapping
        The modal detector's audit record, whose ``"significance"`` sub-record carries
        the lead count.
    generic_records : mapping
        Each generic detector label mapped to its permutation-significance record.
    corpus : mapping
        Corpus metadata: source, transition and null counts, dropped-scenario count,
        sampling rate, segment length, and the labelling description.
    operating_point : mapping
        The modal detector's aggregation, recency weighting, and selection provenance.
    held_out_validation : mapping
        The modal detector's held-out lead count and p-value — the unbiased estimate.
    target_fa : float
        The matched false-alarm rate both detectors are calibrated to.
    n_permutations : int
        Number of random relabellings behind every p-value.
    seed : int
        Seed of the resampling, so every p-value is reproducible.

    Returns
    -------
    dict
        The JSON-safe payload with a ``content_hash`` field sealing the record.
    """
    modal_significance = modal_record["significance"]
    assert isinstance(modal_significance, dict)
    generic_payload = {name: dict(record) for name, record in generic_records.items()}
    payload: dict[str, object] = {
        "benchmark": BENCHMARK,
        "question": (
            "At a matched false alarm on the identical non-circular disturbance-type "
            "split, does the domain-specific modal envelope-growth detector beat the "
            "generic SCPN early-warning suite?"
        ),
        "corpus": dict(corpus),
        "operating_point": dict(operating_point),
        "target_false_alarm": target_fa,
        "n_permutations": n_permutations,
        "seed": seed,
        "modal": dict(modal_record),
        "generic_suite": generic_payload,
        "held_out_validation": dict(held_out_validation),
        "verdict": modal_versus_generic_verdict(modal_significance, generic_records),
    }
    payload["content_hash"] = canonical_record_hash(payload)
    return payload


def _partition_by_disturbance_type(  # pragma: no cover - I/O shell over ingestion
    data_dir: Path, *, segment_seconds: float
) -> _BuiltSegments:
    """Read the PSML corpus into the identical modal/generic transition and null sets.

    Every scenario is classified by disturbance type (generator trip = transition,
    damped bus fault or branch trip = null); scenarios with a non-physical rate are
    dropped and counted. For each kept scenario the same two-second pre-onset window
    yields a raw voltage segment (for the modal detector) and a sliced observable field
    (for the generic suite), so both detectors read the identical segments.
    """
    import numpy as np

    from bench.early_warning_domain import slice_observables
    from bench.early_warning_leadtime_grid import (
        DAMPED_TYPES,
        GEN_TRIP,
        GridPhaseAdapter,
        bus_voltages,
        discover_scenarios,
        oscillation_info,
    )

    transition_raw: list[FloatArray] = []
    null_raw: list[FloatArray] = []
    transition_obs: list[SuiteObservables] = []
    null_obs: list[SuiteObservables] = []
    rate = 0.0
    dropped = 0
    for scenario in discover_scenarios(data_dir):
        info = oscillation_info(scenario)
        try:
            end_s = float(info["end"])
        except (KeyError, ValueError):
            continue
        kind = info.get("type", "")
        if kind != GEN_TRIP and kind not in DAMPED_TYPES:
            continue
        scenario_rate, volts = bus_voltages(scenario)
        physical = np.isfinite(scenario_rate) and (
            RATE_BAND_HZ[0] < scenario_rate < RATE_BAND_HZ[1]
        )
        if not physical:
            dropped += 1
            continue
        rate = scenario_rate
        segment_samples = int(round(segment_seconds * scenario_rate))
        onset = int(round(end_s * scenario_rate))
        if onset < segment_samples or onset > volts.shape[1]:
            continue
        raw = np.ascontiguousarray(volts[:, onset - segment_samples : onset])
        if raw.shape[1] < segment_samples:
            continue
        adapter = GridPhaseAdapter(sampling_rate_hz=scenario_rate)
        sliced = slice_observables(
            adapter.observables(volts), start=onset - segment_samples, stop=onset
        )
        if kind == GEN_TRIP:
            transition_raw.append(raw)
            transition_obs.append(sliced)
        else:
            null_raw.append(raw)
            null_obs.append(sliced)
    return _BuiltSegments(
        transition_raw=transition_raw,
        null_raw=null_raw,
        transition_obs=transition_obs,
        null_obs=null_obs,
        rate=rate,
        dropped=dropped,
    )


def evaluate_head_to_head(  # pragma: no cover - I/O shell over the tested detectors
    data_dir: str | Path,
    *,
    segment_seconds: float = 2.0,
    target_fa: float = DEFAULT_TARGET_FALSE_ALARM,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_PERMUTATION_SEED,
) -> dict[str, object]:
    """Run the modal detector and the generic suite on the identical split and seal it.

    Reads the PSML corpus, builds the identical two-second pre-onset transition and null
    segments, scores them with the modal envelope-growth detector at its validated
    operating point and with the generic suite at the same matched false alarm,
    validates the modal operating point on a held-out half, and returns the sealed
    payload.
    """
    from bench.early_warning_domain import (
        calibrate_detectors,
        permutation_significance_by_detector,
    )
    from bench.early_warning_leadtime_grid import (
        SEGMENT_BASELINE_FRACTION,
        STEP_SECONDS,
        WINDOW_SECONDS,
    )
    from bench.grid_oscillation_detector import modal_growth_significance

    built = _partition_by_disturbance_type(
        Path(data_dir), segment_seconds=segment_seconds
    )
    rate = built.rate

    modal = modal_growth_significance(
        built.transition_raw,
        built.null_raw,
        rate=rate,
        aggregation=DEFAULT_AGGREGATION,
        recency_top=DEFAULT_RECENCY_TOP,
        target_fa=target_fa,
        n_permutations=n_permutations,
        seed=seed,
    )

    # Unbiased held-out estimate: the operating point was chosen on the even (dev) half.
    held = modal_growth_significance(
        built.transition_raw[1::2],
        built.null_raw[1::2],
        rate=rate,
        aggregation=DEFAULT_AGGREGATION,
        recency_top=DEFAULT_RECENCY_TOP,
        target_fa=target_fa,
        n_permutations=n_permutations,
        seed=seed,
    )

    window = max(1, int(round(WINDOW_SECONDS * rate)))
    step = max(1, int(round(STEP_SECONDS * rate)))
    calibration = calibrate_detectors(
        built.null_obs,
        target_fa=target_fa,
        window=window,
        step=step,
        baseline_fraction=SEGMENT_BASELINE_FRACTION,
    )
    generic = permutation_significance_by_detector(
        built.transition_obs,
        built.null_obs,
        thresholds=calibration.thresholds,
        window=window,
        step=step,
        baseline_fraction=SEGMENT_BASELINE_FRACTION,
        n_permutations=n_permutations,
        seed=seed,
    )
    generic_records = {name: sig.to_audit_record() for name, sig in generic.items()}
    held_record = held.significance.to_audit_record()
    return head_to_head_payload(
        modal_record=modal.to_audit_record(),
        generic_records=generic_records,
        corpus={
            "source": "PSML 23-bus power-system dataset (Zheng et al. 2021)",
            "n_transitions": len(built.transition_raw),
            "n_nulls": len(built.null_raw),
            "n_dropped_bad_rate": built.dropped,
            "sampling_rate_hz": rate,
            "segment_seconds": segment_seconds,
            "labelling": (
                "disturbance type — generator trip = transition, damped bus fault or "
                "branch trip = null (independent of the growth statistic)"
            ),
        },
        operating_point={
            "aggregation": DEFAULT_AGGREGATION,
            "recency_top": DEFAULT_RECENCY_TOP,
            "selection": (
                "chosen on the even-index development half, validated on the odd-index "
                "held-out half before this full-corpus comparison"
            ),
        },
        held_out_validation={
            "n_transitions": held_record["n_transitions"],
            "observed_led": held_record["observed_led"],
            "p_value": held_record["p_value"],
        },
        target_fa=target_fa,
        n_permutations=n_permutations,
        seed=seed,
    )


def main(  # pragma: no cover - CLI shell over the tested logic
    data_dir: str | Path, output_dir: str | Path
) -> None:
    """Run the grid modal-vs-generic head-to-head and write the sealed artefact.

    Parameters
    ----------
    data_dir : str or Path
        The raw PSML corpus directory (read, never copied).
    output_dir : str or Path
        Directory the sealed derived artefact is written to.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = evaluate_head_to_head(data_dir)
    (out / "grid_modal_head_to_head.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(payload["verdict"])
    print(f"content hash {payload['content_hash']}")
    print(f"results written to {out}")


if __name__ == "__main__":  # pragma: no cover - CLI shell over the tested logic
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir")
    parser.add_argument("output_dir")
    arguments = parser.parse_args()
    main(arguments.data_dir, arguments.output_dir)

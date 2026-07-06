# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — a sealed real-PSML grid early-warning advisory example

"""Seal a review-only grid early-warning advisory on a real PSML instability.

This closes the streaming-to-decision loop end-to-end on real data: it builds the live
monitor at the certified *streaming* operating point straight from the sealed
operating-point artefact (``GridModalStreamMonitor.from_stream_evidence``), replays a
real PSML generator-trip scenario's bus voltages through it, and turns the first stream
alarm into a claim-bounded, review-only advisory (``advise_from_stream_alarm``).

The advisory is sealed **without** a ground-truth onset — the honest live-deployment
case, where an operator receives the advisory and investigates without knowing when (or
whether) an onset follows. The honest recall and false alarm come from the sealed
streaming artefact, and the operating-point provenance is that artefact's own content
hash, so the example carries no hand-set constants.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from scpn_phase_orchestrator.assurance.grid_early_warning_advisory import (
    advise_from_stream_alarm,
)
from scpn_phase_orchestrator.monitor.grid_modal_stream import GridModalStreamMonitor
from scpn_phase_orchestrator.runtime.grid_modal_replay import replay

if TYPE_CHECKING:  # pragma: no cover - import only for static typing
    from scpn_phase_orchestrator.assurance.grid_early_warning_advisory import (
        GridEarlyWarningAdvisory,
    )

#: The PSML generator-trip scenario the certified monitor leads (a real instability).
#: A subdir-qualified path, because a bare row id can occur in more than one regime.
EXAMPLE_SCENARIO = "Natural Oscillation/row_108"
#: A fixed, deterministic capture timestamp (the artefact must not read a wall clock).
CAPTURE_TIMESTAMP = "2021-01-01T00:00:00Z"


def build_example_advisory(  # pragma: no cover - PSML I/O shell over sealed src code
    data_dir: str,
    evidence_path: str,
    *,
    scenario: str = EXAMPLE_SCENARIO,
) -> GridEarlyWarningAdvisory:
    """Replay a real PSML scenario and seal the first alarm into an advisory."""
    from bench.early_warning_leadtime_grid import bus_voltages, discover_scenarios

    stream = json.loads(Path(evidence_path).read_text(encoding="utf-8"))
    winner = next(
        row
        for row in stream["search"]
        if row["feature"] == "r2gate"
        and row["window_seconds"] == 2.0
        and row["persistence"] == 2
    )
    recall = float(winner["held_out_led"]) / float(winner["n_held_out"])
    false_alarm = float(winner["held_out_false_alarm"])
    provenance = (
        f"psml grid_modal_stream_operating_point.json#{stream['content_hash'][:8]}"
    )

    scenario_dir = next(
        path
        for path in discover_scenarios(data_dir)
        if path.as_posix().endswith(scenario)
    )
    rate, voltages = bus_voltages(scenario_dir)
    monitor = GridModalStreamMonitor.from_stream_evidence(evidence_path, rate=rate)
    alarms = replay(voltages, monitor)
    if not alarms:
        raise ValueError(f"scenario {scenario} raised no alarm at the certified point")
    return advise_from_stream_alarm(
        alarms[0],
        monitor,
        signal_source=f"PSML Millisecond-level PMU / gen_trip / {scenario}",
        captured_at=CAPTURE_TIMESTAMP,
        certified_recall=recall,
        certified_false_alarm=false_alarm,
        certified_operating_point=provenance,
    )


def main() -> None:  # pragma: no cover - CLI shell
    """Build the example advisory and write the sealed artefact."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="PSML scenario root directory")
    parser.add_argument("evidence", help="sealed streaming operating-point JSON")
    parser.add_argument("output", help="path for the sealed advisory JSON")
    args = parser.parse_args()

    advisory = build_example_advisory(args.data_dir, args.evidence)
    Path(args.output).write_text(
        json.dumps(advisory.to_audit_record(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {args.output}: sigma={advisory.growth_rate:.4f} "
        f"bus={advisory.most_unstable_bus}"
    )


if __name__ == "__main__":  # pragma: no cover - CLI shell
    main()

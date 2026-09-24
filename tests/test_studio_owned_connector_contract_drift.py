# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Studio owned-connector contract drift tests

"""An owned connector record never mixes two binding contracts.

The record names the ``contract_hash`` of the replay's connector plan, while
the envelope is built from the binding spec on disk. When that spec was edited
after the replay, the record must be blocked instead of reporting an accepted
exchange under the replay's contract.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from scpn_phase_orchestrator.binding.digital_twin import (
    build_digital_twin_binding_contract,
)
from scpn_phase_orchestrator.binding.loader import load_binding_spec
from scpn_phase_orchestrator.studio.ui_helpers import (
    StudioKnobState,
    build_owned_live_connector_runtime_record,
    run_binding_spec_replay,
)
from scpn_phase_orchestrator.studio.ui_helpers._state import StudioReplayResult

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "domainpacks" / "digital_twin_nchannel"


def _replay_copy(tmp_path: Path) -> tuple[Path, StudioReplayResult]:
    """Copy the digital-twin pack to ``tmp_path`` and replay it."""
    pack = tmp_path / "pack"
    shutil.copytree(PACK, pack)
    spec = pack / "binding_spec.yaml"
    result = run_binding_spec_replay(spec, steps=3, knobs=StudioKnobState(K=1.0))
    return spec, result


def _record(result: StudioReplayResult) -> dict[str, object]:
    """Return an owned REST runtime record for ``result``."""
    return build_owned_live_connector_runtime_record(
        result,
        transport="rest",
        owner="plant-ops",
        auth_policy={"scheme": "bearer", "credential_label": "studio-local-key"},
        payload={"kind": "owned_runtime_probe", "R": 0.82},
        sequence=7,
    )


def test_spec_edited_after_replay_blocks_the_record(tmp_path: Path) -> None:
    """A spec that now yields another contract blocks the exchange."""
    spec, result = _replay_copy(tmp_path)
    text = spec.read_text(encoding="utf-8")
    edited = text.replace("name: digital_twin_nchannel", "name: swapped_twin", 1)
    assert edited != text
    spec.write_text(edited, encoding="utf-8")
    edited_hash = build_digital_twin_binding_contract(
        load_binding_spec(spec)
    ).contract_hash
    assert edited_hash != result.connector_plan["contract_hash"]

    record = _record(result)

    assert record["status"] == "blocked"
    assert record["blocked_reasons"] == [
        "binding spec changed since the replay; replay it again"
    ]
    assert record["contract_hash"] == result.connector_plan["contract_hash"]
    assert record["adapter"] == {}
    assert record["response"] == {}
    assert record["queued_count"] == 0


def test_unchanged_spec_runs_the_adapter_on_the_replay_contract(
    tmp_path: Path,
) -> None:
    """With the spec unchanged, the adapter runs on the recorded contract."""
    _, result = _replay_copy(tmp_path)

    record = _record(result)

    assert record["status"] == "accepted"
    adapter = record["adapter"]
    assert isinstance(adapter, dict)
    assert adapter["contract_hash"] == record["contract_hash"]

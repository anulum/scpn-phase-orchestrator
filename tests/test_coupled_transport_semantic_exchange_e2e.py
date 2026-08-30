# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Real FUSION to SPO semantic exchange

"""Cross-repository public-surface test over canonical real TORAX evidence."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from scpn_phase_orchestrator.reactor_semantics import (
    FUSION_TORAX_REVIEW_SCHEMA,
    RegimeState,
    coupled_transport_handoff_from_fusion_bytes,
    handoff_from_bytes,
    handoff_to_bytes,
)


def _fusion_root() -> Path:
    configured = os.environ.get("SCPN_FUSION_CORE_ROOT")
    candidate = (
        Path(configured).resolve()
        if configured is not None
        else Path(__file__).resolve().parents[2] / "SCPN-FUSION-CORE"
    )
    if not (candidate / "src/scpn_fusion/integrations/torax/review.py").is_file():
        pytest.skip("canonical SCPN-FUSION-CORE checkout is not available")
    return candidate


def _fusion_public_roundtrip(
    fusion_root: Path,
    fixture: Path,
    digest: str,
) -> dict[str, object]:
    environment = os.environ.copy()
    source_path = str(fusion_root / "src")
    previous = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_path if previous is None else source_path + os.pathsep + previous
    )
    program = textwrap.dedent(
        """
        import json
        import sys
        from pathlib import Path

        from scpn_fusion.integrations.torax import (
            review_envelope_from_bytes,
            review_envelope_sha256,
            review_envelope_to_bytes,
        )

        payload = Path(sys.argv[1]).read_bytes()
        envelope = review_envelope_from_bytes(payload, expected_sha256=sys.argv[2])
        assert review_envelope_to_bytes(envelope) == payload
        print(json.dumps({
            "event_id": envelope.event_id,
            "schema": envelope.schema,
            "sha256": review_envelope_sha256(envelope),
            "source_revision": envelope.source_revision,
        }, sort_keys=True))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", program, str(fixture), digest],
        cwd=fusion_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=60.0,
    )
    return json.loads(completed.stdout)


def test_real_fusion_fixture_crosses_public_spo_adapter_without_phase_or_action() -> (
    None
):
    fusion_root = _fusion_root()
    fixture = (
        fusion_root
        / "validation/reference_data/torax/torax_runtime_review_envelope_v1.json"
    )
    report_path = fusion_root / "validation/reports/torax_runtime_contract.json"
    source_bytes = fixture.read_bytes()
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    source = json.loads(source_bytes)
    report = json.loads(report_path.read_bytes())

    assert report["passes_thresholds"] is True
    assert all(report["gates"].values())
    assert report["review_envelope"] == {
        "model_intersection_schema": source["model_intersection_schema"],
        "path": str(fixture.relative_to(fusion_root)),
        "schema": source["schema"],
        "sha256": source_digest,
        "source_revision": source["source_revision"],
    }
    git = shutil.which("git")
    assert git is not None
    subprocess.run(
        [git, "cat-file", "-e", f"{source['source_revision']}^{{commit}}"],
        cwd=fusion_root,
        check=True,
        capture_output=True,
        timeout=10.0,
    )
    fusion_receipt = _fusion_public_roundtrip(fusion_root, fixture, source_digest)
    assert fusion_receipt == {
        "event_id": source["event_id"],
        "schema": FUSION_TORAX_REVIEW_SCHEMA,
        "sha256": source_digest,
        "source_revision": source["source_revision"],
    }

    handoff = coupled_transport_handoff_from_fusion_bytes(
        source_bytes,
        expected_sha256=source_digest,
    )
    handoff_bytes = handoff_to_bytes(handoff)
    decoded = handoff_from_bytes(handoff_bytes)

    assert decoded == handoff
    assert decoded.source_envelope_sha256 == source_digest
    assert decoded.source_revision == source["source_revision"]
    assert len(decoded.observables) == len(decoded.semantics) == 12
    assert decoded.phase_relations == ()
    assert decoded.regime.state is RegimeState.UNKNOWN
    assert decoded.actionable is False

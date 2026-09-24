# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — grid modal monitors verify their sealed artefacts

"""A monitor built "from sealed evidence" must verify the seal and the types.

An edited artefact (threshold raised to 1e9, so the monitor never alarms) used
to configure the monitor, and a ``true`` threshold was coerced to 1.0. Both
constructors now verify ``content_hash`` and pass certified values through
unconverted; the streaming constructor also enforces its documented rule that
the requested false-alarm target matches the sealed one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_phase_orchestrator.assurance import _hashing as assurance_hashing
from scpn_phase_orchestrator.monitor._sealed_record import (
    canonical_record_hash,
    load_sealed_json,
)
from scpn_phase_orchestrator.monitor.grid_modal_stream import GridModalStreamMonitor

_DIR = Path(__file__).resolve().parents[1] / "examples/real_data/psml_modal_growth"
_HEAD_TO_HEAD = _DIR / "grid_modal_head_to_head.json"
_STREAM = _DIR / "grid_modal_stream_operating_point.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(tmp_path: Path, payload: dict, *, reseal: bool) -> Path:
    body = {k: v for k, v in payload.items() if k != "content_hash"}
    out = dict(body)
    out["content_hash"] = (
        canonical_record_hash(body) if reseal else payload["content_hash"]
    )
    path = tmp_path / "artefact.json"
    path.write_text(json.dumps(out), encoding="utf-8")
    return path


def test_edited_head_to_head_threshold_is_refused(tmp_path: Path) -> None:
    payload = _load(_HEAD_TO_HEAD)
    payload["modal"]["score_threshold"] = 1e9
    with pytest.raises(ValueError, match="tampered"):
        GridModalStreamMonitor.from_evidence(
            _write(tmp_path, payload, reseal=False), rate=238.095
        )


def test_edited_stream_search_is_refused(tmp_path: Path) -> None:
    payload = _load(_STREAM)
    for row in payload["search"]:
        row["threshold"] = 1e9
    with pytest.raises(ValueError, match="tampered"):
        GridModalStreamMonitor.from_stream_evidence(
            _write(tmp_path, payload, reseal=False), rate=238.0
        )


@pytest.mark.parametrize("path", [_HEAD_TO_HEAD, _STREAM])
def test_unsealed_artefact_is_refused(tmp_path: Path, path: Path) -> None:
    payload = _load(path)
    payload.pop("content_hash")
    unsealed = tmp_path / "unsealed.json"
    unsealed.write_text(json.dumps(payload), encoding="utf-8")
    build = (
        GridModalStreamMonitor.from_evidence
        if path == _HEAD_TO_HEAD
        else GridModalStreamMonitor.from_stream_evidence
    )
    with pytest.raises(ValueError, match="no content_hash"):
        build(unsealed, rate=238.0)


@pytest.mark.parametrize("value", [True, "1.32"])
def test_mistyped_sealed_threshold_is_refused(tmp_path: Path, value: object) -> None:
    payload = _load(_HEAD_TO_HEAD)
    payload["modal"]["score_threshold"] = value
    with pytest.raises(ValueError, match="threshold"):
        GridModalStreamMonitor.from_evidence(
            _write(tmp_path, payload, reseal=True), rate=238.095
        )


def test_fractional_sealed_persistence_is_refused(tmp_path: Path) -> None:
    payload = _load(_STREAM)
    for row in payload["search"]:
        row["persistence"] = 2.7
    with pytest.raises(ValueError, match="persistence"):
        GridModalStreamMonitor.from_stream_evidence(
            _write(tmp_path, payload, reseal=True), rate=238.0
        )


def test_stream_target_must_match_the_sealed_target() -> None:
    with pytest.raises(ValueError, match="does not match the sealed"):
        GridModalStreamMonitor.from_stream_evidence(
            _STREAM, rate=238.0, target_false_alarm=0.2
        )


def test_shipped_artefacts_still_configure_monitors() -> None:
    per_window = GridModalStreamMonitor.from_evidence(_HEAD_TO_HEAD, rate=238.095)
    streaming = GridModalStreamMonitor.from_stream_evidence(_STREAM, rate=238.0)
    assert per_window.threshold == pytest.approx(
        _load(_HEAD_TO_HEAD)["modal"]["score_threshold"]
    )
    assert streaming.r2_gate == 0.5


def test_core_loader_is_the_one_assurance_re_exports() -> None:
    # One implementation: the runtime assurance package re-exports the core one.
    assert assurance_hashing.canonical_record_hash is canonical_record_hash
    assert assurance_hashing.load_sealed_json is load_sealed_json


@pytest.mark.parametrize("path", [_HEAD_TO_HEAD, _STREAM])
def test_core_loader_verifies_the_shipped_artefacts(path: Path) -> None:
    payload = load_sealed_json(path)
    body = {k: v for k, v in payload.items() if k != "content_hash"}
    assert payload["content_hash"] == canonical_record_hash(body)


def test_core_loader_refuses_non_object_and_non_finite(tmp_path: Path) -> None:
    array = tmp_path / "array.json"
    array.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_sealed_json(array)
    with pytest.raises(ValueError, match="finite"):
        canonical_record_hash({"x": float("nan")})

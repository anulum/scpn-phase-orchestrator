# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — modal sentinel adapter tests

"""Owner tests for :mod:`scpn_phase_orchestrator.runtime.modal_sentinel`.

Synthetic fixtures only: a sealed E2.G-shaped payload (hashed with the real
canonical hasher) configures the sentinel, and synthetic frames exercise the
fail-closed observation contract, the alarm sealing, and the review-only
posture without any raw data or live bridge.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash
from scpn_phase_orchestrator.monitor.grid_modal_stream import GridModalStreamMonitor
from scpn_phase_orchestrator.runtime.modal_sentinel import (
    ModalSentinel,
    load_verified_evidence,
)

CASE = "ISO-NE_case1"
RATE_HZ = 30.0
CHANNELS = ("Sub:1", "Sub:2", "Sub:3")


def _sealed_evidence(
    path: Path,
    *,
    threshold: float = 0.05,
    window_seconds: float = 4.0,
    tamper: bool = False,
    drop_hash: bool = False,
    drop_corpus: bool = False,
    drop_detector: bool = False,
    bad_window: bool = False,
    unknown_case: bool = False,
) -> Path:
    """Write a minimal E2.G-shaped payload sealed with the real hasher."""
    record: dict[str, Any] = {
        "benchmark": "iso_ne_modal_growth_cross_dataset",
        "detector": {"aggregation": "focal", "recency_top": 3.0},
        "corpus": {"transitions": [{"case": CASE, "window_seconds": window_seconds}]},
        "local_calibration": {"threshold": threshold, "n_null": 6},
    }
    if drop_corpus:
        record.pop("corpus")
    if drop_detector:
        record.pop("detector")
    if bad_window:
        record["corpus"]["transitions"][0]["window_seconds"] = "wide"
    if unknown_case:
        record["corpus"]["transitions"][0]["case"] = "other"
    record["content_hash"] = canonical_record_hash(
        {key: value for key, value in record.items() if key != "content_hash"}
    )
    if tamper:
        record["local_calibration"]["threshold"] = threshold / 2.0
    if drop_hash:
        record.pop("content_hash")
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def _growing_frames(
    n_seconds: float = 60.0, onset_s: float = 20.0
) -> list[dict[str, float]]:
    """Frames of ambient noise then a growing 1 Hz oscillation."""
    rng = np.random.default_rng(3)
    n = int(n_seconds * RATE_HZ)
    t = np.arange(n) / RATE_HZ
    ramp = np.clip((t - onset_s) / 10.0, 0.0, 1.0)
    frames = []
    for index in range(n):
        frames.append(
            {
                name: float(
                    60.0
                    + 0.001 * rng.standard_normal()
                    + 0.05
                    * (1.0 - 0.2 * k)
                    * ramp[index]
                    * np.sin(2 * np.pi * 1.0 * t[index] + 0.3 * k)
                )
                for k, name in enumerate(CHANNELS)
            }
        )
    return frames


class TestLoadVerifiedEvidence:
    def test_returns_verified_payload(self, tmp_path: Path) -> None:
        path = _sealed_evidence(tmp_path / "evidence.json")
        assert load_verified_evidence(path)["benchmark"] == (
            "iso_ne_modal_growth_cross_dataset"
        )

    def test_rejects_missing_hash(self, tmp_path: Path) -> None:
        path = _sealed_evidence(tmp_path / "nohash.json", drop_hash=True)
        with pytest.raises(ValueError, match="no content_hash"):
            load_verified_evidence(path)

    def test_rejects_tampered_record(self, tmp_path: Path) -> None:
        path = _sealed_evidence(tmp_path / "tampered.json", tamper=True)
        with pytest.raises(ValueError, match="tampered"):
            load_verified_evidence(path)


class TestFromSealedEvidence:
    def test_carries_sealed_operating_point_and_provenance(
        self, tmp_path: Path
    ) -> None:
        path = _sealed_evidence(
            tmp_path / "evidence.json", threshold=0.07, window_seconds=4.0
        )
        sentinel = ModalSentinel.from_sealed_evidence(
            path, case_id=CASE, rate=RATE_HZ, channels=CHANNELS
        )
        assert sentinel.monitor.threshold == 0.07
        assert sentinel.monitor.window_seconds == pytest.approx(4.0)
        assert sentinel.monitor.step_seconds == pytest.approx(1.0)
        assert sentinel.provenance["case"] == CASE
        assert sentinel.provenance["calibration_n_null"] == 6
        assert sentinel.non_actuating and sentinel.execution_disabled

    @pytest.mark.parametrize(
        ("flag", "message"),
        [
            ("drop_corpus", "corpus.transitions"),
            ("unknown_case", "not in the sealed corpus"),
            ("drop_detector", "detector/local_calibration"),
            ("bad_window", "window_seconds"),
        ],
    )
    def test_rejects_malformed_evidence(
        self, tmp_path: Path, flag: str, message: str
    ) -> None:
        path = _sealed_evidence(tmp_path / "evidence.json", **{flag: True})
        with pytest.raises(ValueError, match=message):
            ModalSentinel.from_sealed_evidence(
                path, case_id=CASE, rate=RATE_HZ, channels=CHANNELS
            )


class TestSentinelContract:
    def _sentinel(self, *, threshold: float = 0.05) -> ModalSentinel:
        monitor = GridModalStreamMonitor(
            rate=RATE_HZ,
            threshold=threshold,
            window_seconds=4.0,
            step_seconds=1.0,
            persistence=1,
        )
        return ModalSentinel(monitor=monitor, channels=CHANNELS)

    def test_rejects_empty_channels(self) -> None:
        monitor = GridModalStreamMonitor(
            rate=RATE_HZ, threshold=1.0, window_seconds=4.0, step_seconds=1.0
        )
        with pytest.raises(ValueError, match="at least one channel"):
            ModalSentinel(monitor=monitor, channels=())

    def test_rejects_duplicate_channels(self) -> None:
        monitor = GridModalStreamMonitor(
            rate=RATE_HZ, threshold=1.0, window_seconds=4.0, step_seconds=1.0
        )
        with pytest.raises(ValueError, match="unique"):
            ModalSentinel(monitor=monitor, channels=("a", "a"))

    def test_rejects_unknown_channel(self) -> None:
        sentinel = self._sentinel()
        frame = dict.fromkeys(CHANNELS, 60.0) | {"intruder": 60.0}
        with pytest.raises(ValueError, match="unknown channels"):
            sentinel.observe(frame)

    def test_rejects_missing_channel(self) -> None:
        sentinel = self._sentinel()
        frame = dict.fromkeys(CHANNELS[:-1], 60.0)
        with pytest.raises(ValueError, match="misses declared channel"):
            sentinel.observe(frame)

    @pytest.mark.parametrize("bad", [True, "60", None])
    def test_rejects_non_real_reading(self, bad: object) -> None:
        sentinel = self._sentinel()
        frame = dict.fromkeys(CHANNELS, 60.0)
        frame[CHANNELS[1]] = bad  # type: ignore[assignment]
        with pytest.raises(ValueError, match="real number"):
            sentinel.observe(frame)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_rejects_non_finite_reading(self, bad: float) -> None:
        sentinel = self._sentinel()
        frame = dict.fromkeys(CHANNELS, 60.0)
        frame[CHANNELS[0]] = bad
        with pytest.raises(ValueError, match="finite"):
            sentinel.observe(frame)

    def test_growing_stream_seals_an_alarm(self, tmp_path: Path) -> None:
        path = _sealed_evidence(tmp_path / "evidence.json", threshold=0.05)
        sentinel = ModalSentinel.from_sealed_evidence(
            path, case_id=CASE, rate=RATE_HZ, channels=CHANNELS
        )
        records = [
            record
            for frame in _growing_frames()
            if (record := sentinel.observe(frame)) is not None
        ]
        assert records, "growing oscillation must seal at least one alarm"
        first = records[0]
        assert first["kind"] == "modal_sentinel_alarm"
        assert first["review_only"] is True
        assert first["focal_channel"] in CHANNELS
        assert first["provenance"]["evidence_content_hash"]  # type: ignore[index]
        sealed = dict(first)
        content_hash = sealed.pop("content_hash")
        assert canonical_record_hash(sealed) == content_hash

    def test_quiet_stream_stays_silent(self) -> None:
        sentinel = self._sentinel(threshold=1e9)
        rng = np.random.default_rng(5)
        for _ in range(int(10 * RATE_HZ)):
            frame = {
                name: float(60.0 + 0.001 * rng.standard_normal()) for name in CHANNELS
            }
            assert sentinel.observe(frame) is None

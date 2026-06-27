# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — scaffold CLI contract tests

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from scpn_phase_orchestrator.runtime.cli import main
from scpn_phase_orchestrator.runtime.cli import scaffold as scaffold_mod


class _HTTPResponse:
    """Small HTTPS response double for scaffold dataset-download tests."""

    def __init__(self, *, status: int, payload: bytes) -> None:
        self.status = status
        self._payload = payload

    def read(self, _size: int) -> bytes:
        """Return the configured payload regardless of requested size."""

        return self._payload


class _HTTPSConnection:
    """HTTPS connection double that records request paths."""

    status = 200
    payload = b"ok"
    requests: list[str] = []
    closed = False

    def __init__(self, host: str, port: int | None, timeout: int) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout

    def request(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str],
    ) -> None:
        """Record the outbound request path and core metadata."""

        assert method == "GET"
        assert "User-Agent" in headers
        _HTTPSConnection.requests.append(path)

    def getresponse(self) -> _HTTPResponse:
        """Return the configured response object."""

        return _HTTPResponse(status=self.status, payload=self.payload)

    def close(self) -> None:
        """Record that the connection was closed."""

        _HTTPSConnection.closed = True


def _heartbeat_csv(rows: tuple[tuple[float, float], ...]) -> str:
    lines = ["rr_ms,hr_bpm"]
    lines.extend(f"{rr_ms},{hr_bpm}" for rr_ms, hr_bpm in rows)
    return "\n".join(lines) + "\n"


def test_scaffold_llm_requires_description_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(main, ["scaffold", "traffic_grid", "--llm"])

    assert result.exit_code != 0
    assert "--description is required with --llm" in result.output
    assert not (tmp_path / "domainpacks" / "traffic_grid").exists()


def test_scaffold_llm_wraps_invalid_offline_provider_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    response_path = tmp_path / "response.json"
    response_path.write_text(
        json.dumps({"name": "traffic_grid", "oscillators": []}),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main,
        [
            "scaffold",
            "traffic_grid",
            "--llm",
            "--description",
            "traffic signals and queue pressure",
            "--llm-response-json",
            str(response_path),
        ],
    )

    assert result.exit_code != 0
    assert "at least one oscillator" in result.output
    assert not (tmp_path / "domainpacks" / "traffic_grid").exists()


def test_real_data_demo_rejects_invalid_controls() -> None:
    with pytest.raises(click.BadParameter, match="steps must be positive"):
        scaffold_mod._run_real_data_demo(
            dataset="heartbeat.csv",
            target="coherence",
            steps=0,
            port=8000,
        )

    with pytest.raises(click.BadParameter, match="only target=coherence"):
        scaffold_mod._run_real_data_demo(
            dataset="heartbeat.csv",
            target="phase",
            steps=1,
            port=8000,
        )


def test_load_demo_dataset_accepts_local_paths_and_https_urls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "heartbeat.csv"
    csv_text = _heartbeat_csv(((800.0, 75.0), (810.0, 74.0), (790.0, 76.0)))
    csv_path.write_text(csv_text, encoding="utf-8")

    local_text, local_source = scaffold_mod._load_demo_dataset(str(csv_path))
    assert local_text == csv_text
    assert local_source == str(csv_path)

    _HTTPSConnection.status = 200
    _HTTPSConnection.payload = b"remote,csv\n"
    _HTTPSConnection.requests = []
    _HTTPSConnection.closed = False
    monkeypatch.setattr(scaffold_mod.http.client, "HTTPSConnection", _HTTPSConnection)

    remote_text, remote_source = scaffold_mod._load_demo_dataset(
        "https://example.test/data.csv?run=1"
    )

    assert remote_text == "remote,csv\n"
    assert remote_source == "https://example.test/data.csv?run=1"
    assert _HTTPSConnection.requests == ["/data.csv?run=1"]
    assert _HTTPSConnection.closed is True


def test_load_demo_dataset_rejects_unknown_dataset() -> None:
    with pytest.raises(click.BadParameter, match="dataset must be heartbeat.csv"):
        scaffold_mod._load_demo_dataset("missing.csv")


def test_download_text_rejects_non_https_and_bad_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(click.ClickException, match="absolute HTTPS URL"):
        scaffold_mod._download_text("http://example.test/data.csv", max_bytes=16)

    monkeypatch.setattr(scaffold_mod.http.client, "HTTPSConnection", _HTTPSConnection)

    _HTTPSConnection.status = 503
    _HTTPSConnection.payload = b"unavailable"
    with pytest.raises(click.ClickException, match="HTTP 503"):
        scaffold_mod._download_text("https://example.test/data.csv", max_bytes=16)

    _HTTPSConnection.status = 200
    _HTTPSConnection.payload = b"0123456789"
    with pytest.raises(click.ClickException, match="too large"):
        scaffold_mod._download_text("https://example.test/data.csv", max_bytes=4)


def test_normalise_heartbeat_csv_limits_rows_and_writes_time_column() -> None:
    normalised = scaffold_mod._normalise_heartbeat_csv(
        _heartbeat_csv(
            (
                (800.0, 75.0),
                (810.0, 74.0),
                (790.0, 76.0),
                (805.0, 75.5),
            )
        ),
        max_rows=3,
    )

    assert normalised.splitlines() == [
        "time,rr_ms,hr_bpm",
        "0.000000,800,75",
        "0.800000,810,74",
        "1.600000,790,76",
    ]


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ("time,value\n0,1\n", "must include rr_ms and hr_bpm"),
        (_heartbeat_csv(((800.0, 75.0), (810.0, 74.0))), "at least 3 rows"),
        (_heartbeat_csv(((0.0, 75.0), (0.0, 74.0), (0.0, 76.0))), "invalid RR"),
    ],
)
def test_normalise_heartbeat_csv_rejects_malformed_payloads(
    raw: str,
    match: str,
) -> None:
    with pytest.raises(click.ClickException, match=match):
        scaffold_mod._normalise_heartbeat_csv(raw, max_rows=256)


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (None, "non-numeric rr_ms"),
        ("not-a-number", "non-numeric rr_ms"),
        ("nan", "non-finite rr_ms"),
    ],
)
def test_finite_csv_float_rejects_invalid_fields(value: object, match: str) -> None:
    with pytest.raises(click.ClickException, match=match):
        scaffold_mod._finite_csv_float(value, "rr_ms")


def test_contained_domainpack_spec_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "domainpacks"
    outside = tmp_path / "outside"
    outside.mkdir(parents=True)
    root.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(click.BadParameter, match="outside domainpack root"):
        scaffold_mod._contained_domainpack_spec(root, "escape")

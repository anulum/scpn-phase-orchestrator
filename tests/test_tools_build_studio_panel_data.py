# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for build_studio_panel_data.py

"""Unit tests for ``tools/build_studio_panel_data.py``.

Covers the snapshot writer, the ``--check`` verifier's in-sync / drift /
missing-file branches, and — as the load-bearing wire-freeze — that the
committed ``studio-web/src/panel/evidence_coverage.json`` snapshot is in sync
with the live producer, so the JavaScript remote never inlines stale evidence.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
SCRIPT = TOOLS_DIR / "build_studio_panel_data.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_build_studio_panel_data_test_mod", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load()


def _redirect(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the tool's snapshot at a temporary tree and return its path."""
    snapshot = tmp_path / "studio-web" / "src" / "panel" / "evidence_coverage.json"
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "SNAPSHOT_PATH", snapshot)
    return snapshot


def test_committed_snapshot_is_in_sync_with_the_producer() -> None:
    # The load-bearing drift-guard: the committed JS-inlined snapshot must equal
    # the live producer output, on the real repository paths.
    assert mod.main(["--check"]) == 0


def test_write_creates_the_snapshot_from_the_producer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot = _redirect(monkeypatch, tmp_path)
    assert not snapshot.exists()
    assert mod.main([]) == 0
    assert snapshot.exists()
    assert snapshot.read_text(encoding="utf-8") == mod.render_panel_data_json()


def test_check_passes_when_in_sync(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot = _redirect(monkeypatch, tmp_path)
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text(mod.render_panel_data_json(), encoding="utf-8")
    assert mod.main(["--check"]) == 0


def test_check_fails_on_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot = _redirect(monkeypatch, tmp_path)
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("{}\n", encoding="utf-8")
    assert mod.main(["--check"]) == 1
    assert "out of sync" in capsys.readouterr().err


def test_check_fails_when_snapshot_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _redirect(monkeypatch, tmp_path)
    assert mod.main(["--check"]) == 1

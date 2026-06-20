# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — WASM Kuramoto playground tests

"""Structural and behavioural checks for the WASM Kuramoto playground.

The playground page and its helper module live in the tracked
``spo-kernel/crates/spo-wasm/example`` directory. The HTML structure is checked
directly; the pure simulation helpers and the WASM integration are checked by
running the ``node --test`` suite through a subprocess so the JavaScript logic is
covered from the Python test run (and from CI) wherever Node.js is available.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DIR = REPO_ROOT / "spo-kernel" / "crates" / "spo-wasm" / "example"
INDEX_HTML = EXAMPLE_DIR / "index.html"
SIMULATION_MJS = EXAMPLE_DIR / "simulation.mjs"
SIMULATION_TEST = EXAMPLE_DIR / "simulation.test.mjs"


def test_playground_assets_exist() -> None:
    assert INDEX_HTML.is_file()
    assert SIMULATION_MJS.is_file()
    assert SIMULATION_TEST.is_file()


def test_index_html_uses_current_wasm_class_api() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    # Current API: the WasmEngine class imported from the built glue.
    assert "import init, { WasmEngine }" in html
    assert "../../../../wasm-pkg/spo_wasm.js" in html
    assert "new WasmEngine(" in html
    assert "engine.step(" in html
    # The stale free-function demo API must not return.
    assert "{ init as spo_init, step, get_phases }" not in html


def test_index_html_has_interactive_controls() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    for control_id in ('id="n"', 'id="coupling"', 'id="dt"', 'id="spread"'):
        assert control_id in html, control_id
    for button_id in ('id="play"', 'id="step"', 'id="reset"'):
        assert button_id in html, button_id
    assert 'id="ring"' in html
    assert 'id="chart"' in html
    assert "requestAnimationFrame" in html


def test_index_html_imports_shared_helpers() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert "./simulation.mjs" in html
    for helper in ("orderParameter", "meanPhase", "phasePoint", "validateParams"):
        assert helper in html, helper


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_node_simulation_suite_passes() -> None:
    node = shutil.which("node")
    assert node is not None  # guarded by the skipif above
    result = subprocess.run(  # noqa: S603 - fixed args, resolved executable, no shell
        [node, "--test", str(SIMULATION_TEST)],
        capture_output=True,
        text=True,
        cwd=str(EXAMPLE_DIR),
        check=False,
    )
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "# fail 0" in combined, combined

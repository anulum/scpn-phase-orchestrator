# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — binding-spec studio export safety tests

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "tools" / "binding_spec_studio.py"


def test_binding_spec_studio_exports_without_browser_driven_file_writes() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "st.download_button(" in source
    assert "write_text(spec_text" not in source
    assert "mkdir(parents=True" not in source
    assert "Root folder" not in source

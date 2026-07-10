#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — STUDIO panel data snapshot generator

"""Write the committed STUDIO panel evidence-coverage snapshot.

The federated studio remote (``studio-web/``) inlines
``studio-web/src/panel/evidence_coverage.json`` at build time. This script
regenerates that snapshot from the single source of truth,
:func:`scpn_phase_orchestrator.studio.panel_data.render_panel_data_json`, so the
panel always restates the repository's own evidence-to-clause coverage map. Run
it after any change to the assurance clause map:

    python tools/build_studio_panel_data.py

With ``--check`` it verifies the committed snapshot is in sync instead of
rewriting it, exiting non-zero on drift (the same guard the test suite enforces).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scpn_phase_orchestrator.studio.panel_data import render_panel_data_json

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = REPO_ROOT / "studio-web" / "src" / "panel" / "evidence_coverage.json"


def main(argv: list[str] | None = None) -> int:
    """Regenerate or check the committed panel snapshot.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        ``0`` on success, ``1`` if ``--check`` finds the snapshot out of sync.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed snapshot is in sync without rewriting it",
    )
    args = parser.parse_args(argv)

    rendered = render_panel_data_json()
    relative = SNAPSHOT_PATH.relative_to(REPO_ROOT)

    if args.check:
        current = (
            SNAPSHOT_PATH.read_text(encoding="utf-8") if SNAPSHOT_PATH.exists() else ""
        )
        if current != rendered:
            print(
                f"studio panel snapshot out of sync: {relative}\n"
                "regenerate with: python tools/build_studio_panel_data.py",
                file=sys.stderr,
            )
            return 1
        print(f"studio panel snapshot in sync: {relative}")
        return 0

    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(rendered, encoding="utf-8")
    print(f"wrote {relative}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — bench driver for detector meta-analysis

"""Generate the cross-domain detector ranking report."""

from __future__ import annotations

import sys
from pathlib import Path

from scpn_phase_orchestrator.evaluation.detector_meta_analysis import main

if __name__ == "__main__":
    # Default to repository root so the script can be invoked from anywhere.
    repo_root = Path(__file__).resolve().parent.parent
    sys.exit(
        main(
            [
                "--root",
                str(repo_root / "examples" / "real_data"),
                "--output",
                str(repo_root / "docs" / "studies" / "detector_ranking_report.md"),
            ]
        )
    )

#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - Policy YAML fuzz target

from __future__ import annotations

import os
import sys
import tempfile
from contextlib import suppress
from pathlib import Path

import atheris

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

with atheris.instrument_imports():
    from scpn_phase_orchestrator.supervisor.policy_rules import load_policy_rules


def TestOneInput(data: bytes) -> None:
    text = data[:8192].decode("utf-8", errors="ignore")
    fd, path_text = tempfile.mkstemp(prefix="spo-policy-", suffix=".yaml")
    path = Path(path_text)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        try:
            rules = load_policy_rules(path)
        except (TypeError, ValueError):
            return
        if not isinstance(rules, list):
            raise AssertionError("policy loader returned a non-list value")
    finally:
        with suppress(FileNotFoundError):
            path.unlink()


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Policy YAML fuzz target

"""Fuzz the canonical policy loader and its real imported dependencies."""

from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Callable
from contextlib import AbstractContextManager, suppress
from importlib import import_module
from pathlib import Path
from typing import NoReturn, Protocol, cast


class _Atheris(Protocol):
    """Typed interface to the compiled fuzzing engine used by this harness."""

    def instrument_imports(self) -> AbstractContextManager[object]:
        """Instrument production modules imported inside the context."""
        ...

    def Setup(self, argv: list[str], callback: Callable[[bytes], None]) -> None:
        """Bind the command-line options and byte-input callback."""
        ...

    def Fuzz(self) -> NoReturn:
        """Run until the engine exits or discovers an unhandled fault."""
        ...


atheris = cast("_Atheris", import_module("atheris"))
with atheris.instrument_imports():
    from scpn_phase_orchestrator.supervisor.policy_rules import load_policy_rules


def TestOneInput(data: bytes) -> None:
    """Exercise the production policy loader with at most 8192 input bytes.

    Parameters
    ----------
    data : bytes
        Candidate UTF-8 YAML, including malformed and adversarial input.
    """
    text = data[:8192].decode("utf-8", errors="ignore")
    fd, path_text = tempfile.mkstemp(prefix="spo-policy-", suffix=".yaml")
    path = Path(path_text)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        try:
            load_policy_rules(path)
        except (TypeError, ValueError):
            return
    finally:
        with suppress(FileNotFoundError):
            path.unlink()


def main() -> None:
    """Start Atheris with the production policy-loader callback."""
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — documented CLI example drift guards

"""Keep first-run documentation commands aligned with the live Click app."""

from __future__ import annotations

import shlex
from collections.abc import Iterator
from pathlib import Path

import click

from scpn_phase_orchestrator.runtime.cli import main

ROOT = Path(__file__).resolve().parents[1]
DOC_PATHS = (
    ROOT / "README.md",
    *(ROOT / "docs" / "getting-started").glob("*.md"),
    *(ROOT / "docs" / "tutorials").glob("*.md"),
)


def _logical_lines(path: Path) -> Iterator[str]:
    pending = ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        pending = f"{pending} {stripped}".strip() if pending else stripped
        if pending.endswith("\\"):
            pending = pending[:-1].rstrip()
            continue
        yield pending
        pending = ""
    if pending:
        yield pending


def _documented_commands() -> Iterator[tuple[Path, str, tuple[str, ...]]]:
    for path in DOC_PATHS:
        for line in _logical_lines(path):
            if not line.startswith("spo "):
                continue
            tokens = tuple(shlex.split(line))
            if len(tokens) < 2 or tokens[1].startswith("-"):
                continue
            yield (
                path,
                tokens[1],
                tuple(
                    token.split("=", maxsplit=1)[0]
                    for token in tokens[2:]
                    if token.startswith("--")
                ),
            )


def _option_names(command: click.Command) -> set[str]:
    names: set[str] = set()
    for parameter in command.params:
        names.update(getattr(parameter, "opts", ()))
        names.update(getattr(parameter, "secondary_opts", ()))
    return names


def test_documented_top_level_commands_and_options_exist() -> None:
    seen: set[str] = set()
    problems: list[str] = []
    for path, name, options in _documented_commands():
        command = main.commands.get(name)
        if command is None:
            problems.append(f"{path.relative_to(ROOT)}: unknown command {name}")
            continue
        seen.add(name)
        valid_options = _option_names(command)
        for option in options:
            if option not in valid_options:
                problems.append(
                    f"{path.relative_to(ROOT)}: {name} has no option {option}"
                )

    assert not problems, problems
    assert {
        "demo",
        "doctor",
        "quickstart",
        "replay",
        "run",
        "scaffold",
        "validate",
    } <= seen

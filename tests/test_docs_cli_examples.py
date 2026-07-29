# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — documented CLI example drift guards

"""Keep first-run documentation commands aligned with the live Click app."""

from __future__ import annotations

import re
import shlex
from collections.abc import Iterator
from pathlib import Path

import click

from scpn_phase_orchestrator.runtime.cli import main

ROOT = Path(__file__).resolve().parents[1]


def _public_doc_paths() -> tuple[Path, ...]:
    docs = (
        path
        for path in (ROOT / "docs").rglob("*.md")
        if "internal" not in path.parts and "superpowers" not in path.parts
    )
    return (ROOT / "README.md", *sorted(docs))


DOC_PATHS = _public_doc_paths()


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


def _documented_commands() -> Iterator[tuple[Path, tuple[str, ...], tuple[str, ...]]]:
    for path in DOC_PATHS:
        for line in _logical_lines(path):
            if not line.startswith("spo "):
                continue
            tokens = tuple(shlex.split(line))
            if len(tokens) < 2 or tokens[1].startswith("-"):
                continue
            command_path = [tokens[1]]
            command = main.commands.get(tokens[1])
            cursor = 2
            while isinstance(command, click.Group) and cursor < len(tokens):
                subcommand = command.commands.get(tokens[cursor])
                if subcommand is None:
                    break
                command_path.append(tokens[cursor])
                command = subcommand
                cursor += 1
            yield (
                path,
                tuple(command_path),
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
    for path, command_path, options in _documented_commands():
        name = command_path[0]
        label = " ".join(command_path)
        relative = path.relative_to(ROOT)
        command = main.commands.get(name)
        if command is None:
            problems.append(f"{relative}: unknown command {name}")
            continue
        seen.add(name)
        valid_options = _option_names(command)
        for subcommand_name in command_path[1:]:
            if not isinstance(command, click.Group):
                problems.append(f"{relative}: {label} is not a command")
                break
            command = command.commands.get(subcommand_name)
            if command is None:
                problems.append(f"{relative}: unknown command {label}")
                break
            valid_options.update(_option_names(command))
        for option in options:
            if option not in valid_options:
                problems.append(f"{relative}: {label} has no option {option}")

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


def test_every_public_literal_spo_command_exists() -> None:
    problems: list[str] = []
    for path in DOC_PATHS:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for match in re.finditer(r"(?<![\w-])spo\s+([a-z][a-z0-9-]*)", line):
                name = match.group(1)
                if name not in main.commands:
                    problems.append(
                        f"{path.relative_to(ROOT)}:{lineno}: unknown command {name}"
                    )

    assert not problems, problems

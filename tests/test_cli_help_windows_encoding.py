# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI help encoding tests

"""Every ``spo`` help page can be written to a Windows ANSI stream.

On Windows, Python writes redirected or piped output in the ANSI code page
(cp1252 on the CI runners and most Western installations). A help text with a
character outside that code page makes ``spo --help > help.txt`` stop with
``UnicodeEncodeError``. The quickstart summary carried a ``→`` and did.
"""

from __future__ import annotations

import click

from scpn_phase_orchestrator.runtime.cli import main


def _help_pages() -> list[tuple[str, str]]:
    """Return ``(command path, rendered help)`` for every CLI command."""
    pages: list[tuple[str, str]] = []

    def walk(command: click.Command, path: tuple[str, ...]) -> None:
        name = " ".join(path)
        pages.append((name, command.get_help(click.Context(command, info_name=name))))
        if isinstance(command, click.Group):
            for sub_name, sub_command in command.commands.items():
                walk(sub_command, (*path, sub_name))

    walk(main, ("spo",))
    return pages


def _outside_cp1252(text: str) -> list[str]:
    """Return the distinct characters of ``text`` that cp1252 cannot encode."""
    missing: set[str] = set()
    for char in text:
        try:
            char.encode("cp1252")
        except UnicodeEncodeError:
            missing.add(char)
    return sorted(missing)


def test_every_help_page_encodes_to_cp1252() -> None:
    """No help page holds a character the Windows ANSI code page lacks."""
    pages = _help_pages()
    assert len(pages) > 40

    unencodable = {name: _outside_cp1252(text) for name, text in pages}

    assert {name: chars for name, chars in unencodable.items() if chars} == {}

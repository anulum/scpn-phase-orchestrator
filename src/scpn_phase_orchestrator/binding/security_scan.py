# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — hard security scan of a domainpack's files

"""Scan a domainpack's user-facing files for dangerous code and config patterns.

``spo validate --security`` rejects executable-looking payloads in the binding
spec itself. The harder ``--security --hard`` pass goes one level further and
statically scans the *files* that ship beside the binding — the domainpack's
Python scenarios and YAML configuration — for the patterns that let an untrusted
domainpack run arbitrary code when it is loaded or executed: dynamic evaluation
(``eval`` / ``exec``), insecure deserialisation (``pickle.load``), unsafe YAML
deep loads (``yaml.load`` without a safe loader, or ``!!python/`` construction
tags), and shell command execution.

The scan is review-only: it reports every match with its file, line and category;
it never edits, executes, or imports the scanned files, so it is safe to run on a
domainpack of unknown provenance before deciding whether to trust it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

__all__ = ["UnsafePatternFinding", "scan_unsafe_patterns"]

_PYTHON_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\beval\s*\("), "dynamic-eval"),
    (re.compile(r"\bexec\s*\("), "dynamic-exec"),
    (re.compile(r"\bpickle\s*\.\s*loads?\s*\("), "insecure-deserialization"),
    (re.compile(r"\bos\s*\.\s*system\s*\("), "shell-exec"),
    (re.compile(r"\bsubprocess\b.*shell\s*=\s*True"), "shell-exec"),
)
_UNSAFE_YAML_LOAD = re.compile(r"\byaml\s*\.\s*load\s*\(")
_SAFE_YAML_MARKER = re.compile(r"safe_load|SafeLoader|CSafeLoader")
_YAML_PYTHON_TAG = re.compile(r"!!python/")
_SCANNED_SUFFIXES = frozenset({".py", ".yaml", ".yml"})


@dataclass(frozen=True)
class UnsafePatternFinding:
    """One dangerous pattern located during a hard security scan.

    Parameters
    ----------
    path : str
        The scanned file, relative to the scan root.
    line : int
        The one-based line number of the match.
    category : str
        The danger class, for example ``"dynamic-eval"`` or ``"unsafe-yaml"``.
    snippet : str
        The stripped source line containing the match.
    """

    path: str
    line: int
    category: str
    snippet: str


def _scan_python_line(line: str) -> str | None:
    """Scan one Python source line for security markers."""
    for pattern, category in _PYTHON_PATTERNS:
        if pattern.search(line):
            return category
    if _UNSAFE_YAML_LOAD.search(line) and not _SAFE_YAML_MARKER.search(line):
        return "unsafe-yaml"
    return None


def scan_unsafe_patterns(root: Path) -> tuple[UnsafePatternFinding, ...]:
    """Scan a directory tree for dangerous code and configuration patterns.

    Parameters
    ----------
    root : pathlib.Path
        The directory (a domainpack) or single file to scan.

    Returns
    -------
    tuple[UnsafePatternFinding, ...]
        Every located pattern, ordered by path then line.

    Raises
    ------
    ValueError
        If ``root`` does not exist.
    """
    if not root.exists():
        raise ValueError(f"scan root does not exist: {root}")
    base = root if root.is_dir() else root.parent
    candidates = [root] if root.is_file() else sorted(root.rglob("*"))
    findings: list[UnsafePatternFinding] = []
    for path in candidates:
        if not path.is_file() or path.suffix not in _SCANNED_SUFFIXES:
            continue
        relative = path.relative_to(base).as_posix() if path != base else path.name
        text = path.read_text(encoding="utf-8", errors="replace")
        for index, line in enumerate(text.splitlines(), start=1):
            category = (
                _scan_python_line(line)
                if path.suffix == ".py"
                else ("unsafe-yaml-tag" if _YAML_PYTHON_TAG.search(line) else None)
            )
            if category is not None:
                findings.append(
                    UnsafePatternFinding(relative, index, category, line.strip())
                )
    return tuple(findings)

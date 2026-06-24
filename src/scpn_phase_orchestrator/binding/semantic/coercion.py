# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Semantic compiler input coercion and validation

"""Prompt, name, path, and bound coercion plus compilation-input validation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")


_MAX_OSCILLATORS_PER_LAYER = 256


_MAX_DRY_RUN_STEPS = 256


_MAX_PROMPT_CHARS = 4000


_PROMPT_INJECTION_PATTERNS = (
    re.compile(r"\bignore\s+(all\s+)?previous\s+instructions\b", re.IGNORECASE),
    re.compile(r"\bdisregard\s+(all\s+)?previous\s+instructions\b", re.IGNORECASE),
    re.compile(r"\breveal\s+(the\s+)?system\s+prompt\b", re.IGNORECASE),
    re.compile(r"\bdeveloper\s+instructions\b", re.IGNORECASE),
    re.compile(r"\bexfiltrat(e|ion)\b", re.IGNORECASE),
    re.compile(r"\bleak\s+(credentials|secrets|api\s+keys)\b", re.IGNORECASE),
    re.compile(r"\bprint\s+(credentials|secrets|api\s+keys)\b", re.IGNORECASE),
)


def _validate_compilation_inputs(
    *,
    prompt: Any,
    name: Any,
    oscillators_per_layer: Any,
    dry_run_steps: Any,
    retrieval_root: str | Path | None,
    docs_root: str | Path | None,
) -> tuple[str, str, int, int, Path | None, Path | None]:
    """Validate the compilation inputs, else raise."""
    prompt_value = _as_prompt(prompt)
    name_value = _as_name(name)
    oscillators = _as_positive_int(
        oscillators_per_layer,
        "oscillators_per_layer",
        max_value=_MAX_OSCILLATORS_PER_LAYER,
    )
    dry_run = _as_positive_int(
        dry_run_steps,
        "dry_run_steps",
        max_value=_MAX_DRY_RUN_STEPS,
    )
    retrieval_path = _as_path(
        retrieval_root,
        "retrieval_root",
        allow_none=True,
    )
    docs_path = _as_path(
        docs_root,
        "docs_root",
        allow_none=True,
    )
    return (
        prompt_value,
        name_value,
        oscillators,
        dry_run,
        retrieval_path,
        docs_path,
    )


def _as_str(value: object, field_name: str) -> str:
    """Return ``value`` as a string, else raise ``ValueError``."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    return value


def _as_prompt(value: object) -> str:
    """Return ``value`` as a validated prompt string, else raise."""
    prompt = _as_str(value, "prompt").replace("\r\n", "\n").replace("\r", "\n")
    if len(prompt) > _MAX_PROMPT_CHARS:
        raise ValueError(f"prompt must be <= {_MAX_PROMPT_CHARS} characters")
    for char in prompt:
        codepoint = ord(char)
        if codepoint < 32 and char not in "\n\t":
            raise ValueError("prompt contains unsupported control characters")
    normalised = " ".join(prompt.split())
    for pattern in _PROMPT_INJECTION_PATTERNS:
        if pattern.search(normalised):
            raise ValueError("prompt contains instruction-injection markers")
    return normalised


def _as_name(value: object) -> str:
    """Return ``value`` as a validated name, else raise."""
    value = _as_str(value, "name")
    if not _NAME_PATTERN.fullmatch(value):
        raise ValueError("name must match [A-Za-z][A-Za-z0-9_-]{0,63}")
    return value


def _as_positive_int(value: object, field_name: str, *, max_value: int) -> int:
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 1:
        raise ValueError(f"{field_name} must be >= 1")
    if value > max_value:
        raise ValueError(f"{field_name} must be <= {max_value}")
    return value


def _as_path(
    value: str | Path | None,
    field_name: str,
    *,
    allow_none: bool = False,
) -> Path | None:
    """Return ``value`` as a validated filesystem path, else raise."""
    if value is None:
        if not allow_none:
            raise TypeError(f"{field_name} must be a string, pathlib.Path, or None")
        return None
    if isinstance(value, Path):
        path = value
    elif isinstance(value, str):
        if not value.strip():
            raise ValueError(f"{field_name} must be a non-empty path")
        path = Path(value)
    else:
        raise TypeError(f"{field_name} must be a string, pathlib.Path, or None")
    if path.exists() and not path.is_dir():
        raise ValueError(f"{field_name} must be a directory when provided")
    return path


def _coerce_output_dir(output_dir: str | Path) -> Path:
    """Return the validated output directory, else raise."""
    if not isinstance(output_dir, (str, Path)):
        raise TypeError("output_dir must be a string or pathlib.Path")
    path = Path(output_dir)
    if path.exists() and not path.is_dir():
        raise ValueError("output_dir must be a directory path")
    return path

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — semantic compiler coercion validation tests

"""Validation tests for the semantic-compiler input coercion helpers.

Exercises the over-length prompt guard, the upper-bound integer guard, the
disallowed-``None`` and empty-string path guards, and the output-directory
type guard.
"""

from __future__ import annotations

from typing import cast

import pytest

import scpn_phase_orchestrator.binding.semantic.coercion as _coercion
from scpn_phase_orchestrator.binding.semantic.coercion import (
    _MAX_PROMPT_CHARS,
    _as_path,
    _as_positive_int,
    _as_prompt,
    _coerce_output_dir,
)

assert _coercion is not None


def test_prompt_rejects_an_over_length_value() -> None:
    with pytest.raises(ValueError, match="prompt must be <= 4000 characters"):
        _as_prompt("a" * (_MAX_PROMPT_CHARS + 1))


def test_positive_int_rejects_a_value_above_the_maximum() -> None:
    with pytest.raises(ValueError, match="oscillators_per_layer must be <= 256"):
        _as_positive_int(300, "oscillators_per_layer", max_value=256)


def test_path_rejects_none_when_not_allowed() -> None:
    with pytest.raises(TypeError, match="must be a string, pathlib.Path, or None"):
        _as_path(None, "retrieval_root", allow_none=False)


def test_path_rejects_a_blank_string() -> None:
    with pytest.raises(ValueError, match="docs_root must be a non-empty path"):
        _as_path("   ", "docs_root")


def test_output_dir_rejects_a_non_path_type() -> None:
    with pytest.raises(TypeError, match="output_dir must be a string or pathlib.Path"):
        _coerce_output_dir(cast("str", 123))

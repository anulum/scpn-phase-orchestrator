# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI value-coercion helper tests

"""Tests for the defensive value-coercion helpers shared by the CLI commands.

These helpers turn loosely-typed audit-record fields into typed values, returning
a safe default rather than raising when the field is missing or the wrong type, so
both branches of each helper are pinned here.
"""

from __future__ import annotations

from scpn_phase_orchestrator.runtime.cli._shared import (
    _count_dict,
    _float_list,
    _float_value,
    _int_value,
    _string_list,
)


def test_string_list_coerces_items_and_defaults_to_empty() -> None:
    assert _string_list([1, "a", 2.5]) == ["1", "a", "2.5"]
    assert _string_list("not-a-list") == []


def test_float_list_filters_non_numeric_and_defaults_to_empty() -> None:
    assert _float_list([1, 2.5, "x", None]) == [1.0, 2.5]
    assert _float_list({"a": 1}) == []


def test_float_value_coerces_numbers_and_defaults_to_zero() -> None:
    assert _float_value(3) == 3.0
    assert _float_value(2.5) == 2.5
    assert _float_value("nope") == 0.0


def test_int_value_passes_ints_and_defaults_to_zero() -> None:
    assert _int_value(7) == 7
    assert _int_value(3.0) == 0
    assert _int_value("nope") == 0


def test_count_dict_keeps_int_counts_and_defaults_to_empty() -> None:
    assert _count_dict({"a": 2, "b": "x", 3: 4}) == {"a": 2, "3": 4}
    assert _count_dict(["not", "a", "dict"]) == {}

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CI Category Result Gate Tests
"""Exercise the aggregate CI gate against every terminal category state."""

from __future__ import annotations

import json

import pytest

from tools import check_ci_category_results as gate


def _results(**states: str) -> str:
    return json.dumps({name: {"result": state} for name, state in states.items()})


def test_all_categories_must_report_success() -> None:
    assert gate.unsuccessful_categories(_results(lint="success", test="success")) == {}


@pytest.mark.parametrize("state", ["failure", "cancelled", "skipped"])
def test_non_success_terminal_states_fail_closed(state: str) -> None:
    assert gate.unsuccessful_categories(_results(category=state)) == {"category": state}


def test_missing_and_malformed_results_fail_closed() -> None:
    assert gate.unsuccessful_categories('{"absent": {}}') == {"absent": "missing"}
    assert gate.unsuccessful_categories('{"bad": "success"}') == {"bad": "malformed"}


@pytest.mark.parametrize("raw", ["[]", "{}", "null", "not-json"])
def test_invalid_result_projection_is_rejected(raw: str) -> None:
    with pytest.raises((json.JSONDecodeError, ValueError)):
        gate.unsuccessful_categories(raw)


def test_main_reports_failure_without_results(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("CATEGORY_RESULTS", raising=False)
    assert gate.main() == 1
    assert "not set" in capsys.readouterr().out


def test_main_reports_each_failed_category(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(
        "CATEGORY_RESULTS",
        _results(static="success", security="failure", coverage="cancelled"),
    )
    assert gate.main() == 1
    output = capsys.readouterr().out
    assert "coverage: cancelled" in output
    assert "security: failure" in output


def test_main_accepts_complete_success_projection(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("CATEGORY_RESULTS", _results(static="success", tests="success"))
    assert gate.main() == 0
    assert "All CI categories succeeded" in capsys.readouterr().out

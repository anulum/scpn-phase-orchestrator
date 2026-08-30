# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Lightweight reactor-semantics import tests

"""Prove that the portable semantic codec has no accelerator import side effects."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import textwrap

import pytest

import scpn_phase_orchestrator as spo


def _run_isolated(source: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-I", "-c", textwrap.dedent(source)],
        check=True,
        capture_output=True,
        text=True,
        timeout=15.0,
    )
    return json.loads(completed.stdout)


def test_reactor_semantics_import_does_not_resolve_optional_runtime_graph() -> None:
    payload = _run_isolated(
        """
        import json
        import sys

        import scpn_phase_orchestrator.reactor_semantics as semantics

        forbidden = (
            "juliacall",
            "juliapkg",
            "scpn_phase_orchestrator.api",
            "scpn_phase_orchestrator.supervisor",
            "scpn_phase_orchestrator.upde._run",
            "scpn_phase_orchestrator.experimental.accelerators",
        )
        loaded = sorted(
            name
            for name in sys.modules
            if any(name == item or name.startswith(f"{item}.") for item in forbidden)
        )
        print(json.dumps({
            "codec_callable": callable(semantics.handoff_from_bytes),
            "forbidden_modules": loaded,
        }, sort_keys=True))
        """
    )

    assert payload == {"codec_callable": True, "forbidden_modules": []}


def test_package_root_resolves_and_caches_only_requested_compatibility_export() -> None:
    payload = _run_isolated(
        """
        import json
        import sys

        import scpn_phase_orchestrator as spo

        before = "SPOError" in vars(spo)
        resolved = spo.SPOError
        after = "SPOError" in vars(spo)
        try:
            spo.not_a_public_export
        except AttributeError as exc:
            unknown_error = str(exc)
        else:
            raise AssertionError("unknown package export unexpectedly resolved")
        print(json.dumps({
            "after": after,
            "api_loaded": "scpn_phase_orchestrator.api" in sys.modules,
            "before": before,
            "cached": spo.SPOError is resolved,
            "resolved_name": resolved.__name__,
            "unknown_error": unknown_error,
        }, sort_keys=True))
        """
    )

    assert payload == {
        "after": True,
        "api_loaded": False,
        "before": False,
        "cached": True,
        "resolved_name": "SPOError",
        "unknown_error": (
            "module 'scpn_phase_orchestrator' has no attribute 'not_a_public_export'"
        ),
    }


def test_package_root_lazy_resolver_caches_lists_and_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(vars(spo), "SPOError", raising=False)

    assert set(vars(spo)["_LAZY_EXPORTS"]) == set(spo.__all__)
    assert "SPOError" in dir(spo)
    resolved = spo.__getattr__("SPOError")
    assert resolved.__name__ == "SPOError"
    assert spo.SPOError is resolved

    with pytest.raises(
        AttributeError,
        match="has no attribute 'not_a_public_export'",
    ):
        spo.__getattr__("not_a_public_export")


def test_package_root_preserves_operator_julia_signal_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHON_JULIACALL_HANDLE_SIGNALS", "no")

    assert importlib.reload(spo) is spo
    assert os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] == "no"

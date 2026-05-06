# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Policy YAML fuzzer tests

from __future__ import annotations

import contextlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest


def test_policy_yaml_fuzzer_loads_policy_rules_with_stubs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    atheris = types.ModuleType("atheris")
    atheris.instrument_imports = lambda: contextlib.nullcontext()
    atheris.Setup = lambda *args, **kwargs: None
    atheris.Fuzz = lambda: None
    monkeypatch.setitem(sys.modules, "atheris", atheris)

    path = Path("fuzzers/policy_yaml_fuzzer.py")
    spec = importlib.util.spec_from_file_location("_policy_yaml_fuzzer_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    load_policy_rules = module._load_policy_rules()

    assert load_policy_rules.__name__ == "load_policy_rules"
    assert "scpn_phase_orchestrator.monitor.stl" in sys.modules

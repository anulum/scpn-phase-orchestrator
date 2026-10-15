# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Policy YAML fuzzer tests

"""Check canonical parsing while isolating the instrumentation boundary."""

from __future__ import annotations

import contextlib
import importlib.util
import sys
import types
from pathlib import Path
from types import FrameType

import pytest

from scpn_phase_orchestrator.supervisor.policy_rules import load_policy_rules


def test_policy_yaml_fuzzer_executes_canonical_policy_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observe real parsing; the hosted image separately exercises Atheris."""
    atheris = types.ModuleType("atheris")
    monkeypatch.setattr(
        atheris, "instrument_imports", contextlib.nullcontext, raising=False
    )
    monkeypatch.setitem(sys.modules, "atheris", atheris)

    root = Path(__file__).resolve().parents[1]
    path = root / "fuzzers/policy_yaml_fuzzer.py"
    spec = importlib.util.spec_from_file_location("_policy_yaml_fuzzer_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.load_policy_rules is load_policy_rules

    policy = root / "domainpacks/minimal_domain/policy.yaml"
    expected_rules = load_policy_rules(policy)
    assert expected_rules
    observed_results: list[object] = []

    def observe(frame: FrameType, event: str, result: object) -> None:
        if frame.f_code is load_policy_rules.__code__ and event == "return":
            observed_results.append(result)

    previous_profile = sys.getprofile()
    sys.setprofile(observe)
    try:
        module.TestOneInput(policy.read_bytes())
    finally:
        sys.setprofile(previous_profile)
    assert observed_results == [expected_rules]

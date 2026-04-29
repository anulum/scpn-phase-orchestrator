#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - Policy YAML fuzz target

from __future__ import annotations

import os
import sys
import tempfile
import types
from contextlib import suppress
from importlib import util
from pathlib import Path

import atheris

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"


def _install_policy_rule_stubs() -> None:
    package = types.ModuleType("scpn_phase_orchestrator")
    package.__path__ = []
    sys.modules.setdefault("scpn_phase_orchestrator", package)

    for name in (
        "scpn_phase_orchestrator.actuation",
        "scpn_phase_orchestrator.supervisor",
        "scpn_phase_orchestrator.upde",
    ):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules.setdefault(name, module)

    mapper = types.ModuleType("scpn_phase_orchestrator.actuation.mapper")

    class ControlAction:
        def __init__(self, **kwargs: object) -> None:
            self.__dict__.update(kwargs)

    mapper.ControlAction = ControlAction
    sys.modules.setdefault("scpn_phase_orchestrator.actuation.mapper", mapper)

    regimes = types.ModuleType("scpn_phase_orchestrator.supervisor.regimes")

    class Regime:
        value = "NOMINAL"

    regimes.Regime = Regime
    sys.modules.setdefault("scpn_phase_orchestrator.supervisor.regimes", regimes)

    metrics = types.ModuleType("scpn_phase_orchestrator.upde.metrics")

    class UPDEState:
        pass

    metrics.UPDEState = UPDEState
    sys.modules.setdefault("scpn_phase_orchestrator.upde.metrics", metrics)


def _policy_rules_path() -> Path:
    bundle_root = Path(getattr(sys, "_MEIPASS", PROJECT_ROOT))
    bundled = (
        bundle_root
        / "src"
        / "scpn_phase_orchestrator"
        / "supervisor"
        / "policy_rules.py"
    )
    if bundled.exists():
        return bundled
    return SRC_ROOT / "scpn_phase_orchestrator" / "supervisor" / "policy_rules.py"


def _load_policy_rules() -> object:
    _install_policy_rule_stubs()
    spec = util.spec_from_file_location("_spo_policy_rules_fuzz", _policy_rules_path())
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load policy_rules.py")
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.load_policy_rules


with atheris.instrument_imports():
    load_policy_rules = _load_policy_rules()


def TestOneInput(data: bytes) -> None:
    text = data[:8192].decode("utf-8", errors="ignore")
    fd, path_text = tempfile.mkstemp(prefix="spo-policy-", suffix=".yaml")
    path = Path(path_text)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        try:
            rules = load_policy_rules(path)
        except (TypeError, ValueError):
            return
        if not isinstance(rules, list):
            raise AssertionError("policy loader returned a non-list value")
    finally:
        with suppress(FileNotFoundError):
            path.unlink()


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()

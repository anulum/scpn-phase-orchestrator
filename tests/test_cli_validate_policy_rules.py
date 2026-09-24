# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — spo validate checks the policy.yaml spo run loads

"""``spo validate`` must refuse the ``policy.yaml`` that ``spo run`` refuses.

``spo run`` loads the ``policy.yaml`` next to the binding spec; ``spo validate``
never looked at it, so a pack with a malformed policy printed "Valid" and
then failed on ``spo run`` with "invalid policy rules".
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from scpn_phase_orchestrator.runtime.cli import main

_REPO = Path(__file__).resolve().parents[1]
_PACKS = sorted((_REPO / "domainpacks").glob("*/binding_spec.yaml"))
_BAD_POLICY = """
rules:
  - name: broken
    regime: [NOMINAL]
    condition: {metric: R_good, layer: 0, op: "~", threshold: 0.5}
    action: {knob: zeta, scope: global, value: 0.1, ttl_s: 1.0}
"""


def _pack_with_policy(tmp_path: Path, policy: str) -> Path:
    spec = tmp_path / "binding_spec.yaml"
    shutil.copyfile(
        _REPO / "domainpacks" / "minimal_domain" / "binding_spec.yaml", spec
    )
    (tmp_path / "policy.yaml").write_text(policy, encoding="utf-8")
    return spec


def test_malformed_policy_fails_validate_like_run(tmp_path: Path) -> None:
    spec = _pack_with_policy(tmp_path, _BAD_POLICY)
    validate = CliRunner().invoke(main, ["validate", str(spec)])
    run = CliRunner().invoke(main, ["run", str(spec), "--steps", "2"])
    assert validate.exit_code != 0
    assert "policy.yaml: invalid policy rules" in validate.output
    assert "Valid" not in validate.output
    assert run.exit_code != 0


def test_unparseable_policy_fails_validate(tmp_path: Path) -> None:
    spec = _pack_with_policy(tmp_path, "rules: [unclosed\n")
    validate = CliRunner().invoke(main, ["validate", str(spec)])
    assert validate.exit_code != 0
    assert "policy.yaml:" in validate.output


@pytest.mark.parametrize("path", _PACKS, ids=[p.parent.name for p in _PACKS])
def test_shipped_pack_validates_with_its_policy(path: Path) -> None:
    result = CliRunner().invoke(main, ["validate", str(path)])
    assert result.exit_code == 0, result.output

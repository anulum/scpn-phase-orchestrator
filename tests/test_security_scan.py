# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — hard security scan tests

"""Tests for the domainpack hard security scan.

The scan is checked to flag every dangerous category, to ignore the safe
counterparts (safe YAML loading, ordinary code), to handle a single file and an
empty tree, and to reject a missing root.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scpn_phase_orchestrator.binding.security_scan import (
    UnsafePatternFinding,
    scan_unsafe_patterns,
)

_DANGEROUS_PY = (
    "import pickle, yaml, os, subprocess\n"
    "a = eval(user_input)\n"
    "exec(payload)\n"
    "b = pickle.loads(blob)\n"
    "c = yaml.load(text)\n"
    "os.system('rm -rf /')\n"
    "subprocess.run(cmd, shell=True)\n"
)
_SAFE_PY = (
    "import yaml\n"
    "cfg = yaml.safe_load(open('x'))\n"
    "cfg2 = yaml.load(open('y'), Loader=yaml.SafeLoader)\n"
    "value = evaluate(metric)\n"  # a function whose name merely contains 'eval'
)


def _domainpack(tmp_path: Path) -> Path:
    (tmp_path / "run.py").write_text(_DANGEROUS_PY, encoding="utf-8")
    (tmp_path / "safe.py").write_text(_SAFE_PY, encoding="utf-8")
    (tmp_path / "policy.yaml").write_text(
        "rules:\n  - x: !!python/object/apply:os.system ['id']\n", encoding="utf-8"
    )
    (tmp_path / "clean.yml").write_text("rules:\n  - name: ok\n", encoding="utf-8")
    return tmp_path


def test_scan_flags_every_dangerous_category(tmp_path: Path) -> None:
    findings = scan_unsafe_patterns(_domainpack(tmp_path))
    categories = {finding.category for finding in findings}
    assert categories == {
        "dynamic-eval",
        "dynamic-exec",
        "insecure-deserialization",
        "shell-exec",
        "unsafe-yaml",
        "unsafe-yaml-tag",
    }
    assert all(isinstance(finding, UnsafePatternFinding) for finding in findings)


def test_scan_records_path_line_and_snippet(tmp_path: Path) -> None:
    findings = scan_unsafe_patterns(_domainpack(tmp_path))
    eval_finding = next(f for f in findings if f.category == "dynamic-eval")
    assert eval_finding.path == "run.py"
    assert eval_finding.line == 2
    assert eval_finding.snippet == "a = eval(user_input)"


def test_scan_ignores_safe_yaml_and_lookalike_names(tmp_path: Path) -> None:
    (tmp_path / "safe.py").write_text(_SAFE_PY, encoding="utf-8")
    (tmp_path / "clean.yml").write_text("rules:\n  - name: ok\n", encoding="utf-8")
    assert scan_unsafe_patterns(tmp_path) == ()


def test_scan_accepts_a_single_file(tmp_path: Path) -> None:
    target = tmp_path / "run.py"
    target.write_text("x = eval(z)\n", encoding="utf-8")
    findings = scan_unsafe_patterns(target)
    assert len(findings) == 1
    assert findings[0].path == "run.py"
    assert findings[0].category == "dynamic-eval"


def test_scan_skips_non_scanned_suffixes(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("eval(this is text)\n", encoding="utf-8")
    assert scan_unsafe_patterns(tmp_path) == ()


def test_scan_rejects_a_missing_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        scan_unsafe_patterns(tmp_path / "absent")

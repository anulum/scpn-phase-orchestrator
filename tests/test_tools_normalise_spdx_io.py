# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for normalise_spdx_headers.py (I/O paths)

"""I/O behaviour tests for ``tools/normalise_spdx_headers.py``.

Complements ``test_tools_normalise_spdx.py`` (which covers the pure-
function split / exclusion / iteration surface). This file focuses
on the file-touching paths: ``normalise_file`` (dry-run vs apply,
typo fix, HEADER_SCAN_LIMIT boundary), and the ``verify`` exit-code
behaviour.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"


def _load_normaliser() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_normalise_spdx_io_test_mod",
        TOOLS_DIR / "normalise_spdx_headers.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load_normaliser()


def _sample_merged(typo: bool = False) -> str:
    email = "protoscience@" + "any" + "lum.li" if typo else "protoscience@anulum.li"
    return (
        "# SPDX-License-Identifier: AGPL-3.0-or-later "
        "| Commercial license available\n"
        "# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.\n"
        "# © Code 2020–2026 Miroslav Šotek. All rights reserved.\n"
        "# ORCID: 0009-0009-3560-0851\n"
        f"# Contact: www.anulum.li | {email}\n"
        "# SCPN Phase Orchestrator — example\n\n"
        "print('hello')\n"
    )


# ---------------------------------------------------------------------
# normalise_file
# ---------------------------------------------------------------------


def test_dry_run_does_not_modify_file(tmp_path: Path) -> None:
    f = tmp_path / "sample.py"
    original = _sample_merged()
    f.write_text(original, encoding="utf-8")
    result = mod.normalise_file(f, apply=False)
    assert result.spdx_split is True
    assert f.read_text(encoding="utf-8") == original


def test_apply_splits_and_fixes_typo(tmp_path: Path) -> None:
    f = tmp_path / "sample.py"
    f.write_text(_sample_merged(typo=True), encoding="utf-8")
    result = mod.normalise_file(f, apply=True)
    assert result.spdx_split is True
    assert result.typo_fixes == 1
    content = f.read_text(encoding="utf-8")
    assert "SPDX-License-Identifier: AGPL-3.0-or-later\n" in content
    assert "Commercial license available\n" in content
    assert "| Commercial license available" not in content
    assert "protoscience@" + "any" + "lum.li" not in content
    assert "protoscience@anulum.li" in content


def test_deep_merged_line_not_split(tmp_path: Path) -> None:
    """Merged SPDX past HEADER_SCAN_LIMIT stays put — the header-only
    invariant keeps the tool focused and avoids touching user strings."""
    padding = "\n".join(f"# pad line {i}" for i in range(15)) + "\n"
    merged = (
        "# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available\n"
    )
    content = padding + merged
    f = tmp_path / "deep.py"
    f.write_text(content, encoding="utf-8")
    result = mod.normalise_file(f, apply=True)
    assert result.spdx_split is False
    assert f.read_text(encoding="utf-8") == content


def test_typo_anywhere_is_fixed_even_without_split(tmp_path: Path) -> None:
    """The typo fix runs independently of the SPDX split — files that
    are already split but contain the old email still get corrected."""
    typo = "protoscience@" + "any" + "lum.li"
    content = (
        "# SPDX-License-Identifier: AGPL-3.0-or-later\n"
        "# Commercial license available\n"
        f"# Contact: {typo}\n\n"
        "print('ok')\n"
    )
    f = tmp_path / "already_split.py"
    f.write_text(content, encoding="utf-8")
    result = mod.normalise_file(f, apply=True)
    assert result.spdx_split is False
    assert result.typo_fixes == 1
    assert typo not in f.read_text(encoding="utf-8")
    assert "protoscience@anulum.li" in f.read_text(encoding="utf-8")


def test_no_change_file_reports_no_change(tmp_path: Path) -> None:
    content = "# SPDX-License-Identifier: MIT\n\nprint('x')\n"
    f = tmp_path / "unrelated.py"
    f.write_text(content, encoding="utf-8")
    result = mod.normalise_file(f, apply=True)
    assert result.spdx_split is False
    assert result.typo_fixes == 0
    assert result.changed is False
    assert f.read_text(encoding="utf-8") == content


# ---------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------


def test_verify_clean_returns_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    f = tmp_path / "a.py"
    f.write_text(
        "# SPDX-License-Identifier: AGPL-3.0-or-later\n"
        "# Commercial license available\n",
        encoding="utf-8",
    )
    assert mod.verify(tmp_path) == 0
    assert "All files normalised" in capsys.readouterr().out


def test_verify_detects_residual_merged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    f = tmp_path / "residual.py"
    f.write_text(
        "# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available\n",
        encoding="utf-8",
    )
    assert mod.verify(tmp_path) == 1
    assert "still contain merged SPDX" in capsys.readouterr().out


def test_verify_detects_typo(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    f = tmp_path / "typo.py"
    typo = "protoscience@" + "any" + "lum.li"
    f.write_text(f"# Contact: {typo}\n", encoding="utf-8")
    assert mod.verify(tmp_path) == 1
    assert "still contain" in capsys.readouterr().out

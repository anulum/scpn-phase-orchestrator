# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — `spo demo` bundled-domainpack packaging tests

"""Tests that ``spo demo`` ships and resolves a bundled domainpack for pip installs.

The repository ``domainpacks/`` tree is not shipped in the wheel, so a pip install
running ``spo demo`` outside a checkout used to reach an unguarded ``.iterdir()`` on
an absent path and crash with a raw ``FileNotFoundError``. The fix bundles the demo
domainpack inside the package and makes the search fail closed. These tests pin both
halves: the bundled binding is byte-identical to its source, the demo runs from the
bundle when the repository tree is absent, and a missing domain exits cleanly.
"""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from scpn_phase_orchestrator.runtime.cli import main
from scpn_phase_orchestrator.runtime.cli import scaffold as scaffold_mod

_REPO_MINIMAL_BINDING = (
    Path(scaffold_mod.__file__).resolve().parents[4]
    / "domainpacks"
    / "minimal_domain"
    / "binding_spec.yaml"
)
_BUNDLED_MINIMAL_BINDING = (
    scaffold_mod._BUNDLED_DEMO_DOMAINPACKS / "minimal_domain" / "binding_spec.yaml"
)


def test_bundled_minimal_domain_is_shipped_and_matches_source() -> None:
    """The wheel-shipped demo binding is byte-identical to the repository source.

    A drift here means the bundled copy has diverged from the canonical
    ``domainpacks/minimal_domain/binding_spec.yaml`` — re-copy the source file.
    """
    assert _BUNDLED_MINIMAL_BINDING.is_file()
    assert _BUNDLED_MINIMAL_BINDING.read_bytes() == _REPO_MINIMAL_BINDING.read_bytes()


def test_demo_search_order_is_source_then_cwd_then_bundle() -> None:
    """The default search roots are the source tree, the cwd, then the wheel bundle."""
    roots = scaffold_mod._demo_domainpack_roots()
    assert roots[0] == Path(scaffold_mod.__file__).resolve().parents[4] / "domainpacks"
    assert Path("domainpacks") in roots
    assert roots[-1] == scaffold_mod._BUNDLED_DEMO_DOMAINPACKS


def test_demo_runs_bundled_domain_when_repository_tree_is_absent(monkeypatch) -> None:
    """A pip-style layout (no repository domainpacks) still runs the bundled demo."""
    monkeypatch.setattr(
        scaffold_mod,
        "_demo_domainpack_roots",
        lambda: (
            Path("/nonexistent-spo-demo-source"),
            scaffold_mod._BUNDLED_DEMO_DOMAINPACKS,
        ),
    )
    result = CliRunner().invoke(
        main, ["demo", "--domain", "minimal_domain", "--steps", "3"]
    )
    assert result.exit_code == 0, result.output
    assert "SPO Demo" in result.output
    assert "Final R=" in result.output


def test_demo_missing_domain_fails_closed_without_traceback(
    monkeypatch, tmp_path
) -> None:
    """An absent domain with no existing root exits cleanly, not with a traceback."""
    monkeypatch.setattr(
        scaffold_mod,
        "_demo_domainpack_roots",
        lambda: (tmp_path / "absent",),
    )
    result = CliRunner().invoke(main, ["demo", "--domain", "minimal_domain"])
    assert result.exit_code == 1
    assert "Domainpack 'minimal_domain' not found." in result.output
    assert "Available:" in result.output
    assert not isinstance(result.exception, FileNotFoundError)


def test_demo_available_listing_dedupes_across_roots(monkeypatch) -> None:
    """A domain present in more than one search root is listed once."""
    bundle = scaffold_mod._BUNDLED_DEMO_DOMAINPACKS
    monkeypatch.setattr(
        scaffold_mod, "_demo_domainpack_roots", lambda: (bundle, bundle)
    )
    result = CliRunner().invoke(main, ["demo", "--domain", "does_not_exist"])
    assert result.exit_code == 1
    available_line = next(
        line for line in result.output.splitlines() if line.startswith("Available:")
    )
    assert available_line.count("minimal_domain") == 1

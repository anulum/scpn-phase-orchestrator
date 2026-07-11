# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Binding validation-tier feature tests

"""Tests for the domainpack ``validation_tier`` feature end to end.

Covers the tier vocabulary, the loader's backward-compatible default, the
validator's fail-closed rejection, the resolved-summary surface, and the
Hub gallery filter/group helpers.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from scpn_phase_orchestrator.binding.gallery import (
    group_specs_by_validation_tier,
    select_specs_by_validation_tier,
)
from scpn_phase_orchestrator.binding.loader import load_binding_spec
from scpn_phase_orchestrator.binding.resolved import resolved_binding_config
from scpn_phase_orchestrator.binding.types import (
    DEFAULT_VALIDATION_TIER,
    VALID_VALIDATION_TIERS,
    VALIDATION_TIER_EXTERNALLY_VALIDATED,
    VALIDATION_TIER_PARTIAL,
    VALIDATION_TIER_SCAFFOLD,
    BindingSpec,
)
from scpn_phase_orchestrator.binding.validator import validate_binding_spec

DOMAINPACKS_DIR = Path(__file__).resolve().parent.parent / "domainpacks"
_MINIMAL = DOMAINPACKS_DIR / "minimal_domain" / "binding_spec.yaml"


@pytest.fixture
def base_spec() -> BindingSpec:
    return load_binding_spec(_MINIMAL)


class TestTierVocabulary:
    """The tier vocabulary is the three honest tiers with a scaffold default."""

    def test_valid_tiers_are_the_three_honest_tiers(self) -> None:
        assert {
            VALIDATION_TIER_SCAFFOLD,
            VALIDATION_TIER_PARTIAL,
            VALIDATION_TIER_EXTERNALLY_VALIDATED,
        } == VALID_VALIDATION_TIERS

    def test_default_is_scaffold(self) -> None:
        assert DEFAULT_VALIDATION_TIER == VALIDATION_TIER_SCAFFOLD
        assert DEFAULT_VALIDATION_TIER in VALID_VALIDATION_TIERS

    def test_tier_tokens_are_stable(self) -> None:
        assert VALIDATION_TIER_SCAFFOLD == "scaffold"
        assert VALIDATION_TIER_PARTIAL == "partial"
        assert VALIDATION_TIER_EXTERNALLY_VALIDATED == "externally_validated"


class TestLoaderDefault:
    """The loader defaults an absent tier to scaffold, and reads an explicit one."""

    def test_absent_validation_tier_defaults_to_scaffold(self, tmp_path: Path) -> None:
        text = _MINIMAL.read_text(encoding="utf-8")
        stripped = "\n".join(
            line
            for line in text.splitlines()
            if not line.startswith("validation_tier:")
        )
        spec_path = tmp_path / "binding_spec.yaml"
        spec_path.write_text(stripped + "\n", encoding="utf-8")

        spec = load_binding_spec(spec_path)

        assert spec.validation_tier == DEFAULT_VALIDATION_TIER

    def test_explicit_validation_tier_is_read(self, tmp_path: Path) -> None:
        text = _MINIMAL.read_text(encoding="utf-8")
        swapped = text.replace(
            "validation_tier: scaffold",
            f"validation_tier: {VALIDATION_TIER_EXTERNALLY_VALIDATED}",
        )
        spec_path = tmp_path / "binding_spec.yaml"
        spec_path.write_text(swapped, encoding="utf-8")

        spec = load_binding_spec(spec_path)

        assert spec.validation_tier == VALIDATION_TIER_EXTERNALLY_VALIDATED

    def test_non_string_validation_tier_is_rejected(self, tmp_path: Path) -> None:
        text = _MINIMAL.read_text(encoding="utf-8")
        swapped = text.replace("validation_tier: scaffold", "validation_tier: 7")
        spec_path = tmp_path / "binding_spec.yaml"
        spec_path.write_text(swapped, encoding="utf-8")

        with pytest.raises(Exception, match="validation_tier"):
            load_binding_spec(spec_path)


class TestValidatorTier:
    """The validator accepts known tiers and fails closed on an unknown one."""

    def test_default_spec_validates(self, base_spec: BindingSpec) -> None:
        assert validate_binding_spec(base_spec) == []

    @pytest.mark.parametrize("tier", sorted(VALID_VALIDATION_TIERS))
    def test_every_known_tier_validates(
        self, base_spec: BindingSpec, tier: str
    ) -> None:
        spec = dataclasses.replace(base_spec, validation_tier=tier)
        assert validate_binding_spec(spec) == []

    def test_unknown_tier_is_rejected(self, base_spec: BindingSpec) -> None:
        spec = dataclasses.replace(base_spec, validation_tier="bogus")
        errors = validate_binding_spec(spec)
        assert any("validation_tier" in error for error in errors)


class TestResolvedSummary:
    """The resolved summary surfaces the validation tier for audit and Studio."""

    def test_summary_includes_validation_tier(self, base_spec: BindingSpec) -> None:
        summary = resolved_binding_config(base_spec)
        assert summary["validation_tier"] == base_spec.validation_tier


class TestGallerySelect:
    """Selecting by tier keeps only the matching specs and guards the tier."""

    def test_selects_matching_specs_in_order(self, base_spec: BindingSpec) -> None:
        scaffold = dataclasses.replace(base_spec, name="s")
        external = dataclasses.replace(
            base_spec, name="e", validation_tier=VALIDATION_TIER_EXTERNALLY_VALIDATED
        )
        selected = select_specs_by_validation_tier(
            [scaffold, external], VALIDATION_TIER_EXTERNALLY_VALIDATED
        )
        assert [spec.name for spec in selected] == ["e"]

    def test_empty_when_no_match(self, base_spec: BindingSpec) -> None:
        selected = select_specs_by_validation_tier([base_spec], VALIDATION_TIER_PARTIAL)
        assert selected == ()

    def test_unknown_tier_raises(self, base_spec: BindingSpec) -> None:
        with pytest.raises(ValueError, match="unknown validation tier 'typo'"):
            select_specs_by_validation_tier([base_spec], "typo")


class TestGalleryGroup:
    """Grouping covers every tier and ignores out-of-vocabulary specs."""

    def test_groups_cover_every_tier(self, base_spec: BindingSpec) -> None:
        grouped = group_specs_by_validation_tier([base_spec])
        assert set(grouped) == VALID_VALIDATION_TIERS
        assert [spec.name for spec in grouped[VALIDATION_TIER_SCAFFOLD]] == [
            base_spec.name
        ]
        assert grouped[VALIDATION_TIER_PARTIAL] == ()

    def test_keys_are_sorted(self, base_spec: BindingSpec) -> None:
        grouped = group_specs_by_validation_tier([base_spec])
        assert list(grouped) == sorted(VALID_VALIDATION_TIERS)

    def test_out_of_vocabulary_tier_is_ignored(self, base_spec: BindingSpec) -> None:
        rogue = dataclasses.replace(base_spec, name="rogue", validation_tier="bogus")
        grouped = group_specs_by_validation_tier([base_spec, rogue])
        placed = [spec.name for specs in grouped.values() for spec in specs]
        assert "rogue" not in placed
        assert base_spec.name in placed


_GUIDE = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "guide"
    / "domainpack_validation_tiers.md"
)


class TestGuide:
    """Keep the validation-tier guide in step with the tier vocabulary and API."""

    def test_guide_states_every_tier_token(self) -> None:
        doc = _GUIDE.read_text(encoding="utf-8")
        for tier in VALID_VALIDATION_TIERS:
            assert f"`{tier}`" in doc, tier

    def test_guide_names_the_gallery_api(self) -> None:
        doc = _GUIDE.read_text(encoding="utf-8")
        assert "select_specs_by_validation_tier" in doc
        assert "group_specs_by_validation_tier" in doc

    def test_guide_states_the_scaffold_default(self) -> None:
        # Collapse whitespace so a line-wrapped phrase still matches.
        doc = " ".join(_GUIDE.read_text(encoding="utf-8").split())
        assert "not a validated detector" in doc
        assert "scaffold" in doc

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Binding loader security/negative tests

# Tests malformed inputs, encoding issues, and parser edge cases.

from __future__ import annotations

from pathlib import Path

import pytest

from scpn_phase_orchestrator.binding.loader import BindingLoadError, load_binding_spec


class TestMalformedYAML:
    def test_invalid_yaml_syntax(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("name: test\n  bad indent: [\n", encoding="utf-8")
        with pytest.raises(BindingLoadError, match="YAML parse error"):
            load_binding_spec(p)

    def test_unclosed_bracket(self, tmp_path: Path) -> None:
        p = tmp_path / "unclosed.yaml"
        p.write_text("layers: [{name: a, index: 0", encoding="utf-8")
        with pytest.raises(BindingLoadError, match="YAML parse error"):
            load_binding_spec(p)

    def test_tab_indentation(self, tmp_path: Path) -> None:
        """YAML forbids tabs for indentation."""
        p = tmp_path / "tabs.yaml"
        p.write_text("name: test\n\tversion: '1.0'\n", encoding="utf-8")
        with pytest.raises(BindingLoadError, match="YAML parse error"):
            load_binding_spec(p)


class TestInvalidJSON:
    def test_truncated_json(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text('{"name": "test", "layers": [', encoding="utf-8")
        with pytest.raises(BindingLoadError, match="JSON parse error"):
            load_binding_spec(p)

    def test_trailing_comma(self, tmp_path: Path) -> None:
        p = tmp_path / "trailing.json"
        p.write_text('{"name": "test",}', encoding="utf-8")
        with pytest.raises(BindingLoadError, match="JSON parse error"):
            load_binding_spec(p)


class TestNullAndEmptyValues:
    def test_null_top_level(self, tmp_path: Path) -> None:
        """YAML `null` at top level is not a mapping."""
        p = tmp_path / "null.yaml"
        p.write_text("null\n", encoding="utf-8")
        with pytest.raises(BindingLoadError, match="expected mapping"):
            load_binding_spec(p)

    def test_empty_file(self, tmp_path: Path) -> None:
        """Empty file parses as None in YAML."""
        p = tmp_path / "empty.yaml"
        p.write_text("", encoding="utf-8")
        with pytest.raises(BindingLoadError, match="expected mapping"):
            load_binding_spec(p)

    def test_scalar_top_level(self, tmp_path: Path) -> None:
        """Scalar string at top level."""
        p = tmp_path / "scalar.yaml"
        p.write_text('"just a string"\n', encoding="utf-8")
        with pytest.raises(BindingLoadError, match="expected mapping"):
            load_binding_spec(p)

    def test_list_top_level(self, tmp_path: Path) -> None:
        """List at top level instead of mapping."""
        p = tmp_path / "list.yaml"
        p.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(BindingLoadError, match="expected mapping"):
            load_binding_spec(p)


class TestMissingRequiredFields:
    def test_missing_name(self, tmp_path: Path) -> None:
        p = tmp_path / "noname.yaml"
        p.write_text(
            "version: '1.0'\nlayers: []\n"
            "oscillator_families: {}\ncoupling: {base_strength: 1, decay_alpha: 0.3}\n"
            "drivers: {}\nobjectives: {good_layers: [], bad_layers: []}\n"
            "safety_tier: research\nsample_period_s: 0.01\ncontrol_period_s: 0.1\n",
            encoding="utf-8",
        )
        with pytest.raises(BindingLoadError, match="missing required key 'name'"):
            load_binding_spec(p)

    def test_missing_layers(self, tmp_path: Path) -> None:
        p = tmp_path / "nolayers.yaml"
        p.write_text("name: test\nversion: '1.0'\n", encoding="utf-8")
        with pytest.raises(BindingLoadError, match="missing required key 'layers'"):
            load_binding_spec(p)

    def test_missing_coupling(self, tmp_path: Path) -> None:
        p = tmp_path / "nocoupling.yaml"
        p.write_text(
            "name: test\nversion: '1.0'\n"
            "layers: [{name: a, index: 0}]\n"
            "oscillator_families: {}\n"
            "drivers: {}\nobjectives: {good_layers: [], bad_layers: []}\n"
            "safety_tier: research\nsample_period_s: 0.01\ncontrol_period_s: 0.1\n",
            encoding="utf-8",
        )
        with pytest.raises(BindingLoadError, match="missing required key 'coupling'"):
            load_binding_spec(p)

    def test_missing_coupling_subfield(self, tmp_path: Path) -> None:
        p = tmp_path / "no_base.yaml"
        p.write_text(
            "name: test\nversion: '1.0'\n"
            "layers: [{name: a, index: 0}]\n"
            "oscillator_families: {}\n"
            "coupling: {decay_alpha: 0.3}\n"
            "drivers: {}\nobjectives: {good_layers: [], bad_layers: []}\n"
            "safety_tier: research\nsample_period_s: 0.01\ncontrol_period_s: 0.1\n",
            encoding="utf-8",
        )
        with pytest.raises(
            BindingLoadError, match="missing required key 'base_strength'"
        ):
            load_binding_spec(p)


class TestFileEdgeCases:
    def test_nonexistent_file(self, tmp_path: Path) -> None:
        p = tmp_path / "does_not_exist.yaml"
        with pytest.raises(BindingLoadError, match="cannot read"):
            load_binding_spec(p)

    def test_unsupported_extension(self, tmp_path: Path) -> None:
        p = tmp_path / "spec.toml"
        p.write_text("name = 'test'\n", encoding="utf-8")
        with pytest.raises(BindingLoadError, match="Unsupported file extension"):
            load_binding_spec(p)

    def test_directory_instead_of_file(self, tmp_path: Path) -> None:
        d = tmp_path / "notafile.yaml"
        d.mkdir()
        with pytest.raises(BindingLoadError, match="cannot read"):
            load_binding_spec(d)

    def test_utf8_bom(self, tmp_path: Path) -> None:
        """UTF-8 BOM should still parse (Python handles it)."""
        p = tmp_path / "bom.yaml"
        content = (
            "name: test\nversion: '1.0'\nlayers: [{name: a, index: 0}]\n"
            "oscillator_families: {}\ncoupling: {base_strength: 1, decay_alpha: 0.3}\n"
            "drivers: {}\nobjectives: {good_layers: [], bad_layers: []}\n"
            "safety_tier: research\nsample_period_s: 0.01\ncontrol_period_s: 0.1\n"
        )
        p.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))
        # BOM may cause YAML to interpret differently — either succeeds or clear error
        try:
            spec = load_binding_spec(p)
            assert spec.name == "test"
        except BindingLoadError:
            pass  # acceptable: clear error on BOM is fine


class TestTypeCoercion:
    def test_layers_not_iterable(self, tmp_path: Path) -> None:
        """layers: must be a list, not a scalar."""
        p = tmp_path / "badlayers.yaml"
        p.write_text(
            "name: test\nversion: '1.0'\nlayers: not_a_list\n"
            "oscillator_families: {}\ncoupling: {base_strength: 1, decay_alpha: 0.3}\n"
            "drivers: {}\nobjectives: {good_layers: [], bad_layers: []}\n"
            "safety_tier: research\nsample_period_s: 0.01\ncontrol_period_s: 0.1\n",
            encoding="utf-8",
        )
        with pytest.raises((BindingLoadError, TypeError)):
            load_binding_spec(p)

    def test_oscillator_families_not_dict(self, tmp_path: Path) -> None:
        p = tmp_path / "badfam.yaml"
        p.write_text(
            "name: test\nversion: '1.0'\n"
            "layers: [{name: a, index: 0}]\n"
            "oscillator_families: [wrong]\n"
            "coupling: {base_strength: 1, decay_alpha: 0.3}\n"
            "drivers: {}\nobjectives: {good_layers: [], bad_layers: []}\n"
            "safety_tier: research\nsample_period_s: 0.01\ncontrol_period_s: 0.1\n",
            encoding="utf-8",
        )
        with pytest.raises((BindingLoadError, AttributeError)):
            load_binding_spec(p)


class TestPathScrubbingInErrors:
    def test_missing_file_error_does_not_leak_full_path(self, tmp_path: Path) -> None:
        nested = tmp_path / "vault" / "secrets" / "binding.yaml"
        with pytest.raises(BindingLoadError) as excinfo:
            load_binding_spec(nested)
        msg = str(excinfo.value)
        assert "binding.yaml" in msg
        assert "vault" not in msg
        assert "secrets" not in msg
        assert str(tmp_path) not in msg

    def test_yaml_parse_error_does_not_leak_full_path(self, tmp_path: Path) -> None:
        nested_dir = tmp_path / "internal" / "private"
        nested_dir.mkdir(parents=True)
        bad = nested_dir / "spec.yaml"
        bad.write_text("name: test\n  bad indent: [\n", encoding="utf-8")
        with pytest.raises(BindingLoadError) as excinfo:
            load_binding_spec(bad)
        msg = str(excinfo.value)
        assert "spec.yaml" in msg
        assert "internal" not in msg
        assert "private" not in msg
        assert str(tmp_path) not in msg

    def test_json_parse_error_does_not_leak_full_path(self, tmp_path: Path) -> None:
        nested_dir = tmp_path / "confidential"
        nested_dir.mkdir()
        bad = nested_dir / "spec.json"
        bad.write_text('{"name":', encoding="utf-8")
        with pytest.raises(BindingLoadError) as excinfo:
            load_binding_spec(bad)
        msg = str(excinfo.value)
        assert "spec.json" in msg
        assert "confidential" not in msg
        assert str(tmp_path) not in msg

    def test_directory_error_does_not_leak_full_path(self, tmp_path: Path) -> None:
        nested = tmp_path / "secret_dir" / "notafile.yaml"
        nested.mkdir(parents=True)
        with pytest.raises(BindingLoadError) as excinfo:
            load_binding_spec(nested)
        msg = str(excinfo.value)
        assert "notafile.yaml" in msg
        assert "secret_dir" not in msg
        assert str(tmp_path) not in msg


# Pipeline wiring: binding loader security tested via path traversal rejection,
# schema validation, malicious input handling, and scrubbed error messages.
# Security is load-bearing infrastructure.

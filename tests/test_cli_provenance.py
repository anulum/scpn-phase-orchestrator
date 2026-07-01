# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI provenance attestation command tests

"""Coverage for the ``provenance-attest`` / ``provenance-verify`` CLI commands.

The spec-parsing rejection paths and the seed/key source guards carry no crypto and
run everywhere; the missing-backend path is exercised by faking an ``ImportError``
from the signer. The full attest → verify round-trip needs an ML-DSA backend and is
guarded by ``requires_mldsa``.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

import scpn_phase_orchestrator.runtime.cli.provenance as provenance_cli
from scpn_phase_orchestrator.assurance.dsse import (
    DSSE_PAYLOAD_TYPE,
    DsseEnvelope,
    DsseSignature,
    _pae,
)
from scpn_phase_orchestrator.runtime.audit_pqc import (
    generate_signing_seed,
    sign_bytes,
    signing_key_from_seed,
)
from scpn_phase_orchestrator.runtime.cli import main
from tests.test_audit_pqc import requires_mldsa

_A = "a" * 64


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _spec(**overrides: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "subjects": [{"name": "scpn-1.0-py3-none-any.whl", "sha256": _A}],
        "build_type": "https://slsa.dev/build/pypi@v1",
        "external_parameters": {"ref": "refs/tags/v1.0"},
        "resolved_dependencies": [
            {"uri": "git+https://github.com/anulum/spo@" + "b" * 40, "sha256": "c" * 64}
        ],
        "builder_id": "https://github.com/anulum/spo/ci",
        "invocation_id": "run-42",
    }
    spec.update(overrides)
    return spec


def _write(tmp_path: Path, name: str, payload: object) -> Path:
    path = tmp_path / name
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --- attest: spec rejection paths -----------------------------------------


class TestAttestSpecGuards:
    def test_rejects_invalid_json(self, runner: CliRunner, tmp_path: Path) -> None:
        spec_path = _write(tmp_path, "spec.json", "{not json")
        result = runner.invoke(
            main, ["provenance-attest", str(spec_path), "--signing-seed", "00" * 32]
        )
        assert result.exit_code == 1
        assert "not valid JSON" in result.output

    def test_rejects_non_object_spec(self, runner: CliRunner, tmp_path: Path) -> None:
        spec_path = _write(tmp_path, "spec.json", [1, 2])
        result = runner.invoke(
            main, ["provenance-attest", str(spec_path), "--signing-seed", "00" * 32]
        )
        assert result.exit_code == 1
        assert "must be a JSON object" in result.output

    @pytest.mark.parametrize(
        ("mutation", "message"),
        [
            ({"subjects": None}, "missing 'subjects'"),
            ({"subjects": "x"}, "'subjects' must be a list"),
            ({"subjects": []}, "'subjects' must be non-empty"),
            ({"subjects": ["x"]}, "each subject must be an object"),
            ({"subjects": [{"name": "w", "sha256": "short"}]}, "subject sha256"),
            ({"build_type": ""}, "'build_type' must be a non-empty string"),
            ({"external_parameters": None}, "missing 'external_parameters'"),
            ({"external_parameters": "x"}, "'external_parameters' must be an object"),
            ({"internal_parameters": "x"}, "'internal_parameters' must be an object"),
            ({"resolved_dependencies": "x"}, "'resolved_dependencies' must be a list"),
            ({"resolved_dependencies": ["x"]}, "each resolved_dependencies entry"),
            (
                {"resolved_dependencies": [{"uri": "u", "sha256": "short"}]},
                "resolved dependency sha256",
            ),
            ({"builder_dependencies": "x"}, "'builder_dependencies' must be a list"),
            ({"builder_dependencies": ["x"]}, "each builder_dependencies entry"),
            ({"byproducts": "x"}, "'byproducts' must be a list"),
            ({"byproducts": ["x"]}, "each byproducts entry"),
            (
                {"byproducts": [{"uri": "u", "sha256": "short"}]},
                "resolved dependency sha256",
            ),
            ({"builder_version": "x"}, "'builder_version' must be an object"),
            ({"builder_version": {"k": 3}}, "builder_version value for 'k'"),
            ({"builder_id": ""}, "'builder_id' must be a non-empty string"),
            ({"invocation_id": ""}, "'invocation_id' must be a non-empty string"),
        ],
    )
    def test_rejects_malformed_spec_fields(
        self, runner: CliRunner, tmp_path: Path, mutation: dict[str, Any], message: str
    ) -> None:
        spec_path = _write(tmp_path, "spec.json", _spec(**mutation))
        result = runner.invoke(
            main, ["provenance-attest", str(spec_path), "--signing-seed", "00" * 32]
        )
        assert result.exit_code == 1
        assert message in result.output


# --- attest: seed source + signing errors ---------------------------------


class TestAttestSeedSources:
    def test_rejects_both_seed_sources(self, runner: CliRunner, tmp_path: Path) -> None:
        spec_path = _write(tmp_path, "spec.json", _spec())
        seed_path = _write(tmp_path, "seed.txt", "00" * 32)
        result = runner.invoke(
            main,
            [
                "provenance-attest",
                str(spec_path),
                "--signing-seed",
                "00" * 32,
                "--signing-seed-file",
                str(seed_path),
            ],
        )
        assert result.exit_code == 1
        assert "only one of --signing-seed" in result.output

    def test_rejects_missing_seed(self, runner: CliRunner, tmp_path: Path) -> None:
        spec_path = _write(tmp_path, "spec.json", _spec())
        result = runner.invoke(main, ["provenance-attest", str(spec_path)])
        assert result.exit_code == 1
        assert "a signing seed is required" in result.output

    def test_rejects_bad_seed_hex(self, runner: CliRunner, tmp_path: Path) -> None:
        spec_path = _write(tmp_path, "spec.json", _spec())
        result = runner.invoke(
            main, ["provenance-attest", str(spec_path), "--signing-seed", "nothex"]
        )
        assert result.exit_code == 1
        assert "seed" in result.output.lower()

    def test_rejects_bad_seed_from_file(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        spec_path = _write(tmp_path, "spec.json", _spec())
        seed_path = _write(tmp_path, "seed.txt", "nothex")
        result = runner.invoke(
            main,
            [
                "provenance-attest",
                str(spec_path),
                "--signing-seed-file",
                str(seed_path),
            ],
        )
        assert result.exit_code == 1
        assert "seed" in result.output.lower()

    def test_spec_without_resolved_dependencies_defaults_empty(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        # Omitting resolved_dependencies exercises the optional-list default; the
        # command then fails only for the missing seed, proving the spec parsed.
        spec = _spec()
        del spec["resolved_dependencies"]
        spec_path = _write(tmp_path, "spec.json", spec)
        result = runner.invoke(main, ["provenance-attest", str(spec_path)])
        assert result.exit_code == 1
        assert "a signing seed is required" in result.output

    def test_reports_missing_pqc_backend(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raise(*_args: Any, **_kwargs: Any) -> Any:
            raise ImportError("install scpn-phase-orchestrator[pqc]")

        monkeypatch.setattr(provenance_cli, "signing_key_from_seed", _raise)
        spec_path = _write(tmp_path, "spec.json", _spec())
        result = runner.invoke(
            main, ["provenance-attest", str(spec_path), "--signing-seed", "00" * 32]
        )
        assert result.exit_code == 1
        assert "scpn-phase-orchestrator[pqc]" in result.output


# --- verify: envelope + key guards ----------------------------------------


class TestVerifyGuards:
    def test_rejects_non_object_envelope(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        env_path = _write(tmp_path, "env.json", [1, 2])
        result = runner.invoke(
            main, ["provenance-verify", str(env_path), "--public-key", _A]
        )
        assert result.exit_code == 1
        # A JSON list fails the spec-object guard before envelope parsing.
        assert "must be a JSON object" in result.output

    def test_rejects_malformed_envelope(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        env_path = _write(tmp_path, "env.json", {"envelope": {"payload": "x"}})
        result = runner.invoke(
            main, ["provenance-verify", str(env_path), "--public-key", _A]
        )
        assert result.exit_code == 1
        assert "missing field" in result.output

    def test_rejects_both_key_sources(self, runner: CliRunner, tmp_path: Path) -> None:
        env = _bare_envelope_dict()
        env_path = _write(tmp_path, "env.json", env)
        key_path = _write(tmp_path, "key.txt", _A)
        result = runner.invoke(
            main,
            [
                "provenance-verify",
                str(env_path),
                "--public-key",
                _A,
                "--public-key-file",
                str(key_path),
            ],
        )
        assert result.exit_code == 1
        assert "only one of --public-key" in result.output

    def test_rejects_missing_key(self, runner: CliRunner, tmp_path: Path) -> None:
        env_path = _write(tmp_path, "env.json", _bare_envelope_dict())
        result = runner.invoke(main, ["provenance-verify", str(env_path)])
        assert result.exit_code == 1
        assert "a trusted public key is required" in result.output

    def test_rejects_non_hex_key(self, runner: CliRunner, tmp_path: Path) -> None:
        env_path = _write(tmp_path, "env.json", _bare_envelope_dict())
        result = runner.invoke(
            main, ["provenance-verify", str(env_path), "--public-key", "zz"]
        )
        assert result.exit_code == 1
        assert "valid hex" in result.output

    def test_rejects_non_hex_key_from_file(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        env_path = _write(tmp_path, "env.json", _bare_envelope_dict())
        key_path = _write(tmp_path, "key.txt", "zz")
        result = runner.invoke(
            main,
            ["provenance-verify", str(env_path), "--public-key-file", str(key_path)],
        )
        assert result.exit_code == 1
        assert "valid hex" in result.output

    def test_rejects_unverifiable_signature(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        env_path = _write(tmp_path, "env.json", _bare_envelope_dict())
        result = runner.invoke(
            main, ["provenance-verify", str(env_path), "--public-key", "ab" * 32]
        )
        assert result.exit_code == 1
        assert "signature is not valid" in result.output


def _bare_envelope_dict() -> dict[str, Any]:
    """Return a structurally valid (but unverifiable) DSSE envelope mapping."""
    envelope = DsseEnvelope(
        payload_b64=base64.b64encode(b'{"ok": true}').decode("ascii"),
        payload_type=DSSE_PAYLOAD_TYPE,
        signatures=(
            DsseSignature(
                keyid="deadbeef",
                algorithm="ml-dsa-65",
                signature_b64=base64.b64encode(b"sig").decode("ascii"),
            ),
        ),
    )
    return envelope.to_dict()


# --- attest -> verify round-trip (crypto) ---------------------------------


@requires_mldsa
class TestAttestVerifyRoundTrip:
    def test_attest_then_verify_with_flags(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        spec_path = _write(tmp_path, "spec.json", _spec())
        seed = generate_signing_seed()
        attest = runner.invoke(
            main, ["provenance-attest", str(spec_path), "--signing-seed", seed]
        )
        assert attest.exit_code == 0, attest.output
        out = json.loads(attest.output)
        assert out["statement_subject"] == ["scpn-1.0-py3-none-any.whl"]
        env_path = _write(tmp_path, "env.json", out)

        verify = runner.invoke(
            main,
            ["provenance-verify", str(env_path), "--public-key", out["public_key_hex"]],
        )
        assert verify.exit_code == 0, verify.output
        verified = json.loads(verify.output)
        assert verified["verified"] is True
        assert verified["statement"]["predicateType"].startswith("https://slsa.dev")

    def test_attest_and_verify_with_files(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        spec_path = _write(tmp_path, "spec.json", _spec())
        seed_path = _write(tmp_path, "seed.txt", generate_signing_seed())
        attest = runner.invoke(
            main,
            [
                "provenance-attest",
                str(spec_path),
                "--signing-seed-file",
                str(seed_path),
                "--algorithm",
                "ml-dsa-65",
            ],
        )
        assert attest.exit_code == 0, attest.output
        out = json.loads(attest.output)
        env_path = _write(tmp_path, "env.json", out["envelope"])
        key_path = _write(tmp_path, "key.txt", out["public_key_hex"])

        verify = runner.invoke(
            main,
            ["provenance-verify", str(env_path), "--public-key-file", str(key_path)],
        )
        assert verify.exit_code == 0, verify.output
        assert json.loads(verify.output)["verified"] is True

    def test_verify_rejects_wrong_key(self, runner: CliRunner, tmp_path: Path) -> None:
        spec_path = _write(tmp_path, "spec.json", _spec())
        attest = runner.invoke(
            main,
            [
                "provenance-attest",
                str(spec_path),
                "--signing-seed",
                generate_signing_seed(),
            ],
        )
        out = json.loads(attest.output)
        env_path = _write(tmp_path, "env.json", out["envelope"])
        other_pub = (
            signing_key_from_seed(generate_signing_seed())
            .public_key()
            .public_bytes_raw()
            .hex()
        )
        verify = runner.invoke(
            main, ["provenance-verify", str(env_path), "--public-key", other_pub]
        )
        assert verify.exit_code == 1
        assert "signature is not valid" in verify.output

    def test_attest_carries_extended_run_details(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        # builder_version, builder_dependencies, and byproducts must survive the
        # spec → statement → signed-envelope → verified-statement round-trip.
        spec = _spec(
            builder_version={"python": "3.12", "runner": "ubuntu-latest"},
            builder_dependencies=[
                {"uri": "pkg:pypi/build@1.5.0", "sha256": "d" * 64, "name": "build"}
            ],
            byproducts=[
                {"uri": "file:sbom.json", "sha256": "e" * 64, "name": "sbom.json"}
            ],
        )
        spec_path = _write(tmp_path, "spec.json", spec)
        attest = runner.invoke(
            main,
            [
                "provenance-attest",
                str(spec_path),
                "--signing-seed",
                generate_signing_seed(),
            ],
        )
        assert attest.exit_code == 0, attest.output
        env_path = _write(tmp_path, "env.json", json.loads(attest.output))
        verify = runner.invoke(
            main,
            [
                "provenance-verify",
                str(env_path),
                "--public-key",
                json.loads(attest.output)["public_key_hex"],
            ],
        )
        assert verify.exit_code == 0, verify.output
        run_details = json.loads(verify.output)["statement"]["predicate"]["runDetails"]
        assert run_details["builder"]["version"] == {
            "python": "3.12",
            "runner": "ubuntu-latest",
        }
        assert run_details["builder"]["builderDependencies"][0]["name"] == "build"
        assert run_details["byproducts"][0]["name"] == "sbom.json"

    def test_verify_reports_non_object_signed_payload(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        # A validly-signed envelope whose payload is a JSON array (not an object):
        # the signature verifies, but extracting the statement then fails.
        key = signing_key_from_seed(generate_signing_seed())
        pub = key.public_key().public_bytes_raw().hex()
        body = b"[1, 2, 3]"
        signature = sign_bytes(_pae(DSSE_PAYLOAD_TYPE, body), key)
        from scpn_phase_orchestrator.runtime.audit_pqc import public_key_id

        envelope = DsseEnvelope(
            payload_b64=base64.b64encode(body).decode("ascii"),
            payload_type=DSSE_PAYLOAD_TYPE,
            signatures=(
                DsseSignature(
                    keyid=public_key_id(key.public_key().public_bytes_raw()),
                    algorithm="ml-dsa-65",
                    signature_b64=base64.b64encode(signature).decode("ascii"),
                ),
            ),
        )
        env_path = _write(tmp_path, "env.json", envelope.to_dict())
        result = runner.invoke(
            main, ["provenance-verify", str(env_path), "--public-key", pub]
        )
        assert result.exit_code == 1
        assert "not a JSON object" in result.output

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — assurance-case CLI command tests

"""CliRunner tests for the ``spo assurance-case`` command."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

import scpn_phase_orchestrator.runtime.cli.assurance as cli_assurance
from scpn_phase_orchestrator.assurance import (
    SignedCertificationEnvelope,
    verify_signed_certification_envelope,
)
from scpn_phase_orchestrator.runtime.audit_pqc import (
    generate_signing_seed,
    signing_key_from_seed,
)
from scpn_phase_orchestrator.runtime.cli import main
from scpn_phase_orchestrator.supervisor.formal_export import (
    FormalSafetyProperty,
    FormalTextArtifact,
    build_formal_verification_package,
)


def _mldsa_supported() -> bool:
    """Return whether the platform cryptography backend implements ML-DSA."""
    try:
        from cryptography.exceptions import UnsupportedAlgorithm
        from cryptography.hazmat.primitives.asymmetric import mldsa
    except ImportError:
        return False
    try:
        mldsa.MLDSA65PrivateKey.generate()
    except UnsupportedAlgorithm:
        return False
    return True


requires_mldsa = pytest.mark.skipif(
    not _mldsa_supported(),
    reason="ML-DSA requires an OpenSSL 3.5+ cryptography backend",
)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _write_chain_log(path: Path, *, tip: str = "ab" * 32) -> None:
    path.write_text(
        json.dumps({"event": "close", "_hash": tip}) + "\n", encoding="utf-8"
    )


def _write_evidence(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "evidence_id": "twin",
                    "category": "twin_confidence",
                    "summary": "twin healthy",
                    "record": {"coverage": 0.9},
                },
                {
                    "evidence_id": "cptc",
                    "category": "conformal_gate",
                    "summary": "gate calibrated",
                    "record": {"alpha": 0.1},
                },
            ]
        ),
        encoding="utf-8",
    )


def test_cli_module_exposes_command() -> None:
    assert cli_assurance.assurance_case.name == "assurance-case"
    assert cli_assurance.certification_evidence.name == "certification-evidence"


def test_build_from_evidence_file_to_stdout(runner: CliRunner, tmp_path: Path) -> None:
    evidence = tmp_path / "ev.json"
    _write_evidence(evidence)
    result = runner.invoke(
        main, ["assurance-case", "--system", "Sys", "--evidence-file", str(evidence)]
    )
    assert result.exit_code == 0, result.output
    bundle = json.loads(result.output)
    assert bundle["schema"] == "scpn_assurance_case_bundle_v1"
    assert bundle["actuation_permitted"] is False
    assert bundle["system_name"] == "Sys"


def test_build_to_output_file(runner: CliRunner, tmp_path: Path) -> None:
    evidence = tmp_path / "ev.json"
    _write_evidence(evidence)
    out = tmp_path / "bundle.json"
    result = runner.invoke(
        main,
        [
            "assurance-case",
            "--system",
            "Sys",
            "--evidence-file",
            str(evidence),
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    bundle = json.loads(out.read_text(encoding="utf-8"))
    assert bundle["bundle_hash"] in result.output


def test_build_with_report_out(runner: CliRunner, tmp_path: Path) -> None:
    evidence = tmp_path / "ev.json"
    _write_evidence(evidence)
    bundle_path = tmp_path / "bundle.json"
    report_path = tmp_path / "conformity.md"
    result = runner.invoke(
        main,
        [
            "assurance-case",
            "--system",
            "Sys",
            "--evidence-file",
            str(evidence),
            "--output",
            str(bundle_path),
            "--report-out",
            str(report_path),
        ],
    )
    assert result.exit_code == 0, result.output
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")
    assert report.startswith("# Conformity Evidence Report — Sys")
    assert f"`{bundle['bundle_hash']}`" in report
    assert f"Wrote conformity report to {report_path}" in result.output


def test_build_with_report_pdf_out(runner: CliRunner, tmp_path: Path) -> None:
    evidence = tmp_path / "ev.json"
    _write_evidence(evidence)
    pdf_path = tmp_path / "conformity.pdf"
    result = runner.invoke(
        main,
        [
            "assurance-case",
            "--system",
            "Sys",
            "--evidence-file",
            str(evidence),
            "--report-pdf-out",
            str(pdf_path),
        ],
    )
    assert result.exit_code == 0, result.output
    pdf = pdf_path.read_bytes()
    assert pdf.startswith(b"%PDF-1.4")
    assert b"CONFORMITY EVIDENCE REPORT" in pdf
    assert f"Wrote conformity report PDF to {pdf_path}" in result.output


def test_build_from_run_result(runner: CliRunner, tmp_path: Path) -> None:
    run_result = tmp_path / "run.json"
    run_result.write_text(
        json.dumps(
            {
                "spec_name": "grid",
                "audit_event_stream_integrity": {
                    "integrity_ok": True,
                    "verified_records": 12,
                },
                "conformal_admission_total": 6,
                "conformal_admission_rejections": 1,
                "last_conformal_admission": {"admitted": True},
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "bundle.json"
    result = runner.invoke(
        main,
        [
            "assurance-case",
            "--system",
            "Grid",
            "--run-result",
            str(run_result),
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    bundle = json.loads(out.read_text(encoding="utf-8"))
    evidence_ids = {item["evidence_id"] for item in bundle["evidence"]}
    assert evidence_ids == {"run-audit-stream-integrity", "run-conformal-admission"}


def test_run_result_without_trust_evidence_is_rejected(
    runner: CliRunner, tmp_path: Path
) -> None:
    run_result = tmp_path / "run.json"
    run_result.write_text(
        json.dumps({"spec_name": "grid", "conformal_admission_total": 0}),
        encoding="utf-8",
    )
    result = runner.invoke(
        main,
        ["assurance-case", "--system", "Grid", "--run-result", str(run_result)],
    )
    assert result.exit_code != 0
    assert (
        "no audit-integrity, conformal-gate, or control-envelope evidence"
        in result.output
    )


def test_build_from_run_result_with_control_envelope(
    runner: CliRunner, tmp_path: Path
) -> None:
    run_result = tmp_path / "run.json"
    run_result.write_text(
        json.dumps(
            {
                "spec_name": "grid",
                "policy_enabled": True,
                "control_mode": "projected",
                "action_total": 5,
                "boundary_violation_total": 0,
                "final_regime": "nominal",
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "bundle.json"
    result = runner.invoke(
        main,
        [
            "assurance-case",
            "--system",
            "Grid",
            "--run-result",
            str(run_result),
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    bundle = json.loads(out.read_text(encoding="utf-8"))
    evidence = {item["evidence_id"]: item for item in bundle["evidence"]}
    assert "run-control-envelope" in evidence
    assert evidence["run-control-envelope"]["category"] == "control_envelope"
    assert evidence["run-control-envelope"]["record"]["action_total"] == 5


def _write_formal_package(path: Path) -> None:
    package = build_formal_verification_package(
        {"safety": FormalTextArtifact("smt2", "(assert (= x x))")},
        [
            FormalSafetyProperty(
                name="bounded",
                artifact_name="safety",
                checker="smt",
                expression="(check-sat)",
            )
        ],
    )
    path.write_text(json.dumps(package.to_audit_record()), encoding="utf-8")


def test_build_from_formal_package(runner: CliRunner, tmp_path: Path) -> None:
    manifest = tmp_path / "formal.json"
    _write_formal_package(manifest)
    out = tmp_path / "bundle.json"
    result = runner.invoke(
        main,
        [
            "assurance-case",
            "--system",
            "Sys",
            "--formal-package",
            str(manifest),
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    bundle = json.loads(out.read_text(encoding="utf-8"))
    formal = [
        item for item in bundle["evidence"] if item["category"] == "formal_verification"
    ]
    assert len(formal) == 1
    assert formal[0]["evidence_id"] == "formal-verification-package"


def test_formal_package_non_object_is_rejected(
    runner: CliRunner, tmp_path: Path
) -> None:
    bad = tmp_path / "formal.json"
    bad.write_text(json.dumps(["not-an-object"]), encoding="utf-8")
    result = runner.invoke(
        main, ["assurance-case", "--system", "Sys", "--formal-package", str(bad)]
    )
    assert result.exit_code != 0
    assert "must be a FormalVerificationPackage JSON manifest" in result.output


def test_formal_package_invalid_manifest_is_rejected(
    runner: CliRunner, tmp_path: Path
) -> None:
    bad = tmp_path / "formal.json"
    bad.write_text(json.dumps({"package_name": "x"}), encoding="utf-8")
    result = runner.invoke(
        main, ["assurance-case", "--system", "Sys", "--formal-package", str(bad)]
    )
    assert result.exit_code != 0
    assert "is not a valid manifest" in result.output


def test_build_certification_evidence_package(
    runner: CliRunner, tmp_path: Path
) -> None:
    evidence = tmp_path / "ev.json"
    _write_evidence(evidence)
    out = tmp_path / "review_package"
    result = runner.invoke(
        main,
        [
            "certification-evidence",
            "--system",
            "Sys",
            "--evidence-file",
            str(evidence),
            "--output-dir",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    bundle = json.loads((out / "assurance_bundle.json").read_text(encoding="utf-8"))
    vectors = json.loads((out / "test_vectors.json").read_text(encoding="utf-8"))

    assert manifest["schema"] == "scpn_certification_evidence_package_v1"
    assert manifest["system_name"] == "Sys"
    assert manifest["assurance_bundle_hash"] == bundle["bundle_hash"]
    assert {row["path"] for row in manifest["files"]} == {
        "assurance_bundle.json",
        "conformity_report.md",
        "conformity_report.pdf",
        "test_vectors.json",
    }
    assert [row["evidence_id"] for row in vectors["evidence_hash_vectors"]] == [
        "cptc",
        "twin",
    ]
    assert "technical evidence-mapping package" in manifest["disclaimer"]

    # The human-readable conformity report is sealed into the package.
    report = (out / "conformity_report.md").read_text(encoding="utf-8")
    assert report.startswith("# Conformity Evidence Report — Sys")
    assert f"`{bundle['bundle_hash']}`" in report
    assert "## Coverage summary" in report
    report_row = next(
        row for row in manifest["files"] if row["path"] == "conformity_report.md"
    )
    assert report_row["sha256"] == hashlib.sha256(report.encode("utf-8")).hexdigest()

    # The filable conformity-report PDF is sealed into the package too.
    report_pdf = (out / "conformity_report.pdf").read_bytes()
    assert report_pdf.startswith(b"%PDF-1.4")
    pdf_row = next(
        row for row in manifest["files"] if row["path"] == "conformity_report.pdf"
    )
    assert pdf_row["sha256"] == hashlib.sha256(report_pdf).hexdigest()


def test_certification_evidence_refuses_non_empty_output_dir(
    runner: CliRunner, tmp_path: Path
) -> None:
    evidence = tmp_path / "ev.json"
    _write_evidence(evidence)
    out = tmp_path / "review_package"
    out.mkdir()
    (out / "stale.json").write_text("{}", encoding="utf-8")

    result = runner.invoke(
        main,
        [
            "certification-evidence",
            "--system",
            "Sys",
            "--evidence-file",
            str(evidence),
            "--output-dir",
            str(out),
        ],
    )

    assert result.exit_code != 0
    assert "output directory is not empty" in result.output


def test_build_from_audit_log(runner: CliRunner, tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    log.write_text(
        json.dumps({"header": True, "n_oscillators": 2, "dt": 0.01}) + "\n",
        encoding="utf-8",
    )
    result = runner.invoke(
        main, ["assurance-case", "--system", "Sys", "--audit-log", str(log)]
    )
    assert result.exit_code == 0, result.output
    bundle = json.loads(result.output)
    categories = {item["category"] for item in bundle["evidence"]}
    assert "audit_logging" in categories


def _write_header_log(path: Path, *, amplitude_mode: bool = False) -> None:
    header: dict[str, object] = {"header": True, "n_oscillators": 2, "dt": 0.01}
    if amplitude_mode:
        header["amplitude_mode"] = True
    path.write_text(json.dumps(header) + "\n", encoding="utf-8")


def test_replay_determinism_evidence_for_an_upde_log(tmp_path: Path) -> None:
    log = tmp_path / "upde.jsonl"
    _write_header_log(log)
    item = cli_assurance._replay_determinism_evidence(str(log))
    assert item.category == "replay_determinism"
    assert item.record["deterministic"] is True
    assert item.record["verified_transitions"] == 0


def test_replay_determinism_evidence_for_a_stuart_landau_log(tmp_path: Path) -> None:
    log = tmp_path / "sl.jsonl"
    _write_header_log(log, amplitude_mode=True)
    item = cli_assurance._replay_determinism_evidence(str(log))
    assert item.category == "replay_determinism"
    assert "verified_transitions" in item.record


def test_replay_determinism_evidence_rejects_a_log_without_a_header(
    tmp_path: Path,
) -> None:
    log = tmp_path / "nohdr.jsonl"
    log.write_text(json.dumps({"step": 0}) + "\n", encoding="utf-8")
    with pytest.raises(click.ClickException, match="no header record"):
        cli_assurance._replay_determinism_evidence(str(log))


def test_build_with_verify_determinism(runner: CliRunner, tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    _write_header_log(log)
    result = runner.invoke(
        main,
        [
            "assurance-case",
            "--system",
            "Sys",
            "--audit-log",
            str(log),
            "--verify-determinism",
        ],
    )
    assert result.exit_code == 0, result.output
    bundle = json.loads(result.output)
    categories = {item["category"] for item in bundle["evidence"]}
    assert "replay_determinism" in categories
    assert "audit_logging" in categories


def test_no_evidence_is_an_error(runner: CliRunner) -> None:
    result = runner.invoke(main, ["assurance-case", "--system", "Sys"])
    assert result.exit_code != 0
    assert "no evidence supplied" in result.output


def test_missing_field_is_reported(runner: CliRunner, tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps([{"evidence_id": "x", "category": "twin_confidence"}]),
        encoding="utf-8",
    )
    result = runner.invoke(
        main, ["assurance-case", "--system", "Sys", "--evidence-file", str(bad)]
    )
    assert result.exit_code != 0
    assert "missing required field" in result.output


def test_non_object_row_is_reported(runner: CliRunner, tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(["not-an-object"]), encoding="utf-8")
    result = runner.invoke(
        main, ["assurance-case", "--system", "Sys", "--evidence-file", str(bad)]
    )
    assert result.exit_code != 0
    assert "must be a JSON object" in result.output


def test_certification_evidence_writes_anchor_only_envelope(
    runner: CliRunner, tmp_path: Path
) -> None:
    evidence = tmp_path / "ev.json"
    _write_evidence(evidence)
    log = tmp_path / "audit.jsonl"
    tip = "ab" * 32
    _write_chain_log(log, tip=tip)
    out = tmp_path / "review_package"
    result = runner.invoke(
        main,
        [
            "certification-evidence",
            "--system",
            "Sys",
            "--evidence-file",
            str(evidence),
            "--audit-log",
            str(log),
            "--sign-envelope",
            "--output-dir",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    envelope_record = json.loads(
        (out / "signed_envelope.json").read_text(encoding="utf-8")
    )
    assert envelope_record["schema"] == "scpn_signed_certification_envelope_v1"
    assert envelope_record["package_hash"] == manifest["package_hash"]
    assert envelope_record["audit_chain_tip"] == tip
    assert envelope_record["seal"] is None

    envelope = SignedCertificationEnvelope(
        package_hash=envelope_record["package_hash"],
        audit_chain_tip=envelope_record["audit_chain_tip"],
        audit_record_count=envelope_record["audit_record_count"],
        seal=envelope_record["seal"],
        envelope_hash=envelope_record["envelope_hash"],
    )
    assert verify_signed_certification_envelope(
        envelope, package_hash=manifest["package_hash"]
    )


def test_sign_envelope_without_audit_log_is_rejected(
    runner: CliRunner, tmp_path: Path
) -> None:
    evidence = tmp_path / "ev.json"
    _write_evidence(evidence)
    out = tmp_path / "review_package"
    result = runner.invoke(
        main,
        [
            "certification-evidence",
            "--system",
            "Sys",
            "--evidence-file",
            str(evidence),
            "--sign-envelope",
            "--output-dir",
            str(out),
        ],
    )
    assert result.exit_code != 0
    assert "require --audit-log" in result.output


def test_anchor_envelope_rejects_log_without_chain_tip(
    runner: CliRunner, tmp_path: Path
) -> None:
    evidence = tmp_path / "ev.json"
    _write_evidence(evidence)
    log = tmp_path / "audit.jsonl"
    log.write_text(json.dumps({"header": True}) + "\n", encoding="utf-8")
    out = tmp_path / "review_package"
    result = runner.invoke(
        main,
        [
            "certification-evidence",
            "--system",
            "Sys",
            "--evidence-file",
            str(evidence),
            "--audit-log",
            str(log),
            "--sign-envelope",
            "--output-dir",
            str(out),
        ],
    )
    assert result.exit_code != 0
    assert "cannot anchor an envelope" in result.output


def test_signed_envelope_rejects_a_malformed_seed(
    runner: CliRunner, tmp_path: Path
) -> None:
    evidence = tmp_path / "ev.json"
    _write_evidence(evidence)
    log = tmp_path / "audit.jsonl"
    _write_chain_log(log)
    seed_file = tmp_path / "seed.hex"
    seed_file.write_text("not-valid-hex", encoding="utf-8")
    out = tmp_path / "review_package"
    result = runner.invoke(
        main,
        [
            "certification-evidence",
            "--system",
            "Sys",
            "--evidence-file",
            str(evidence),
            "--audit-log",
            str(log),
            "--signing-seed-file",
            str(seed_file),
            "--output-dir",
            str(out),
        ],
    )
    assert result.exit_code != 0
    assert "post-quantum envelope signing failed" in result.output


@requires_mldsa
def test_certification_evidence_signs_envelope_with_seed(
    runner: CliRunner, tmp_path: Path
) -> None:
    evidence = tmp_path / "ev.json"
    _write_evidence(evidence)
    log = tmp_path / "audit.jsonl"
    tip = "ab" * 32
    _write_chain_log(log, tip=tip)
    seed = generate_signing_seed()
    seed_file = tmp_path / "seed.hex"
    seed_file.write_text(seed, encoding="utf-8")
    out = tmp_path / "review_package"
    result = runner.invoke(
        main,
        [
            "certification-evidence",
            "--system",
            "Sys",
            "--evidence-file",
            str(evidence),
            "--audit-log",
            str(log),
            "--signing-seed-file",
            str(seed_file),
            "--output-dir",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    envelope_record = json.loads(
        (out / "signed_envelope.json").read_text(encoding="utf-8")
    )
    assert envelope_record["seal"] is not None
    assert envelope_record["seal"]["tip_hash"] == tip

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    public_hex = signing_key_from_seed(seed).public_key().public_bytes_raw().hex()
    envelope = SignedCertificationEnvelope(
        package_hash=envelope_record["package_hash"],
        audit_chain_tip=envelope_record["audit_chain_tip"],
        audit_record_count=envelope_record["audit_record_count"],
        seal=envelope_record["seal"],
        envelope_hash=envelope_record["envelope_hash"],
    )
    assert verify_signed_certification_envelope(
        envelope,
        package_hash=manifest["package_hash"],
        trusted_public_key_hex=public_hex,
    )


def test_invalid_category_is_reported(runner: CliRunner, tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "evidence_id": "x",
                "category": "not_a_category",
                "summary": "s",
                "record": {"v": 1},
            }
        ),
        encoding="utf-8",
    )
    result = runner.invoke(
        main, ["assurance-case", "--system", "Sys", "--evidence-file", str(bad)]
    )
    assert result.exit_code != 0
    assert "is invalid" in result.output

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — OpenQASM 3 conformance checker tests

"""Behaviour and edge-case tests for the OpenQASM 3 conformance checker."""

from __future__ import annotations

import pytest

from scpn_phase_orchestrator.adapters.openqasm_conformance import (
    BACKEND_EXTENSION_GATES,
    STANDARD_LIBRARY_GATES,
    OpenQasm3ConformanceReport,
    check_openqasm3,
)

EMITTER_PROGRAM = (
    "OPENQASM 3.0;\n"
    'include "stdgates.inc";\n'
    "qubit[2] q;\n"
    "rz(0.125000000000) q[0];\n"
    "rz(-0.062500000000) q[1];\n"
    "rxx(0.046875000000) q[0], q[1];\n"
    "ryy(0.046875000000) q[0], q[1];\n"
)


class TestVerifiedGateTables:
    """The gate registries reflect the source-verified OpenQASM 3 library."""

    def test_stdgates_excludes_two_qubit_pauli_rotations(self):
        # Verified at source (openqasm/openqasm, 2026-07-21): stdgates.inc does
        # not define rxx/ryy/rzz/rzx.
        for gate in ("rxx", "ryy", "rzz", "rzx"):
            assert gate not in STANDARD_LIBRARY_GATES
            assert gate in BACKEND_EXTENSION_GATES

    def test_representative_stdgate_arities(self):
        assert STANDARD_LIBRARY_GATES["rz"] == (1, 1)
        assert STANDARD_LIBRARY_GATES["cx"] == (0, 2)
        assert STANDARD_LIBRARY_GATES["u3"] == (3, 1)
        assert STANDARD_LIBRARY_GATES["ccx"] == (0, 3)
        assert BACKEND_EXTENSION_GATES["rxx"] == (1, 2)


class TestConformantPrograms:
    """Well-formed programs are reported conformant."""

    def test_emitter_style_program_is_conformant(self):
        report = check_openqasm3(EMITTER_PROGRAM)
        assert isinstance(report, OpenQasm3ConformanceReport)
        assert report.conformant is True
        assert report.qasm_version == "3.0"
        assert report.includes == ("stdgates.inc",)
        assert report.qubit_registers == (("q", 2),)
        assert report.gate_call_count == 4
        assert report.stdgates_used == ("rz",)
        assert report.extension_gates_used == ("rxx", "ryy")
        assert report.custom_gates_declared == ()
        assert report.issues == ()
        assert report.uses_non_stdgates_extensions is True

    def test_pure_stdgates_program_has_no_extensions(self):
        report = check_openqasm3(
            'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[2] q;\n'
            "h q[0];\ncx q[0], q[1];\n"
        )
        assert report.conformant is True
        assert report.extension_gates_used == ()
        assert report.uses_non_stdgates_extensions is False

    def test_comments_and_whitespace_are_ignored(self):
        program = (
            "// leading line comment\n"
            "OPENQASM 3.0;\n"
            'include "stdgates.inc"; /* inline block */\n'
            "qubit[1] q;\n"
            "/* multi\nline\ncomment */\n"
            "rx (0.5) q[0];\n"
        )
        report = check_openqasm3(program)
        assert report.conformant is True
        assert report.stdgates_used == ("rx",)

    def test_builtin_gates_need_no_include(self):
        report = check_openqasm3(
            "OPENQASM 3.0;\nqubit[1] q;\nU(0, 0, 0) q[0];\ngphase(0.5);\n"
        )
        assert report.conformant is True
        # U/gphase are recorded as stdgates-tier builtins.
        assert report.stdgates_used == ("U", "gphase")

    def test_legacy_qreg_and_measure_are_tolerated(self):
        report = check_openqasm3(
            'OPENQASM 3.0;\ninclude "stdgates.inc";\n'
            "qreg q[2];\nbit[2] c;\n"
            "h q[0];\nbarrier;\nmeasure q[0] -> c[0];\n"
        )
        assert report.conformant is True
        assert report.qubit_registers == (("q", 2),)

    def test_scalar_qubit_declaration_defaults_to_size_one(self):
        report = check_openqasm3(
            'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit q;\nqreg r;\nx q;\nx r;\n'
        )
        assert report.conformant is True
        assert report.qubit_registers == (("q", 1), ("r", 1))


class TestCustomGateDeclarations:
    """In-program gate declarations extend the resolvable registry."""

    def test_custom_gate_declaration_and_use(self):
        program = (
            "OPENQASM 3.0;\n"
            'include "stdgates.inc";\n'
            "gate my_iswap(theta) a, b { rxx(theta) a, b; ryy(theta) a, b; }\n"
            "qubit[2] q;\n"
            "my_iswap(0.3) q[0], q[1];\n"
        )
        report = check_openqasm3(program)
        assert report.conformant is True
        assert report.custom_gates_declared == ("my_iswap",)
        assert report.gate_call_count == 1

    def test_custom_gate_without_qubit_parameter_is_rejected(self):
        report = check_openqasm3("OPENQASM 3.0;\ngate bad(theta) { }\n")
        assert report.conformant is False
        assert any("at least one qubit parameter" in issue for issue in report.issues)

    def test_malformed_gate_declaration_is_reported(self):
        # A ``gate`` keyword with no brace body cannot be parsed as a declaration.
        report = check_openqasm3("OPENQASM 3.0;\ngate broken a b\n")
        assert report.conformant is False
        assert any("malformed declaration" in issue for issue in report.issues)

    def test_unparseable_body_statement_is_unchecked(self):
        program = "OPENQASM 3.0;\ngate g a { 2bad; }\nqubit[1] q;\ng q[0];\n"
        report = check_openqasm3(program)
        assert "2bad" in report.unchecked_statements
        assert report.custom_gates_declared == ("g",)


class TestStructuralViolations:
    """Structural errors flip ``conformant`` to ``False`` with an issue."""

    def test_missing_version_header(self):
        report = check_openqasm3('include "stdgates.inc";\nqubit[1] q;\n')
        assert report.conformant is False
        assert report.qasm_version is None
        assert any("missing OPENQASM version" in issue for issue in report.issues)

    def test_version_must_be_first_statement(self):
        report = check_openqasm3('include "stdgates.inc";\nOPENQASM 3.0;\n')
        assert report.conformant is False
        assert any("must be the first statement" in issue for issue in report.issues)

    def test_unknown_gate_is_reported(self):
        report = check_openqasm3("OPENQASM 3.0;\nqubit[1] q;\nnope q[0];\n")
        assert report.conformant is False
        assert any("unknown gate 'nope'" in issue for issue in report.issues)

    def test_parameter_count_mismatch(self):
        report = check_openqasm3(
            'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[1] q;\nrz q[0];\n'
        )
        assert report.conformant is False
        assert any("expects 1 parameter(s), got 0" in issue for issue in report.issues)

    def test_too_many_parameters(self):
        report = check_openqasm3(
            'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[1] q;\nrz(0.1, 0.2) q[0];\n'
        )
        assert report.conformant is False
        assert any("expects 1 parameter(s), got 2" in issue for issue in report.issues)

    def test_qubit_operand_count_mismatch(self):
        report = check_openqasm3(
            'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[2] q;\ncx q[0];\n'
        )
        assert report.conformant is False
        assert any(
            "expects 2 qubit operand(s), got 1" in issue for issue in report.issues
        )

    def test_qubit_index_out_of_range(self):
        report = check_openqasm3(
            'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[2] q;\nrz(0.1) q[5];\n'
        )
        assert report.conformant is False
        assert any("out of range" in issue for issue in report.issues)

    def test_undeclared_register(self):
        report = check_openqasm3(
            'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[1] q;\nrz(0.1) p[0];\n'
        )
        assert report.conformant is False
        assert any("undeclared qubit register 'p'" in issue for issue in report.issues)

    def test_malformed_operand(self):
        report = check_openqasm3(
            'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[1] q;\nrx(0.1) q[0][1];\n'
        )
        assert report.conformant is False
        assert any("malformed qubit operand" in issue for issue in report.issues)

    def test_stdgates_used_without_include(self):
        report = check_openqasm3("OPENQASM 3.0;\nqubit[1] q;\nh q[0];\n")
        assert report.conformant is False
        assert any('without include "stdgates.inc"' in issue for issue in report.issues)

    def test_non_gate_statement_is_unchecked_not_failed(self):
        report = check_openqasm3("OPENQASM 3.0;\nqubit[1] q;\n2 + 2\n")
        assert "2 + 2" in report.unchecked_statements
        # An unchecked statement alone does not flip conformance.
        assert report.conformant is True

    def test_nested_brace_block_is_unchecked_not_split(self):
        # A ``cal``/``box`` block nests braces; it must be kept as one statement
        # (not split at the inner ``}``) and treated as outside the checked
        # subset rather than mis-parsed as a gate application.
        report = check_openqasm3("OPENQASM 3.0;\ncal { inner { } }\n")
        assert report.unchecked_statements == ("cal { inner { } }",)
        assert report.conformant is True


class TestReportContract:
    """The report exposes a deterministic audit record and validates input."""

    def test_to_audit_record_is_json_safe_and_complete(self):
        record = check_openqasm3(EMITTER_PROGRAM).to_audit_record()
        assert record["conformant"] is True
        assert record["qasm_version"] == "3.0"
        assert record["includes"] == ["stdgates.inc"]
        assert record["qubit_registers"] == [{"name": "q", "size": 2}]
        assert record["gate_call_count"] == 4
        assert record["stdgates_used"] == ["rz"]
        assert record["extension_gates_used"] == ["rxx", "ryy"]
        assert record["uses_non_stdgates_extensions"] is True
        assert record["issue_count"] == 0
        assert record["issues"] == []
        assert record["unchecked_statement_count"] == 0

    def test_audit_record_reports_issue_details(self):
        record = check_openqasm3("qubit[1] q;\n").to_audit_record()
        assert record["conformant"] is False
        assert record["issue_count"] == 1
        assert record["qasm_version"] is None

    def test_non_string_input_raises_type_error(self):
        with pytest.raises(TypeError, match="program must be a string"):
            check_openqasm3(42)

    def test_empty_program_is_non_conformant(self):
        report = check_openqasm3("")
        assert report.conformant is False
        assert report.qasm_version is None
        assert report.gate_call_count == 0

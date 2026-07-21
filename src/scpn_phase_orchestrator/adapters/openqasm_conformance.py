# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — OpenQASM 3 structural conformance checker

"""Deterministic structural conformance checker for OpenQASM 3 programs.

This module validates the *structural* conformance of an OpenQASM 3 program:
version header, includes, quantum-register declarations, custom ``gate``
declarations, and gate applications (``measure`` / ``reset`` / ``barrier`` are
recognised as non-gate operations). For every gate application it resolves the
gate name against a known registry and checks that the classical-parameter count
and qubit-operand count match the gate's arity, and that every indexed qubit
operand refers to a declared register within bounds.

Scope and honesty boundary
---------------------------
The checker is a static structural validator, **not** a full OpenQASM 3 parser,
type checker, or simulator: it does not evaluate parameter expressions, classical
control flow (``if`` / ``for`` / ``while``), subroutines (``def``), timing, or
pulse-level constructs. Statements outside the checked subset are surfaced in
:attr:`OpenQasm3ConformanceReport.unchecked_statements` and never silently pass
as "conformant".

The known-gate registry is split into two honestly-labelled tiers:

* :data:`STANDARD_LIBRARY_GATES` — the gates the OpenQASM 3 standard library
  header ``stdgates.inc`` actually defines. This table is transcribed from the
  reference definitions at
  https://github.com/openqasm/openqasm/blob/main/examples/stdgates.inc and the
  standard-library documentation, verified at source on 2026-07-21. It notably
  does **not** contain the two-qubit Pauli-rotation gates ``rxx`` / ``ryy`` /
  ``rzz`` / ``rzx``.
* :data:`BACKEND_EXTENSION_GATES` — the two-qubit Pauli-rotation gates that
  common OpenQASM 3 target backends (Qiskit's ``qiskit.qasm3`` importer and
  PennyLane's QASM loader) provide as builtins even though they are absent from
  ``stdgates.inc``. A program that uses these is portable to those backends but
  is **not** pure-``stdgates.inc`` conformant, so the report records their use
  explicitly via :attr:`OpenQasm3ConformanceReport.extension_gates_used` rather
  than blurring the distinction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = [
    "BACKEND_EXTENSION_GATES",
    "OPENQASM_BUILTIN_GATES",
    "STANDARD_LIBRARY_GATES",
    "OpenQasm3ConformanceReport",
    "check_openqasm3",
]

#: Gate name -> (classical-parameter count, qubit-operand count) for the two
#: primitive gates every OpenQASM 3 program may use without any include.
OPENQASM_BUILTIN_GATES: dict[str, tuple[int, int]] = {
    "U": (3, 1),
    "gphase": (1, 0),
}

#: Gate name -> (classical-parameter count, qubit-operand count) for the gates
#: defined by ``stdgates.inc``. Verified at source (openqasm/openqasm, 2026-07-21).
STANDARD_LIBRARY_GATES: dict[str, tuple[int, int]] = {
    # single-qubit, no parameters
    "x": (0, 1),
    "y": (0, 1),
    "z": (0, 1),
    "h": (0, 1),
    "s": (0, 1),
    "sdg": (0, 1),
    "t": (0, 1),
    "tdg": (0, 1),
    "sx": (0, 1),
    "id": (0, 1),
    # single-qubit, parametrised
    "p": (1, 1),
    "rx": (1, 1),
    "ry": (1, 1),
    "rz": (1, 1),
    "phase": (1, 1),
    "u1": (1, 1),
    "u2": (2, 1),
    "u3": (3, 1),
    # two-qubit, no parameters
    "cx": (0, 2),
    "cy": (0, 2),
    "cz": (0, 2),
    "ch": (0, 2),
    "swap": (0, 2),
    "CX": (0, 2),
    # two-qubit, parametrised
    "cp": (1, 2),
    "cphase": (1, 2),
    "crx": (1, 2),
    "cry": (1, 2),
    "crz": (1, 2),
    "cu": (4, 2),
    # three-qubit
    "ccx": (0, 3),
    "cswap": (0, 3),
}

#: Two-qubit Pauli-rotation gates absent from ``stdgates.inc`` but provided as
#: builtins by common target backends (Qiskit, PennyLane). Verified at source
#: (2026-07-21): none of these appear in ``stdgates.inc``.
BACKEND_EXTENSION_GATES: dict[str, tuple[int, int]] = {
    "rxx": (1, 2),
    "ryy": (1, 2),
    "rzz": (1, 2),
    "rzx": (1, 2),
}

_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_VERSION_RE = re.compile(r"^OPENQASM\s+(\d+(?:\.\d+)?)$")
_INCLUDE_RE = re.compile(r'^include\s+"([^"]+)"$')
_QUBIT_DECL_RE = re.compile(r"^qubit(?:\[(\d+)\])?\s+([A-Za-z_]\w*)$")
_QREG_DECL_RE = re.compile(r"^qreg\s+([A-Za-z_]\w*)(?:\[(\d+)\])?$")
_BIT_DECL_RE = re.compile(r"^(?:bit(?:\[\d+\])?|creg\s+[A-Za-z_]\w*(?:\[\d+\])?)\b")
_GATEDECL_HEAD_RE = re.compile(
    r"^gate\s+([A-Za-z_]\w*)\s*(?:\(([^)]*)\))?\s*([^{]*)\{(.*)\}$",
    re.DOTALL,
)
_CALL_RE = re.compile(r"^([A-Za-z_]\w*)\s*(?:\(([^)]*)\))?\s*(.*)$")
_OPERAND_RE = re.compile(r"^([A-Za-z_]\w*)(?:\[(\d+)\])?$")


def _strip_comments(program: str) -> str:
    """Return *program* with block and line comments removed.

    Parameters
    ----------
    program : str
        Raw OpenQASM 3 source text.

    Returns
    -------
    str
        The source with ``/* ... */`` and ``// ...`` comments stripped.
    """
    without_block = _BLOCK_COMMENT_RE.sub(" ", program)
    return _LINE_COMMENT_RE.sub("", without_block)


def _split_statements(program: str) -> list[str]:
    """Split *program* into trimmed top-level statements.

    Simple statements terminate at a ``;`` at brace depth zero; a ``gate``
    declaration terminates at the ``}`` that returns brace depth to zero.

    Parameters
    ----------
    program : str
        Comment-stripped OpenQASM 3 source text.

    Returns
    -------
    list[str]
        The non-empty, whitespace-collapsed statements in source order. A
        ``gate`` declaration retains its ``{ ... }`` body as one statement.
    """
    statements: list[str] = []
    buffer: list[str] = []
    depth = 0
    for char in program:
        if char == "{":
            depth += 1
            buffer.append(char)
        elif char == "}":
            depth = max(0, depth - 1)
            buffer.append(char)
            if depth == 0:
                statements.append("".join(buffer))
                buffer = []
        elif char == ";" and depth == 0:
            statements.append("".join(buffer))
            buffer = []
        else:
            buffer.append(char)
    trailing = "".join(buffer).strip()
    if trailing:
        statements.append(trailing)
    return [collapsed for stmt in statements if (collapsed := _collapse(stmt))]


def _collapse(text: str) -> str:
    """Return *text* with runs of whitespace collapsed to single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def _split_top_level(text: str) -> list[str]:
    """Split a comma-separated list, ignoring commas inside parentheses.

    Parameters
    ----------
    text : str
        A comma-separated fragment such as a parameter or operand list.

    Returns
    -------
    list[str]
        The non-empty comma-separated items with surrounding whitespace removed.
    """
    items: list[str] = []
    buffer: list[str] = []
    depth = 0
    for char in text:
        if char in "([{":
            depth += 1
            buffer.append(char)
        elif char in ")]}":
            depth = max(0, depth - 1)
            buffer.append(char)
        elif char == "," and depth == 0:
            items.append("".join(buffer))
            buffer = []
        else:
            buffer.append(char)
    items.append("".join(buffer))
    return [stripped for item in items if (stripped := item.strip())]


@dataclass(frozen=True)
class OpenQasm3ConformanceReport:
    """Structured result of an OpenQASM 3 structural conformance check."""

    conformant: bool
    qasm_version: str | None
    includes: tuple[str, ...]
    qubit_registers: tuple[tuple[str, int], ...]
    gate_call_count: int
    stdgates_used: tuple[str, ...]
    extension_gates_used: tuple[str, ...]
    custom_gates_declared: tuple[str, ...]
    issues: tuple[str, ...] = field(default_factory=tuple)
    unchecked_statements: tuple[str, ...] = field(default_factory=tuple)

    @property
    def uses_non_stdgates_extensions(self) -> bool:
        """Return whether the program uses gates absent from ``stdgates.inc``.

        Returns
        -------
        bool
            ``True`` when at least one applied gate is a backend extension
            (:data:`BACKEND_EXTENSION_GATES`) rather than a standard-library or
            in-program gate.
        """
        return bool(self.extension_gates_used)

    def to_audit_record(self) -> dict[str, object]:
        """Return a deterministic JSON-safe audit mapping of this report.

        Returns
        -------
        dict[str, object]
            A sorted, JSON-serialisable mapping suitable for embedding in a
            review manifest.
        """
        return {
            "conformant": self.conformant,
            "qasm_version": self.qasm_version,
            "includes": list(self.includes),
            "qubit_registers": [
                {"name": name, "size": size} for name, size in self.qubit_registers
            ],
            "gate_call_count": self.gate_call_count,
            "stdgates_used": list(self.stdgates_used),
            "extension_gates_used": list(self.extension_gates_used),
            "custom_gates_declared": list(self.custom_gates_declared),
            "uses_non_stdgates_extensions": self.uses_non_stdgates_extensions,
            "issue_count": len(self.issues),
            "issues": list(self.issues),
            "unchecked_statement_count": len(self.unchecked_statements),
        }


class _ConformanceState:
    """Mutable accumulator threaded through statement processing."""

    def __init__(self) -> None:
        self.version: str | None = None
        self.includes: list[str] = []
        self.qubit_registers: dict[str, int] = {}
        self.custom_gates: dict[str, tuple[int, int]] = {}
        self.gate_calls = 0
        self.stdgates_used: set[str] = set()
        self.extension_gates_used: set[str] = set()
        self.issues: list[str] = []
        self.unchecked: list[str] = []

    def resolve_gate(self, name: str) -> tuple[int, int] | None:
        """Return the ``(params, qubits)`` arity of *name*, or ``None``.

        Parameters
        ----------
        name : str
            The gate identifier used in an application.

        Returns
        -------
        tuple[int, int] or None
            The resolved arity from the builtin, standard-library, backend
            extension, or in-program registry; ``None`` when unresolved.
        """
        if name in OPENQASM_BUILTIN_GATES:
            return OPENQASM_BUILTIN_GATES[name]
        if name in STANDARD_LIBRARY_GATES:
            return STANDARD_LIBRARY_GATES[name]
        if name in BACKEND_EXTENSION_GATES:
            return BACKEND_EXTENSION_GATES[name]
        return self.custom_gates.get(name)


def _record_gate_family(state: _ConformanceState, name: str) -> None:
    """Record which known family *name* belongs to for the report tallies."""
    if name in STANDARD_LIBRARY_GATES or name in OPENQASM_BUILTIN_GATES:
        state.stdgates_used.add(name)
    elif name in BACKEND_EXTENSION_GATES:
        state.extension_gates_used.add(name)


def _check_operand(
    state: _ConformanceState,
    operand: str,
    *,
    context: str,
    local_qubits: frozenset[str],
) -> None:
    """Validate one qubit operand against declared registers/local qubits."""
    match = _OPERAND_RE.match(operand)
    if match is None:
        state.issues.append(f"{context}: malformed qubit operand {operand!r}")
        return
    name, index = match.group(1), match.group(2)
    if name in local_qubits:
        return
    if name not in state.qubit_registers:
        state.issues.append(f"{context}: undeclared qubit register {name!r}")
        return
    if index is not None:
        size = state.qubit_registers[name]
        if int(index) >= size:
            state.issues.append(
                f"{context}: qubit index {name}[{index}] out of range "
                f"(register size {size})"
            )


def _check_gate_application(
    state: _ConformanceState,
    statement: str,
    *,
    context: str,
    local_qubits: frozenset[str] = frozenset(),
    tally: bool = True,
) -> bool:
    """Validate a gate-application statement; return ``True`` if it parsed.

    Parameters
    ----------
    state : _ConformanceState
        The accumulator receiving issues and tallies.
    statement : str
        A single, comment-free, whitespace-collapsed gate application without
        its terminating ``;``.
    context : str
        Human-readable location prefix used in issue messages.
    local_qubits : frozenset[str], optional
        Formal qubit names in scope (non-empty only inside a gate body).
    tally : bool, optional
        Whether to count the call and record its gate family (suppressed for
        gate-body statements so report tallies reflect the top-level circuit).

    Returns
    -------
    bool
        ``True`` when the statement is shaped like a gate application (even if
        it raised issues); ``False`` when it does not look like one.
    """
    call = _CALL_RE.match(statement)
    if call is None:
        return False
    name, params, operands = call.group(1), call.group(2), call.group(3)
    arity = state.resolve_gate(name)
    if arity is None:
        state.issues.append(f"{context}: unknown gate {name!r}")
        return True
    expected_params, expected_qubits = arity
    param_items = _split_top_level(params) if params is not None else []
    operand_items = _split_top_level(operands)
    if len(param_items) != expected_params:
        state.issues.append(
            f"{context}: gate {name!r} expects {expected_params} parameter(s), "
            f"got {len(param_items)}"
        )
    if len(operand_items) != expected_qubits:
        state.issues.append(
            f"{context}: gate {name!r} expects {expected_qubits} qubit operand(s), "
            f"got {len(operand_items)}"
        )
    for operand in operand_items:
        _check_operand(state, operand, context=context, local_qubits=local_qubits)
    if tally:
        state.gate_calls += 1
        _record_gate_family(state, name)
    return True


def _process_gate_declaration(state: _ConformanceState, statement: str) -> None:
    """Parse and validate a custom ``gate`` declaration."""
    head = _GATEDECL_HEAD_RE.match(statement)
    if head is None:
        state.issues.append(f"gate declaration: malformed declaration {statement!r}")
        return
    name, params, qubits, body = head.groups()
    param_items = _split_top_level(params) if params else []
    qubit_items = _split_top_level(qubits) if qubits else []
    if not qubit_items:
        state.issues.append(
            f"gate declaration {name!r}: at least one qubit parameter is required"
        )
        return
    state.custom_gates[name] = (len(param_items), len(qubit_items))
    local_qubits = frozenset(qubit_items)
    for raw_body_stmt in body.split(";"):
        body_stmt = _collapse(raw_body_stmt)
        if not body_stmt:
            continue
        parsed = _check_gate_application(
            state,
            body_stmt,
            context=f"gate {name!r} body",
            local_qubits=local_qubits,
            tally=False,
        )
        if not parsed:
            state.unchecked.append(body_stmt)


def _process_statement(state: _ConformanceState, statement: str, index: int) -> None:
    """Classify and validate a single top-level statement."""
    version = _VERSION_RE.match(statement)
    if version is not None:
        if index != 0:
            state.issues.append(
                "OPENQASM version statement must be the first statement"
            )
        state.version = version.group(1)
        return
    include = _INCLUDE_RE.match(statement)
    if include is not None:
        state.includes.append(include.group(1))
        return
    if statement.startswith("gate "):
        _process_gate_declaration(state, statement)
        return
    qubit = _QUBIT_DECL_RE.match(statement)
    if qubit is not None:
        size = int(qubit.group(1)) if qubit.group(1) is not None else 1
        state.qubit_registers[qubit.group(2)] = size
        return
    qreg = _QREG_DECL_RE.match(statement)
    if qreg is not None:
        size = int(qreg.group(2)) if qreg.group(2) is not None else 1
        state.qubit_registers[qreg.group(1)] = size
        return
    if _BIT_DECL_RE.match(statement):
        return
    first_token = statement.split(maxsplit=1)[0]
    if first_token in {"measure", "reset", "barrier"} or "->" in statement:
        return
    if "{" in statement:
        # A block construct that is not a gate declaration (e.g. ``cal`` /
        # ``box``) is outside the checked subset, not a gate application.
        state.unchecked.append(statement)
        return
    if not _check_gate_application(state, statement, context=f"statement {index}"):
        state.unchecked.append(statement)


def check_openqasm3(program: str) -> OpenQasm3ConformanceReport:
    """Check the structural conformance of an OpenQASM 3 program.

    The function never raises: any structural violation is recorded as an issue
    and reflected in :attr:`OpenQasm3ConformanceReport.conformant`.

    Parameters
    ----------
    program : str
        The OpenQASM 3 source text to validate.

    Returns
    -------
    OpenQasm3ConformanceReport
        The structural conformance report. ``conformant`` is ``True`` only when
        a version header is present and no structural issue was found.

    Raises
    ------
    TypeError
        If *program* is not a string.
    """
    if not isinstance(program, str):
        raise TypeError("program must be a string")
    state = _ConformanceState()
    statements = _split_statements(_strip_comments(program))
    for index, statement in enumerate(statements):
        _process_statement(state, statement, index)
    if state.version is None:
        state.issues.append("missing OPENQASM version header")
    # Only ``stdgates.inc`` gates require the include; the ``U``/``gphase``
    # builtins are available without any include.
    uses_stdgates = any(gate in STANDARD_LIBRARY_GATES for gate in state.stdgates_used)
    if uses_stdgates and "stdgates.inc" not in state.includes:
        state.issues.append(
            'standard-library gates used without include "stdgates.inc"'
        )
    return OpenQasm3ConformanceReport(
        conformant=not state.issues,
        qasm_version=state.version,
        includes=tuple(state.includes),
        qubit_registers=tuple(sorted(state.qubit_registers.items())),
        gate_call_count=state.gate_calls,
        stdgates_used=tuple(sorted(state.stdgates_used)),
        extension_gates_used=tuple(sorted(state.extension_gates_used)),
        custom_gates_declared=tuple(sorted(state.custom_gates)),
        issues=tuple(state.issues),
        unchecked_statements=tuple(state.unchecked),
    )

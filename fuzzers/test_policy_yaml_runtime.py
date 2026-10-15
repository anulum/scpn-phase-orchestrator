# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Policy fuzzer production-runtime contracts

"""Exercise the policy fuzz callback through its canonical loader dependency."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import FrameType

from fuzzers import policy_yaml_fuzzer
from scpn_phase_orchestrator.supervisor.policy_rules import (
    PolicyRule,
    load_policy_rules,
)

ROOT = Path(__file__).resolve().parents[1]


class PolicyFuzzerRuntimeTests(unittest.TestCase):
    """Verify canonical parsing, bounded input conversion and file cleanup."""

    def test_canonical_loader_executes_and_cleans_each_input(self) -> None:
        """Observe actual loader arguments/results across the callback boundary."""
        valid = (ROOT / "domainpacks/minimal_domain/policy.yaml").read_bytes()
        expected_rules = load_policy_rules(
            ROOT / "domainpacks/minimal_domain/policy.yaml"
        )
        self.assertTrue(expected_rules)
        bounded = valid.ljust(8192, b" ")
        cases: tuple[tuple[str, bytes, bytes, list[PolicyRule] | None], ...] = (
            ("domainpack", valid, valid, expected_rules),
            ("malformed", b"[unterminated", b"[unterminated", None),
            ("byte-limit", bounded + b"[unterminated", bounded, expected_rules),
            ("invalid-utf8", b"\xff" + valid + b"\xfe", valid, expected_rules),
            ("empty", b"", b"", []),
        )
        observed_inputs: list[bytes] = []
        observed_results: list[object] = []

        def observe(frame: FrameType, event: str, value: object) -> None:
            if frame.f_code is not load_policy_rules.__code__:
                return
            if event == "call":
                parameter: object = frame.f_locals["path"]
                if not isinstance(parameter, (str, Path)):
                    raise AssertionError("policy loader received no file path")
                observed_inputs.append(Path(parameter).read_bytes())
            elif event == "return":
                observed_results.append(value)

        for label, payload, written, result in cases:
            with self.subTest(input=label):
                temporary_root = Path(tempfile.gettempdir())
                before = set(temporary_root.glob("spo-policy-*.yaml"))
                observed_inputs.clear()
                observed_results.clear()
                previous_profile = sys.getprofile()
                sys.setprofile(observe)
                try:
                    policy_yaml_fuzzer.TestOneInput(payload)
                finally:
                    sys.setprofile(previous_profile)
                self.assertEqual(observed_inputs, [written])
                self.assertEqual(observed_results, [result])
                self.assertEqual(set(temporary_root.glob("spo-policy-*.yaml")), before)

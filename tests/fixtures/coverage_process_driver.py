# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — subprocess coverage integration

"""Exercise the public rate limiter from an isolated Python child process."""

from __future__ import annotations

import json
import subprocess
import sys


def test_child_rate_limit_validation() -> None:
    """A child accepts valid requests and rejects an overflowing clock interval."""
    child = """
import json
from scpn_phase_orchestrator.runtime.network_security import TokenBucketRateLimiter

limiter = TokenBucketRateLimiter(limit_per_minute=2, burst_capacity=2)
accepted = [limiter.allow("valid", now=0.0) for _ in range(3)]
assert limiter.allow("overflow", now=-1e308)
try:
    limiter.allow("overflow", now=1e308)
except ValueError as exc:
    error = str(exc)
else:
    raise AssertionError("overflowing clock interval was accepted")
print(json.dumps({"accepted": accepted, "error": error}))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", child],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    payload = json.loads(completed.stdout)
    assert payload["accepted"] == [True, True, False]
    assert payload["error"] == "rate-limit timestamp difference must remain finite"


if __name__ == "__main__":
    test_child_rate_limit_validation()

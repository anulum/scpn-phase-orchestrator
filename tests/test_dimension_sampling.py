# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Actual threaded Rust pair sampling

"""Bind native draw quotas, retries and remainder allocation to scalar counts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest


def _requested_distances(quota: int, seed: int, threads: int) -> list[float]:
    """Evaluate the documented legacy modular LCG law without a numerical kernel."""
    modulus = 2**64
    chunks = min(threads, quota)
    distances = []
    for chunk in range(chunks):
        needed = quota // chunks + int(chunk < quota % chunks)
        state = seed ^ ((chunk * 0x9E3779B97F4A7C15) % modulus)
        accepted = 0
        for _ in range(1024):
            if accepted == needed:
                break
            state = (state * 6364136223846793005 + 1442695040888963407) % modulus
            first = (state >> 33) % 12
            state = (state * 6364136223846793005 + 1442695040888963407) % modulus
            second = (state >> 33) % 12
            if first != second:
                distances.append(float(abs(first - second)))
                accepted += 1
        assert accepted == needed
    assert len(distances) == quota
    return distances


@pytest.mark.native_runtime
@pytest.mark.parametrize("threads", [1, 3])
def test_actual_rust_sample_budget_remainder_and_normalization(threads: int) -> None:
    """Fresh Rayon pools consume the exact valid-pair quota, including remainders."""
    kernel = import_module("spo_kernel")
    filename = kernel.__file__
    assert isinstance(filename, str)
    root = Path(filename).resolve().parent.parent
    program = """
import importlib, inspect, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, sys.argv[1])
kernel = importlib.import_module('spo_kernel')
assert Path(kernel.__file__).resolve().is_relative_to(Path(sys.argv[1]))
assert inspect.isbuiltin(kernel.correlation_integral_rust)
rows = []
for seed in [0, 7, 42, 2**64 - 1]:
    for quota in [1, 2, 5, 8, 37]:
        values = kernel.correlation_integral_rust(
            np.arange(12, dtype=np.float64), 12, 1,
            np.array([0., 2., 4., 8., 20.]), quota, seed)
        rows.append({'seed': seed, 'quota': quota, 'integral': values.tolist()})
print(json.dumps(rows))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program, str(root)],
        env=dict(os.environ, RAYON_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS="1"),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    rows = json.loads(completed.stdout)
    assert len(rows) == 20
    for row in rows:
        distances = _requested_distances(int(row["quota"]), int(row["seed"]), threads)
        expected = [
            sum(distance < radius for distance in distances) / len(distances)
            for radius in [0.0, 2.0, 4.0, 8.0, 20.0]
        ]
        np.testing.assert_allclose(row["integral"], expected, rtol=0.0, atol=1e-15)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Validated sleep API dispatch measurements

"""Compare validated public sleep APIs, including Python/native marshalling.

Run from the repository root with the release kernel installed:
``taskset -c 0 .venv/bin/python benchmarks/sleep_staging_dispatch.py``.
JSON is emitted on stdout. CPU affinity is recorded but is not core isolation;
these measurements are local regression evidence, not production speed claims.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import statistics
import sys
import timeit
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import numpy as np

from scpn_phase_orchestrator.monitor.sleep_staging import (
    classify_sleep_stage,
    ultradian_phase,
)


def measure(
    calls: dict[str, Callable[[], str | float]], *, number: int, repeat: int
) -> dict[str, object]:
    """Measure full API calls after verifying exact output agreement.

    Parameters
    ----------
    calls : dict
        Named public default, Python and Rust invocations for one input.
    number : int
        Invocations per sample.
    repeat : int
        Sample count; execution order reverses between samples.

    Returns
    -------
    dict
        Outputs, raw microseconds per call and median timings.

    Raises
    ------
    ValueError
        If the backend outputs disagree.
    """
    outputs = {name: call() for name, call in calls.items()}
    if len(set(outputs.values())) != 1:
        raise ValueError(f"sleep backend output mismatch: {outputs}")
    samples: dict[str, list[float]] = {name: [] for name in calls}
    order = list(calls)
    for iteration in range(repeat):
        for name in order if iteration % 2 == 0 else order[::-1]:
            elapsed = timeit.timeit(calls[name], number=number)
            samples[name].append(elapsed * 1e6 / number)
    return {
        "outputs": outputs,
        "number": number,
        "samples_us": samples,
        "median_us": {
            name: statistics.median(values) for name, values in samples.items()
        },
    }


def main() -> None:
    """Emit reproducible, non-isolated public-API timing and parity evidence.

    Raises
    ------
    RuntimeError
        If the optional native extension is unavailable.
    ValueError
        If Python, Rust and default results disagree.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--number", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--epochs", type=int, nargs="+", default=[0, 2, 1000, 10000])
    arguments = parser.parse_args()
    if (
        arguments.number < 1
        or arguments.repeat < 1
        or any(size < 0 for size in arguments.epochs)
    ):
        parser.error("number and repeat must be positive; epochs must be non-negative")
    kernel = (
        importlib.util.find_spec("spo_kernel.spo_kernel")
        if importlib.util.find_spec("spo_kernel") is not None
        else None
    )
    if kernel is None or kernel.origin is None:
        raise RuntimeError("the release spo_kernel extension is required")
    extension = Path(kernel.origin)
    affinity = sorted(os.sched_getaffinity(0))
    governors = {
        str(cpu): governor.read_text().strip()
        for cpu in affinity
        if (
            governor := Path(
                f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"
            )
        ).exists()
    }
    cpu_model = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        "unknown",
    )
    load_before = os.getloadavg()
    results: list[dict[str, object]] = []
    for order_parameter, desync in [
        (0.1, False),
        (0.25, True),
        (0.35, False),
        (0.5, False),
        (0.8, False),
    ]:
        calls: dict[str, Callable[[], str | float]] = {
            "default": partial(classify_sleep_stage, order_parameter, desync),
            "python": partial(
                classify_sleep_stage, order_parameter, desync, backend="python"
            ),
            "rust": partial(
                classify_sleep_stage, order_parameter, desync, backend="rust"
            ),
        }
        results.append(
            {
                "api": "classify_sleep_stage",
                "order_parameter": order_parameter,
                "functional_desync": desync,
                **measure(calls, number=arguments.number * 10, repeat=arguments.repeat),
            }
        )
    for epochs in arguments.epochs:
        timestamps = np.arange(epochs, dtype=np.float64) * 30.0
        for position in ("none", "first", "last"):
            stages = ["N2"] * epochs
            if epochs and position != "none":
                stages[0 if position == "first" else -1] = "N3"
            calls = {
                "default": partial(ultradian_phase, timestamps, stages),
                "python": partial(
                    ultradian_phase, timestamps, stages, backend="python"
                ),
                "rust": partial(ultradian_phase, timestamps, stages, backend="rust"),
            }
            results.append(
                {
                    "api": "ultradian_phase",
                    "epochs": epochs,
                    "n3_position": position,
                    "executes_backend": epochs > 0,
                    **measure(calls, number=arguments.number, repeat=arguments.repeat),
                }
            )
    print(
        json.dumps(
            {
                "recorded_at": datetime.now(UTC).isoformat(),
                "command": [
                    "python",
                    "benchmarks/sleep_staging_dispatch.py",
                    *sys.argv[1:],
                ],
                "method": (
                    "validated public APIs; warm output check; alternating timing order"
                ),
                "isolation": "non-isolated; affinity only, no reserved cores",
                "other_heavy_jobs": "not controlled",
                "cpu_model": cpu_model,
                "affinity": affinity,
                "governors": governors,
                "load_before": load_before,
                "load_after": os.getloadavg(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "spo_kernel": importlib.metadata.version("spo-kernel"),
                "extension_name": extension.name,
                "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
                "source_sha256": hashlib.sha256(
                    (
                        Path(__file__).resolve().parents[1]
                        / "src/scpn_phase_orchestrator/monitor/sleep_staging.py"
                    ).read_bytes()
                ).hexdigest(),
                "results": results,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()

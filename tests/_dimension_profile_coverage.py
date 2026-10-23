# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Source-bound dimension child measurements

"""Retain measured child arcs only when executed and retained sources match."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.util import find_spec
from pathlib import Path

from coverage import Coverage, CoverageData

ROOT = Path(__file__).resolve().parents[1]


def profile_environment() -> dict[str, str]:
    """Declare the retained parent installation used to bind child measurements.

    Returns
    -------
    dict[str, str]
        Inherited process environment with serial numerical threads and actual
        parent package/benchmark roots, unless explicitly supplied by the caller.
    """
    spec = find_spec("scpn_phase_orchestrator")
    assert spec is not None and spec.origin is not None
    return dict(
        os.environ,
        JULIA_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        SPO_DIMENSION_COVERAGE_SOURCE_ROOT=os.environ.get(
            "SPO_DIMENSION_COVERAGE_SOURCE_ROOT",
            str(Path(spec.origin).resolve().parent),
        ),
        SPO_DIMENSION_COVERAGE_BENCHMARK_ROOT=os.environ.get(
            "SPO_DIMENSION_COVERAGE_BENCHMARK_ROOT", str(ROOT / "benchmarks")
        ),
    )


@contextmanager
def measure_profile(exported: Path, label: str) -> Iterator[None]:
    """Measure a real profile and bind its child sources before export removal.

    Parameters
    ----------
    exported : Path
        Finite source export containing the actual executed package and any
        benchmark namespace modules. Julia must initialize before collection.
    label : str
        Controlled profile/context label, consisting of letters and hyphens.

    Yields
    ------
    None
        Execute the public profile while branch measurement is active.

    Raises
    ------
    ValueError
        If an executed file is outside the export or differs from its retained
        counterpart. No child database is published on a failed source binding.

    Notes
    -----
    Collection is enabled by COVERAGE_FILE. The source-root environment overrides
    identify the actual retained parent package and benchmark directories; the
    default roots support source-checkout runs. Each child creates a distinct
    combinable database and a provenance sidecar outside the retired export.
    SPO_DIMENSION_COVERAGE_PACKAGE_ONLY=1 excludes benchmark scripts when the
    receiving CI profile admits only installed package members.
    """
    requested = os.environ.get("COVERAGE_FILE")
    if not requested:
        yield
        return
    package = (exported / "src/scpn_phase_orchestrator").resolve()
    benchmarks = (exported / "benchmarks").resolve()
    retained_package = Path(
        os.environ.get(
            "SPO_DIMENSION_COVERAGE_SOURCE_ROOT",
            str(ROOT / "src/scpn_phase_orchestrator"),
        )
    ).resolve()
    retained_benchmarks = Path(
        os.environ.get(
            "SPO_DIMENSION_COVERAGE_BENCHMARK_ROOT", str(ROOT / "benchmarks")
        )
    ).resolve()
    measurement = Coverage(
        config_file=False,
        data_file=str(exported / ".coverage.profile-local"),
        branch=True,
        include=[
            str(package / "monitor/dimension.py"),
            str(package / "monitor/_dimension_numerics.py"),
            str(package / "experimental/accelerators/monitor/_dimension_*.py"),
            *(
                [
                    str(benchmarks / "dimension_benchmark.py"),
                    str(benchmarks / "_dimension_reference.py"),
                ]
                if os.environ.get("SPO_DIMENSION_COVERAGE_PACKAGE_ONLY") != "1"
                else []
            ),
        ],
    )
    measurement.start()
    measurement.switch_context(f"dimension-profile:{label}")
    try:
        yield
    finally:
        measurement.stop()
        bindings: dict[str, dict[str, str]] = {}

        def bind_source(filename: str) -> str:
            """Accept only exact executed bytes at the declared retained root."""
            source = Path(filename).resolve()
            if source.is_relative_to(package):
                retained = retained_package / source.relative_to(package)
            elif source.is_relative_to(benchmarks):
                retained = retained_benchmarks / source.relative_to(benchmarks)
            else:
                raise ValueError("dimension child source is outside its export")
            executed = source.read_bytes()
            if executed != retained.read_bytes():
                raise ValueError("dimension child source differs from retained source")
            bindings[str(source)] = {
                "retained": str(retained),
                "sha256": hashlib.sha256(executed).hexdigest(),
            }
            return str(retained)

        measured = measurement.get_data()
        for filename in measured.measured_files():
            bind_source(filename)
        directory = Path(requested).resolve().parent
        directory.mkdir(parents=True, exist_ok=True)
        suffix = f"{label}-{os.getpid()}-{time.time_ns()}"
        destination = directory / f".coverage.dimension-{suffix}"
        bound = CoverageData(basename=str(destination))
        bound.update(measured, map_path=bind_source)
        bound.write()
        receipt = {
            "label": label,
            "database": str(destination),
            "database_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "branch": bound.has_arcs(),
            "bindings": bindings,
        }
        with (directory / f"dimension-coverage-{suffix}.json").open("x") as output:
            output.write(json.dumps(receipt, indent=2) + "\n")

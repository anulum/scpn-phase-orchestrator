# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Executed browser WebGPU phase benchmark

"""Expose the real browser phase benchmark through the benchmarks namespace."""

from tools.run_webgpu_phase_benchmark import benchmark_webgpu_phase_wrapping, main

__all__ = ["benchmark_webgpu_phase_wrapping", "main"]

if __name__ == "__main__":
    raise SystemExit(main())

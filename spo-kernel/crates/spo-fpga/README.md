<!--
SPDX-License-Identifier: AGPL-3.0-or-later
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# spo-fpga — FPGA Kuramoto Solver

Verilog implementation of an N=16 coupled-oscillator Kuramoto solver
targeting the Xilinx Zynq-7020 SoC (~15K LUT budget).

## Architecture

```
                 AXI-Lite bus (PS ↔ PL)
                        │
              ┌─────────┴─────────┐
              │   kuramoto_core    │
              │                    │
              │  ┌──────────────┐  │
              │  │  K_ij BRAM   │  │  256 × 32-bit coupling coefficients
              │  │  (N×N = 256) │  │  written via AXI-Lite from ARM core
              │  └──────┬───────┘  │
              │         │          │
              │  ┌──────┴───────┐  │
              │  │  FSM         │  │  IDLE → DIFF → CORDIC → ACCUM → UPDATE → DONE
              │  │  (6 states)  │  │
              │  └──────┬───────┘  │
              │         │          │
              │  ┌──────┴───────┐  │
              │  │ cordic_sincos│  │  4-stage pipelined CORDIC (Q16.16)
              │  │ (4 stages)   │  │  computes sin(θ_j − θ_i), cos(θ_j − θ_i)
              │  └──────────────┘  │
              │                    │
              │  phase_in [N×32]   │  ← from PS or external ADC
              │  phase_out [N×32]  │  → to PS or DAC
              │  omega_in [N×32]   │  ← natural frequencies
              │  dt [32]           │  ← integration step size
              └────────────────────┘
```

## Design Decisions

**Fixed-point Q16.16.** Avoids floating-point DSP usage entirely.
The 16-bit fractional part gives ~1.5×10⁻⁵ rad resolution,
sufficient for Kuramoto dynamics where phase differences of O(0.01) matter.

**4-stage CORDIC.** Each stage adds one bit of angular precision.
Four stages give ~0.05 rad worst-case error — acceptable for the
coupling term sin(θ_j − θ_i) where the sigmoid-like shape near zero
is most important. Extending to 8 stages doubles latency for <0.001 rad
gain; not worth the pipeline depth on Zynq-7020.

**Quadrant folding.** Input angles are folded to [-π/2, π/2] before
CORDIC rotation, with output sign correction. This keeps the CORDIC
convergence domain valid without requiring extra iterations.

**Sequential i,j sweep.** The FSM iterates over all (i, j≠i) pairs
sequentially, reusing a single CORDIC instance. For N=16 this means
16×15 = 240 CORDIC evaluations per Kuramoto step. At 100 MHz with
~6 cycles per evaluation, one full step takes ~15 μs — well within
the 1 ms control deadlines typical for phase-locked loop applications.

**AXI-Lite coupling matrix.** The N×N coupling matrix K_ij lives in
block RAM and is programmed from the ARM PS via AXI-Lite. This allows
runtime reconfiguration of coupling topology without PL re-synthesis.
Address map: `K[i][j]` at byte address `(i*16 + j) * 4`.

## Resource Estimate (Zynq-7020)

| Resource     | Estimated | Available | Utilisation |
|-------------|-----------|-----------|-------------|
| LUTs        | ~8K       | 17,600    | ~45%        |
| FFs         | ~4K       | 35,200    | ~11%        |
| BRAM (36Kb) | 1         | 60        | ~2%         |
| DSP48       | 2         | 80        | ~3%         |

The dominant cost is the 32-bit multiplier in the accumulation stage
(K_ij × sin result). Two DSP48 slices handle the 32×32→64 bit multiply.

## Files

| File                    | Description                              |
|------------------------|------------------------------------------|
| `src/kuramoto_core.v`  | Top-level module + CORDIC sub-module     |

## Integration

The module expects the Zynq PS to:

1. Write coupling coefficients K_ij via AXI-Lite before starting.
2. Load `phase_in` and `omega_in` bus values (directly or via DMA).
3. Assert `start` for one clock cycle.
4. Wait for `done` to read `phase_out`.

Typical use: the PS runs the high-level SCPN supervisor in Python/Rust,
offloading the inner Kuramoto integration loop to the PL for
deterministic sub-millisecond latency.

## Simulation

Use any Verilog simulator (Verilator, Icarus, Vivado XSIM):

```bash
iverilog -o kuramoto_tb src/kuramoto_core.v tb/kuramoto_core_tb.v
vvp kuramoto_tb
```

A testbench is not yet included — synthesis-only verification against
the Python/Rust reference implementations is the current validation path.

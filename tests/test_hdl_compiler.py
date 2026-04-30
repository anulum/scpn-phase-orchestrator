# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — HDL compiler tests

from __future__ import annotations

import re
from typing import get_type_hints

import numpy as np
import pytest

from scpn_phase_orchestrator.actuation.hdl_compiler import (
    KuramotoVerilogCompiler,
    _q16_16,
)


class TestQ1616Encoding:
    def test_zero(self) -> None:
        assert _q16_16(0.0) == 0x00000000

    def test_one(self) -> None:
        assert _q16_16(1.0) == 0x00010000

    def test_pi_over_two(self) -> None:
        assert _q16_16(np.pi / 2.0) == 0x00019220

    def test_negative_one(self) -> None:
        assert _q16_16(-1.0) == 0xFFFF0000

    def test_quarter(self) -> None:
        assert _q16_16(0.25) == 0x00004000

    def test_saturates_positive(self) -> None:
        assert _q16_16(1e9) == 0x7FFFFFFF

    def test_saturates_negative(self) -> None:
        assert _q16_16(-1e9) == 0x80000000


class TestModuleStructure:
    def test_public_array_contracts_are_parameterised(self) -> None:
        hints = get_type_hints(KuramotoVerilogCompiler.compile)
        for param in ("knm", "omegas"):
            assert "numpy.ndarray" in str(hints[param])
            assert "float64" in str(hints[param])

    def test_module_header_uses_n(self) -> None:
        compiler = KuramotoVerilogCompiler(n_oscillators=4)
        verilog = compiler.compile(np.zeros((4, 4)), np.ones(4), 0.01)
        assert "module kuramoto_mesh_4" in verilog
        assert "endmodule" in verilog

    def test_module_emits_n_state_registers(self) -> None:
        compiler = KuramotoVerilogCompiler(n_oscillators=5)
        verilog = compiler.compile(np.zeros((5, 5)), np.ones(5), 0.01)
        for i in range(5):
            assert f"reg signed [WIDTH-1:0] theta_{i};" in verilog

    def test_emits_output_wiring_per_oscillator(self) -> None:
        compiler = KuramotoVerilogCompiler(n_oscillators=3)
        verilog = compiler.compile(np.zeros((3, 3)), np.ones(3), 0.01)
        for i in range(3):
            assert f"assign theta_out[{i}] = theta_{i};" in verilog


class TestSynthesisability:
    def test_no_system_tasks(self) -> None:
        compiler = KuramotoVerilogCompiler(n_oscillators=4)
        knm = np.ones((4, 4)) * 0.3
        np.fill_diagonal(knm, 0.0)
        verilog = compiler.compile(knm, np.ones(4), 0.01)
        # $sin, $cos, $display, $finish, $time etc. are simulation-only.
        # $signed / $unsigned are cast *operators* and are synthesisable.
        simulation_only = (
            r"\$(sin|cos|tan|exp|log|sqrt|display|write|finish|stop|monitor|"
            r"time|random|fopen|fclose|fwrite|fscanf|readmemh|readmemb|dumpfile)"
            r"\b"
        )
        assert not re.search(simulation_only, verilog), (
            "simulation-only system tasks must not appear in synthesisable output"
        )

    def test_no_real_type(self) -> None:
        compiler = KuramotoVerilogCompiler(n_oscillators=4)
        knm = np.eye(4) * 0.5
        verilog = compiler.compile(knm, np.ones(4), 0.01)
        assert not re.search(r"\breal\b", verilog)

    def test_fixed_point_dt_constant(self) -> None:
        compiler = KuramotoVerilogCompiler(n_oscillators=2)
        verilog = compiler.compile(np.zeros((2, 2)), np.ones(2), 0.01)
        expected = f"{_q16_16(0.01):08X}"
        assert f"DT_Q = 32'h{expected};" in verilog

    def test_fixed_point_omega_constants(self) -> None:
        compiler = KuramotoVerilogCompiler(n_oscillators=2)
        omegas = np.array([0.5, -0.25])
        verilog = compiler.compile(np.zeros((2, 2)), omegas, 0.01)
        assert f"OMEGA_0 = 32'h{_q16_16(0.5):08X};" in verilog
        assert f"OMEGA_1 = 32'h{_q16_16(-0.25):08X};" in verilog

    def test_fixed_point_coupling_constants(self) -> None:
        compiler = KuramotoVerilogCompiler(n_oscillators=3)
        knm = np.zeros((3, 3))
        knm[0, 1] = 0.7
        knm[2, 1] = -0.3
        verilog = compiler.compile(knm, np.ones(3), 0.01)
        assert f"K_0_1 = 32'h{_q16_16(0.7):08X};" in verilog
        assert f"K_2_1 = 32'h{_q16_16(-0.3):08X};" in verilog


class TestCordicInstantiation:
    def test_active_edges_get_cordic_instances(self) -> None:
        compiler = KuramotoVerilogCompiler(n_oscillators=4)
        knm = np.zeros((4, 4))
        knm[0, 1] = 0.5
        knm[1, 2] = 0.5
        knm[2, 3] = 0.5
        verilog = compiler.compile(knm, np.ones(4), 0.01)
        assert "u_cordic_0_1" in verilog
        assert "u_cordic_1_2" in verilog
        assert "u_cordic_2_3" in verilog
        # Non-edges must NOT produce CORDIC instances.
        assert "u_cordic_3_0" not in verilog

    def test_zero_coupling_produces_no_cordic(self) -> None:
        compiler = KuramotoVerilogCompiler(n_oscillators=3)
        verilog = compiler.compile(np.zeros((3, 3)), np.ones(3), 0.01)
        # Comment mentioning the CORDIC primitive is OK; what must not appear
        # is an instantiation: `cordic_sincos #(...) u_name (...)`.
        assert "u_cordic_" not in verilog
        assert not re.search(r"cordic_sincos\s+#", verilog)

    def test_diagonal_coupling_ignored(self) -> None:
        compiler = KuramotoVerilogCompiler(n_oscillators=3)
        knm = np.eye(3) * 0.5
        verilog = compiler.compile(knm, np.ones(3), 0.01)
        for i in range(3):
            assert f"u_cordic_{i}_{i}" not in verilog

    def test_cordic_params_propagate(self) -> None:
        compiler = KuramotoVerilogCompiler(n_oscillators=2, cordic_stages=6)
        knm = np.zeros((2, 2))
        knm[0, 1] = 0.4
        verilog = compiler.compile(knm, np.ones(2), 0.01)
        assert "CORDIC_STAGES = 6" in verilog
        assert ".STAGES(CORDIC_STAGES)" in verilog


class TestSummationChain:
    def test_each_oscillator_gets_dtheta(self) -> None:
        compiler = KuramotoVerilogCompiler(n_oscillators=3)
        knm = np.zeros((3, 3))
        knm[0, 1] = 0.5
        knm[2, 1] = 0.5
        verilog = compiler.compile(knm, np.ones(3), 0.01)
        for i in range(3):
            assert f"assign dtheta_{i} = " in verilog

    def test_dtheta_scaled_by_dt(self) -> None:
        compiler = KuramotoVerilogCompiler(n_oscillators=2)
        knm = np.zeros((2, 2))
        knm[0, 1] = 0.3
        verilog = compiler.compile(knm, np.ones(2), 0.01)
        assert "dtheta_dt_0 = $signed(dtheta_0) * $signed(DT_Q);" in verilog
        assert "dtheta_dt_1 = $signed(dtheta_1) * $signed(DT_Q);" in verilog

    def test_theta_update_is_additive(self) -> None:
        compiler = KuramotoVerilogCompiler(n_oscillators=2)
        knm = np.zeros((2, 2))
        knm[0, 1] = 0.3
        verilog = compiler.compile(knm, np.ones(2), 0.01)
        assert "theta_0 <= theta_0 + dtheta_scaled_0;" in verilog
        assert "theta_1 <= theta_1 + dtheta_scaled_1;" in verilog

    def test_reset_initialises_to_zero(self) -> None:
        compiler = KuramotoVerilogCompiler(n_oscillators=3)
        verilog = compiler.compile(np.zeros((3, 3)), np.ones(3), 0.01)
        for i in range(3):
            assert f"theta_{i} <= 32'sd0;" in verilog


class TestValidation:
    def test_rejects_zero_oscillators(self) -> None:
        with pytest.raises(ValueError, match="n_oscillators must be >= 1"):
            KuramotoVerilogCompiler(n_oscillators=0)

    def test_rejects_non_32_bit_width(self) -> None:
        with pytest.raises(ValueError, match="32-bit Q16.16"):
            KuramotoVerilogCompiler(n_oscillators=4, bit_width=16)

    def test_rejects_zero_cordic_stages(self) -> None:
        with pytest.raises(ValueError, match="cordic_stages must be >= 1"):
            KuramotoVerilogCompiler(n_oscillators=4, cordic_stages=0)

    def test_rejects_shape_mismatch(self) -> None:
        compiler = KuramotoVerilogCompiler(n_oscillators=3)
        with pytest.raises(ValueError, match="knm shape"):
            compiler.compile(np.zeros((2, 2)), np.ones(3), 0.01)

    def test_rejects_omega_shape_mismatch(self) -> None:
        compiler = KuramotoVerilogCompiler(n_oscillators=3)
        with pytest.raises(ValueError, match="omegas shape"):
            compiler.compile(np.zeros((3, 3)), np.ones(2), 0.01)

    def test_rejects_non_positive_dt(self) -> None:
        compiler = KuramotoVerilogCompiler(n_oscillators=2)
        with pytest.raises(ValueError, match="dt must be positive"):
            compiler.compile(np.zeros((2, 2)), np.ones(2), 0.0)


class TestHdlCompilerPipelineWiring:
    """Pipeline wiring: HDL compiler consumes topology from CouplingBuilder,
    emits Verilog for FPGA actuation path."""

    def test_full_dense_topology_synthesisable(self) -> None:
        """Full-connectivity mesh still produces clean synthesisable code."""
        from scpn_phase_orchestrator.coupling.knm import CouplingBuilder

        builder = CouplingBuilder()
        state = builder.build(n_layers=4, base_strength=0.4, decay_alpha=0.3)
        omegas = np.linspace(0.8, 1.2, 4)
        compiler = KuramotoVerilogCompiler(n_oscillators=4)
        verilog = compiler.compile(state.knm, omegas, 0.01)

        assert "module kuramoto_mesh_4" in verilog
        assert "endmodule" in verilog
        assert "cordic_sincos" in verilog
        assert not re.search(r"\$sin\b|\$cos\b|\breal\b", verilog)

# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996-2026 Miroslav Šotek. All rights reserved.
# © Code 2020-2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - HDL Compiler tests

import numpy as np
from scpn_phase_orchestrator.actuation.hdl_compiler import KuramotoVerilogCompiler

def test_verilog_compilation():
    n = 4
    compiler = KuramotoVerilogCompiler(n_oscillators=n)
    knm = np.eye(n) * 0.5
    omegas = np.ones(n)
    dt = 0.01
    
    verilog = compiler.compile(knm, omegas, dt)
    assert "module kuramoto_mesh_4" in verilog
    assert "reg [31:0] theta_0" in verilog
    assert "theta_0 <= theta_0 + ((1.000000) * DT)" in verilog
    
    knm[0, 1] = 0.8
    verilog = compiler.compile(knm, omegas, dt)
    assert "0.800000 * $sin(theta_1 - theta_0)" in verilog

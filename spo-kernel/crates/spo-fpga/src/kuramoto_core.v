// SPDX-License-Identifier: AGPL-3.0-or-later
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — FPGA Kuramoto solver (Zynq-7020 target)

// Kuramoto model: dθ_i/dt = ω_i + (1/N) Σ_j K_ij sin(θ_j − θ_i)
// N=16 oscillators, Q16.16 fixed-point, 4-stage CORDIC pipeline.

`timescale 1ns / 1ps

// ---------- CORDIC sin/cos (4 pipeline stages, Q16.16) ----------
module cordic_sincos #(
    parameter WIDTH = 32,  // Q16.16 fixed-point
    parameter STAGES = 4
)(
    input  wire              clk,
    input  wire              rst_n,
    input  wire              valid_in,
    input  wire [WIDTH-1:0]  angle_in,   // radians in Q16.16
    output reg               valid_out,
    output reg  [WIDTH-1:0]  sin_out,
    output reg  [WIDTH-1:0]  cos_out
);

    // CORDIC rotation angles: atan(2^-k) in Q16.16
    // k=0: 0.7854 → 0x0000C910, k=1: 0.4636 → 0x000076B2
    // k=2: 0.2450 → 0x00003EB7, k=3: 0.1244 → 0x00001FD6
    wire [WIDTH-1:0] ATAN_LUT [0:STAGES-1];
    assign ATAN_LUT[0] = 32'h0000C910;
    assign ATAN_LUT[1] = 32'h000076B2;
    assign ATAN_LUT[2] = 32'h00003EB7;
    assign ATAN_LUT[3] = 32'h00001FD6;

    // CORDIC gain inverse (1/K_4 ≈ 0.6073) in Q16.16 → 0x00009B78
    localparam [WIDTH-1:0] CORDIC_GAIN_INV = 32'h00009B78;

    // Pipeline registers: x, y, z per stage
    reg signed [WIDTH-1:0] x_pipe [0:STAGES];
    reg signed [WIDTH-1:0] y_pipe [0:STAGES];
    reg signed [WIDTH-1:0] z_pipe [0:STAGES];
    reg                    valid_pipe [0:STAGES];

    // Quadrant folding: map angle to [-π/2, π/2]
    wire signed [WIDTH-1:0] angle_signed = angle_in;
    localparam signed [WIDTH-1:0] PI_HALF = 32'h00019220;   // π/2 in Q16.16
    localparam signed [WIDTH-1:0] PI_VAL  = 32'h00032440;   // π   in Q16.16
    localparam signed [WIDTH-1:0] NEG_PI_HALF = -PI_HALF;

    wire fold_upper = (angle_signed > PI_HALF);
    wire fold_lower = (angle_signed < NEG_PI_HALF);

    wire signed [WIDTH-1:0] folded_angle =
        fold_upper ? (angle_signed - PI_VAL) :
        fold_lower ? (angle_signed + PI_VAL) :
        angle_signed;

    wire negate_result = fold_upper | fold_lower;

    // Stage 0 init: x=gain_inv, y=0, z=folded_angle
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x_pipe[0]     <= 0;
            y_pipe[0]     <= 0;
            z_pipe[0]     <= 0;
            valid_pipe[0] <= 0;
        end else begin
            x_pipe[0]     <= CORDIC_GAIN_INV;
            y_pipe[0]     <= 0;
            z_pipe[0]     <= folded_angle;
            valid_pipe[0] <= valid_in;
        end
    end

    // CORDIC rotation stages
    genvar k;
    generate
        for (k = 0; k < STAGES; k = k + 1) begin : cordic_stage
            wire sigma = (z_pipe[k] >= 0) ? 1'b0 : 1'b1;

            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    x_pipe[k+1]     <= 0;
                    y_pipe[k+1]     <= 0;
                    z_pipe[k+1]     <= 0;
                    valid_pipe[k+1] <= 0;
                end else begin
                    if (!sigma) begin
                        x_pipe[k+1] <= x_pipe[k] - (y_pipe[k] >>> k);
                        y_pipe[k+1] <= y_pipe[k] + (x_pipe[k] >>> k);
                        z_pipe[k+1] <= z_pipe[k] - $signed(ATAN_LUT[k]);
                    end else begin
                        x_pipe[k+1] <= x_pipe[k] + (y_pipe[k] >>> k);
                        y_pipe[k+1] <= y_pipe[k] - (x_pipe[k] >>> k);
                        z_pipe[k+1] <= z_pipe[k] + $signed(ATAN_LUT[k]);
                    end
                    valid_pipe[k+1] <= valid_pipe[k];
                end
            end
        end
    endgenerate

    // Output with quadrant correction
    reg negate_delay [0:STAGES];
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            negate_delay[0] <= 0;
        else
            negate_delay[0] <= negate_result;
    end

    generate
        for (k = 0; k < STAGES; k = k + 1) begin : neg_delay
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    negate_delay[k+1] <= 0;
                else
                    negate_delay[k+1] <= negate_delay[k];
            end
        end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sin_out   <= 0;
            cos_out   <= 0;
            valid_out <= 0;
        end else begin
            sin_out   <= negate_delay[STAGES] ? -y_pipe[STAGES] : y_pipe[STAGES];
            cos_out   <= negate_delay[STAGES] ? -x_pipe[STAGES] : x_pipe[STAGES];
            valid_out <= valid_pipe[STAGES];
        end
    end
endmodule


// ---------- Kuramoto solver core (N=16 oscillators) ----------
module kuramoto_core #(
    parameter N     = 16,
    parameter WIDTH = 32  // Q16.16 fixed-point
)(
    input  wire                    clk,
    input  wire                    rst_n,

    // Phase I/O: flat bus, N × WIDTH
    input  wire [N*WIDTH-1:0]      phase_in,
    output reg  [N*WIDTH-1:0]      phase_out,

    // Natural frequencies: flat bus, N × WIDTH
    input  wire [N*WIDTH-1:0]      omega_in,

    // Step size (dt in Q16.16)
    input  wire [WIDTH-1:0]        dt,

    // Control
    input  wire                    start,
    output reg                     done,

    // AXI-Lite register interface for coupling matrix K_ij
    input  wire [7:0]              axi_awaddr,
    input  wire                    axi_awvalid,
    output wire                    axi_awready,
    input  wire [WIDTH-1:0]        axi_wdata,
    input  wire                    axi_wvalid,
    output wire                    axi_wready,
    output wire [1:0]              axi_bresp,
    output wire                    axi_bvalid,
    input  wire                    axi_bready,
    input  wire [7:0]              axi_araddr,
    input  wire                    axi_arvalid,
    output wire                    axi_arready,
    output wire [WIDTH-1:0]        axi_rdata,
    output wire [1:0]              axi_rresp,
    output wire                    axi_rvalid,
    input  wire                    axi_rready
);

    // --- Coupling matrix stored in BRAM ---
    // Addressed as K[i][j] = K_mem[i*N + j], 256 entries for N=16
    reg [WIDTH-1:0] K_mem [0:N*N-1];

    // --- AXI-Lite write channel ---
    reg [7:0]  aw_addr_r;
    reg        aw_done, w_done;
    assign axi_awready = !aw_done;
    assign axi_wready  = !w_done;
    assign axi_bresp   = 2'b00;
    assign axi_bvalid  = aw_done && w_done;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            aw_done   <= 0;
            w_done    <= 0;
            aw_addr_r <= 0;
        end else begin
            if (axi_awvalid && !aw_done) begin
                aw_addr_r <= axi_awaddr;
                aw_done   <= 1;
            end
            if (axi_wvalid && !w_done) begin
                w_done <= 1;
            end
            if (aw_done && w_done) begin
                K_mem[aw_addr_r] <= axi_wdata;
                if (axi_bready) begin
                    aw_done <= 0;
                    w_done  <= 0;
                end
            end
        end
    end

    // --- AXI-Lite read channel ---
    reg [7:0]      ar_addr_r;
    reg             ar_valid_r;
    reg [WIDTH-1:0] rdata_r;

    assign axi_arready = !ar_valid_r;
    assign axi_rdata   = rdata_r;
    assign axi_rresp   = 2'b00;
    assign axi_rvalid  = ar_valid_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ar_valid_r <= 0;
            ar_addr_r  <= 0;
            rdata_r    <= 0;
        end else begin
            if (axi_arvalid && !ar_valid_r) begin
                ar_addr_r  <= axi_araddr;
                rdata_r    <= K_mem[axi_araddr];
                ar_valid_r <= 1;
            end
            if (ar_valid_r && axi_rready) begin
                ar_valid_r <= 0;
            end
        end
    end

    // --- Kuramoto iteration FSM ---
    // States: IDLE → COMPUTE_DIFF → CORDIC_WAIT → ACCUMULATE → UPDATE → DONE
    localparam S_IDLE       = 3'd0;
    localparam S_DIFF       = 3'd1;
    localparam S_CORDIC     = 3'd2;
    localparam S_ACCUM      = 3'd3;
    localparam S_UPDATE     = 3'd4;
    localparam S_DONE       = 3'd5;

    reg [2:0]  state;
    reg [3:0]  idx_i, idx_j;
    reg signed [WIDTH-1:0] coupling_sum [0:N-1];
    reg signed [WIDTH-1:0] theta [0:N-1];
    reg signed [WIDTH-1:0] diff_reg;

    // CORDIC instance
    wire             cordic_valid_out;
    wire [WIDTH-1:0] cordic_sin, cordic_cos;
    reg              cordic_valid_in;
    reg  [WIDTH-1:0] cordic_angle;

    cordic_sincos #(.WIDTH(WIDTH), .STAGES(4)) u_cordic (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (cordic_valid_in),
        .angle_in  (cordic_angle),
        .valid_out (cordic_valid_out),
        .sin_out   (cordic_sin),
        .cos_out   (cordic_cos)
    );

    // CORDIC pipeline latency counter (4 stages + 1 output reg = 5 cycles)
    reg [2:0] cordic_wait;

    integer m;

    // Unpack phase_in into theta[] on start
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= S_IDLE;
            done           <= 0;
            idx_i          <= 0;
            idx_j          <= 0;
            cordic_valid_in<= 0;
            cordic_angle   <= 0;
            cordic_wait    <= 0;
            for (m = 0; m < N; m = m + 1) begin
                theta[m]        <= 0;
                coupling_sum[m] <= 0;
                phase_out[m*WIDTH +: WIDTH] <= 0;
            end
        end else begin
            case (state)
                S_IDLE: begin
                    done <= 0;
                    if (start) begin
                        for (m = 0; m < N; m = m + 1) begin
                            theta[m]        <= phase_in[m*WIDTH +: WIDTH];
                            coupling_sum[m] <= 0;
                        end
                        idx_i <= 0;
                        idx_j <= 0;
                        state <= S_DIFF;
                    end
                end

                S_DIFF: begin
                    if (idx_i == idx_j) begin
                        // Skip self-coupling
                        if (idx_j == N - 1) begin
                            idx_j <= 0;
                            if (idx_i == N - 1)
                                state <= S_UPDATE;
                            else
                                idx_i <= idx_i + 1;
                        end else begin
                            idx_j <= idx_j + 1;
                        end
                    end else begin
                        // Compute θ_j − θ_i
                        diff_reg        <= theta[idx_j] - theta[idx_i];
                        cordic_angle    <= theta[idx_j] - theta[idx_i];
                        cordic_valid_in <= 1;
                        cordic_wait     <= 0;
                        state           <= S_CORDIC;
                    end
                end

                S_CORDIC: begin
                    cordic_valid_in <= 0;
                    cordic_wait     <= cordic_wait + 1;
                    if (cordic_valid_out) begin
                        state <= S_ACCUM;
                    end
                end

                // K_ij * sin(θ_j − θ_i), accumulated into coupling_sum[i]
                // Multiply: (K * sin) >> 16 to stay in Q16.16
                S_ACCUM: begin
                    // 64-bit intermediate for Q16.16 × Q16.16
                    coupling_sum[idx_i] <= coupling_sum[idx_i] +
                        (($signed(K_mem[idx_i * N + idx_j]) * $signed(cordic_sin)) >>> 16);

                    if (idx_j == N - 1) begin
                        idx_j <= 0;
                        if (idx_i == N - 1)
                            state <= S_UPDATE;
                        else begin
                            idx_i <= idx_i + 1;
                            state <= S_DIFF;
                        end
                    end else begin
                        idx_j <= idx_j + 1;
                        state <= S_DIFF;
                    end
                end

                // θ_i += dt * (ω_i + (1/N) * Σ_j K_ij sin(θ_j − θ_i))
                // Division by N=16 is a right-shift by 4
                S_UPDATE: begin
                    for (m = 0; m < N; m = m + 1) begin
                        theta[m] <= theta[m] +
                            (($signed(dt) * ($signed(omega_in[m*WIDTH +: WIDTH]) +
                              (coupling_sum[m] >>> 4))) >>> 16);
                        phase_out[m*WIDTH +: WIDTH] <= theta[m] +
                            (($signed(dt) * ($signed(omega_in[m*WIDTH +: WIDTH]) +
                              (coupling_sum[m] >>> 4))) >>> 16);
                    end
                    state <= S_DONE;
                end

                S_DONE: begin
                    done  <= 1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule

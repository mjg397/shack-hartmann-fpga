`timescale 1ns/1ps
// =============================================================================
// shwfs_ref_model.sv -- FUNCTIONAL REFERENCE MODEL
//
// An INDEPENDENT behavioural re-implementation of the SHWFS datapath, written
// from the fixed-point specification rather than from the RTL. The scoreboard
// compares the DUT against this model, so a bug in one is not silently mirrored
// by the other.
//
// It is deliberately NOT structured like the DUT: no pipelining, no valid
// handshaking, no per-cycle scheduling. It computes whole-frame results in one
// pass, which is what makes it an independent check of the DUT's pipelining.
//
// Specification implemented (traced from the design, see EVALUATION_REPORT.md):
//   threshold   pixel < 5      -> 0
//   accumulate  sI, xI, yI per 16x16 tile, exact integers
//   reciprocal  rI = (sI==0) ? 19'h7FFFF : LUT[sI]          Q18, 19 bits
//               the DUT's port truncates this to 18 bits -- modelled here
//   centroid    (xI[18:0] * rec[17:0])[23:0]                Q18, 24 bits
//   slope       (centroid - ref)[24:0]                      Q18, 25 bits signed
//   zernike[m]  sum_s Ex[m,s]*sx[s] + Ey[m,s]*sy[s]
//               products 43b signed, accumulator 52b signed, out = acc[42:18]
// =============================================================================

module shwfs_ref_model #(
    parameter integer FRAME_W     = 256,
    parameter integer FRAME_H     = 256,
    parameter integer SUBAP_PX    = 16,
    parameter integer N_LENS      = 16,
    parameter integer N_TILES     = 256,
    parameter integer N_VALID     = 168,
    parameter integer N_MODES     = 10,
    parameter integer PIXEL_THRESH = 5
);

    // Expected results, filled by compute(). Accessed hierarchically by the
    // scoreboard: array ports are not portable across Icarus and Verilator.
    integer exp_zernike    [0:N_MODES-1];
    integer exp_centroid_x [0:N_TILES-1];
    integer exp_centroid_y [0:N_TILES-1];
    integer exp_slope_x    [0:N_TILES-1];
    integer exp_slope_y    [0:N_TILES-1];
    integer exp_valid_idx  [0:N_VALID-1];
    integer exp_n_valid;

    // ---- ROMs, loaded from the same .mem files the DUT uses ----
    reg [18:0]  recip_lut [0:65535];
    reg [24:0]  ref_x [0:255];
    reg [24:0]  ref_y [0:255];
    reg [255:0] pupil_bitmap [0:0];
    reg [19:0]  e_rom_x [0:N_MODES*N_VALID-1];
    reg [19:0]  e_rom_y [0:N_MODES*N_VALID-1];

    reg [7:0] frame [0:FRAME_W*FRAME_H-1];

    integer i;
    initial begin
        for (i = 0; i < 65536; i = i + 1) recip_lut[i] = 19'd0;
        $readmemh("reciprocal_u16_q18.mem", recip_lut);
        $readmemh("slopes_ref_x.mem",       ref_x);
        $readmemh("slopes_ref_y.mem",       ref_y);
        $readmemh("subaperture_bitmap.mem", pupil_bitmap);
        $readmemh("e_matrix_x.mem",         e_rom_x);
        $readmemh("e_matrix_y.mem",         e_rom_y);
    end

    // ---- sign helpers (Verilog-2001 safe) ----
    function integer signed sext;
        input integer value;
        input integer bits;
        integer m, s;
        begin
            m = (1 << bits) - 1;
            s = 1 << (bits - 1);
            sext = ((value & m) ^ s) - s;
        end
    endfunction

    function integer trunc;
        input integer value;
        input integer bits;
        begin
            trunc = value & ((1 << bits) - 1);
        end
    endfunction

    // -------------------------------------------------------------------------
    // compute(): load a frame and evaluate the whole datapath.
    // -------------------------------------------------------------------------
    task automatic compute(input string frame_path);
        integer r, c, py, px, idx, p, v, m, s;
        integer sI, xI, yI, rI, rec, cx, cy;
        integer signed sx, sy, raw;
        reg signed [51:0] acc;
        reg signed [42:0] prod;
        begin
            $readmemh(frame_path, frame);

            v = 0;
            for (idx = 0; idx < N_TILES; idx = idx + 1) begin
                r = idx / N_LENS;
                c = idx % N_LENS;

                sI = 0; xI = 0; yI = 0;
                for (py = 0; py < SUBAP_PX; py = py + 1)
                    for (px = 0; px < SUBAP_PX; px = px + 1) begin
                        p = frame[(r*SUBAP_PX + py)*FRAME_W + (c*SUBAP_PX + px)];
                        if (p < PIXEL_THRESH) p = 0;
                        sI = sI + p;
                        xI = xI + p * px;
                        yI = yI + p * py;
                    end

                // reciprocal, then the DUT's 18-bit port truncation
                rI  = (sI == 0) ? 19'h7FFFF : recip_lut[sI];
                rec = trunc(rI, 18);

                cx = trunc(trunc(xI, 19) * rec, 24);
                cy = trunc(trunc(yI, 19) * rec, 24);

                sx = sext(trunc(cx - sext(ref_x[idx], 25), 25), 25);
                sy = sext(trunc(cy - sext(ref_y[idx], 25), 25), 25);

                exp_centroid_x[idx] = cx;
                exp_centroid_y[idx] = cy;
                exp_slope_x[idx]    = sx;
                exp_slope_y[idx]    = sy;

                // pupil bit N gates raster tile N (plain bit-select in the DUT)
                if (pupil_bitmap[0][idx]) begin
                    exp_valid_idx[v] = idx;
                    v = v + 1;
                end
            end
            exp_n_valid = v;

            // ---- modal reconstruction over the valid subapertures ----
            for (m = 0; m < N_MODES; m = m + 1) begin
                acc = 52'sd0;
                for (s = 0; s < N_VALID; s = s + 1) begin
                    idx = exp_valid_idx[s];
                    prod = sext(e_rom_x[m*N_VALID + s], 18) * exp_slope_x[idx];
                    acc  = acc + prod;
                    prod = sext(e_rom_y[m*N_VALID + s], 18) * exp_slope_y[idx];
                    acc  = acc + prod;
                end
                exp_zernike[m] = sext(trunc(acc >>> 18, 25), 25);
            end
        end
    endtask

endmodule

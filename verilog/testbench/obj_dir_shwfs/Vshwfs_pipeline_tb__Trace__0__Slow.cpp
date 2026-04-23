// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Tracing implementation internals
#include "verilated_vcd_c.h"
#include "Vshwfs_pipeline_tb__Syms.h"


VL_ATTR_COLD void Vshwfs_pipeline_tb___024root__trace_init_sub__TOP__0(Vshwfs_pipeline_tb___024root* vlSelf, VerilatedVcd* tracep) {
    if (false && vlSelf) {}  // Prevent unused
    Vshwfs_pipeline_tb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vshwfs_pipeline_tb___024root__trace_init_sub__TOP__0\n"); );
    // Init
    const int c = vlSymsp->__Vm_baseCode;
    // Body
    tracep->pushPrefix("shwfs_pipeline_tb", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+53,0,"clk_100",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+54,0,"reset",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+1,0,"xI_reciprocal",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 19,0);
    tracep->declBus(c+2,0,"yI_reciprocal",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 19,0);
    tracep->declBus(c+3,0,"rI_reciprocal",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+4,0,"subaps_done_reciprocal",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->pushPrefix("DUT", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+53,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+54,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+1,0,"xI_reciprocal",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 19,0);
    tracep->declBus(c+2,0,"yI_reciprocal",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 19,0);
    tracep->declBus(c+3,0,"rI_reciprocal",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+4,0,"subaps_done_reciprocal",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBus(c+5,0,"streamer_data",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBit(c+6,0,"streamer_fv",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+7,0,"streamer_lv",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+8,0,"frame_complete",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+9,0,"streamer_valid",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+10,0,"full_frame_complete_accumulator",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+11,0,"subaps_done_accumulator",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBus(c+12,0,"intensity_accumulator",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+13,0,"xI_accumulator",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 19,0);
    tracep->declBus(c+14,0,"yI_accumulator",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 19,0);
    tracep->pushPrefix("accumulator", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+55,0,"NUM_SUBAPETURES_SQRT",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+55,0,"NUM_PIXELS_SUBAPETURE_SQRT",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBit(c+53,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+54,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+9,0,"valid",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+5,0,"data_in",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBit(c+10,0,"full_frame_complete",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+11,0,"subapetures_completed",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBus(c+12,0,"intensity",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+13,0,"x_intensity",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 19,0);
    tracep->declBus(c+14,0,"y_intensity",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 19,0);
    tracep->declBus(c+15,0,"subap_col",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 3,0);
    tracep->declBus(c+16,0,"subap_row",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 3,0);
    tracep->declBus(c+17,0,"count_pixel_h",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 3,0);
    tracep->declBus(c+18,0,"count_pixel_v",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 3,0);
    tracep->declBus(c+19,0,"subap_idx",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBit(c+20,0,"last_pixel_h",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+21,0,"last_pixel_v",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+22,0,"last_subap_col",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+23,0,"last_subap_row",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+24,0,"subap_done_delay",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+25,0,"subap_idx_delay",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBus(c+26,0,"j",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->popPrefix();
    tracep->pushPrefix("reciprocal", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+53,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+54,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+13,0,"xI_in",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 19,0);
    tracep->declBus(c+14,0,"yI_in",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 19,0);
    tracep->declBus(c+12,0,"sI",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+11,0,"centroids_done_in",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBus(c+1,0,"xI_out",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 19,0);
    tracep->declBus(c+2,0,"yI_out",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 19,0);
    tracep->declBus(c+3,0,"rI_out",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+4,0,"centroids_done_out",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBus(c+27,0,"reciprocal_wire",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBit(c+28,0,"divide_by_zero_wire",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+29,0,"saturated_wire",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("recip", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+12,0,"v_u16",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+27,0,"reciprocal_q16",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBit(c+28,0,"divide_by_zero",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+29,0,"saturated",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declArray(c+56,0,"RECIP_SEED_LUT_Q0_16",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 4095,0);
    tracep->declBit(c+28,0,"v_is_zero",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+30,0,"v_safe",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+31,0,"shift_left",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 4,0);
    tracep->declBus(c+32,0,"a_q1_15",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+33,0,"lut_idx",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBus(c+34,0,"x0_q0_16",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+35,0,"x1_q0_16",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBit(c+36,0,"x1_saturated",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+37,0,"msb_index",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 4,0);
    tracep->declBus(c+38,0,"denorm_round_bias",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+39,0,"denorm_numer",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 16,0);
    tracep->declBus(c+40,0,"out_q0_16_ext",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 16,0);
    tracep->declBit(c+41,0,"out_sat",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("u_newton_step_q16", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+32,0,"a_q1_15",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+34,0,"x0_q0_16",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+35,0,"x1_q0_16",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBit(c+36,0,"saturated",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+42,0,"ax_mul",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+43,0,"ax_q1_15",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 16,0);
    tracep->declBus(c+43,0,"ax_q1_15_clamped",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 16,0);
    tracep->declBus(c+44,0,"two_minus_ax_q1_15",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 16,0);
    tracep->declQuad(c+45,0,"prod_mul",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 32,0);
    tracep->declBus(c+47,0,"x1_rounded_ext",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 17,0);
    tracep->popPrefix();
    tracep->popPrefix();
    tracep->popPrefix();
    tracep->pushPrefix("streamer", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+184,0,"HSIZE",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+184,0,"VSIZE",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+185,0,"HBLANK",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+186,0,"VBLANK",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBit(c+53,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+54,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+5,0,"data",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBit(c+6,0,"fv",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+7,0,"lv",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+8,0,"frame_complete",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+9,0,"valid",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+48,0,"line_counter",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBus(c+49,0,"row_counter",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBus(c+50,0,"h_blank_counter",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 1,0);
    tracep->declBus(c+51,0,"v_blank_counter",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBus(c+187,0,"STATE_FRAME_INIT",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 1,0);
    tracep->declBus(c+188,0,"STATE_ACTIVE_FRAME",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 1,0);
    tracep->declBus(c+189,0,"STATE_HOROZONTAL_BLANKING",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 1,0);
    tracep->declBus(c+190,0,"STATE_VERTICAL_BLANKING",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 1,0);
    tracep->declBus(c+52,0,"state",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 1,0);
    tracep->popPrefix();
    tracep->popPrefix();
    tracep->popPrefix();
}

VL_ATTR_COLD void Vshwfs_pipeline_tb___024root__trace_init_top(Vshwfs_pipeline_tb___024root* vlSelf, VerilatedVcd* tracep) {
    if (false && vlSelf) {}  // Prevent unused
    Vshwfs_pipeline_tb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vshwfs_pipeline_tb___024root__trace_init_top\n"); );
    // Body
    Vshwfs_pipeline_tb___024root__trace_init_sub__TOP__0(vlSelf, tracep);
}

VL_ATTR_COLD void Vshwfs_pipeline_tb___024root__trace_const_0(void* voidSelf, VerilatedVcd::Buffer* bufp);
VL_ATTR_COLD void Vshwfs_pipeline_tb___024root__trace_full_0(void* voidSelf, VerilatedVcd::Buffer* bufp);
void Vshwfs_pipeline_tb___024root__trace_chg_0(void* voidSelf, VerilatedVcd::Buffer* bufp);
void Vshwfs_pipeline_tb___024root__trace_cleanup(void* voidSelf, VerilatedVcd* /*unused*/);

VL_ATTR_COLD void Vshwfs_pipeline_tb___024root__trace_register(Vshwfs_pipeline_tb___024root* vlSelf, VerilatedVcd* tracep) {
    if (false && vlSelf) {}  // Prevent unused
    Vshwfs_pipeline_tb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vshwfs_pipeline_tb___024root__trace_register\n"); );
    // Body
    tracep->addConstCb(&Vshwfs_pipeline_tb___024root__trace_const_0, 0U, vlSelf);
    tracep->addFullCb(&Vshwfs_pipeline_tb___024root__trace_full_0, 0U, vlSelf);
    tracep->addChgCb(&Vshwfs_pipeline_tb___024root__trace_chg_0, 0U, vlSelf);
    tracep->addCleanupCb(&Vshwfs_pipeline_tb___024root__trace_cleanup, vlSelf);
}

VL_ATTR_COLD void Vshwfs_pipeline_tb___024root__trace_const_0_sub_0(Vshwfs_pipeline_tb___024root* vlSelf, VerilatedVcd::Buffer* bufp);

VL_ATTR_COLD void Vshwfs_pipeline_tb___024root__trace_const_0(void* voidSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vshwfs_pipeline_tb___024root__trace_const_0\n"); );
    // Init
    Vshwfs_pipeline_tb___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vshwfs_pipeline_tb___024root*>(voidSelf);
    Vshwfs_pipeline_tb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    // Body
    Vshwfs_pipeline_tb___024root__trace_const_0_sub_0((&vlSymsp->TOP), bufp);
}

extern const VlWide<128>/*4095:0*/ Vshwfs_pipeline_tb__ConstPool__CONST_h29f91db3_0;

VL_ATTR_COLD void Vshwfs_pipeline_tb___024root__trace_const_0_sub_0(Vshwfs_pipeline_tb___024root* vlSelf, VerilatedVcd::Buffer* bufp) {
    if (false && vlSelf) {}  // Prevent unused
    Vshwfs_pipeline_tb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vshwfs_pipeline_tb___024root__trace_const_0_sub_0\n"); );
    // Init
    uint32_t* const oldp VL_ATTR_UNUSED = bufp->oldp(vlSymsp->__Vm_baseCode);
    // Body
    bufp->fullIData(oldp+55,(0x10U),32);
    bufp->fullWData(oldp+56,(Vshwfs_pipeline_tb__ConstPool__CONST_h29f91db3_0),4096);
    bufp->fullIData(oldp+184,(0x100U),32);
    bufp->fullIData(oldp+185,(4U),32);
    bufp->fullIData(oldp+186,(0x98U),32);
    bufp->fullCData(oldp+187,(0U),2);
    bufp->fullCData(oldp+188,(1U),2);
    bufp->fullCData(oldp+189,(2U),2);
    bufp->fullCData(oldp+190,(3U),2);
}

VL_ATTR_COLD void Vshwfs_pipeline_tb___024root__trace_full_0_sub_0(Vshwfs_pipeline_tb___024root* vlSelf, VerilatedVcd::Buffer* bufp);

VL_ATTR_COLD void Vshwfs_pipeline_tb___024root__trace_full_0(void* voidSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vshwfs_pipeline_tb___024root__trace_full_0\n"); );
    // Init
    Vshwfs_pipeline_tb___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vshwfs_pipeline_tb___024root*>(voidSelf);
    Vshwfs_pipeline_tb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    // Body
    Vshwfs_pipeline_tb___024root__trace_full_0_sub_0((&vlSymsp->TOP), bufp);
}

VL_ATTR_COLD void Vshwfs_pipeline_tb___024root__trace_full_0_sub_0(Vshwfs_pipeline_tb___024root* vlSelf, VerilatedVcd::Buffer* bufp) {
    if (false && vlSelf) {}  // Prevent unused
    Vshwfs_pipeline_tb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vshwfs_pipeline_tb___024root__trace_full_0_sub_0\n"); );
    // Init
    uint32_t* const oldp VL_ATTR_UNUSED = bufp->oldp(vlSymsp->__Vm_baseCode);
    // Body
    bufp->fullIData(oldp+1,(vlSelf->shwfs_pipeline_tb__DOT__xI_reciprocal),20);
    bufp->fullIData(oldp+2,(vlSelf->shwfs_pipeline_tb__DOT__yI_reciprocal),20);
    bufp->fullSData(oldp+3,(vlSelf->shwfs_pipeline_tb__DOT__rI_reciprocal),16);
    bufp->fullCData(oldp+4,(vlSelf->shwfs_pipeline_tb__DOT__subaps_done_reciprocal),8);
    bufp->fullCData(oldp+5,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer_data),8);
    bufp->fullBit(oldp+6,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer_fv));
    bufp->fullBit(oldp+7,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer_lv));
    bufp->fullBit(oldp+8,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__frame_complete));
    bufp->fullBit(oldp+9,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer_valid));
    bufp->fullBit(oldp+10,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__full_frame_complete_accumulator));
    bufp->fullCData(oldp+11,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__subaps_done_accumulator),8);
    bufp->fullSData(oldp+12,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__intensity_accumulator),16);
    bufp->fullIData(oldp+13,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__xI_accumulator),20);
    bufp->fullIData(oldp+14,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__yI_accumulator),20);
    bufp->fullCData(oldp+15,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_col),4);
    bufp->fullCData(oldp+16,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_row),4);
    bufp->fullCData(oldp+17,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__count_pixel_h),4);
    bufp->fullCData(oldp+18,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__count_pixel_v),4);
    bufp->fullCData(oldp+19,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_idx),8);
    bufp->fullBit(oldp+20,((0xfU == (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__count_pixel_h))));
    bufp->fullBit(oldp+21,((0xfU == (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__count_pixel_v))));
    bufp->fullBit(oldp+22,((0xfU == (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_col))));
    bufp->fullBit(oldp+23,((0xfU == (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_row))));
    bufp->fullBit(oldp+24,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_done_delay));
    bufp->fullCData(oldp+25,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_idx_delay),8);
    bufp->fullIData(oldp+26,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__j),32);
    bufp->fullSData(oldp+27,(((0U == (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__intensity_accumulator))
                               ? 0xffffU : ((0xffffU 
                                             < vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__out_q0_16_ext)
                                             ? 0xffffU
                                             : (0xffffU 
                                                & vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__out_q0_16_ext)))),16);
    bufp->fullBit(oldp+28,((0U == (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__intensity_accumulator))));
    bufp->fullBit(oldp+29,(((0U == (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__intensity_accumulator)) 
                            | (0xffffU < vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__out_q0_16_ext))));
    bufp->fullSData(oldp+30,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__v_safe),16);
    bufp->fullCData(oldp+31,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__shift_left),5);
    bufp->fullSData(oldp+32,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__a_q1_15),16);
    bufp->fullCData(oldp+33,((0xffU & ((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                       >> 7U))),8);
    bufp->fullSData(oldp+34,((0xffffU & (((0U == (0x1fU 
                                                  & VL_SHIFTL_III(12,12,32, 
                                                                  (0xffU 
                                                                   & ((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                      >> 7U)), 4U)))
                                           ? 0U : (
                                                   Vshwfs_pipeline_tb__ConstPool__CONST_h29f91db3_0[
                                                   (((IData)(0xfU) 
                                                     + 
                                                     (0xfffU 
                                                      & VL_SHIFTL_III(12,12,32, 
                                                                      (0xffU 
                                                                       & ((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                          >> 7U)), 4U))) 
                                                    >> 5U)] 
                                                   << 
                                                   ((IData)(0x20U) 
                                                    - 
                                                    (0x1fU 
                                                     & VL_SHIFTL_III(12,12,32, 
                                                                     (0xffU 
                                                                      & ((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                         >> 7U)), 4U))))) 
                                         | (Vshwfs_pipeline_tb__ConstPool__CONST_h29f91db3_0[
                                            (0x7fU 
                                             & (VL_SHIFTL_III(12,12,32, 
                                                              (0xffU 
                                                               & ((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                  >> 7U)), 4U) 
                                                >> 5U))] 
                                            >> (0x1fU 
                                                & VL_SHIFTL_III(12,12,32, 
                                                                (0xffU 
                                                                 & ((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                    >> 7U)), 4U)))))),16);
    bufp->fullSData(oldp+35,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__x1_q0_16),16);
    bufp->fullBit(oldp+36,((0xffffU < (0x3ffffU & (IData)(
                                                          (0x3ffffULL 
                                                           & ((0x4000ULL 
                                                               + vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__u_newton_step_q16__DOT__prod_mul) 
                                                              >> 0xfU)))))));
    bufp->fullCData(oldp+37,((0x1fU & ((IData)(0xfU) 
                                       - (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__shift_left)))),5);
    bufp->fullSData(oldp+38,(((0U == (0x1fU & ((IData)(0xfU) 
                                               - (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__shift_left))))
                               ? 0U : (0xffffU & ((IData)(1U) 
                                                  << 
                                                  (0x1fU 
                                                   & (((IData)(0xfU) 
                                                       - (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__shift_left)) 
                                                      - (IData)(1U))))))),16);
    bufp->fullIData(oldp+39,((0x1ffffU & ((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__x1_q0_16) 
                                          + ((0U == 
                                              (0x1fU 
                                               & ((IData)(0xfU) 
                                                  - (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__shift_left))))
                                              ? 0U : 
                                             (0xffffU 
                                              & ((IData)(1U) 
                                                 << 
                                                 (0x1fU 
                                                  & (((IData)(0xfU) 
                                                      - (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__shift_left)) 
                                                     - (IData)(1U))))))))),17);
    bufp->fullIData(oldp+40,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__out_q0_16_ext),17);
    bufp->fullBit(oldp+41,((0xffffU < vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__out_q0_16_ext)));
    bufp->fullIData(oldp+42,(((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                              * (0xffffU & (((0U == 
                                              (0x1fU 
                                               & VL_SHIFTL_III(12,12,32, 
                                                               (0xffU 
                                                                & ((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                   >> 7U)), 4U)))
                                              ? 0U : 
                                             (Vshwfs_pipeline_tb__ConstPool__CONST_h29f91db3_0[
                                              (((IData)(0xfU) 
                                                + (0xfffU 
                                                   & VL_SHIFTL_III(12,12,32, 
                                                                   (0xffU 
                                                                    & ((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                       >> 7U)), 4U))) 
                                               >> 5U)] 
                                              << ((IData)(0x20U) 
                                                  - 
                                                  (0x1fU 
                                                   & VL_SHIFTL_III(12,12,32, 
                                                                   (0xffU 
                                                                    & ((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                       >> 7U)), 4U))))) 
                                            | (Vshwfs_pipeline_tb__ConstPool__CONST_h29f91db3_0[
                                               (0x7fU 
                                                & (VL_SHIFTL_III(12,12,32, 
                                                                 (0xffU 
                                                                  & ((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                     >> 7U)), 4U) 
                                                   >> 5U))] 
                                               >> (0x1fU 
                                                   & VL_SHIFTL_III(12,12,32, 
                                                                   (0xffU 
                                                                    & ((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                       >> 7U)), 4U))))))),32);
    bufp->fullIData(oldp+43,((((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                               * (0xffffU & (((0U == 
                                               (0x1fU 
                                                & VL_SHIFTL_III(12,12,32, 
                                                                (0xffU 
                                                                 & ((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                    >> 7U)), 4U)))
                                               ? 0U
                                               : (Vshwfs_pipeline_tb__ConstPool__CONST_h29f91db3_0[
                                                  (((IData)(0xfU) 
                                                    + 
                                                    (0xfffU 
                                                     & VL_SHIFTL_III(12,12,32, 
                                                                     (0xffU 
                                                                      & ((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                         >> 7U)), 4U))) 
                                                   >> 5U)] 
                                                  << 
                                                  ((IData)(0x20U) 
                                                   - 
                                                   (0x1fU 
                                                    & VL_SHIFTL_III(12,12,32, 
                                                                    (0xffU 
                                                                     & ((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                        >> 7U)), 4U))))) 
                                             | (Vshwfs_pipeline_tb__ConstPool__CONST_h29f91db3_0[
                                                (0x7fU 
                                                 & (VL_SHIFTL_III(12,12,32, 
                                                                  (0xffU 
                                                                   & ((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                      >> 7U)), 4U) 
                                                    >> 5U))] 
                                                >> 
                                                (0x1fU 
                                                 & VL_SHIFTL_III(12,12,32, 
                                                                 (0xffU 
                                                                  & ((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                     >> 7U)), 4U)))))) 
                              >> 0x10U)),17);
    bufp->fullIData(oldp+44,((0x1ffffU & ((IData)(0x10000U) 
                                          - (((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                              * (0xffffU 
                                                 & (((0U 
                                                      == 
                                                      (0x1fU 
                                                       & VL_SHIFTL_III(12,12,32, 
                                                                       (0xffU 
                                                                        & ((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                           >> 7U)), 4U)))
                                                      ? 0U
                                                      : 
                                                     (Vshwfs_pipeline_tb__ConstPool__CONST_h29f91db3_0[
                                                      (((IData)(0xfU) 
                                                        + 
                                                        (0xfffU 
                                                         & VL_SHIFTL_III(12,12,32, 
                                                                         (0xffU 
                                                                          & ((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                             >> 7U)), 4U))) 
                                                       >> 5U)] 
                                                      << 
                                                      ((IData)(0x20U) 
                                                       - 
                                                       (0x1fU 
                                                        & VL_SHIFTL_III(12,12,32, 
                                                                        (0xffU 
                                                                         & ((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                            >> 7U)), 4U))))) 
                                                    | (Vshwfs_pipeline_tb__ConstPool__CONST_h29f91db3_0[
                                                       (0x7fU 
                                                        & (VL_SHIFTL_III(12,12,32, 
                                                                         (0xffU 
                                                                          & ((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                             >> 7U)), 4U) 
                                                           >> 5U))] 
                                                       >> 
                                                       (0x1fU 
                                                        & VL_SHIFTL_III(12,12,32, 
                                                                        (0xffU 
                                                                         & ((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                            >> 7U)), 4U)))))) 
                                             >> 0x10U)))),17);
    bufp->fullQData(oldp+45,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__u_newton_step_q16__DOT__prod_mul),33);
    bufp->fullIData(oldp+47,((0x3ffffU & (IData)((0x3ffffULL 
                                                  & ((0x4000ULL 
                                                      + vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__u_newton_step_q16__DOT__prod_mul) 
                                                     >> 0xfU))))),18);
    bufp->fullCData(oldp+48,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__line_counter),8);
    bufp->fullCData(oldp+49,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__row_counter),8);
    bufp->fullCData(oldp+50,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__h_blank_counter),2);
    bufp->fullCData(oldp+51,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__v_blank_counter),8);
    bufp->fullCData(oldp+52,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__state),2);
    bufp->fullBit(oldp+53,(vlSelf->shwfs_pipeline_tb__DOT__clk_100));
    bufp->fullBit(oldp+54,(vlSelf->shwfs_pipeline_tb__DOT__reset));
}

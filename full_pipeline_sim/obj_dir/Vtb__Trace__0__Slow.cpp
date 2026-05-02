// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Tracing implementation internals
#include "verilated_vcd_c.h"
#include "Vtb__Syms.h"


VL_ATTR_COLD void Vtb___024root__trace_init_sub__TOP__0(Vtb___024root* vlSelf, VerilatedVcd* tracep) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb___024root__trace_init_sub__TOP__0\n"); );
    // Init
    const int c = vlSymsp->__Vm_baseCode;
    // Body
    tracep->pushPrefix("tb", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+251,0,"clk",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+252,0,"key_reset",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+253,0,"hps_reset",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+254,0,"i",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->pushPrefix("dut", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+251,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+253,0,"hps_reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+252,0,"key_reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+255,0,"hex3_hex0",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+256,0,"result_rdata",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+21,0,"result_wdata",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+22,0,"result_addr",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 10,0);
    tracep->declBit(c+257,0,"result_clken",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+23,0,"result_write",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+257,0,"result_chipsel",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+258,0,"result_bytesel",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 3,0);
    tracep->declBus(c+24,0,"streamer_data",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBit(c+25,0,"streamer_fv",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+26,0,"streamer_lv",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+27,0,"frame_complete",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+28,0,"streamer_valid",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+27,0,"frame_complete_w",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+259,0,"ctrl_reg_h2f",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBus(c+29,0,"ctrl_reg_f2h",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBit(c+30,0,"reset",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+31,0,"start_sync",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 1,0);
    tracep->declBit(c+32,0,"start_synced",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+33,0,"full_frame_complete_accumulator",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+34,0,"subaps_done_accumulator",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBus(c+35,0,"intensity_accumulator",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+36,0,"xI_accumulator",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 19,0);
    tracep->declBus(c+37,0,"yI_accumulator",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 19,0);
    tracep->declBus(c+38,0,"xI_reciprocal",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 19,0);
    tracep->declBus(c+39,0,"yI_reciprocal",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 19,0);
    tracep->declBus(c+40,0,"rI_reciprocal",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+41,0,"subaps_done_reciprocal",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBus(c+42,0,"x_centroid",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+43,0,"y_centroid",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+44,0,"x_slopes",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+45,0,"y_slopes",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBit(c+46,0,"new_subapeture",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declArray(c+47,0,"zernike_out",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 269,0);
    tracep->declBit(c+56,0,"done",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("zernike_out_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 10; ++i) {
        tracep->declBus(c+57+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 26,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+67,0,"x_centroid_out",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+68,0,"y_centroid_out",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+69,0,"x_slopes_out",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+70,0,"y_slopes_out",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+71,0,"resw_write_idx",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 10,0);
    tracep->declBus(c+260,0,"SLOPE_X_OFFSET",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 10,0);
    tracep->declBus(c+261,0,"SLOPE_Y_OFFSET",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 10,0);
    tracep->declBus(c+262,0,"CENTROID_X_OFFSET",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 10,0);
    tracep->declBus(c+263,0,"CENTROID_Y_OFFSET",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 10,0);
    tracep->declBus(c+264,0,"ZERNIKE_OFFSET",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 10,0);
    tracep->declBus(c+265,0,"RESULT_W_WAIT",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 4,0);
    tracep->declBus(c+266,0,"RESULT_W_SX",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 4,0);
    tracep->declBus(c+267,0,"RESULT_W_SY",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 4,0);
    tracep->declBus(c+268,0,"RESULT_W_CX",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 4,0);
    tracep->declBus(c+269,0,"RESULT_W_CY",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 4,0);
    tracep->declBus(c+270,0,"RESULT_W_ZK",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 4,0);
    tracep->declBus(c+271,0,"RESULT_W_DONE",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 4,0);
    tracep->declBus(c+72,0,"resw_next_state",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 4,0);
    tracep->declBit(c+73,0,"resw_start",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+74,0,"resw_state",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 4,0);
    tracep->declBit(c+75,0,"write_zernike_latch",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+76,0,"write_zernike_count",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 10,0);
    tracep->declBit(c+77,0,"reset_from_key",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+78,0,"reset_from_hps",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("accumulator", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+272,0,"NUM_SUBAPETURES_SQRT",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+272,0,"NUM_PIXELS_SUBAPETURE_SQRT",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBit(c+251,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+30,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+28,0,"valid",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+24,0,"data_in",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBit(c+33,0,"full_frame_complete",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+34,0,"subapetures_completed",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBus(c+35,0,"intensity",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+36,0,"x_intensity",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 19,0);
    tracep->declBus(c+37,0,"y_intensity",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 19,0);
    tracep->declBus(c+79,0,"subap_col",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 3,0);
    tracep->declBus(c+80,0,"subap_row",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 3,0);
    tracep->declBus(c+81,0,"count_pixel_h",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 3,0);
    tracep->declBus(c+82,0,"count_pixel_v",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 3,0);
    tracep->declBus(c+83,0,"subap_idx",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBit(c+84,0,"last_pixel_h",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+85,0,"last_pixel_v",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+86,0,"last_subap_col",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+87,0,"last_subap_row",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+88,0,"subap_done_delay",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+89,0,"subap_idx_delay",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBus(c+90,0,"j",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->popPrefix();
    tracep->pushPrefix("em", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+273,0,"NUM_MODES",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+274,0,"NUM_SUBS",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+275,0,"NUM_SLOPES",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBit(c+251,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+30,0,"rst",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+46,0,"sub_valid",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+44,0,"x_slope",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+45,0,"y_slope",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declArray(c+47,0,"zernike_out",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 269,0);
    tracep->declBit(c+56,0,"done",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+276,0,"SUB_BITS",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declBus(c+277,0,"ACC_WIDTH",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declBus(c+278,0,"STATE_IDLE",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declBus(c+279,0,"STATE_DONE",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declBus(c+280,0,"LAST_SUB",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBit(c+91,0,"state",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+92,0,"sub_counter",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->pushPrefix("mac_sum", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 10; ++i) {
        tracep->declQuad(c+93+i*2,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 54,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("acc", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 10; ++i) {
        tracep->declQuad(c+113+i*2,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 54,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("acc_next", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 10; ++i) {
        tracep->declQuad(c+133+i*2,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 54,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+153,0,"i",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->pushPrefix("mac_gen[0]", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+1,0,"ex",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 17,0);
    tracep->declBus(c+2,0,"ey",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 17,0);
    tracep->declQuad(c+154,0,"prod_x",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 44,0);
    tracep->declQuad(c+156,0,"prod_y",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 44,0);
    tracep->popPrefix();
    tracep->pushPrefix("mac_gen[1]", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+3,0,"ex",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 17,0);
    tracep->declBus(c+4,0,"ey",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 17,0);
    tracep->declQuad(c+158,0,"prod_x",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 44,0);
    tracep->declQuad(c+160,0,"prod_y",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 44,0);
    tracep->popPrefix();
    tracep->pushPrefix("mac_gen[2]", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+5,0,"ex",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 17,0);
    tracep->declBus(c+6,0,"ey",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 17,0);
    tracep->declQuad(c+162,0,"prod_x",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 44,0);
    tracep->declQuad(c+164,0,"prod_y",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 44,0);
    tracep->popPrefix();
    tracep->pushPrefix("mac_gen[3]", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+7,0,"ex",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 17,0);
    tracep->declBus(c+8,0,"ey",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 17,0);
    tracep->declQuad(c+166,0,"prod_x",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 44,0);
    tracep->declQuad(c+168,0,"prod_y",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 44,0);
    tracep->popPrefix();
    tracep->pushPrefix("mac_gen[4]", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+9,0,"ex",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 17,0);
    tracep->declBus(c+10,0,"ey",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 17,0);
    tracep->declQuad(c+170,0,"prod_x",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 44,0);
    tracep->declQuad(c+172,0,"prod_y",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 44,0);
    tracep->popPrefix();
    tracep->pushPrefix("mac_gen[5]", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+11,0,"ex",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 17,0);
    tracep->declBus(c+12,0,"ey",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 17,0);
    tracep->declQuad(c+174,0,"prod_x",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 44,0);
    tracep->declQuad(c+176,0,"prod_y",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 44,0);
    tracep->popPrefix();
    tracep->pushPrefix("mac_gen[6]", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+13,0,"ex",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 17,0);
    tracep->declBus(c+14,0,"ey",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 17,0);
    tracep->declQuad(c+178,0,"prod_x",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 44,0);
    tracep->declQuad(c+180,0,"prod_y",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 44,0);
    tracep->popPrefix();
    tracep->pushPrefix("mac_gen[7]", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+15,0,"ex",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 17,0);
    tracep->declBus(c+16,0,"ey",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 17,0);
    tracep->declQuad(c+182,0,"prod_x",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 44,0);
    tracep->declQuad(c+184,0,"prod_y",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 44,0);
    tracep->popPrefix();
    tracep->pushPrefix("mac_gen[8]", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+17,0,"ex",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 17,0);
    tracep->declBus(c+18,0,"ey",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 17,0);
    tracep->declQuad(c+186,0,"prod_x",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 44,0);
    tracep->declQuad(c+188,0,"prod_y",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 44,0);
    tracep->popPrefix();
    tracep->pushPrefix("mac_gen[9]", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+19,0,"ex",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 17,0);
    tracep->declBus(c+20,0,"ey",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 17,0);
    tracep->declQuad(c+190,0,"prod_x",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 44,0);
    tracep->declQuad(c+192,0,"prod_y",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 44,0);
    tracep->popPrefix();
    tracep->popPrefix();
    tracep->pushPrefix("hps_reset_sync", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+253,0,"reset_async",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+251,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+78,0,"reset_sync",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+194,0,"sync_ff",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 1,0);
    tracep->declBit(c+195,0,"sync_d",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+196,0,"reset_count",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 3,0);
    tracep->declBit(c+197,0,"sync_level",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+198,0,"rising_edge",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->popPrefix();
    tracep->pushPrefix("key_reset_sync", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+252,0,"reset_async",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+251,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+77,0,"reset_sync",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+199,0,"sync_ff",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 1,0);
    tracep->declBit(c+200,0,"sync_d",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+201,0,"reset_count",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 3,0);
    tracep->declBit(c+202,0,"sync_level",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+203,0,"rising_edge",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->popPrefix();
    tracep->pushPrefix("reciprocal", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+251,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+30,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+36,0,"xI_in",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 19,0);
    tracep->declBus(c+37,0,"yI_in",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 19,0);
    tracep->declBus(c+35,0,"sI",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+34,0,"centroids_done_in",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBus(c+38,0,"xI_out",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 19,0);
    tracep->declBus(c+39,0,"yI_out",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 19,0);
    tracep->declBus(c+40,0,"rI_out",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+41,0,"centroids_done_out",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBus(c+204,0,"reciprocal_wire",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBit(c+205,0,"divide_by_zero_wire",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+206,0,"saturated_wire",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("recip", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+35,0,"v_u16",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+204,0,"reciprocal_q27",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBit(c+205,0,"divide_by_zero",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+206,0,"saturated",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declArray(c+281,0,"RECIP_SEED_LUT_Q0_16",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 4095,0);
    tracep->declBit(c+205,0,"v_is_zero",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+207,0,"v_safe",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+208,0,"shift_left",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 4,0);
    tracep->declBus(c+209,0,"a_q1_15",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+210,0,"a_q1_26",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+211,0,"lut_idx",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBus(c+212,0,"seed_q0_16",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+213,0,"x0_u27",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+214,0,"x1_q0_27",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBit(c+215,0,"x1_saturated",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+216,0,"x2_q0_27",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBit(c+217,0,"x2_saturated",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+218,0,"msb_index",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 4,0);
    tracep->declBus(c+219,0,"denorm_round_bias",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+220,0,"denorm_numer",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 27,0);
    tracep->declBus(c+221,0,"out_q0_27_ext",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 27,0);
    tracep->declBit(c+222,0,"out_sat",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("u_newton_step_0", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+210,0,"a_q1_26",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+213,0,"x0_u27",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+214,0,"x1_q0_27",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBit(c+215,0,"saturated",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declQuad(c+223,0,"ax_mul",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 53,0);
    tracep->declBus(c+225,0,"ax_q1_26",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 27,0);
    tracep->declBus(c+225,0,"ax_q1_26_clamped",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 27,0);
    tracep->declBus(c+226,0,"two_minus_ax_q1_26",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 27,0);
    tracep->declQuad(c+227,0,"prod_mul",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 54,0);
    tracep->declBus(c+229,0,"x1_rounded_ext",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 28,0);
    tracep->popPrefix();
    tracep->pushPrefix("u_newton_step_1", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+210,0,"a_q1_26",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+214,0,"x0_u27",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+216,0,"x1_q0_27",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBit(c+217,0,"saturated",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declQuad(c+230,0,"ax_mul",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 53,0);
    tracep->declBus(c+232,0,"ax_q1_26",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 27,0);
    tracep->declBus(c+232,0,"ax_q1_26_clamped",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 27,0);
    tracep->declBus(c+233,0,"two_minus_ax_q1_26",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 27,0);
    tracep->declQuad(c+234,0,"prod_mul",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 54,0);
    tracep->declBus(c+236,0,"x1_rounded_ext",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 28,0);
    tracep->popPrefix();
    tracep->popPrefix();
    tracep->popPrefix();
    tracep->pushPrefix("slopes", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+251,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+30,0,"rst",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+41,0,"subapetures_completed",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBit(c+27,0,"frame_complete",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+40,0,"rec_intensity",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+38,0,"x_intensity",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 19,0);
    tracep->declBus(c+39,0,"y_intensity",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 19,0);
    tracep->declBus(c+42,0,"x_centroid",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+43,0,"y_centroid",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+44,0,"x_slope",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+45,0,"y_slope",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBit(c+46,0,"new_subapeture",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+409,0,"SCALE",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declBus(c+410,0,"x_ref",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+410,0,"y_ref",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+237,0,"x_centroid_mult",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+238,0,"y_centroid_mult",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+239,0,"raw_x_slope",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 27,0);
    tracep->declBus(c+240,0,"raw_y_slope",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 27,0);
    tracep->declBus(c+241,0,"current_subap",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->pushPrefix("x_mult", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+237,0,"out",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+38,0,"a",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 19,0);
    tracep->declBus(c+40,0,"b",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declQuad(c+242,0,"mult_out",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 46,0);
    tracep->popPrefix();
    tracep->pushPrefix("y_mult", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+238,0,"out",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+39,0,"a",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 19,0);
    tracep->declBus(c+40,0,"b",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declQuad(c+244,0,"mult_out",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 46,0);
    tracep->popPrefix();
    tracep->popPrefix();
    tracep->pushPrefix("streamer", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+411,0,"HSIZE",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+411,0,"VSIZE",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+412,0,"HBLANK",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+413,0,"VBLANK",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBit(c+251,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+30,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+24,0,"data",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBit(c+25,0,"fv",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+26,0,"lv",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+27,0,"frame_complete",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+28,0,"valid",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+246,0,"line_counter",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBus(c+247,0,"row_counter",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBus(c+248,0,"h_blank_counter",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 1,0);
    tracep->declBus(c+249,0,"v_blank_counter",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBus(c+414,0,"STATE_FRAME_INIT",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 1,0);
    tracep->declBus(c+415,0,"STATE_ACTIVE_FRAME",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 1,0);
    tracep->declBus(c+416,0,"STATE_HOROZONTAL_BLANKING",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 1,0);
    tracep->declBus(c+417,0,"STATE_VERTICAL_BLANKING",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 1,0);
    tracep->declBus(c+250,0,"state",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 1,0);
    tracep->popPrefix();
    tracep->popPrefix();
    tracep->popPrefix();
}

VL_ATTR_COLD void Vtb___024root__trace_init_top(Vtb___024root* vlSelf, VerilatedVcd* tracep) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb___024root__trace_init_top\n"); );
    // Body
    Vtb___024root__trace_init_sub__TOP__0(vlSelf, tracep);
}

VL_ATTR_COLD void Vtb___024root__trace_const_0(void* voidSelf, VerilatedVcd::Buffer* bufp);
VL_ATTR_COLD void Vtb___024root__trace_full_0(void* voidSelf, VerilatedVcd::Buffer* bufp);
void Vtb___024root__trace_chg_0(void* voidSelf, VerilatedVcd::Buffer* bufp);
void Vtb___024root__trace_cleanup(void* voidSelf, VerilatedVcd* /*unused*/);

VL_ATTR_COLD void Vtb___024root__trace_register(Vtb___024root* vlSelf, VerilatedVcd* tracep) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb___024root__trace_register\n"); );
    // Body
    tracep->addConstCb(&Vtb___024root__trace_const_0, 0U, vlSelf);
    tracep->addFullCb(&Vtb___024root__trace_full_0, 0U, vlSelf);
    tracep->addChgCb(&Vtb___024root__trace_chg_0, 0U, vlSelf);
    tracep->addCleanupCb(&Vtb___024root__trace_cleanup, vlSelf);
}

VL_ATTR_COLD void Vtb___024root__trace_const_0_sub_0(Vtb___024root* vlSelf, VerilatedVcd::Buffer* bufp);

VL_ATTR_COLD void Vtb___024root__trace_const_0(void* voidSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb___024root__trace_const_0\n"); );
    // Init
    Vtb___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vtb___024root*>(voidSelf);
    Vtb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    // Body
    Vtb___024root__trace_const_0_sub_0((&vlSymsp->TOP), bufp);
}

extern const VlWide<128>/*4095:0*/ Vtb__ConstPool__CONST_h29f91db3_0;

VL_ATTR_COLD void Vtb___024root__trace_const_0_sub_0(Vtb___024root* vlSelf, VerilatedVcd::Buffer* bufp) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb___024root__trace_const_0_sub_0\n"); );
    // Init
    uint32_t* const oldp VL_ATTR_UNUSED = bufp->oldp(vlSymsp->__Vm_baseCode);
    // Body
    bufp->fullSData(oldp+255,(0U),16);
    bufp->fullIData(oldp+256,(vlSelf->tb__DOT__dut__DOT__result_rdata),32);
    bufp->fullBit(oldp+257,(1U));
    bufp->fullCData(oldp+258,(0xfU),4);
    bufp->fullCData(oldp+259,(vlSelf->tb__DOT__dut__DOT__ctrl_reg_h2f),8);
    bufp->fullSData(oldp+260,(0U),11);
    bufp->fullSData(oldp+261,(0x100U),11);
    bufp->fullSData(oldp+262,(0x200U),11);
    bufp->fullSData(oldp+263,(0x300U),11);
    bufp->fullSData(oldp+264,(0x400U),11);
    bufp->fullCData(oldp+265,(0U),5);
    bufp->fullCData(oldp+266,(1U),5);
    bufp->fullCData(oldp+267,(2U),5);
    bufp->fullCData(oldp+268,(3U),5);
    bufp->fullCData(oldp+269,(4U),5);
    bufp->fullCData(oldp+270,(5U),5);
    bufp->fullCData(oldp+271,(6U),5);
    bufp->fullIData(oldp+272,(0x10U),32);
    bufp->fullIData(oldp+273,(0xaU),32);
    bufp->fullIData(oldp+274,(0xa8U),32);
    bufp->fullIData(oldp+275,(0x188U),32);
    bufp->fullIData(oldp+276,(8U),32);
    bufp->fullIData(oldp+277,(0x37U),32);
    bufp->fullIData(oldp+278,(0U),32);
    bufp->fullIData(oldp+279,(1U),32);
    bufp->fullCData(oldp+280,(0xa7U),8);
    bufp->fullWData(oldp+281,(Vtb__ConstPool__CONST_h29f91db3_0),4096);
    bufp->fullIData(oldp+409,(0x800000U),32);
    bufp->fullIData(oldp+410,(0x3c00000U),32);
    bufp->fullIData(oldp+411,(0x100U),32);
    bufp->fullIData(oldp+412,(4U),32);
    bufp->fullIData(oldp+413,(0x98U),32);
    bufp->fullCData(oldp+414,(0U),2);
    bufp->fullCData(oldp+415,(1U),2);
    bufp->fullCData(oldp+416,(2U),2);
    bufp->fullCData(oldp+417,(3U),2);
}

VL_ATTR_COLD void Vtb___024root__trace_full_0_sub_0(Vtb___024root* vlSelf, VerilatedVcd::Buffer* bufp);

VL_ATTR_COLD void Vtb___024root__trace_full_0(void* voidSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb___024root__trace_full_0\n"); );
    // Init
    Vtb___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vtb___024root*>(voidSelf);
    Vtb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    // Body
    Vtb___024root__trace_full_0_sub_0((&vlSymsp->TOP), bufp);
}

VL_ATTR_COLD void Vtb___024root__trace_full_0_sub_0(Vtb___024root* vlSelf, VerilatedVcd::Buffer* bufp) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb___024root__trace_full_0_sub_0\n"); );
    // Init
    uint32_t* const oldp VL_ATTR_UNUSED = bufp->oldp(vlSymsp->__Vm_baseCode);
    // Body
    bufp->fullIData(oldp+1,(vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x
                            [vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter]),18);
    bufp->fullIData(oldp+2,(vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y
                            [vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter]),18);
    bufp->fullIData(oldp+3,(((0x68fU >= (0x7ffU & ((IData)(0xa8U) 
                                                   + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                              ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x
                             [(0x7ffU & ((IData)(0xa8U) 
                                         + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                              : 0U)),18);
    bufp->fullIData(oldp+4,(((0x68fU >= (0x7ffU & ((IData)(0xa8U) 
                                                   + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                              ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y
                             [(0x7ffU & ((IData)(0xa8U) 
                                         + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                              : 0U)),18);
    bufp->fullIData(oldp+5,(((0x68fU >= (0x7ffU & ((IData)(0x150U) 
                                                   + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                              ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x
                             [(0x7ffU & ((IData)(0x150U) 
                                         + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                              : 0U)),18);
    bufp->fullIData(oldp+6,(((0x68fU >= (0x7ffU & ((IData)(0x150U) 
                                                   + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                              ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y
                             [(0x7ffU & ((IData)(0x150U) 
                                         + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                              : 0U)),18);
    bufp->fullIData(oldp+7,(((0x68fU >= (0x7ffU & ((IData)(0x1f8U) 
                                                   + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                              ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x
                             [(0x7ffU & ((IData)(0x1f8U) 
                                         + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                              : 0U)),18);
    bufp->fullIData(oldp+8,(((0x68fU >= (0x7ffU & ((IData)(0x1f8U) 
                                                   + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                              ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y
                             [(0x7ffU & ((IData)(0x1f8U) 
                                         + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                              : 0U)),18);
    bufp->fullIData(oldp+9,(((0x68fU >= (0x7ffU & ((IData)(0x2a0U) 
                                                   + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                              ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x
                             [(0x7ffU & ((IData)(0x2a0U) 
                                         + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                              : 0U)),18);
    bufp->fullIData(oldp+10,(((0x68fU >= (0x7ffU & 
                                          ((IData)(0x2a0U) 
                                           + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                               ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y
                              [(0x7ffU & ((IData)(0x2a0U) 
                                          + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                               : 0U)),18);
    bufp->fullIData(oldp+11,(((0x68fU >= (0x7ffU & 
                                          ((IData)(0x348U) 
                                           + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                               ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x
                              [(0x7ffU & ((IData)(0x348U) 
                                          + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                               : 0U)),18);
    bufp->fullIData(oldp+12,(((0x68fU >= (0x7ffU & 
                                          ((IData)(0x348U) 
                                           + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                               ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y
                              [(0x7ffU & ((IData)(0x348U) 
                                          + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                               : 0U)),18);
    bufp->fullIData(oldp+13,(((0x68fU >= (0x7ffU & 
                                          ((IData)(0x3f0U) 
                                           + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                               ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x
                              [(0x7ffU & ((IData)(0x3f0U) 
                                          + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                               : 0U)),18);
    bufp->fullIData(oldp+14,(((0x68fU >= (0x7ffU & 
                                          ((IData)(0x3f0U) 
                                           + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                               ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y
                              [(0x7ffU & ((IData)(0x3f0U) 
                                          + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                               : 0U)),18);
    bufp->fullIData(oldp+15,(((0x68fU >= (0x7ffU & 
                                          ((IData)(0x498U) 
                                           + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                               ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x
                              [(0x7ffU & ((IData)(0x498U) 
                                          + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                               : 0U)),18);
    bufp->fullIData(oldp+16,(((0x68fU >= (0x7ffU & 
                                          ((IData)(0x498U) 
                                           + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                               ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y
                              [(0x7ffU & ((IData)(0x498U) 
                                          + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                               : 0U)),18);
    bufp->fullIData(oldp+17,(((0x68fU >= (0x7ffU & 
                                          ((IData)(0x540U) 
                                           + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                               ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x
                              [(0x7ffU & ((IData)(0x540U) 
                                          + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                               : 0U)),18);
    bufp->fullIData(oldp+18,(((0x68fU >= (0x7ffU & 
                                          ((IData)(0x540U) 
                                           + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                               ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y
                              [(0x7ffU & ((IData)(0x540U) 
                                          + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                               : 0U)),18);
    bufp->fullIData(oldp+19,(((0x68fU >= (0x7ffU & 
                                          ((IData)(0x5e8U) 
                                           + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                               ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x
                              [(0x7ffU & ((IData)(0x5e8U) 
                                          + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                               : 0U)),18);
    bufp->fullIData(oldp+20,(((0x68fU >= (0x7ffU & 
                                          ((IData)(0x5e8U) 
                                           + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                               ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y
                              [(0x7ffU & ((IData)(0x5e8U) 
                                          + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                               : 0U)),18);
    bufp->fullIData(oldp+21,(vlSelf->tb__DOT__dut__DOT__result_wdata),32);
    bufp->fullSData(oldp+22,(vlSelf->tb__DOT__dut__DOT__result_addr),11);
    bufp->fullBit(oldp+23,(vlSelf->tb__DOT__dut__DOT__result_write));
    bufp->fullCData(oldp+24,(vlSelf->tb__DOT__dut__DOT__streamer_data),8);
    bufp->fullBit(oldp+25,(vlSelf->tb__DOT__dut__DOT__streamer_fv));
    bufp->fullBit(oldp+26,(vlSelf->tb__DOT__dut__DOT__streamer_lv));
    bufp->fullBit(oldp+27,(vlSelf->tb__DOT__dut__DOT__frame_complete));
    bufp->fullBit(oldp+28,(vlSelf->tb__DOT__dut__DOT__streamer_valid));
    bufp->fullCData(oldp+29,(vlSelf->tb__DOT__dut__DOT__ctrl_reg_f2h),8);
    bufp->fullBit(oldp+30,(vlSelf->tb__DOT__dut__DOT__reset));
    bufp->fullCData(oldp+31,(vlSelf->tb__DOT__dut__DOT__start_sync),2);
    bufp->fullBit(oldp+32,((1U & ((IData)(vlSelf->tb__DOT__dut__DOT__start_sync) 
                                  >> 1U))));
    bufp->fullBit(oldp+33,(vlSelf->tb__DOT__dut__DOT__full_frame_complete_accumulator));
    bufp->fullCData(oldp+34,(vlSelf->tb__DOT__dut__DOT__subaps_done_accumulator),8);
    bufp->fullSData(oldp+35,(vlSelf->tb__DOT__dut__DOT__intensity_accumulator),16);
    bufp->fullIData(oldp+36,(vlSelf->tb__DOT__dut__DOT__xI_accumulator),20);
    bufp->fullIData(oldp+37,(vlSelf->tb__DOT__dut__DOT__yI_accumulator),20);
    bufp->fullIData(oldp+38,(vlSelf->tb__DOT__dut__DOT__xI_reciprocal),20);
    bufp->fullIData(oldp+39,(vlSelf->tb__DOT__dut__DOT__yI_reciprocal),20);
    bufp->fullIData(oldp+40,(vlSelf->tb__DOT__dut__DOT__rI_reciprocal),27);
    bufp->fullCData(oldp+41,(vlSelf->tb__DOT__dut__DOT__subaps_done_reciprocal),8);
    bufp->fullIData(oldp+42,(vlSelf->tb__DOT__dut__DOT__x_centroid),27);
    bufp->fullIData(oldp+43,(vlSelf->tb__DOT__dut__DOT__y_centroid),27);
    bufp->fullIData(oldp+44,(vlSelf->tb__DOT__dut__DOT__x_slopes),27);
    bufp->fullIData(oldp+45,(vlSelf->tb__DOT__dut__DOT__y_slopes),27);
    bufp->fullBit(oldp+46,(vlSelf->tb__DOT__dut__DOT__new_subapeture));
    bufp->fullWData(oldp+47,(vlSelf->tb__DOT__dut__DOT__zernike_out),270);
    bufp->fullBit(oldp+56,(vlSelf->tb__DOT__dut__DOT__done));
    bufp->fullIData(oldp+57,(vlSelf->tb__DOT__dut__DOT__zernike_out_reg[0]),27);
    bufp->fullIData(oldp+58,(vlSelf->tb__DOT__dut__DOT__zernike_out_reg[1]),27);
    bufp->fullIData(oldp+59,(vlSelf->tb__DOT__dut__DOT__zernike_out_reg[2]),27);
    bufp->fullIData(oldp+60,(vlSelf->tb__DOT__dut__DOT__zernike_out_reg[3]),27);
    bufp->fullIData(oldp+61,(vlSelf->tb__DOT__dut__DOT__zernike_out_reg[4]),27);
    bufp->fullIData(oldp+62,(vlSelf->tb__DOT__dut__DOT__zernike_out_reg[5]),27);
    bufp->fullIData(oldp+63,(vlSelf->tb__DOT__dut__DOT__zernike_out_reg[6]),27);
    bufp->fullIData(oldp+64,(vlSelf->tb__DOT__dut__DOT__zernike_out_reg[7]),27);
    bufp->fullIData(oldp+65,(vlSelf->tb__DOT__dut__DOT__zernike_out_reg[8]),27);
    bufp->fullIData(oldp+66,(vlSelf->tb__DOT__dut__DOT__zernike_out_reg[9]),27);
    bufp->fullIData(oldp+67,(vlSelf->tb__DOT__dut__DOT__x_centroid_out),27);
    bufp->fullIData(oldp+68,(vlSelf->tb__DOT__dut__DOT__y_centroid_out),27);
    bufp->fullIData(oldp+69,(vlSelf->tb__DOT__dut__DOT__x_slopes_out),27);
    bufp->fullIData(oldp+70,(vlSelf->tb__DOT__dut__DOT__y_slopes_out),27);
    bufp->fullSData(oldp+71,(vlSelf->tb__DOT__dut__DOT__resw_write_idx),11);
    bufp->fullCData(oldp+72,(((0x10U & (IData)(vlSelf->tb__DOT__dut__DOT__resw_state))
                               ? 0U : ((8U & (IData)(vlSelf->tb__DOT__dut__DOT__resw_state))
                                        ? 0U : ((4U 
                                                 & (IData)(vlSelf->tb__DOT__dut__DOT__resw_state))
                                                 ? 
                                                ((2U 
                                                  & (IData)(vlSelf->tb__DOT__dut__DOT__resw_state))
                                                  ? 0U
                                                  : 
                                                 ((1U 
                                                   & (IData)(vlSelf->tb__DOT__dut__DOT__resw_state))
                                                   ? 
                                                  ((9U 
                                                    == (IData)(vlSelf->tb__DOT__dut__DOT__write_zernike_count))
                                                    ? 6U
                                                    : 5U)
                                                   : 
                                                  ((IData)(vlSelf->tb__DOT__dut__DOT__write_zernike_latch)
                                                    ? 5U
                                                    : 6U)))
                                                 : 
                                                ((2U 
                                                  & (IData)(vlSelf->tb__DOT__dut__DOT__resw_state))
                                                  ? 
                                                 ((1U 
                                                   & (IData)(vlSelf->tb__DOT__dut__DOT__resw_state))
                                                   ? 4U
                                                   : 3U)
                                                  : 
                                                 ((1U 
                                                   & (IData)(vlSelf->tb__DOT__dut__DOT__resw_state))
                                                   ? 2U
                                                   : 
                                                  ((IData)(vlSelf->tb__DOT__dut__DOT__resw_start)
                                                    ? 1U
                                                    : 0U))))))),5);
    bufp->fullBit(oldp+73,(vlSelf->tb__DOT__dut__DOT__resw_start));
    bufp->fullCData(oldp+74,(vlSelf->tb__DOT__dut__DOT__resw_state),5);
    bufp->fullBit(oldp+75,(vlSelf->tb__DOT__dut__DOT__write_zernike_latch));
    bufp->fullSData(oldp+76,(vlSelf->tb__DOT__dut__DOT__write_zernike_count),11);
    bufp->fullBit(oldp+77,((0U != (IData)(vlSelf->tb__DOT__dut__DOT__key_reset_sync__DOT__reset_count))));
    bufp->fullBit(oldp+78,((0U != (IData)(vlSelf->tb__DOT__dut__DOT__hps_reset_sync__DOT__reset_count))));
    bufp->fullCData(oldp+79,(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_col),4);
    bufp->fullCData(oldp+80,(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_row),4);
    bufp->fullCData(oldp+81,(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__count_pixel_h),4);
    bufp->fullCData(oldp+82,(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__count_pixel_v),4);
    bufp->fullCData(oldp+83,(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_idx),8);
    bufp->fullBit(oldp+84,((0xfU == (IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__count_pixel_h))));
    bufp->fullBit(oldp+85,((0xfU == (IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__count_pixel_v))));
    bufp->fullBit(oldp+86,((0xfU == (IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_col))));
    bufp->fullBit(oldp+87,((0xfU == (IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_row))));
    bufp->fullBit(oldp+88,(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_done_delay));
    bufp->fullCData(oldp+89,(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_idx_delay),8);
    bufp->fullIData(oldp+90,(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__j),32);
    bufp->fullBit(oldp+91,(vlSelf->tb__DOT__dut__DOT__em__DOT__state));
    bufp->fullCData(oldp+92,(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter),8);
    bufp->fullQData(oldp+93,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_sum[0]),55);
    bufp->fullQData(oldp+95,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_sum[1]),55);
    bufp->fullQData(oldp+97,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_sum[2]),55);
    bufp->fullQData(oldp+99,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_sum[3]),55);
    bufp->fullQData(oldp+101,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_sum[4]),55);
    bufp->fullQData(oldp+103,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_sum[5]),55);
    bufp->fullQData(oldp+105,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_sum[6]),55);
    bufp->fullQData(oldp+107,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_sum[7]),55);
    bufp->fullQData(oldp+109,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_sum[8]),55);
    bufp->fullQData(oldp+111,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_sum[9]),55);
    bufp->fullQData(oldp+113,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc[0]),55);
    bufp->fullQData(oldp+115,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc[1]),55);
    bufp->fullQData(oldp+117,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc[2]),55);
    bufp->fullQData(oldp+119,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc[3]),55);
    bufp->fullQData(oldp+121,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc[4]),55);
    bufp->fullQData(oldp+123,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc[5]),55);
    bufp->fullQData(oldp+125,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc[6]),55);
    bufp->fullQData(oldp+127,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc[7]),55);
    bufp->fullQData(oldp+129,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc[8]),55);
    bufp->fullQData(oldp+131,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc[9]),55);
    bufp->fullQData(oldp+133,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next[0]),55);
    bufp->fullQData(oldp+135,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next[1]),55);
    bufp->fullQData(oldp+137,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next[2]),55);
    bufp->fullQData(oldp+139,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next[3]),55);
    bufp->fullQData(oldp+141,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next[4]),55);
    bufp->fullQData(oldp+143,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next[5]),55);
    bufp->fullQData(oldp+145,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next[6]),55);
    bufp->fullQData(oldp+147,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next[7]),55);
    bufp->fullQData(oldp+149,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next[8]),55);
    bufp->fullQData(oldp+151,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next[9]),55);
    bufp->fullIData(oldp+153,(vlSelf->tb__DOT__dut__DOT__em__DOT__i),32);
    bufp->fullQData(oldp+154,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__0__KET____DOT__prod_x),45);
    bufp->fullQData(oldp+156,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__0__KET____DOT__prod_y),45);
    bufp->fullQData(oldp+158,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__1__KET____DOT__prod_x),45);
    bufp->fullQData(oldp+160,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__1__KET____DOT__prod_y),45);
    bufp->fullQData(oldp+162,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__2__KET____DOT__prod_x),45);
    bufp->fullQData(oldp+164,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__2__KET____DOT__prod_y),45);
    bufp->fullQData(oldp+166,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__3__KET____DOT__prod_x),45);
    bufp->fullQData(oldp+168,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__3__KET____DOT__prod_y),45);
    bufp->fullQData(oldp+170,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__4__KET____DOT__prod_x),45);
    bufp->fullQData(oldp+172,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__4__KET____DOT__prod_y),45);
    bufp->fullQData(oldp+174,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__5__KET____DOT__prod_x),45);
    bufp->fullQData(oldp+176,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__5__KET____DOT__prod_y),45);
    bufp->fullQData(oldp+178,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__6__KET____DOT__prod_x),45);
    bufp->fullQData(oldp+180,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__6__KET____DOT__prod_y),45);
    bufp->fullQData(oldp+182,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__7__KET____DOT__prod_x),45);
    bufp->fullQData(oldp+184,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__7__KET____DOT__prod_y),45);
    bufp->fullQData(oldp+186,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__8__KET____DOT__prod_x),45);
    bufp->fullQData(oldp+188,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__8__KET____DOT__prod_y),45);
    bufp->fullQData(oldp+190,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__9__KET____DOT__prod_x),45);
    bufp->fullQData(oldp+192,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__9__KET____DOT__prod_y),45);
    bufp->fullCData(oldp+194,(vlSelf->tb__DOT__dut__DOT__hps_reset_sync__DOT__sync_ff),2);
    bufp->fullBit(oldp+195,(vlSelf->tb__DOT__dut__DOT__hps_reset_sync__DOT__sync_d));
    bufp->fullCData(oldp+196,(vlSelf->tb__DOT__dut__DOT__hps_reset_sync__DOT__reset_count),4);
    bufp->fullBit(oldp+197,((1U & ((IData)(vlSelf->tb__DOT__dut__DOT__hps_reset_sync__DOT__sync_ff) 
                                   >> 1U))));
    bufp->fullBit(oldp+198,(((~ (IData)(vlSelf->tb__DOT__dut__DOT__hps_reset_sync__DOT__sync_d)) 
                             & ((IData)(vlSelf->tb__DOT__dut__DOT__hps_reset_sync__DOT__sync_ff) 
                                >> 1U))));
    bufp->fullCData(oldp+199,(vlSelf->tb__DOT__dut__DOT__key_reset_sync__DOT__sync_ff),2);
    bufp->fullBit(oldp+200,(vlSelf->tb__DOT__dut__DOT__key_reset_sync__DOT__sync_d));
    bufp->fullCData(oldp+201,(vlSelf->tb__DOT__dut__DOT__key_reset_sync__DOT__reset_count),4);
    bufp->fullBit(oldp+202,((1U & ((IData)(vlSelf->tb__DOT__dut__DOT__key_reset_sync__DOT__sync_ff) 
                                   >> 1U))));
    bufp->fullBit(oldp+203,(((~ (IData)(vlSelf->tb__DOT__dut__DOT__key_reset_sync__DOT__sync_d)) 
                             & ((IData)(vlSelf->tb__DOT__dut__DOT__key_reset_sync__DOT__sync_ff) 
                                >> 1U))));
    bufp->fullIData(oldp+204,(((0U == (IData)(vlSelf->tb__DOT__dut__DOT__intensity_accumulator))
                                ? 0x7ffffffU : ((0x7ffffffU 
                                                 < vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__out_q0_27_ext)
                                                 ? 0x7ffffffU
                                                 : 
                                                (0x7ffffffU 
                                                 & vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__out_q0_27_ext)))),27);
    bufp->fullBit(oldp+205,((0U == (IData)(vlSelf->tb__DOT__dut__DOT__intensity_accumulator))));
    bufp->fullBit(oldp+206,(((0U == (IData)(vlSelf->tb__DOT__dut__DOT__intensity_accumulator)) 
                             | (0x7ffffffU < vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__out_q0_27_ext))));
    bufp->fullSData(oldp+207,(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__v_safe),16);
    bufp->fullCData(oldp+208,(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__shift_left),5);
    bufp->fullSData(oldp+209,(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15),16);
    bufp->fullIData(oldp+210,(((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                               << 0xbU)),27);
    bufp->fullCData(oldp+211,((0xffU & ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                        >> 7U))),8);
    bufp->fullSData(oldp+212,((0xffffU & (((0U == (0x1fU 
                                                   & VL_SHIFTL_III(12,12,32, 
                                                                   (0xffU 
                                                                    & ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                       >> 7U)), 4U)))
                                            ? 0U : 
                                           (Vtb__ConstPool__CONST_h29f91db3_0[
                                            (((IData)(0xfU) 
                                              + (0xfffU 
                                                 & VL_SHIFTL_III(12,12,32, 
                                                                 (0xffU 
                                                                  & ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                     >> 7U)), 4U))) 
                                             >> 5U)] 
                                            << ((IData)(0x20U) 
                                                - (0x1fU 
                                                   & VL_SHIFTL_III(12,12,32, 
                                                                   (0xffU 
                                                                    & ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                       >> 7U)), 4U))))) 
                                          | (Vtb__ConstPool__CONST_h29f91db3_0[
                                             (0x7fU 
                                              & (VL_SHIFTL_III(12,12,32, 
                                                               (0xffU 
                                                                & ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                   >> 7U)), 4U) 
                                                 >> 5U))] 
                                             >> (0x1fU 
                                                 & VL_SHIFTL_III(12,12,32, 
                                                                 (0xffU 
                                                                  & ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                     >> 7U)), 4U)))))),16);
    bufp->fullIData(oldp+213,((0x7fff800U & ((((0U 
                                                == 
                                                (0x1fU 
                                                 & VL_SHIFTL_III(12,12,32, 
                                                                 (0xffU 
                                                                  & ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                     >> 7U)), 4U)))
                                                ? 0U
                                                : (
                                                   Vtb__ConstPool__CONST_h29f91db3_0[
                                                   (((IData)(0xfU) 
                                                     + 
                                                     (0xfffU 
                                                      & VL_SHIFTL_III(12,12,32, 
                                                                      (0xffU 
                                                                       & ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                          >> 7U)), 4U))) 
                                                    >> 5U)] 
                                                   << 
                                                   ((IData)(0x20U) 
                                                    - 
                                                    (0x1fU 
                                                     & VL_SHIFTL_III(12,12,32, 
                                                                     (0xffU 
                                                                      & ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                         >> 7U)), 4U))))) 
                                              | (Vtb__ConstPool__CONST_h29f91db3_0[
                                                 (0x7fU 
                                                  & (VL_SHIFTL_III(12,12,32, 
                                                                   (0xffU 
                                                                    & ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                       >> 7U)), 4U) 
                                                     >> 5U))] 
                                                 >> 
                                                 (0x1fU 
                                                  & VL_SHIFTL_III(12,12,32, 
                                                                  (0xffU 
                                                                   & ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                      >> 7U)), 4U)))) 
                                             << 0xbU))),27);
    bufp->fullIData(oldp+214,(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__x1_q0_27),27);
    bufp->fullBit(oldp+215,((0x7ffffffU < (0x1fffffffU 
                                           & (IData)(
                                                     (0x1fffffffULL 
                                                      & ((0x2000000ULL 
                                                          + vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__u_newton_step_0__DOT__prod_mul) 
                                                         >> 0x1aU)))))));
    bufp->fullIData(oldp+216,(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__x2_q0_27),27);
    bufp->fullBit(oldp+217,((0x7ffffffU < (0x1fffffffU 
                                           & (IData)(
                                                     (0x1fffffffULL 
                                                      & ((0x2000000ULL 
                                                          + vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__u_newton_step_1__DOT__prod_mul) 
                                                         >> 0x1aU)))))));
    bufp->fullCData(oldp+218,((0x1fU & ((IData)(0xfU) 
                                        - (IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__shift_left)))),5);
    bufp->fullIData(oldp+219,(((0U == (0x1fU & ((IData)(0xfU) 
                                                - (IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__shift_left))))
                                ? 0U : (0x7ffffffU 
                                        & ((IData)(1U) 
                                           << (0x1fU 
                                               & (((IData)(0xfU) 
                                                   - (IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__shift_left)) 
                                                  - (IData)(1U))))))),27);
    bufp->fullIData(oldp+220,((0xfffffffU & (vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__x2_q0_27 
                                             + ((0U 
                                                 == 
                                                 (0x1fU 
                                                  & ((IData)(0xfU) 
                                                     - (IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__shift_left))))
                                                 ? 0U
                                                 : 
                                                (0x7ffffffU 
                                                 & ((IData)(1U) 
                                                    << 
                                                    (0x1fU 
                                                     & (((IData)(0xfU) 
                                                         - (IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__shift_left)) 
                                                        - (IData)(1U))))))))),28);
    bufp->fullIData(oldp+221,(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__out_q0_27_ext),28);
    bufp->fullBit(oldp+222,((0x7ffffffU < vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__out_q0_27_ext)));
    bufp->fullQData(oldp+223,((0x3fffffffffffffULL 
                               & ((QData)((IData)(((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                   << 0xbU))) 
                                  * (QData)((IData)(
                                                    (0x7fff800U 
                                                     & ((((0U 
                                                           == 
                                                           (0x1fU 
                                                            & VL_SHIFTL_III(12,12,32, 
                                                                            (0xffU 
                                                                             & ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                                >> 7U)), 4U)))
                                                           ? 0U
                                                           : 
                                                          (Vtb__ConstPool__CONST_h29f91db3_0[
                                                           (((IData)(0xfU) 
                                                             + 
                                                             (0xfffU 
                                                              & VL_SHIFTL_III(12,12,32, 
                                                                              (0xffU 
                                                                               & ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                                >> 7U)), 4U))) 
                                                            >> 5U)] 
                                                           << 
                                                           ((IData)(0x20U) 
                                                            - 
                                                            (0x1fU 
                                                             & VL_SHIFTL_III(12,12,32, 
                                                                             (0xffU 
                                                                              & ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                                >> 7U)), 4U))))) 
                                                         | (Vtb__ConstPool__CONST_h29f91db3_0[
                                                            (0x7fU 
                                                             & (VL_SHIFTL_III(12,12,32, 
                                                                              (0xffU 
                                                                               & ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                                >> 7U)), 4U) 
                                                                >> 5U))] 
                                                            >> 
                                                            (0x1fU 
                                                             & VL_SHIFTL_III(12,12,32, 
                                                                             (0xffU 
                                                                              & ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                                >> 7U)), 4U)))) 
                                                        << 0xbU))))))),54);
    bufp->fullIData(oldp+225,((0x7ffffffU & (IData)(
                                                    (0x7ffffffULL 
                                                     & (((QData)((IData)(
                                                                         ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                          << 0xbU))) 
                                                         * (QData)((IData)(
                                                                           (0x7fff800U 
                                                                            & ((((0U 
                                                                                == 
                                                                                (0x1fU 
                                                                                & VL_SHIFTL_III(12,12,32, 
                                                                                (0xffU 
                                                                                & ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                                >> 7U)), 4U)))
                                                                                 ? 0U
                                                                                 : 
                                                                                (Vtb__ConstPool__CONST_h29f91db3_0[
                                                                                (((IData)(0xfU) 
                                                                                + 
                                                                                (0xfffU 
                                                                                & VL_SHIFTL_III(12,12,32, 
                                                                                (0xffU 
                                                                                & ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                                >> 7U)), 4U))) 
                                                                                >> 5U)] 
                                                                                << 
                                                                                ((IData)(0x20U) 
                                                                                - 
                                                                                (0x1fU 
                                                                                & VL_SHIFTL_III(12,12,32, 
                                                                                (0xffU 
                                                                                & ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                                >> 7U)), 4U))))) 
                                                                                | (Vtb__ConstPool__CONST_h29f91db3_0[
                                                                                (0x7fU 
                                                                                & (VL_SHIFTL_III(12,12,32, 
                                                                                (0xffU 
                                                                                & ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                                >> 7U)), 4U) 
                                                                                >> 5U))] 
                                                                                >> 
                                                                                (0x1fU 
                                                                                & VL_SHIFTL_III(12,12,32, 
                                                                                (0xffU 
                                                                                & ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                                >> 7U)), 4U)))) 
                                                                               << 0xbU))))) 
                                                        >> 0x1bU))))),28);
    bufp->fullIData(oldp+226,((0xfffffffU & ((IData)(0x8000000U) 
                                             - (0x7ffffffU 
                                                & (IData)(
                                                          (0x7ffffffULL 
                                                           & (((QData)((IData)(
                                                                               ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                                << 0xbU))) 
                                                               * (QData)((IData)(
                                                                                (0x7fff800U 
                                                                                & ((((0U 
                                                                                == 
                                                                                (0x1fU 
                                                                                & VL_SHIFTL_III(12,12,32, 
                                                                                (0xffU 
                                                                                & ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                                >> 7U)), 4U)))
                                                                                 ? 0U
                                                                                 : 
                                                                                (Vtb__ConstPool__CONST_h29f91db3_0[
                                                                                (((IData)(0xfU) 
                                                                                + 
                                                                                (0xfffU 
                                                                                & VL_SHIFTL_III(12,12,32, 
                                                                                (0xffU 
                                                                                & ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                                >> 7U)), 4U))) 
                                                                                >> 5U)] 
                                                                                << 
                                                                                ((IData)(0x20U) 
                                                                                - 
                                                                                (0x1fU 
                                                                                & VL_SHIFTL_III(12,12,32, 
                                                                                (0xffU 
                                                                                & ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                                >> 7U)), 4U))))) 
                                                                                | (Vtb__ConstPool__CONST_h29f91db3_0[
                                                                                (0x7fU 
                                                                                & (VL_SHIFTL_III(12,12,32, 
                                                                                (0xffU 
                                                                                & ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                                >> 7U)), 4U) 
                                                                                >> 5U))] 
                                                                                >> 
                                                                                (0x1fU 
                                                                                & VL_SHIFTL_III(12,12,32, 
                                                                                (0xffU 
                                                                                & ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                                >> 7U)), 4U)))) 
                                                                                << 0xbU))))) 
                                                              >> 0x1bU))))))),28);
    bufp->fullQData(oldp+227,(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__u_newton_step_0__DOT__prod_mul),55);
    bufp->fullIData(oldp+229,((0x1fffffffU & (IData)(
                                                     (0x1fffffffULL 
                                                      & ((0x2000000ULL 
                                                          + vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__u_newton_step_0__DOT__prod_mul) 
                                                         >> 0x1aU))))),29);
    bufp->fullQData(oldp+230,((0x3fffffffffffffULL 
                               & ((QData)((IData)(((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                   << 0xbU))) 
                                  * (QData)((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__x1_q0_27))))),54);
    bufp->fullIData(oldp+232,((0x7ffffffU & (IData)(
                                                    (0x7ffffffULL 
                                                     & (((QData)((IData)(
                                                                         ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                          << 0xbU))) 
                                                         * (QData)((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__x1_q0_27))) 
                                                        >> 0x1bU))))),28);
    bufp->fullIData(oldp+233,((0xfffffffU & ((IData)(0x8000000U) 
                                             - (0x7ffffffU 
                                                & (IData)(
                                                          (0x7ffffffULL 
                                                           & (((QData)((IData)(
                                                                               ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                                << 0xbU))) 
                                                               * (QData)((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__x1_q0_27))) 
                                                              >> 0x1bU))))))),28);
    bufp->fullQData(oldp+234,(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__u_newton_step_1__DOT__prod_mul),55);
    bufp->fullIData(oldp+236,((0x1fffffffU & (IData)(
                                                     (0x1fffffffULL 
                                                      & ((0x2000000ULL 
                                                          + vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__u_newton_step_1__DOT__prod_mul) 
                                                         >> 0x1aU))))),29);
    bufp->fullIData(oldp+237,((0x7ffffffU & (IData)(
                                                    (vlSelf->tb__DOT__dut__DOT__slopes__DOT__x_mult__DOT__mult_out 
                                                     >> 4U)))),27);
    bufp->fullIData(oldp+238,((0x7ffffffU & (IData)(
                                                    (vlSelf->tb__DOT__dut__DOT__slopes__DOT__y_mult__DOT__mult_out 
                                                     >> 4U)))),27);
    bufp->fullIData(oldp+239,((0xfffffffU & (VL_EXTENDS_II(28,27, 
                                                           (0x7ffffffU 
                                                            & (IData)(
                                                                      (vlSelf->tb__DOT__dut__DOT__slopes__DOT__x_mult__DOT__mult_out 
                                                                       >> 4U)))) 
                                             - (IData)(0x3c00000U)))),28);
    bufp->fullIData(oldp+240,((0xfffffffU & (VL_EXTENDS_II(28,27, 
                                                           (0x7ffffffU 
                                                            & (IData)(
                                                                      (vlSelf->tb__DOT__dut__DOT__slopes__DOT__y_mult__DOT__mult_out 
                                                                       >> 4U)))) 
                                             - (IData)(0x3c00000U)))),28);
    bufp->fullCData(oldp+241,(vlSelf->tb__DOT__dut__DOT__slopes__DOT__current_subap),8);
    bufp->fullQData(oldp+242,(vlSelf->tb__DOT__dut__DOT__slopes__DOT__x_mult__DOT__mult_out),47);
    bufp->fullQData(oldp+244,(vlSelf->tb__DOT__dut__DOT__slopes__DOT__y_mult__DOT__mult_out),47);
    bufp->fullCData(oldp+246,(vlSelf->tb__DOT__dut__DOT__streamer__DOT__line_counter),8);
    bufp->fullCData(oldp+247,(vlSelf->tb__DOT__dut__DOT__streamer__DOT__row_counter),8);
    bufp->fullCData(oldp+248,(vlSelf->tb__DOT__dut__DOT__streamer__DOT__h_blank_counter),2);
    bufp->fullCData(oldp+249,(vlSelf->tb__DOT__dut__DOT__streamer__DOT__v_blank_counter),8);
    bufp->fullCData(oldp+250,(vlSelf->tb__DOT__dut__DOT__streamer__DOT__state),2);
    bufp->fullBit(oldp+251,(vlSelf->tb__DOT__clk));
    bufp->fullBit(oldp+252,(vlSelf->tb__DOT__key_reset));
    bufp->fullBit(oldp+253,(vlSelf->tb__DOT__hps_reset));
    bufp->fullIData(oldp+254,(vlSelf->tb__DOT__i),32);
}

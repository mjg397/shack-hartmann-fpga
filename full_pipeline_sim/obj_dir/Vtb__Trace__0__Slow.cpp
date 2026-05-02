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
    tracep->declBit(c+268,0,"clk",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+269,0,"key_reset",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+270,0,"hps_reset",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+271,0,"i",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->pushPrefix("dut", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+268,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+270,0,"hps_reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+269,0,"key_reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+272,0,"hex3_hex0",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+273,0,"result_rdata",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+37,0,"result_wdata",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+38,0,"result_addr",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 10,0);
    tracep->declBit(c+274,0,"result_clken",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+39,0,"result_write",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+274,0,"result_chipsel",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+275,0,"result_bytesel",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 3,0);
    tracep->declBus(c+40,0,"streamer_data",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBit(c+41,0,"streamer_fv",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+42,0,"streamer_lv",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+43,0,"frame_complete",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+44,0,"streamer_valid",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+43,0,"frame_complete_w",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+276,0,"ctrl_reg_h2f",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBus(c+45,0,"ctrl_reg_f2h",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBit(c+46,0,"reset",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+47,0,"start_sync",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 1,0);
    tracep->declBit(c+48,0,"start_synced",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+49,0,"full_frame_complete_accumulator",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+50,0,"subaps_done_accumulator",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBus(c+51,0,"intensity_accumulator",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+52,0,"xI_accumulator",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 19,0);
    tracep->declBus(c+53,0,"yI_accumulator",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 19,0);
    tracep->declBus(c+54,0,"xI_reciprocal",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 19,0);
    tracep->declBus(c+55,0,"yI_reciprocal",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 19,0);
    tracep->declBus(c+56,0,"rI_reciprocal",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+57,0,"subaps_done_reciprocal",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBus(c+58,0,"x_centroid",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+59,0,"y_centroid",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+60,0,"x_slopes",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+61,0,"y_slopes",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBit(c+62,0,"new_subapeture",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+63,0,"subap_valid",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declArray(c+64,0,"zernike_out",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 269,0);
    tracep->declBit(c+73,0,"done",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("zernike_out_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 10; ++i) {
        tracep->declBus(c+74+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 26,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+84,0,"x_centroid_out",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+85,0,"y_centroid_out",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+86,0,"x_slopes_out",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+87,0,"y_slopes_out",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+88,0,"resw_write_idx",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 10,0);
    tracep->declBus(c+277,0,"SLOPE_X_OFFSET",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 10,0);
    tracep->declBus(c+278,0,"SLOPE_Y_OFFSET",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 10,0);
    tracep->declBus(c+279,0,"CENTROID_X_OFFSET",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 10,0);
    tracep->declBus(c+280,0,"CENTROID_Y_OFFSET",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 10,0);
    tracep->declBus(c+281,0,"ZERNIKE_OFFSET",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 10,0);
    tracep->declBus(c+282,0,"RESULT_W_WAIT",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 4,0);
    tracep->declBus(c+283,0,"RESULT_W_SX",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 4,0);
    tracep->declBus(c+284,0,"RESULT_W_SY",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 4,0);
    tracep->declBus(c+285,0,"RESULT_W_CX",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 4,0);
    tracep->declBus(c+286,0,"RESULT_W_CY",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 4,0);
    tracep->declBus(c+287,0,"RESULT_W_ZK",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 4,0);
    tracep->declBus(c+288,0,"RESULT_W_DONE",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 4,0);
    tracep->declBus(c+89,0,"resw_next_state",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 4,0);
    tracep->declBit(c+90,0,"resw_start",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+91,0,"resw_state",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 4,0);
    tracep->declBit(c+92,0,"write_zernike_latch",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+93,0,"write_zernike_count",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 10,0);
    tracep->declBit(c+94,0,"reset_from_key",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+95,0,"reset_from_hps",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("accumulator", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+289,0,"NUM_SUBAPETURES_SQRT",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+289,0,"NUM_PIXELS_SUBAPETURE_SQRT",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBit(c+268,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+46,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+44,0,"valid",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+40,0,"data_in",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBit(c+49,0,"full_frame_complete",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+50,0,"subapetures_completed",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBus(c+51,0,"intensity",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+52,0,"x_intensity",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 19,0);
    tracep->declBus(c+53,0,"y_intensity",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 19,0);
    tracep->declBus(c+96,0,"subap_col",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 3,0);
    tracep->declBus(c+97,0,"subap_row",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 3,0);
    tracep->declBus(c+98,0,"count_pixel_h",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 3,0);
    tracep->declBus(c+99,0,"count_pixel_v",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 3,0);
    tracep->declBus(c+100,0,"subap_idx",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBit(c+101,0,"last_pixel_h",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+102,0,"last_pixel_v",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+103,0,"last_subap_col",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+104,0,"last_subap_row",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+105,0,"subap_done_delay",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+106,0,"subap_idx_delay",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBus(c+107,0,"j",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->popPrefix();
    tracep->pushPrefix("em", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+290,0,"NUM_MODES",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+291,0,"NUM_SUBS",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+292,0,"NUM_SLOPES",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBit(c+268,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+46,0,"rst",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+63,0,"sub_valid",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+60,0,"x_slope",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+61,0,"y_slope",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declArray(c+64,0,"zernike_out",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 269,0);
    tracep->declBit(c+73,0,"done",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+293,0,"SUB_BITS",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declBus(c+294,0,"ACC_WIDTH",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declBus(c+295,0,"STATE_IDLE",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declBus(c+296,0,"STATE_DONE",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declBus(c+297,0,"LAST_SUB",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBit(c+108,0,"state",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+109,0,"sub_counter",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->pushPrefix("mac_sum", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 10; ++i) {
        tracep->declQuad(c+110+i*2,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 54,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("acc", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 10; ++i) {
        tracep->declQuad(c+130+i*2,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 54,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("acc_next", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 10; ++i) {
        tracep->declQuad(c+150+i*2,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 54,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+170,0,"i",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->pushPrefix("mac_gen[0]", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+17,0,"ex",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 17,0);
    tracep->declBus(c+18,0,"ey",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 17,0);
    tracep->declQuad(c+171,0,"prod_x",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 44,0);
    tracep->declQuad(c+173,0,"prod_y",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 44,0);
    tracep->popPrefix();
    tracep->pushPrefix("mac_gen[1]", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+19,0,"ex",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 17,0);
    tracep->declBus(c+20,0,"ey",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 17,0);
    tracep->declQuad(c+175,0,"prod_x",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 44,0);
    tracep->declQuad(c+177,0,"prod_y",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 44,0);
    tracep->popPrefix();
    tracep->pushPrefix("mac_gen[2]", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+21,0,"ex",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 17,0);
    tracep->declBus(c+22,0,"ey",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 17,0);
    tracep->declQuad(c+179,0,"prod_x",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 44,0);
    tracep->declQuad(c+181,0,"prod_y",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 44,0);
    tracep->popPrefix();
    tracep->pushPrefix("mac_gen[3]", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+23,0,"ex",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 17,0);
    tracep->declBus(c+24,0,"ey",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 17,0);
    tracep->declQuad(c+183,0,"prod_x",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 44,0);
    tracep->declQuad(c+185,0,"prod_y",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 44,0);
    tracep->popPrefix();
    tracep->pushPrefix("mac_gen[4]", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+25,0,"ex",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 17,0);
    tracep->declBus(c+26,0,"ey",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 17,0);
    tracep->declQuad(c+187,0,"prod_x",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 44,0);
    tracep->declQuad(c+189,0,"prod_y",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 44,0);
    tracep->popPrefix();
    tracep->pushPrefix("mac_gen[5]", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+27,0,"ex",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 17,0);
    tracep->declBus(c+28,0,"ey",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 17,0);
    tracep->declQuad(c+191,0,"prod_x",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 44,0);
    tracep->declQuad(c+193,0,"prod_y",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 44,0);
    tracep->popPrefix();
    tracep->pushPrefix("mac_gen[6]", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+29,0,"ex",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 17,0);
    tracep->declBus(c+30,0,"ey",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 17,0);
    tracep->declQuad(c+195,0,"prod_x",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 44,0);
    tracep->declQuad(c+197,0,"prod_y",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 44,0);
    tracep->popPrefix();
    tracep->pushPrefix("mac_gen[7]", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+31,0,"ex",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 17,0);
    tracep->declBus(c+32,0,"ey",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 17,0);
    tracep->declQuad(c+199,0,"prod_x",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 44,0);
    tracep->declQuad(c+201,0,"prod_y",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 44,0);
    tracep->popPrefix();
    tracep->pushPrefix("mac_gen[8]", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+33,0,"ex",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 17,0);
    tracep->declBus(c+34,0,"ey",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 17,0);
    tracep->declQuad(c+203,0,"prod_x",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 44,0);
    tracep->declQuad(c+205,0,"prod_y",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 44,0);
    tracep->popPrefix();
    tracep->pushPrefix("mac_gen[9]", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+35,0,"ex",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 17,0);
    tracep->declBus(c+36,0,"ey",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 17,0);
    tracep->declQuad(c+207,0,"prod_x",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 44,0);
    tracep->declQuad(c+209,0,"prod_y",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 44,0);
    tracep->popPrefix();
    tracep->popPrefix();
    tracep->pushPrefix("hps_reset_sync", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+270,0,"reset_async",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+268,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+95,0,"reset_sync",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+211,0,"sync_ff",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 1,0);
    tracep->declBit(c+212,0,"sync_d",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+213,0,"reset_count",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 3,0);
    tracep->declBit(c+214,0,"sync_level",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+215,0,"rising_edge",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->popPrefix();
    tracep->pushPrefix("key_reset_sync", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+269,0,"reset_async",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+268,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+94,0,"reset_sync",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+216,0,"sync_ff",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 1,0);
    tracep->declBit(c+217,0,"sync_d",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+218,0,"reset_count",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 3,0);
    tracep->declBit(c+219,0,"sync_level",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+220,0,"rising_edge",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->popPrefix();
    tracep->pushPrefix("reciprocal", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+268,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+46,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+52,0,"xI_in",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 19,0);
    tracep->declBus(c+53,0,"yI_in",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 19,0);
    tracep->declBus(c+51,0,"sI",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+50,0,"centroids_done_in",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBus(c+54,0,"xI_out",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 19,0);
    tracep->declBus(c+55,0,"yI_out",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 19,0);
    tracep->declBus(c+56,0,"rI_out",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+57,0,"centroids_done_out",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBus(c+221,0,"reciprocal_wire",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBit(c+222,0,"divide_by_zero_wire",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+223,0,"saturated_wire",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("recip", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+51,0,"v_u16",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+221,0,"reciprocal_q27",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBit(c+222,0,"divide_by_zero",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+223,0,"saturated",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declArray(c+298,0,"RECIP_SEED_LUT_Q0_16",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 4095,0);
    tracep->declBit(c+222,0,"v_is_zero",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+224,0,"v_safe",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+225,0,"shift_left",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 4,0);
    tracep->declBus(c+226,0,"a_q1_15",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+227,0,"a_q1_26",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+228,0,"lut_idx",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBus(c+229,0,"seed_q0_16",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+230,0,"x0_u27",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+231,0,"x1_q0_27",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBit(c+232,0,"x1_saturated",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+233,0,"x2_q0_27",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBit(c+234,0,"x2_saturated",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+235,0,"msb_index",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 4,0);
    tracep->declBus(c+236,0,"denorm_round_bias",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+237,0,"denorm_numer",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 27,0);
    tracep->declBus(c+238,0,"out_q0_27_ext",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 27,0);
    tracep->declBit(c+239,0,"out_sat",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("u_newton_step_0", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+227,0,"a_q1_26",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+230,0,"x0_u27",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+231,0,"x1_q0_27",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBit(c+232,0,"saturated",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declQuad(c+240,0,"ax_mul",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 53,0);
    tracep->declBus(c+242,0,"ax_q1_26",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 27,0);
    tracep->declBus(c+242,0,"ax_q1_26_clamped",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 27,0);
    tracep->declBus(c+243,0,"two_minus_ax_q1_26",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 27,0);
    tracep->declQuad(c+244,0,"prod_mul",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 54,0);
    tracep->declBus(c+246,0,"x1_rounded_ext",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 28,0);
    tracep->popPrefix();
    tracep->pushPrefix("u_newton_step_1", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+227,0,"a_q1_26",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+231,0,"x0_u27",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+233,0,"x1_q0_27",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBit(c+234,0,"saturated",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declQuad(c+247,0,"ax_mul",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 53,0);
    tracep->declBus(c+249,0,"ax_q1_26",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 27,0);
    tracep->declBus(c+249,0,"ax_q1_26_clamped",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 27,0);
    tracep->declBus(c+250,0,"two_minus_ax_q1_26",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 27,0);
    tracep->declQuad(c+251,0,"prod_mul",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 54,0);
    tracep->declBus(c+253,0,"x1_rounded_ext",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 28,0);
    tracep->popPrefix();
    tracep->popPrefix();
    tracep->popPrefix();
    tracep->pushPrefix("slopes", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+268,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+46,0,"rst",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+57,0,"subapetures_completed",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBit(c+43,0,"frame_complete",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+56,0,"rec_intensity",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+54,0,"x_intensity",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 19,0);
    tracep->declBus(c+55,0,"y_intensity",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 19,0);
    tracep->declBus(c+58,0,"x_centroid",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+59,0,"y_centroid",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+60,0,"x_slope",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+61,0,"y_slope",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBit(c+62,0,"new_subapeture",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+63,0,"subap_valid",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("subap_bitmap_mem", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 1; ++i) {
        tracep->declArray(c+1+i*8,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 255,0);
    }
    tracep->popPrefix();
    tracep->declArray(c+9,0,"subap_bitmap",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 255,0);
    tracep->declBus(c+426,0,"SCALE",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->declBus(c+427,0,"x_ref",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+427,0,"y_ref",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+254,0,"x_centroid_mult",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+255,0,"y_centroid_mult",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+256,0,"raw_x_slope",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 27,0);
    tracep->declBus(c+257,0,"raw_y_slope",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 27,0);
    tracep->declBus(c+258,0,"current_subap",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->pushPrefix("x_mult", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+254,0,"out",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+54,0,"a",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 19,0);
    tracep->declBus(c+56,0,"b",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declQuad(c+259,0,"mult_out",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 46,0);
    tracep->popPrefix();
    tracep->pushPrefix("y_mult", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+255,0,"out",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declBus(c+55,0,"a",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 19,0);
    tracep->declBus(c+56,0,"b",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 26,0);
    tracep->declQuad(c+261,0,"mult_out",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 46,0);
    tracep->popPrefix();
    tracep->popPrefix();
    tracep->pushPrefix("streamer", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+428,0,"HSIZE",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+428,0,"VSIZE",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+429,0,"HBLANK",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+430,0,"VBLANK",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBit(c+268,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+46,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+40,0,"data",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBit(c+41,0,"fv",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+42,0,"lv",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+43,0,"frame_complete",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+44,0,"valid",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+263,0,"line_counter",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBus(c+264,0,"row_counter",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBus(c+265,0,"h_blank_counter",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 1,0);
    tracep->declBus(c+266,0,"v_blank_counter",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBus(c+431,0,"STATE_FRAME_INIT",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 1,0);
    tracep->declBus(c+432,0,"STATE_ACTIVE_FRAME",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 1,0);
    tracep->declBus(c+433,0,"STATE_HOROZONTAL_BLANKING",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 1,0);
    tracep->declBus(c+434,0,"STATE_VERTICAL_BLANKING",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 1,0);
    tracep->declBus(c+267,0,"state",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 1,0);
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
    bufp->fullSData(oldp+272,(0U),16);
    bufp->fullIData(oldp+273,(vlSelf->tb__DOT__dut__DOT__result_rdata),32);
    bufp->fullBit(oldp+274,(1U));
    bufp->fullCData(oldp+275,(0xfU),4);
    bufp->fullCData(oldp+276,(vlSelf->tb__DOT__dut__DOT__ctrl_reg_h2f),8);
    bufp->fullSData(oldp+277,(0U),11);
    bufp->fullSData(oldp+278,(0x100U),11);
    bufp->fullSData(oldp+279,(0x200U),11);
    bufp->fullSData(oldp+280,(0x300U),11);
    bufp->fullSData(oldp+281,(0x400U),11);
    bufp->fullCData(oldp+282,(0U),5);
    bufp->fullCData(oldp+283,(1U),5);
    bufp->fullCData(oldp+284,(2U),5);
    bufp->fullCData(oldp+285,(3U),5);
    bufp->fullCData(oldp+286,(4U),5);
    bufp->fullCData(oldp+287,(5U),5);
    bufp->fullCData(oldp+288,(6U),5);
    bufp->fullIData(oldp+289,(0x10U),32);
    bufp->fullIData(oldp+290,(0xaU),32);
    bufp->fullIData(oldp+291,(0x84U),32);
    bufp->fullIData(oldp+292,(0x188U),32);
    bufp->fullIData(oldp+293,(8U),32);
    bufp->fullIData(oldp+294,(0x37U),32);
    bufp->fullIData(oldp+295,(0U),32);
    bufp->fullIData(oldp+296,(1U),32);
    bufp->fullCData(oldp+297,(0x83U),8);
    bufp->fullWData(oldp+298,(Vtb__ConstPool__CONST_h29f91db3_0),4096);
    bufp->fullIData(oldp+426,(0x800000U),32);
    bufp->fullIData(oldp+427,(0x3c00000U),32);
    bufp->fullIData(oldp+428,(0x100U),32);
    bufp->fullIData(oldp+429,(4U),32);
    bufp->fullIData(oldp+430,(0x98U),32);
    bufp->fullCData(oldp+431,(0U),2);
    bufp->fullCData(oldp+432,(1U),2);
    bufp->fullCData(oldp+433,(2U),2);
    bufp->fullCData(oldp+434,(3U),2);
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
    bufp->fullWData(oldp+1,(vlSelf->tb__DOT__dut__DOT__slopes__DOT__subap_bitmap_mem[0]),256);
    bufp->fullWData(oldp+9,(vlSelf->tb__DOT__dut__DOT__slopes__DOT__subap_bitmap_mem
                            [0U]),256);
    bufp->fullIData(oldp+17,(vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x
                             [vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter]),18);
    bufp->fullIData(oldp+18,(vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y
                             [vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter]),18);
    bufp->fullIData(oldp+19,(((0x527U >= (0x7ffU & 
                                          ((IData)(0x84U) 
                                           + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                               ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x
                              [(0x7ffU & ((IData)(0x84U) 
                                          + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                               : 0U)),18);
    bufp->fullIData(oldp+20,(((0x527U >= (0x7ffU & 
                                          ((IData)(0x84U) 
                                           + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                               ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y
                              [(0x7ffU & ((IData)(0x84U) 
                                          + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                               : 0U)),18);
    bufp->fullIData(oldp+21,(((0x527U >= (0x7ffU & 
                                          ((IData)(0x108U) 
                                           + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                               ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x
                              [(0x7ffU & ((IData)(0x108U) 
                                          + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                               : 0U)),18);
    bufp->fullIData(oldp+22,(((0x527U >= (0x7ffU & 
                                          ((IData)(0x108U) 
                                           + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                               ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y
                              [(0x7ffU & ((IData)(0x108U) 
                                          + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                               : 0U)),18);
    bufp->fullIData(oldp+23,(((0x527U >= (0x7ffU & 
                                          ((IData)(0x18cU) 
                                           + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                               ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x
                              [(0x7ffU & ((IData)(0x18cU) 
                                          + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                               : 0U)),18);
    bufp->fullIData(oldp+24,(((0x527U >= (0x7ffU & 
                                          ((IData)(0x18cU) 
                                           + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                               ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y
                              [(0x7ffU & ((IData)(0x18cU) 
                                          + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                               : 0U)),18);
    bufp->fullIData(oldp+25,(((0x527U >= (0x7ffU & 
                                          ((IData)(0x210U) 
                                           + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                               ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x
                              [(0x7ffU & ((IData)(0x210U) 
                                          + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                               : 0U)),18);
    bufp->fullIData(oldp+26,(((0x527U >= (0x7ffU & 
                                          ((IData)(0x210U) 
                                           + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                               ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y
                              [(0x7ffU & ((IData)(0x210U) 
                                          + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                               : 0U)),18);
    bufp->fullIData(oldp+27,(((0x527U >= (0x7ffU & 
                                          ((IData)(0x294U) 
                                           + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                               ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x
                              [(0x7ffU & ((IData)(0x294U) 
                                          + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                               : 0U)),18);
    bufp->fullIData(oldp+28,(((0x527U >= (0x7ffU & 
                                          ((IData)(0x294U) 
                                           + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                               ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y
                              [(0x7ffU & ((IData)(0x294U) 
                                          + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                               : 0U)),18);
    bufp->fullIData(oldp+29,(((0x527U >= (0x7ffU & 
                                          ((IData)(0x318U) 
                                           + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                               ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x
                              [(0x7ffU & ((IData)(0x318U) 
                                          + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                               : 0U)),18);
    bufp->fullIData(oldp+30,(((0x527U >= (0x7ffU & 
                                          ((IData)(0x318U) 
                                           + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                               ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y
                              [(0x7ffU & ((IData)(0x318U) 
                                          + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                               : 0U)),18);
    bufp->fullIData(oldp+31,(((0x527U >= (0x7ffU & 
                                          ((IData)(0x39cU) 
                                           + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                               ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x
                              [(0x7ffU & ((IData)(0x39cU) 
                                          + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                               : 0U)),18);
    bufp->fullIData(oldp+32,(((0x527U >= (0x7ffU & 
                                          ((IData)(0x39cU) 
                                           + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                               ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y
                              [(0x7ffU & ((IData)(0x39cU) 
                                          + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                               : 0U)),18);
    bufp->fullIData(oldp+33,(((0x527U >= (0x7ffU & 
                                          ((IData)(0x420U) 
                                           + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                               ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x
                              [(0x7ffU & ((IData)(0x420U) 
                                          + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                               : 0U)),18);
    bufp->fullIData(oldp+34,(((0x527U >= (0x7ffU & 
                                          ((IData)(0x420U) 
                                           + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                               ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y
                              [(0x7ffU & ((IData)(0x420U) 
                                          + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                               : 0U)),18);
    bufp->fullIData(oldp+35,(((0x527U >= (0x7ffU & 
                                          ((IData)(0x4a4U) 
                                           + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                               ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x
                              [(0x7ffU & ((IData)(0x4a4U) 
                                          + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                               : 0U)),18);
    bufp->fullIData(oldp+36,(((0x527U >= (0x7ffU & 
                                          ((IData)(0x4a4U) 
                                           + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                               ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y
                              [(0x7ffU & ((IData)(0x4a4U) 
                                          + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                               : 0U)),18);
    bufp->fullIData(oldp+37,(vlSelf->tb__DOT__dut__DOT__result_wdata),32);
    bufp->fullSData(oldp+38,(vlSelf->tb__DOT__dut__DOT__result_addr),11);
    bufp->fullBit(oldp+39,(vlSelf->tb__DOT__dut__DOT__result_write));
    bufp->fullCData(oldp+40,(vlSelf->tb__DOT__dut__DOT__streamer_data),8);
    bufp->fullBit(oldp+41,(vlSelf->tb__DOT__dut__DOT__streamer_fv));
    bufp->fullBit(oldp+42,(vlSelf->tb__DOT__dut__DOT__streamer_lv));
    bufp->fullBit(oldp+43,(vlSelf->tb__DOT__dut__DOT__frame_complete));
    bufp->fullBit(oldp+44,(vlSelf->tb__DOT__dut__DOT__streamer_valid));
    bufp->fullCData(oldp+45,(vlSelf->tb__DOT__dut__DOT__ctrl_reg_f2h),8);
    bufp->fullBit(oldp+46,(vlSelf->tb__DOT__dut__DOT__reset));
    bufp->fullCData(oldp+47,(vlSelf->tb__DOT__dut__DOT__start_sync),2);
    bufp->fullBit(oldp+48,((1U & ((IData)(vlSelf->tb__DOT__dut__DOT__start_sync) 
                                  >> 1U))));
    bufp->fullBit(oldp+49,(vlSelf->tb__DOT__dut__DOT__full_frame_complete_accumulator));
    bufp->fullCData(oldp+50,(vlSelf->tb__DOT__dut__DOT__subaps_done_accumulator),8);
    bufp->fullSData(oldp+51,(vlSelf->tb__DOT__dut__DOT__intensity_accumulator),16);
    bufp->fullIData(oldp+52,(vlSelf->tb__DOT__dut__DOT__xI_accumulator),20);
    bufp->fullIData(oldp+53,(vlSelf->tb__DOT__dut__DOT__yI_accumulator),20);
    bufp->fullIData(oldp+54,(vlSelf->tb__DOT__dut__DOT__xI_reciprocal),20);
    bufp->fullIData(oldp+55,(vlSelf->tb__DOT__dut__DOT__yI_reciprocal),20);
    bufp->fullIData(oldp+56,(vlSelf->tb__DOT__dut__DOT__rI_reciprocal),27);
    bufp->fullCData(oldp+57,(vlSelf->tb__DOT__dut__DOT__subaps_done_reciprocal),8);
    bufp->fullIData(oldp+58,(vlSelf->tb__DOT__dut__DOT__x_centroid),27);
    bufp->fullIData(oldp+59,(vlSelf->tb__DOT__dut__DOT__y_centroid),27);
    bufp->fullIData(oldp+60,(vlSelf->tb__DOT__dut__DOT__x_slopes),27);
    bufp->fullIData(oldp+61,(vlSelf->tb__DOT__dut__DOT__y_slopes),27);
    bufp->fullBit(oldp+62,(vlSelf->tb__DOT__dut__DOT__new_subapeture));
    bufp->fullBit(oldp+63,(vlSelf->tb__DOT__dut__DOT__subap_valid));
    bufp->fullWData(oldp+64,(vlSelf->tb__DOT__dut__DOT__zernike_out),270);
    bufp->fullBit(oldp+73,(vlSelf->tb__DOT__dut__DOT__done));
    bufp->fullIData(oldp+74,(vlSelf->tb__DOT__dut__DOT__zernike_out_reg[0]),27);
    bufp->fullIData(oldp+75,(vlSelf->tb__DOT__dut__DOT__zernike_out_reg[1]),27);
    bufp->fullIData(oldp+76,(vlSelf->tb__DOT__dut__DOT__zernike_out_reg[2]),27);
    bufp->fullIData(oldp+77,(vlSelf->tb__DOT__dut__DOT__zernike_out_reg[3]),27);
    bufp->fullIData(oldp+78,(vlSelf->tb__DOT__dut__DOT__zernike_out_reg[4]),27);
    bufp->fullIData(oldp+79,(vlSelf->tb__DOT__dut__DOT__zernike_out_reg[5]),27);
    bufp->fullIData(oldp+80,(vlSelf->tb__DOT__dut__DOT__zernike_out_reg[6]),27);
    bufp->fullIData(oldp+81,(vlSelf->tb__DOT__dut__DOT__zernike_out_reg[7]),27);
    bufp->fullIData(oldp+82,(vlSelf->tb__DOT__dut__DOT__zernike_out_reg[8]),27);
    bufp->fullIData(oldp+83,(vlSelf->tb__DOT__dut__DOT__zernike_out_reg[9]),27);
    bufp->fullIData(oldp+84,(vlSelf->tb__DOT__dut__DOT__x_centroid_out),27);
    bufp->fullIData(oldp+85,(vlSelf->tb__DOT__dut__DOT__y_centroid_out),27);
    bufp->fullIData(oldp+86,(vlSelf->tb__DOT__dut__DOT__x_slopes_out),27);
    bufp->fullIData(oldp+87,(vlSelf->tb__DOT__dut__DOT__y_slopes_out),27);
    bufp->fullSData(oldp+88,(vlSelf->tb__DOT__dut__DOT__resw_write_idx),11);
    bufp->fullCData(oldp+89,(((0x10U & (IData)(vlSelf->tb__DOT__dut__DOT__resw_state))
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
    bufp->fullBit(oldp+90,(vlSelf->tb__DOT__dut__DOT__resw_start));
    bufp->fullCData(oldp+91,(vlSelf->tb__DOT__dut__DOT__resw_state),5);
    bufp->fullBit(oldp+92,(vlSelf->tb__DOT__dut__DOT__write_zernike_latch));
    bufp->fullSData(oldp+93,(vlSelf->tb__DOT__dut__DOT__write_zernike_count),11);
    bufp->fullBit(oldp+94,((0U != (IData)(vlSelf->tb__DOT__dut__DOT__key_reset_sync__DOT__reset_count))));
    bufp->fullBit(oldp+95,((0U != (IData)(vlSelf->tb__DOT__dut__DOT__hps_reset_sync__DOT__reset_count))));
    bufp->fullCData(oldp+96,(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_col),4);
    bufp->fullCData(oldp+97,(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_row),4);
    bufp->fullCData(oldp+98,(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__count_pixel_h),4);
    bufp->fullCData(oldp+99,(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__count_pixel_v),4);
    bufp->fullCData(oldp+100,(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_idx),8);
    bufp->fullBit(oldp+101,((0xfU == (IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__count_pixel_h))));
    bufp->fullBit(oldp+102,((0xfU == (IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__count_pixel_v))));
    bufp->fullBit(oldp+103,((0xfU == (IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_col))));
    bufp->fullBit(oldp+104,((0xfU == (IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_row))));
    bufp->fullBit(oldp+105,(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_done_delay));
    bufp->fullCData(oldp+106,(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_idx_delay),8);
    bufp->fullIData(oldp+107,(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__j),32);
    bufp->fullBit(oldp+108,(vlSelf->tb__DOT__dut__DOT__em__DOT__state));
    bufp->fullCData(oldp+109,(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter),8);
    bufp->fullQData(oldp+110,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_sum[0]),55);
    bufp->fullQData(oldp+112,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_sum[1]),55);
    bufp->fullQData(oldp+114,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_sum[2]),55);
    bufp->fullQData(oldp+116,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_sum[3]),55);
    bufp->fullQData(oldp+118,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_sum[4]),55);
    bufp->fullQData(oldp+120,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_sum[5]),55);
    bufp->fullQData(oldp+122,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_sum[6]),55);
    bufp->fullQData(oldp+124,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_sum[7]),55);
    bufp->fullQData(oldp+126,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_sum[8]),55);
    bufp->fullQData(oldp+128,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_sum[9]),55);
    bufp->fullQData(oldp+130,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc[0]),55);
    bufp->fullQData(oldp+132,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc[1]),55);
    bufp->fullQData(oldp+134,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc[2]),55);
    bufp->fullQData(oldp+136,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc[3]),55);
    bufp->fullQData(oldp+138,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc[4]),55);
    bufp->fullQData(oldp+140,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc[5]),55);
    bufp->fullQData(oldp+142,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc[6]),55);
    bufp->fullQData(oldp+144,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc[7]),55);
    bufp->fullQData(oldp+146,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc[8]),55);
    bufp->fullQData(oldp+148,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc[9]),55);
    bufp->fullQData(oldp+150,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next[0]),55);
    bufp->fullQData(oldp+152,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next[1]),55);
    bufp->fullQData(oldp+154,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next[2]),55);
    bufp->fullQData(oldp+156,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next[3]),55);
    bufp->fullQData(oldp+158,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next[4]),55);
    bufp->fullQData(oldp+160,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next[5]),55);
    bufp->fullQData(oldp+162,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next[6]),55);
    bufp->fullQData(oldp+164,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next[7]),55);
    bufp->fullQData(oldp+166,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next[8]),55);
    bufp->fullQData(oldp+168,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next[9]),55);
    bufp->fullIData(oldp+170,(vlSelf->tb__DOT__dut__DOT__em__DOT__i),32);
    bufp->fullQData(oldp+171,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__0__KET____DOT__prod_x),45);
    bufp->fullQData(oldp+173,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__0__KET____DOT__prod_y),45);
    bufp->fullQData(oldp+175,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__1__KET____DOT__prod_x),45);
    bufp->fullQData(oldp+177,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__1__KET____DOT__prod_y),45);
    bufp->fullQData(oldp+179,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__2__KET____DOT__prod_x),45);
    bufp->fullQData(oldp+181,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__2__KET____DOT__prod_y),45);
    bufp->fullQData(oldp+183,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__3__KET____DOT__prod_x),45);
    bufp->fullQData(oldp+185,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__3__KET____DOT__prod_y),45);
    bufp->fullQData(oldp+187,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__4__KET____DOT__prod_x),45);
    bufp->fullQData(oldp+189,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__4__KET____DOT__prod_y),45);
    bufp->fullQData(oldp+191,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__5__KET____DOT__prod_x),45);
    bufp->fullQData(oldp+193,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__5__KET____DOT__prod_y),45);
    bufp->fullQData(oldp+195,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__6__KET____DOT__prod_x),45);
    bufp->fullQData(oldp+197,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__6__KET____DOT__prod_y),45);
    bufp->fullQData(oldp+199,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__7__KET____DOT__prod_x),45);
    bufp->fullQData(oldp+201,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__7__KET____DOT__prod_y),45);
    bufp->fullQData(oldp+203,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__8__KET____DOT__prod_x),45);
    bufp->fullQData(oldp+205,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__8__KET____DOT__prod_y),45);
    bufp->fullQData(oldp+207,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__9__KET____DOT__prod_x),45);
    bufp->fullQData(oldp+209,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__9__KET____DOT__prod_y),45);
    bufp->fullCData(oldp+211,(vlSelf->tb__DOT__dut__DOT__hps_reset_sync__DOT__sync_ff),2);
    bufp->fullBit(oldp+212,(vlSelf->tb__DOT__dut__DOT__hps_reset_sync__DOT__sync_d));
    bufp->fullCData(oldp+213,(vlSelf->tb__DOT__dut__DOT__hps_reset_sync__DOT__reset_count),4);
    bufp->fullBit(oldp+214,((1U & ((IData)(vlSelf->tb__DOT__dut__DOT__hps_reset_sync__DOT__sync_ff) 
                                   >> 1U))));
    bufp->fullBit(oldp+215,(((~ (IData)(vlSelf->tb__DOT__dut__DOT__hps_reset_sync__DOT__sync_d)) 
                             & ((IData)(vlSelf->tb__DOT__dut__DOT__hps_reset_sync__DOT__sync_ff) 
                                >> 1U))));
    bufp->fullCData(oldp+216,(vlSelf->tb__DOT__dut__DOT__key_reset_sync__DOT__sync_ff),2);
    bufp->fullBit(oldp+217,(vlSelf->tb__DOT__dut__DOT__key_reset_sync__DOT__sync_d));
    bufp->fullCData(oldp+218,(vlSelf->tb__DOT__dut__DOT__key_reset_sync__DOT__reset_count),4);
    bufp->fullBit(oldp+219,((1U & ((IData)(vlSelf->tb__DOT__dut__DOT__key_reset_sync__DOT__sync_ff) 
                                   >> 1U))));
    bufp->fullBit(oldp+220,(((~ (IData)(vlSelf->tb__DOT__dut__DOT__key_reset_sync__DOT__sync_d)) 
                             & ((IData)(vlSelf->tb__DOT__dut__DOT__key_reset_sync__DOT__sync_ff) 
                                >> 1U))));
    bufp->fullIData(oldp+221,(((0U == (IData)(vlSelf->tb__DOT__dut__DOT__intensity_accumulator))
                                ? 0x7ffffffU : ((0x7ffffffU 
                                                 < vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__out_q0_27_ext)
                                                 ? 0x7ffffffU
                                                 : 
                                                (0x7ffffffU 
                                                 & vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__out_q0_27_ext)))),27);
    bufp->fullBit(oldp+222,((0U == (IData)(vlSelf->tb__DOT__dut__DOT__intensity_accumulator))));
    bufp->fullBit(oldp+223,(((0U == (IData)(vlSelf->tb__DOT__dut__DOT__intensity_accumulator)) 
                             | (0x7ffffffU < vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__out_q0_27_ext))));
    bufp->fullSData(oldp+224,(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__v_safe),16);
    bufp->fullCData(oldp+225,(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__shift_left),5);
    bufp->fullSData(oldp+226,(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15),16);
    bufp->fullIData(oldp+227,(((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                               << 0xbU)),27);
    bufp->fullCData(oldp+228,((0xffU & ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                        >> 7U))),8);
    bufp->fullSData(oldp+229,((0xffffU & (((0U == (0x1fU 
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
    bufp->fullIData(oldp+230,((0x7fff800U & ((((0U 
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
    bufp->fullIData(oldp+231,(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__x1_q0_27),27);
    bufp->fullBit(oldp+232,((0x7ffffffU < (0x1fffffffU 
                                           & (IData)(
                                                     (0x1fffffffULL 
                                                      & ((0x2000000ULL 
                                                          + vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__u_newton_step_0__DOT__prod_mul) 
                                                         >> 0x1aU)))))));
    bufp->fullIData(oldp+233,(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__x2_q0_27),27);
    bufp->fullBit(oldp+234,((0x7ffffffU < (0x1fffffffU 
                                           & (IData)(
                                                     (0x1fffffffULL 
                                                      & ((0x2000000ULL 
                                                          + vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__u_newton_step_1__DOT__prod_mul) 
                                                         >> 0x1aU)))))));
    bufp->fullCData(oldp+235,((0x1fU & ((IData)(0xfU) 
                                        - (IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__shift_left)))),5);
    bufp->fullIData(oldp+236,(((0U == (0x1fU & ((IData)(0xfU) 
                                                - (IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__shift_left))))
                                ? 0U : (0x7ffffffU 
                                        & ((IData)(1U) 
                                           << (0x1fU 
                                               & (((IData)(0xfU) 
                                                   - (IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__shift_left)) 
                                                  - (IData)(1U))))))),27);
    bufp->fullIData(oldp+237,((0xfffffffU & (vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__x2_q0_27 
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
    bufp->fullIData(oldp+238,(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__out_q0_27_ext),28);
    bufp->fullBit(oldp+239,((0x7ffffffU < vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__out_q0_27_ext)));
    bufp->fullQData(oldp+240,((0x3fffffffffffffULL 
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
    bufp->fullIData(oldp+242,((0x7ffffffU & (IData)(
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
    bufp->fullIData(oldp+243,((0xfffffffU & ((IData)(0x8000000U) 
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
    bufp->fullQData(oldp+244,(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__u_newton_step_0__DOT__prod_mul),55);
    bufp->fullIData(oldp+246,((0x1fffffffU & (IData)(
                                                     (0x1fffffffULL 
                                                      & ((0x2000000ULL 
                                                          + vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__u_newton_step_0__DOT__prod_mul) 
                                                         >> 0x1aU))))),29);
    bufp->fullQData(oldp+247,((0x3fffffffffffffULL 
                               & ((QData)((IData)(((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                   << 0xbU))) 
                                  * (QData)((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__x1_q0_27))))),54);
    bufp->fullIData(oldp+249,((0x7ffffffU & (IData)(
                                                    (0x7ffffffULL 
                                                     & (((QData)((IData)(
                                                                         ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                          << 0xbU))) 
                                                         * (QData)((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__x1_q0_27))) 
                                                        >> 0x1bU))))),28);
    bufp->fullIData(oldp+250,((0xfffffffU & ((IData)(0x8000000U) 
                                             - (0x7ffffffU 
                                                & (IData)(
                                                          (0x7ffffffULL 
                                                           & (((QData)((IData)(
                                                                               ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                                << 0xbU))) 
                                                               * (QData)((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__x1_q0_27))) 
                                                              >> 0x1bU))))))),28);
    bufp->fullQData(oldp+251,(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__u_newton_step_1__DOT__prod_mul),55);
    bufp->fullIData(oldp+253,((0x1fffffffU & (IData)(
                                                     (0x1fffffffULL 
                                                      & ((0x2000000ULL 
                                                          + vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__u_newton_step_1__DOT__prod_mul) 
                                                         >> 0x1aU))))),29);
    bufp->fullIData(oldp+254,((0x7ffffffU & (IData)(
                                                    (vlSelf->tb__DOT__dut__DOT__slopes__DOT__x_mult__DOT__mult_out 
                                                     >> 4U)))),27);
    bufp->fullIData(oldp+255,((0x7ffffffU & (IData)(
                                                    (vlSelf->tb__DOT__dut__DOT__slopes__DOT__y_mult__DOT__mult_out 
                                                     >> 4U)))),27);
    bufp->fullIData(oldp+256,((0xfffffffU & (VL_EXTENDS_II(28,27, 
                                                           (0x7ffffffU 
                                                            & (IData)(
                                                                      (vlSelf->tb__DOT__dut__DOT__slopes__DOT__x_mult__DOT__mult_out 
                                                                       >> 4U)))) 
                                             - (IData)(0x3c00000U)))),28);
    bufp->fullIData(oldp+257,((0xfffffffU & (VL_EXTENDS_II(28,27, 
                                                           (0x7ffffffU 
                                                            & (IData)(
                                                                      (vlSelf->tb__DOT__dut__DOT__slopes__DOT__y_mult__DOT__mult_out 
                                                                       >> 4U)))) 
                                             - (IData)(0x3c00000U)))),28);
    bufp->fullCData(oldp+258,(vlSelf->tb__DOT__dut__DOT__slopes__DOT__current_subap),8);
    bufp->fullQData(oldp+259,(vlSelf->tb__DOT__dut__DOT__slopes__DOT__x_mult__DOT__mult_out),47);
    bufp->fullQData(oldp+261,(vlSelf->tb__DOT__dut__DOT__slopes__DOT__y_mult__DOT__mult_out),47);
    bufp->fullCData(oldp+263,(vlSelf->tb__DOT__dut__DOT__streamer__DOT__line_counter),8);
    bufp->fullCData(oldp+264,(vlSelf->tb__DOT__dut__DOT__streamer__DOT__row_counter),8);
    bufp->fullCData(oldp+265,(vlSelf->tb__DOT__dut__DOT__streamer__DOT__h_blank_counter),2);
    bufp->fullCData(oldp+266,(vlSelf->tb__DOT__dut__DOT__streamer__DOT__v_blank_counter),8);
    bufp->fullCData(oldp+267,(vlSelf->tb__DOT__dut__DOT__streamer__DOT__state),2);
    bufp->fullBit(oldp+268,(vlSelf->tb__DOT__clk));
    bufp->fullBit(oldp+269,(vlSelf->tb__DOT__key_reset));
    bufp->fullBit(oldp+270,(vlSelf->tb__DOT__hps_reset));
    bufp->fullIData(oldp+271,(vlSelf->tb__DOT__i),32);
}

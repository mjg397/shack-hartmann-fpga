// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Tracing implementation internals
#include "verilated_vcd_c.h"
#include "Vtb__Syms.h"


void Vtb___024root__trace_chg_0_sub_0(Vtb___024root* vlSelf, VerilatedVcd::Buffer* bufp);

void Vtb___024root__trace_chg_0(void* voidSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb___024root__trace_chg_0\n"); );
    // Init
    Vtb___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vtb___024root*>(voidSelf);
    Vtb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    if (VL_UNLIKELY(!vlSymsp->__Vm_activity)) return;
    // Body
    Vtb___024root__trace_chg_0_sub_0((&vlSymsp->TOP), bufp);
}

extern const VlWide<128>/*4095:0*/ Vtb__ConstPool__CONST_h29f91db3_0;

void Vtb___024root__trace_chg_0_sub_0(Vtb___024root* vlSelf, VerilatedVcd::Buffer* bufp) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb___024root__trace_chg_0_sub_0\n"); );
    // Init
    uint32_t* const oldp VL_ATTR_UNUSED = bufp->oldp(vlSymsp->__Vm_baseCode + 1);
    // Body
    if (VL_UNLIKELY(vlSelf->__Vm_traceActivity[1U])) {
        bufp->chgWData(oldp+0,(vlSelf->tb__DOT__dut__DOT__slopes__DOT__subap_bitmap_mem[0]),256);
        bufp->chgWData(oldp+8,(vlSelf->tb__DOT__dut__DOT__slopes__DOT__subap_bitmap_mem
                               [0U]),256);
    }
    if (VL_UNLIKELY((vlSelf->__Vm_traceActivity[1U] 
                     | vlSelf->__Vm_traceActivity[2U]))) {
        bufp->chgIData(oldp+16,(vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x
                                [vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter]),18);
        bufp->chgIData(oldp+17,(vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y
                                [vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter]),18);
        bufp->chgIData(oldp+18,(((0x527U >= (0x7ffU 
                                             & ((IData)(0x84U) 
                                                + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                                  ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x
                                 [(0x7ffU & ((IData)(0x84U) 
                                             + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                                  : 0U)),18);
        bufp->chgIData(oldp+19,(((0x527U >= (0x7ffU 
                                             & ((IData)(0x84U) 
                                                + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                                  ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y
                                 [(0x7ffU & ((IData)(0x84U) 
                                             + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                                  : 0U)),18);
        bufp->chgIData(oldp+20,(((0x527U >= (0x7ffU 
                                             & ((IData)(0x108U) 
                                                + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                                  ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x
                                 [(0x7ffU & ((IData)(0x108U) 
                                             + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                                  : 0U)),18);
        bufp->chgIData(oldp+21,(((0x527U >= (0x7ffU 
                                             & ((IData)(0x108U) 
                                                + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                                  ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y
                                 [(0x7ffU & ((IData)(0x108U) 
                                             + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                                  : 0U)),18);
        bufp->chgIData(oldp+22,(((0x527U >= (0x7ffU 
                                             & ((IData)(0x18cU) 
                                                + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                                  ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x
                                 [(0x7ffU & ((IData)(0x18cU) 
                                             + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                                  : 0U)),18);
        bufp->chgIData(oldp+23,(((0x527U >= (0x7ffU 
                                             & ((IData)(0x18cU) 
                                                + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                                  ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y
                                 [(0x7ffU & ((IData)(0x18cU) 
                                             + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                                  : 0U)),18);
        bufp->chgIData(oldp+24,(((0x527U >= (0x7ffU 
                                             & ((IData)(0x210U) 
                                                + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                                  ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x
                                 [(0x7ffU & ((IData)(0x210U) 
                                             + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                                  : 0U)),18);
        bufp->chgIData(oldp+25,(((0x527U >= (0x7ffU 
                                             & ((IData)(0x210U) 
                                                + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                                  ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y
                                 [(0x7ffU & ((IData)(0x210U) 
                                             + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                                  : 0U)),18);
        bufp->chgIData(oldp+26,(((0x527U >= (0x7ffU 
                                             & ((IData)(0x294U) 
                                                + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                                  ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x
                                 [(0x7ffU & ((IData)(0x294U) 
                                             + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                                  : 0U)),18);
        bufp->chgIData(oldp+27,(((0x527U >= (0x7ffU 
                                             & ((IData)(0x294U) 
                                                + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                                  ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y
                                 [(0x7ffU & ((IData)(0x294U) 
                                             + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                                  : 0U)),18);
        bufp->chgIData(oldp+28,(((0x527U >= (0x7ffU 
                                             & ((IData)(0x318U) 
                                                + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                                  ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x
                                 [(0x7ffU & ((IData)(0x318U) 
                                             + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                                  : 0U)),18);
        bufp->chgIData(oldp+29,(((0x527U >= (0x7ffU 
                                             & ((IData)(0x318U) 
                                                + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                                  ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y
                                 [(0x7ffU & ((IData)(0x318U) 
                                             + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                                  : 0U)),18);
        bufp->chgIData(oldp+30,(((0x527U >= (0x7ffU 
                                             & ((IData)(0x39cU) 
                                                + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                                  ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x
                                 [(0x7ffU & ((IData)(0x39cU) 
                                             + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                                  : 0U)),18);
        bufp->chgIData(oldp+31,(((0x527U >= (0x7ffU 
                                             & ((IData)(0x39cU) 
                                                + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                                  ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y
                                 [(0x7ffU & ((IData)(0x39cU) 
                                             + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                                  : 0U)),18);
        bufp->chgIData(oldp+32,(((0x527U >= (0x7ffU 
                                             & ((IData)(0x420U) 
                                                + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                                  ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x
                                 [(0x7ffU & ((IData)(0x420U) 
                                             + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                                  : 0U)),18);
        bufp->chgIData(oldp+33,(((0x527U >= (0x7ffU 
                                             & ((IData)(0x420U) 
                                                + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                                  ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y
                                 [(0x7ffU & ((IData)(0x420U) 
                                             + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                                  : 0U)),18);
        bufp->chgIData(oldp+34,(((0x527U >= (0x7ffU 
                                             & ((IData)(0x4a4U) 
                                                + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                                  ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x
                                 [(0x7ffU & ((IData)(0x4a4U) 
                                             + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                                  : 0U)),18);
        bufp->chgIData(oldp+35,(((0x527U >= (0x7ffU 
                                             & ((IData)(0x4a4U) 
                                                + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                                  ? vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y
                                 [(0x7ffU & ((IData)(0x4a4U) 
                                             + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                                  : 0U)),18);
    }
    if (VL_UNLIKELY(vlSelf->__Vm_traceActivity[2U])) {
        bufp->chgIData(oldp+36,(vlSelf->tb__DOT__dut__DOT__result_wdata),32);
        bufp->chgSData(oldp+37,(vlSelf->tb__DOT__dut__DOT__result_addr),11);
        bufp->chgBit(oldp+38,(vlSelf->tb__DOT__dut__DOT__result_write));
        bufp->chgCData(oldp+39,(vlSelf->tb__DOT__dut__DOT__streamer_data),8);
        bufp->chgBit(oldp+40,(vlSelf->tb__DOT__dut__DOT__streamer_fv));
        bufp->chgBit(oldp+41,(vlSelf->tb__DOT__dut__DOT__streamer_lv));
        bufp->chgBit(oldp+42,(vlSelf->tb__DOT__dut__DOT__frame_complete));
        bufp->chgBit(oldp+43,(vlSelf->tb__DOT__dut__DOT__streamer_valid));
        bufp->chgCData(oldp+44,(vlSelf->tb__DOT__dut__DOT__ctrl_reg_f2h),8);
        bufp->chgBit(oldp+45,(vlSelf->tb__DOT__dut__DOT__reset));
        bufp->chgCData(oldp+46,(vlSelf->tb__DOT__dut__DOT__start_sync),2);
        bufp->chgBit(oldp+47,((1U & ((IData)(vlSelf->tb__DOT__dut__DOT__start_sync) 
                                     >> 1U))));
        bufp->chgBit(oldp+48,(vlSelf->tb__DOT__dut__DOT__full_frame_complete_accumulator));
        bufp->chgCData(oldp+49,(vlSelf->tb__DOT__dut__DOT__subaps_done_accumulator),8);
        bufp->chgSData(oldp+50,(vlSelf->tb__DOT__dut__DOT__intensity_accumulator),16);
        bufp->chgIData(oldp+51,(vlSelf->tb__DOT__dut__DOT__xI_accumulator),20);
        bufp->chgIData(oldp+52,(vlSelf->tb__DOT__dut__DOT__yI_accumulator),20);
        bufp->chgIData(oldp+53,(vlSelf->tb__DOT__dut__DOT__xI_reciprocal),20);
        bufp->chgIData(oldp+54,(vlSelf->tb__DOT__dut__DOT__yI_reciprocal),20);
        bufp->chgIData(oldp+55,(vlSelf->tb__DOT__dut__DOT__rI_reciprocal),27);
        bufp->chgCData(oldp+56,(vlSelf->tb__DOT__dut__DOT__subaps_done_reciprocal),8);
        bufp->chgIData(oldp+57,(vlSelf->tb__DOT__dut__DOT__x_centroid),27);
        bufp->chgIData(oldp+58,(vlSelf->tb__DOT__dut__DOT__y_centroid),27);
        bufp->chgIData(oldp+59,(vlSelf->tb__DOT__dut__DOT__x_slopes),27);
        bufp->chgIData(oldp+60,(vlSelf->tb__DOT__dut__DOT__y_slopes),27);
        bufp->chgBit(oldp+61,(vlSelf->tb__DOT__dut__DOT__new_subapeture));
        bufp->chgBit(oldp+62,(vlSelf->tb__DOT__dut__DOT__subap_valid));
        bufp->chgWData(oldp+63,(vlSelf->tb__DOT__dut__DOT__zernike_out),270);
        bufp->chgBit(oldp+72,(vlSelf->tb__DOT__dut__DOT__done));
        bufp->chgIData(oldp+73,(vlSelf->tb__DOT__dut__DOT__zernike_out_reg[0]),27);
        bufp->chgIData(oldp+74,(vlSelf->tb__DOT__dut__DOT__zernike_out_reg[1]),27);
        bufp->chgIData(oldp+75,(vlSelf->tb__DOT__dut__DOT__zernike_out_reg[2]),27);
        bufp->chgIData(oldp+76,(vlSelf->tb__DOT__dut__DOT__zernike_out_reg[3]),27);
        bufp->chgIData(oldp+77,(vlSelf->tb__DOT__dut__DOT__zernike_out_reg[4]),27);
        bufp->chgIData(oldp+78,(vlSelf->tb__DOT__dut__DOT__zernike_out_reg[5]),27);
        bufp->chgIData(oldp+79,(vlSelf->tb__DOT__dut__DOT__zernike_out_reg[6]),27);
        bufp->chgIData(oldp+80,(vlSelf->tb__DOT__dut__DOT__zernike_out_reg[7]),27);
        bufp->chgIData(oldp+81,(vlSelf->tb__DOT__dut__DOT__zernike_out_reg[8]),27);
        bufp->chgIData(oldp+82,(vlSelf->tb__DOT__dut__DOT__zernike_out_reg[9]),27);
        bufp->chgIData(oldp+83,(vlSelf->tb__DOT__dut__DOT__x_centroid_out),27);
        bufp->chgIData(oldp+84,(vlSelf->tb__DOT__dut__DOT__y_centroid_out),27);
        bufp->chgIData(oldp+85,(vlSelf->tb__DOT__dut__DOT__x_slopes_out),27);
        bufp->chgIData(oldp+86,(vlSelf->tb__DOT__dut__DOT__y_slopes_out),27);
        bufp->chgSData(oldp+87,(vlSelf->tb__DOT__dut__DOT__resw_write_idx),11);
        bufp->chgCData(oldp+88,(((0x10U & (IData)(vlSelf->tb__DOT__dut__DOT__resw_state))
                                  ? 0U : ((8U & (IData)(vlSelf->tb__DOT__dut__DOT__resw_state))
                                           ? 0U : (
                                                   (4U 
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
        bufp->chgBit(oldp+89,(vlSelf->tb__DOT__dut__DOT__resw_start));
        bufp->chgCData(oldp+90,(vlSelf->tb__DOT__dut__DOT__resw_state),5);
        bufp->chgBit(oldp+91,(vlSelf->tb__DOT__dut__DOT__write_zernike_latch));
        bufp->chgSData(oldp+92,(vlSelf->tb__DOT__dut__DOT__write_zernike_count),11);
        bufp->chgBit(oldp+93,((0U != (IData)(vlSelf->tb__DOT__dut__DOT__key_reset_sync__DOT__reset_count))));
        bufp->chgBit(oldp+94,((0U != (IData)(vlSelf->tb__DOT__dut__DOT__hps_reset_sync__DOT__reset_count))));
        bufp->chgCData(oldp+95,(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_col),4);
        bufp->chgCData(oldp+96,(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_row),4);
        bufp->chgCData(oldp+97,(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__count_pixel_h),4);
        bufp->chgCData(oldp+98,(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__count_pixel_v),4);
        bufp->chgCData(oldp+99,(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_idx),8);
        bufp->chgBit(oldp+100,((0xfU == (IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__count_pixel_h))));
        bufp->chgBit(oldp+101,((0xfU == (IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__count_pixel_v))));
        bufp->chgBit(oldp+102,((0xfU == (IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_col))));
        bufp->chgBit(oldp+103,((0xfU == (IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_row))));
        bufp->chgBit(oldp+104,(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_done_delay));
        bufp->chgCData(oldp+105,(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_idx_delay),8);
        bufp->chgIData(oldp+106,(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__j),32);
        bufp->chgBit(oldp+107,(vlSelf->tb__DOT__dut__DOT__em__DOT__state));
        bufp->chgCData(oldp+108,(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter),8);
        bufp->chgQData(oldp+109,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_sum[0]),55);
        bufp->chgQData(oldp+111,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_sum[1]),55);
        bufp->chgQData(oldp+113,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_sum[2]),55);
        bufp->chgQData(oldp+115,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_sum[3]),55);
        bufp->chgQData(oldp+117,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_sum[4]),55);
        bufp->chgQData(oldp+119,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_sum[5]),55);
        bufp->chgQData(oldp+121,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_sum[6]),55);
        bufp->chgQData(oldp+123,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_sum[7]),55);
        bufp->chgQData(oldp+125,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_sum[8]),55);
        bufp->chgQData(oldp+127,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_sum[9]),55);
        bufp->chgQData(oldp+129,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc[0]),55);
        bufp->chgQData(oldp+131,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc[1]),55);
        bufp->chgQData(oldp+133,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc[2]),55);
        bufp->chgQData(oldp+135,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc[3]),55);
        bufp->chgQData(oldp+137,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc[4]),55);
        bufp->chgQData(oldp+139,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc[5]),55);
        bufp->chgQData(oldp+141,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc[6]),55);
        bufp->chgQData(oldp+143,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc[7]),55);
        bufp->chgQData(oldp+145,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc[8]),55);
        bufp->chgQData(oldp+147,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc[9]),55);
        bufp->chgQData(oldp+149,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next[0]),55);
        bufp->chgQData(oldp+151,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next[1]),55);
        bufp->chgQData(oldp+153,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next[2]),55);
        bufp->chgQData(oldp+155,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next[3]),55);
        bufp->chgQData(oldp+157,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next[4]),55);
        bufp->chgQData(oldp+159,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next[5]),55);
        bufp->chgQData(oldp+161,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next[6]),55);
        bufp->chgQData(oldp+163,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next[7]),55);
        bufp->chgQData(oldp+165,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next[8]),55);
        bufp->chgQData(oldp+167,(vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next[9]),55);
        bufp->chgIData(oldp+169,(vlSelf->tb__DOT__dut__DOT__em__DOT__i),32);
        bufp->chgQData(oldp+170,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__0__KET____DOT__prod_x),45);
        bufp->chgQData(oldp+172,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__0__KET____DOT__prod_y),45);
        bufp->chgQData(oldp+174,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__1__KET____DOT__prod_x),45);
        bufp->chgQData(oldp+176,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__1__KET____DOT__prod_y),45);
        bufp->chgQData(oldp+178,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__2__KET____DOT__prod_x),45);
        bufp->chgQData(oldp+180,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__2__KET____DOT__prod_y),45);
        bufp->chgQData(oldp+182,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__3__KET____DOT__prod_x),45);
        bufp->chgQData(oldp+184,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__3__KET____DOT__prod_y),45);
        bufp->chgQData(oldp+186,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__4__KET____DOT__prod_x),45);
        bufp->chgQData(oldp+188,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__4__KET____DOT__prod_y),45);
        bufp->chgQData(oldp+190,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__5__KET____DOT__prod_x),45);
        bufp->chgQData(oldp+192,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__5__KET____DOT__prod_y),45);
        bufp->chgQData(oldp+194,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__6__KET____DOT__prod_x),45);
        bufp->chgQData(oldp+196,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__6__KET____DOT__prod_y),45);
        bufp->chgQData(oldp+198,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__7__KET____DOT__prod_x),45);
        bufp->chgQData(oldp+200,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__7__KET____DOT__prod_y),45);
        bufp->chgQData(oldp+202,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__8__KET____DOT__prod_x),45);
        bufp->chgQData(oldp+204,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__8__KET____DOT__prod_y),45);
        bufp->chgQData(oldp+206,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__9__KET____DOT__prod_x),45);
        bufp->chgQData(oldp+208,(vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__9__KET____DOT__prod_y),45);
        bufp->chgCData(oldp+210,(vlSelf->tb__DOT__dut__DOT__hps_reset_sync__DOT__sync_ff),2);
        bufp->chgBit(oldp+211,(vlSelf->tb__DOT__dut__DOT__hps_reset_sync__DOT__sync_d));
        bufp->chgCData(oldp+212,(vlSelf->tb__DOT__dut__DOT__hps_reset_sync__DOT__reset_count),4);
        bufp->chgBit(oldp+213,((1U & ((IData)(vlSelf->tb__DOT__dut__DOT__hps_reset_sync__DOT__sync_ff) 
                                      >> 1U))));
        bufp->chgBit(oldp+214,(((~ (IData)(vlSelf->tb__DOT__dut__DOT__hps_reset_sync__DOT__sync_d)) 
                                & ((IData)(vlSelf->tb__DOT__dut__DOT__hps_reset_sync__DOT__sync_ff) 
                                   >> 1U))));
        bufp->chgCData(oldp+215,(vlSelf->tb__DOT__dut__DOT__key_reset_sync__DOT__sync_ff),2);
        bufp->chgBit(oldp+216,(vlSelf->tb__DOT__dut__DOT__key_reset_sync__DOT__sync_d));
        bufp->chgCData(oldp+217,(vlSelf->tb__DOT__dut__DOT__key_reset_sync__DOT__reset_count),4);
        bufp->chgBit(oldp+218,((1U & ((IData)(vlSelf->tb__DOT__dut__DOT__key_reset_sync__DOT__sync_ff) 
                                      >> 1U))));
        bufp->chgBit(oldp+219,(((~ (IData)(vlSelf->tb__DOT__dut__DOT__key_reset_sync__DOT__sync_d)) 
                                & ((IData)(vlSelf->tb__DOT__dut__DOT__key_reset_sync__DOT__sync_ff) 
                                   >> 1U))));
        bufp->chgIData(oldp+220,(((0U == (IData)(vlSelf->tb__DOT__dut__DOT__intensity_accumulator))
                                   ? 0x7ffffffU : (
                                                   (0x7ffffffU 
                                                    < vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__out_q0_27_ext)
                                                    ? 0x7ffffffU
                                                    : 
                                                   (0x7ffffffU 
                                                    & vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__out_q0_27_ext)))),27);
        bufp->chgBit(oldp+221,((0U == (IData)(vlSelf->tb__DOT__dut__DOT__intensity_accumulator))));
        bufp->chgBit(oldp+222,(((0U == (IData)(vlSelf->tb__DOT__dut__DOT__intensity_accumulator)) 
                                | (0x7ffffffU < vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__out_q0_27_ext))));
        bufp->chgSData(oldp+223,(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__v_safe),16);
        bufp->chgCData(oldp+224,(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__shift_left),5);
        bufp->chgSData(oldp+225,(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15),16);
        bufp->chgIData(oldp+226,(((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                  << 0xbU)),27);
        bufp->chgCData(oldp+227,((0xffU & ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                           >> 7U))),8);
        bufp->chgSData(oldp+228,((0xffffU & (((0U == 
                                               (0x1fU 
                                                & VL_SHIFTL_III(12,12,32, 
                                                                (0xffU 
                                                                 & ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                    >> 7U)), 4U)))
                                               ? 0U
                                               : (Vtb__ConstPool__CONST_h29f91db3_0[
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
                                                                     >> 7U)), 4U)))))),16);
        bufp->chgIData(oldp+229,((0x7fff800U & ((((0U 
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
                                                << 0xbU))),27);
        bufp->chgIData(oldp+230,(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__x1_q0_27),27);
        bufp->chgBit(oldp+231,((0x7ffffffU < (0x1fffffffU 
                                              & (IData)(
                                                        (0x1fffffffULL 
                                                         & ((0x2000000ULL 
                                                             + vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__u_newton_step_0__DOT__prod_mul) 
                                                            >> 0x1aU)))))));
        bufp->chgIData(oldp+232,(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__x2_q0_27),27);
        bufp->chgBit(oldp+233,((0x7ffffffU < (0x1fffffffU 
                                              & (IData)(
                                                        (0x1fffffffULL 
                                                         & ((0x2000000ULL 
                                                             + vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__u_newton_step_1__DOT__prod_mul) 
                                                            >> 0x1aU)))))));
        bufp->chgCData(oldp+234,((0x1fU & ((IData)(0xfU) 
                                           - (IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__shift_left)))),5);
        bufp->chgIData(oldp+235,(((0U == (0x1fU & ((IData)(0xfU) 
                                                   - (IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__shift_left))))
                                   ? 0U : (0x7ffffffU 
                                           & ((IData)(1U) 
                                              << (0x1fU 
                                                  & (((IData)(0xfU) 
                                                      - (IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__shift_left)) 
                                                     - (IData)(1U))))))),27);
        bufp->chgIData(oldp+236,((0xfffffffU & (vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__x2_q0_27 
                                                + (
                                                   (0U 
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
        bufp->chgIData(oldp+237,(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__out_q0_27_ext),28);
        bufp->chgBit(oldp+238,((0x7ffffffU < vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__out_q0_27_ext)));
        bufp->chgQData(oldp+239,((0x3fffffffffffffULL 
                                  & ((QData)((IData)(
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
                                                           << 0xbU))))))),54);
        bufp->chgIData(oldp+241,((0x7ffffffU & (IData)(
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
        bufp->chgIData(oldp+242,((0xfffffffU & ((IData)(0x8000000U) 
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
        bufp->chgQData(oldp+243,(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__u_newton_step_0__DOT__prod_mul),55);
        bufp->chgIData(oldp+245,((0x1fffffffU & (IData)(
                                                        (0x1fffffffULL 
                                                         & ((0x2000000ULL 
                                                             + vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__u_newton_step_0__DOT__prod_mul) 
                                                            >> 0x1aU))))),29);
        bufp->chgQData(oldp+246,((0x3fffffffffffffULL 
                                  & ((QData)((IData)(
                                                     ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                      << 0xbU))) 
                                     * (QData)((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__x1_q0_27))))),54);
        bufp->chgIData(oldp+248,((0x7ffffffU & (IData)(
                                                       (0x7ffffffULL 
                                                        & (((QData)((IData)(
                                                                            ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                             << 0xbU))) 
                                                            * (QData)((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__x1_q0_27))) 
                                                           >> 0x1bU))))),28);
        bufp->chgIData(oldp+249,((0xfffffffU & ((IData)(0x8000000U) 
                                                - (0x7ffffffU 
                                                   & (IData)(
                                                             (0x7ffffffULL 
                                                              & (((QData)((IData)(
                                                                                ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                                << 0xbU))) 
                                                                  * (QData)((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__x1_q0_27))) 
                                                                 >> 0x1bU))))))),28);
        bufp->chgQData(oldp+250,(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__u_newton_step_1__DOT__prod_mul),55);
        bufp->chgIData(oldp+252,((0x1fffffffU & (IData)(
                                                        (0x1fffffffULL 
                                                         & ((0x2000000ULL 
                                                             + vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__u_newton_step_1__DOT__prod_mul) 
                                                            >> 0x1aU))))),29);
        bufp->chgIData(oldp+253,((0x7ffffffU & (IData)(
                                                       (vlSelf->tb__DOT__dut__DOT__slopes__DOT__x_mult__DOT__mult_out 
                                                        >> 4U)))),27);
        bufp->chgIData(oldp+254,((0x7ffffffU & (IData)(
                                                       (vlSelf->tb__DOT__dut__DOT__slopes__DOT__y_mult__DOT__mult_out 
                                                        >> 4U)))),27);
        bufp->chgIData(oldp+255,((0xfffffffU & (VL_EXTENDS_II(28,27, 
                                                              (0x7ffffffU 
                                                               & (IData)(
                                                                         (vlSelf->tb__DOT__dut__DOT__slopes__DOT__x_mult__DOT__mult_out 
                                                                          >> 4U)))) 
                                                - (IData)(0x3c00000U)))),28);
        bufp->chgIData(oldp+256,((0xfffffffU & (VL_EXTENDS_II(28,27, 
                                                              (0x7ffffffU 
                                                               & (IData)(
                                                                         (vlSelf->tb__DOT__dut__DOT__slopes__DOT__y_mult__DOT__mult_out 
                                                                          >> 4U)))) 
                                                - (IData)(0x3c00000U)))),28);
        bufp->chgCData(oldp+257,(vlSelf->tb__DOT__dut__DOT__slopes__DOT__current_subap),8);
        bufp->chgQData(oldp+258,(vlSelf->tb__DOT__dut__DOT__slopes__DOT__x_mult__DOT__mult_out),47);
        bufp->chgQData(oldp+260,(vlSelf->tb__DOT__dut__DOT__slopes__DOT__y_mult__DOT__mult_out),47);
        bufp->chgCData(oldp+262,(vlSelf->tb__DOT__dut__DOT__streamer__DOT__line_counter),8);
        bufp->chgCData(oldp+263,(vlSelf->tb__DOT__dut__DOT__streamer__DOT__row_counter),8);
        bufp->chgCData(oldp+264,(vlSelf->tb__DOT__dut__DOT__streamer__DOT__h_blank_counter),2);
        bufp->chgCData(oldp+265,(vlSelf->tb__DOT__dut__DOT__streamer__DOT__v_blank_counter),8);
        bufp->chgCData(oldp+266,(vlSelf->tb__DOT__dut__DOT__streamer__DOT__state),2);
    }
    bufp->chgBit(oldp+267,(vlSelf->tb__DOT__clk));
    bufp->chgBit(oldp+268,(vlSelf->tb__DOT__key_reset));
    bufp->chgBit(oldp+269,(vlSelf->tb__DOT__hps_reset));
    bufp->chgIData(oldp+270,(vlSelf->tb__DOT__i),32);
}

void Vtb___024root__trace_cleanup(void* voidSelf, VerilatedVcd* /*unused*/) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb___024root__trace_cleanup\n"); );
    // Init
    Vtb___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vtb___024root*>(voidSelf);
    Vtb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    // Body
    vlSymsp->__Vm_activity = false;
    vlSymsp->TOP.__Vm_traceActivity[0U] = 0U;
    vlSymsp->TOP.__Vm_traceActivity[1U] = 0U;
    vlSymsp->TOP.__Vm_traceActivity[2U] = 0U;
}

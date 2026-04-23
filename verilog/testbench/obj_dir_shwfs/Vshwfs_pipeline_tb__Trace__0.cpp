// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Tracing implementation internals
#include "verilated_vcd_c.h"
#include "Vshwfs_pipeline_tb__Syms.h"


void Vshwfs_pipeline_tb___024root__trace_chg_0_sub_0(Vshwfs_pipeline_tb___024root* vlSelf, VerilatedVcd::Buffer* bufp);

void Vshwfs_pipeline_tb___024root__trace_chg_0(void* voidSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vshwfs_pipeline_tb___024root__trace_chg_0\n"); );
    // Init
    Vshwfs_pipeline_tb___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vshwfs_pipeline_tb___024root*>(voidSelf);
    Vshwfs_pipeline_tb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    if (VL_UNLIKELY(!vlSymsp->__Vm_activity)) return;
    // Body
    Vshwfs_pipeline_tb___024root__trace_chg_0_sub_0((&vlSymsp->TOP), bufp);
}

extern const VlWide<128>/*4095:0*/ Vshwfs_pipeline_tb__ConstPool__CONST_h29f91db3_0;

void Vshwfs_pipeline_tb___024root__trace_chg_0_sub_0(Vshwfs_pipeline_tb___024root* vlSelf, VerilatedVcd::Buffer* bufp) {
    if (false && vlSelf) {}  // Prevent unused
    Vshwfs_pipeline_tb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vshwfs_pipeline_tb___024root__trace_chg_0_sub_0\n"); );
    // Init
    uint32_t* const oldp VL_ATTR_UNUSED = bufp->oldp(vlSymsp->__Vm_baseCode + 1);
    // Body
    if (VL_UNLIKELY(vlSelf->__Vm_traceActivity[1U])) {
        bufp->chgIData(oldp+0,(vlSelf->shwfs_pipeline_tb__DOT__xI_reciprocal),20);
        bufp->chgIData(oldp+1,(vlSelf->shwfs_pipeline_tb__DOT__yI_reciprocal),20);
        bufp->chgSData(oldp+2,(vlSelf->shwfs_pipeline_tb__DOT__rI_reciprocal),16);
        bufp->chgCData(oldp+3,(vlSelf->shwfs_pipeline_tb__DOT__subaps_done_reciprocal),8);
        bufp->chgCData(oldp+4,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer_data),8);
        bufp->chgBit(oldp+5,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer_fv));
        bufp->chgBit(oldp+6,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer_lv));
        bufp->chgBit(oldp+7,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__frame_complete));
        bufp->chgBit(oldp+8,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer_valid));
        bufp->chgBit(oldp+9,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__full_frame_complete_accumulator));
        bufp->chgCData(oldp+10,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__subaps_done_accumulator),8);
        bufp->chgSData(oldp+11,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__intensity_accumulator),16);
        bufp->chgIData(oldp+12,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__xI_accumulator),20);
        bufp->chgIData(oldp+13,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__yI_accumulator),20);
        bufp->chgCData(oldp+14,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_col),4);
        bufp->chgCData(oldp+15,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_row),4);
        bufp->chgCData(oldp+16,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__count_pixel_h),4);
        bufp->chgCData(oldp+17,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__count_pixel_v),4);
        bufp->chgCData(oldp+18,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_idx),8);
        bufp->chgBit(oldp+19,((0xfU == (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__count_pixel_h))));
        bufp->chgBit(oldp+20,((0xfU == (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__count_pixel_v))));
        bufp->chgBit(oldp+21,((0xfU == (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_col))));
        bufp->chgBit(oldp+22,((0xfU == (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_row))));
        bufp->chgBit(oldp+23,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_done_delay));
        bufp->chgCData(oldp+24,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_idx_delay),8);
        bufp->chgIData(oldp+25,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__j),32);
        bufp->chgSData(oldp+26,(((0U == (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__intensity_accumulator))
                                  ? 0xffffU : ((0xffffU 
                                                < vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__out_q0_16_ext)
                                                ? 0xffffU
                                                : (0xffffU 
                                                   & vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__out_q0_16_ext)))),16);
        bufp->chgBit(oldp+27,((0U == (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__intensity_accumulator))));
        bufp->chgBit(oldp+28,(((0U == (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__intensity_accumulator)) 
                               | (0xffffU < vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__out_q0_16_ext))));
        bufp->chgSData(oldp+29,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__v_safe),16);
        bufp->chgCData(oldp+30,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__shift_left),5);
        bufp->chgSData(oldp+31,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__a_q1_15),16);
        bufp->chgCData(oldp+32,((0xffU & ((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                          >> 7U))),8);
        bufp->chgSData(oldp+33,((0xffffU & (((0U == 
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
                                                                       >> 7U)), 4U)))))),16);
        bufp->chgSData(oldp+34,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__x1_q0_16),16);
        bufp->chgBit(oldp+35,((0xffffU < (0x3ffffU 
                                          & (IData)(
                                                    (0x3ffffULL 
                                                     & ((0x4000ULL 
                                                         + vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__u_newton_step_q16__DOT__prod_mul) 
                                                        >> 0xfU)))))));
        bufp->chgCData(oldp+36,((0x1fU & ((IData)(0xfU) 
                                          - (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__shift_left)))),5);
        bufp->chgSData(oldp+37,(((0U == (0x1fU & ((IData)(0xfU) 
                                                  - (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__shift_left))))
                                  ? 0U : (0xffffU & 
                                          ((IData)(1U) 
                                           << (0x1fU 
                                               & (((IData)(0xfU) 
                                                   - (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__shift_left)) 
                                                  - (IData)(1U))))))),16);
        bufp->chgIData(oldp+38,((0x1ffffU & ((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__x1_q0_16) 
                                             + ((0U 
                                                 == 
                                                 (0x1fU 
                                                  & ((IData)(0xfU) 
                                                     - (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__shift_left))))
                                                 ? 0U
                                                 : 
                                                (0xffffU 
                                                 & ((IData)(1U) 
                                                    << 
                                                    (0x1fU 
                                                     & (((IData)(0xfU) 
                                                         - (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__shift_left)) 
                                                        - (IData)(1U))))))))),17);
        bufp->chgIData(oldp+39,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__out_q0_16_ext),17);
        bufp->chgBit(oldp+40,((0xffffU < vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__out_q0_16_ext)));
        bufp->chgIData(oldp+41,(((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                 * (0xffffU & (((0U 
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
                                                                       >> 7U)), 4U))))))),32);
        bufp->chgIData(oldp+42,((((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                  * (0xffffU & (((0U 
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
                                 >> 0x10U)),17);
        bufp->chgIData(oldp+43,((0x1ffffU & ((IData)(0x10000U) 
                                             - (((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                 * 
                                                 (0xffffU 
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
        bufp->chgQData(oldp+44,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__u_newton_step_q16__DOT__prod_mul),33);
        bufp->chgIData(oldp+46,((0x3ffffU & (IData)(
                                                    (0x3ffffULL 
                                                     & ((0x4000ULL 
                                                         + vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__u_newton_step_q16__DOT__prod_mul) 
                                                        >> 0xfU))))),18);
        bufp->chgCData(oldp+47,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__line_counter),8);
        bufp->chgCData(oldp+48,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__row_counter),8);
        bufp->chgCData(oldp+49,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__h_blank_counter),2);
        bufp->chgCData(oldp+50,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__v_blank_counter),8);
        bufp->chgCData(oldp+51,(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__state),2);
    }
    bufp->chgBit(oldp+52,(vlSelf->shwfs_pipeline_tb__DOT__clk_100));
    bufp->chgBit(oldp+53,(vlSelf->shwfs_pipeline_tb__DOT__reset));
}

void Vshwfs_pipeline_tb___024root__trace_cleanup(void* voidSelf, VerilatedVcd* /*unused*/) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vshwfs_pipeline_tb___024root__trace_cleanup\n"); );
    // Init
    Vshwfs_pipeline_tb___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vshwfs_pipeline_tb___024root*>(voidSelf);
    Vshwfs_pipeline_tb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    // Body
    vlSymsp->__Vm_activity = false;
    vlSymsp->TOP.__Vm_traceActivity[0U] = 0U;
    vlSymsp->TOP.__Vm_traceActivity[1U] = 0U;
}

// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtb.h for the primary calling header

#include "Vtb__pch.h"
#include "Vtb___024root.h"

VL_ATTR_COLD void Vtb___024root___eval_static(Vtb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb___024root___eval_static\n"); );
}

extern const VlWide<19>/*607:0*/ Vtb__ConstPool__CONST_hf3894275_0;
extern const VlWide<18>/*575:0*/ Vtb__ConstPool__CONST_heb68ca02_0;
extern const VlWide<16>/*511:0*/ Vtb__ConstPool__CONST_h70d4d4c2_0;
extern const VlWide<16>/*511:0*/ Vtb__ConstPool__CONST_ha15ce332_0;

VL_ATTR_COLD void Vtb___024root___eval_initial__TOP(Vtb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb___024root___eval_initial__TOP\n"); );
    // Body
    VL_READMEM_N(true, 8, 65536, 0, VL_CVT_PACK_STR_NW(19, Vtb__ConstPool__CONST_hf3894275_0)
                 ,  &(vlSelf->tb__DOT__dut__DOT__streamer__DOT__mem)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 256, 1, 0, VL_CVT_PACK_STR_NW(18, Vtb__ConstPool__CONST_heb68ca02_0)
                 ,  &(vlSelf->tb__DOT__dut__DOT__slopes__DOT__subap_bitmap_mem)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 18, 1320, 0, VL_CVT_PACK_STR_NW(16, Vtb__ConstPool__CONST_h70d4d4c2_0)
                 ,  &(vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 18, 1320, 0, VL_CVT_PACK_STR_NW(16, Vtb__ConstPool__CONST_ha15ce332_0)
                 ,  &(vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y)
                 , 0, ~0ULL);
}

VL_ATTR_COLD void Vtb___024root___eval_final(Vtb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb___024root___eval_final\n"); );
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb___024root___dump_triggers__stl(Vtb___024root* vlSelf);
#endif  // VL_DEBUG
VL_ATTR_COLD bool Vtb___024root___eval_phase__stl(Vtb___024root* vlSelf);

VL_ATTR_COLD void Vtb___024root___eval_settle(Vtb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb___024root___eval_settle\n"); );
    // Init
    IData/*31:0*/ __VstlIterCount;
    CData/*0:0*/ __VstlContinue;
    // Body
    __VstlIterCount = 0U;
    vlSelf->__VstlFirstIteration = 1U;
    __VstlContinue = 1U;
    while (__VstlContinue) {
        if (VL_UNLIKELY((0x64U < __VstlIterCount))) {
#ifdef VL_DEBUG
            Vtb___024root___dump_triggers__stl(vlSelf);
#endif
            VL_FATAL_MT("tb.v", 3, "", "Settle region did not converge.");
        }
        __VstlIterCount = ((IData)(1U) + __VstlIterCount);
        __VstlContinue = 0U;
        if (Vtb___024root___eval_phase__stl(vlSelf)) {
            __VstlContinue = 1U;
        }
        vlSelf->__VstlFirstIteration = 0U;
    }
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb___024root___dump_triggers__stl(Vtb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb___024root___dump_triggers__stl\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VstlTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelf->__VstlTriggered.word(0U))) {
        VL_DBG_MSGF("         'stl' region trigger index 0 is active: Internal 'stl' trigger - first iteration\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD void Vtb___024root___stl_sequent__TOP__0(Vtb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb___024root___stl_sequent__TOP__0\n"); );
    // Init
    CData/*4:0*/ __Vfunc_tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__clz16__0__Vfuncout;
    __Vfunc_tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__clz16__0__Vfuncout = 0;
    SData/*15:0*/ __Vfunc_tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__clz16__0__x;
    __Vfunc_tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__clz16__0__x = 0;
    // Body
    vlSelf->tb__DOT__dut__DOT__ctrl_reg_f2h = ((0xfeU 
                                                & (IData)(vlSelf->tb__DOT__dut__DOT__ctrl_reg_f2h)) 
                                               | (IData)(vlSelf->tb__DOT__dut__DOT__done));
    vlSelf->tb__DOT__dut__DOT__accumulator__DOT__last_subap_col 
        = (0xfU == (IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_col));
    vlSelf->tb__DOT__dut__DOT__accumulator__DOT__last_subap_row 
        = (0xfU == (IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_row));
    vlSelf->tb__DOT__dut__DOT__accumulator__DOT__last_pixel_h 
        = (0xfU == (IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__count_pixel_h));
    vlSelf->tb__DOT__dut__DOT__accumulator__DOT__last_pixel_v 
        = (0xfU == (IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__count_pixel_v));
    vlSelf->tb__DOT__dut__DOT__reset = ((0U != (IData)(vlSelf->tb__DOT__dut__DOT__hps_reset_sync__DOT__reset_count)) 
                                        | (0U != (IData)(vlSelf->tb__DOT__dut__DOT__key_reset_sync__DOT__reset_count)));
    vlSelf->tb__DOT__dut__DOT__streamer_valid = ((IData)(vlSelf->tb__DOT__dut__DOT__streamer_fv) 
                                                 & (IData)(vlSelf->tb__DOT__dut__DOT__streamer_lv));
    vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_idx 
        = (0xffU & (VL_SHIFTL_III(8,8,32, (IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_row), 4U) 
                    + (IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_col)));
    vlSelf->tb__DOT__dut__DOT__slopes__DOT__x_mult__DOT__mult_out 
        = (0x7fffffffffffULL & VL_MULS_QQQ(47, (0x7fffffffffffULL 
                                                & VL_EXTENDS_QI(47,20, 
                                                                vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__xI_internal
                                                                [3U])), 
                                           (0x7fffffffffffULL 
                                            & VL_EXTENDS_QI(47,27, vlSelf->tb__DOT__dut__DOT__rI_reciprocal))));
    vlSelf->tb__DOT__dut__DOT__slopes__DOT__y_mult__DOT__mult_out 
        = (0x7fffffffffffULL & VL_MULS_QQQ(47, (0x7fffffffffffULL 
                                                & VL_EXTENDS_QI(47,20, 
                                                                vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__yI_internal
                                                                [3U])), 
                                           (0x7fffffffffffULL 
                                            & VL_EXTENDS_QI(47,27, vlSelf->tb__DOT__dut__DOT__rI_reciprocal))));
    vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__u_newton_step_0__DOT__prod_mul 
        = (0x7fffffffffffffULL & ((QData)((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg2_x0_u27)) 
                                  * (QData)((IData)(
                                                    (0xfffffffU 
                                                     & ((IData)(0x8000000U) 
                                                        - 
                                                        (0x7ffffffU 
                                                         & (IData)(
                                                                   (0x7ffffffULL 
                                                                    & (((QData)((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg2_a_q1_26)) 
                                                                        * (QData)((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg2_x0_u27))) 
                                                                       >> 0x1bU))))))))));
    vlSelf->tb__DOT__dut__DOT__resw_next_state = ((0x10U 
                                                   & (IData)(vlSelf->tb__DOT__dut__DOT__resw_state))
                                                   ? 0U
                                                   : 
                                                  ((8U 
                                                    & (IData)(vlSelf->tb__DOT__dut__DOT__resw_state))
                                                    ? 0U
                                                    : 
                                                   ((4U 
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
                                                        : 0U))))));
    vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__u_newton_step_1__DOT__prod_mul 
        = (0x7fffffffffffffULL & ((QData)((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg3_x1_q0_27)) 
                                  * (QData)((IData)(
                                                    (0xfffffffU 
                                                     & ((IData)(0x8000000U) 
                                                        - 
                                                        (0x7ffffffU 
                                                         & (IData)(
                                                                   (0x7ffffffULL 
                                                                    & (((QData)((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg3_a_q1_26)) 
                                                                        * (QData)((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg3_x1_q0_27))) 
                                                                       >> 0x1bU))))))))));
    vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__v_safe 
        = ((0U == (IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg1_v_u16))
            ? 1U : (IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg1_v_u16));
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__0__KET____DOT__prod_x 
        = (0x1fffffffffffULL & VL_MULS_QQQ(45, (0x1fffffffffffULL 
                                                & VL_EXTENDS_QI(45,18, 
                                                                vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x
                                                                [vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter])), 
                                           (0x1fffffffffffULL 
                                            & VL_EXTENDS_QI(45,27, vlSelf->tb__DOT__dut__DOT__x_slopes))));
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__0__KET____DOT__prod_y 
        = (0x1fffffffffffULL & VL_MULS_QQQ(45, (0x1fffffffffffULL 
                                                & VL_EXTENDS_QI(45,18, 
                                                                vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y
                                                                [vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter])), 
                                           (0x1fffffffffffULL 
                                            & VL_EXTENDS_QI(45,27, vlSelf->tb__DOT__dut__DOT__y_slopes))));
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__1__KET____DOT__prod_x 
        = (0x1fffffffffffULL & VL_MULS_QQQ(45, (0x1fffffffffffULL 
                                                & VL_EXTENDS_QI(45,18, 
                                                                ((0x527U 
                                                                  >= 
                                                                  (0x7ffU 
                                                                   & ((IData)(0x84U) 
                                                                      + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                                                                  ? 
                                                                 vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x
                                                                 [
                                                                 (0x7ffU 
                                                                  & ((IData)(0x84U) 
                                                                     + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                                                                  : 0U))), 
                                           (0x1fffffffffffULL 
                                            & VL_EXTENDS_QI(45,27, vlSelf->tb__DOT__dut__DOT__x_slopes))));
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__1__KET____DOT__prod_y 
        = (0x1fffffffffffULL & VL_MULS_QQQ(45, (0x1fffffffffffULL 
                                                & VL_EXTENDS_QI(45,18, 
                                                                ((0x527U 
                                                                  >= 
                                                                  (0x7ffU 
                                                                   & ((IData)(0x84U) 
                                                                      + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                                                                  ? 
                                                                 vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y
                                                                 [
                                                                 (0x7ffU 
                                                                  & ((IData)(0x84U) 
                                                                     + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                                                                  : 0U))), 
                                           (0x1fffffffffffULL 
                                            & VL_EXTENDS_QI(45,27, vlSelf->tb__DOT__dut__DOT__y_slopes))));
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__2__KET____DOT__prod_x 
        = (0x1fffffffffffULL & VL_MULS_QQQ(45, (0x1fffffffffffULL 
                                                & VL_EXTENDS_QI(45,18, 
                                                                ((0x527U 
                                                                  >= 
                                                                  (0x7ffU 
                                                                   & ((IData)(0x108U) 
                                                                      + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                                                                  ? 
                                                                 vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x
                                                                 [
                                                                 (0x7ffU 
                                                                  & ((IData)(0x108U) 
                                                                     + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                                                                  : 0U))), 
                                           (0x1fffffffffffULL 
                                            & VL_EXTENDS_QI(45,27, vlSelf->tb__DOT__dut__DOT__x_slopes))));
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__2__KET____DOT__prod_y 
        = (0x1fffffffffffULL & VL_MULS_QQQ(45, (0x1fffffffffffULL 
                                                & VL_EXTENDS_QI(45,18, 
                                                                ((0x527U 
                                                                  >= 
                                                                  (0x7ffU 
                                                                   & ((IData)(0x108U) 
                                                                      + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                                                                  ? 
                                                                 vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y
                                                                 [
                                                                 (0x7ffU 
                                                                  & ((IData)(0x108U) 
                                                                     + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                                                                  : 0U))), 
                                           (0x1fffffffffffULL 
                                            & VL_EXTENDS_QI(45,27, vlSelf->tb__DOT__dut__DOT__y_slopes))));
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__3__KET____DOT__prod_x 
        = (0x1fffffffffffULL & VL_MULS_QQQ(45, (0x1fffffffffffULL 
                                                & VL_EXTENDS_QI(45,18, 
                                                                ((0x527U 
                                                                  >= 
                                                                  (0x7ffU 
                                                                   & ((IData)(0x18cU) 
                                                                      + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                                                                  ? 
                                                                 vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x
                                                                 [
                                                                 (0x7ffU 
                                                                  & ((IData)(0x18cU) 
                                                                     + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                                                                  : 0U))), 
                                           (0x1fffffffffffULL 
                                            & VL_EXTENDS_QI(45,27, vlSelf->tb__DOT__dut__DOT__x_slopes))));
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__3__KET____DOT__prod_y 
        = (0x1fffffffffffULL & VL_MULS_QQQ(45, (0x1fffffffffffULL 
                                                & VL_EXTENDS_QI(45,18, 
                                                                ((0x527U 
                                                                  >= 
                                                                  (0x7ffU 
                                                                   & ((IData)(0x18cU) 
                                                                      + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                                                                  ? 
                                                                 vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y
                                                                 [
                                                                 (0x7ffU 
                                                                  & ((IData)(0x18cU) 
                                                                     + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                                                                  : 0U))), 
                                           (0x1fffffffffffULL 
                                            & VL_EXTENDS_QI(45,27, vlSelf->tb__DOT__dut__DOT__y_slopes))));
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__4__KET____DOT__prod_x 
        = (0x1fffffffffffULL & VL_MULS_QQQ(45, (0x1fffffffffffULL 
                                                & VL_EXTENDS_QI(45,18, 
                                                                ((0x527U 
                                                                  >= 
                                                                  (0x7ffU 
                                                                   & ((IData)(0x210U) 
                                                                      + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                                                                  ? 
                                                                 vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x
                                                                 [
                                                                 (0x7ffU 
                                                                  & ((IData)(0x210U) 
                                                                     + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                                                                  : 0U))), 
                                           (0x1fffffffffffULL 
                                            & VL_EXTENDS_QI(45,27, vlSelf->tb__DOT__dut__DOT__x_slopes))));
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__4__KET____DOT__prod_y 
        = (0x1fffffffffffULL & VL_MULS_QQQ(45, (0x1fffffffffffULL 
                                                & VL_EXTENDS_QI(45,18, 
                                                                ((0x527U 
                                                                  >= 
                                                                  (0x7ffU 
                                                                   & ((IData)(0x210U) 
                                                                      + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                                                                  ? 
                                                                 vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y
                                                                 [
                                                                 (0x7ffU 
                                                                  & ((IData)(0x210U) 
                                                                     + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                                                                  : 0U))), 
                                           (0x1fffffffffffULL 
                                            & VL_EXTENDS_QI(45,27, vlSelf->tb__DOT__dut__DOT__y_slopes))));
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__5__KET____DOT__prod_x 
        = (0x1fffffffffffULL & VL_MULS_QQQ(45, (0x1fffffffffffULL 
                                                & VL_EXTENDS_QI(45,18, 
                                                                ((0x527U 
                                                                  >= 
                                                                  (0x7ffU 
                                                                   & ((IData)(0x294U) 
                                                                      + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                                                                  ? 
                                                                 vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x
                                                                 [
                                                                 (0x7ffU 
                                                                  & ((IData)(0x294U) 
                                                                     + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                                                                  : 0U))), 
                                           (0x1fffffffffffULL 
                                            & VL_EXTENDS_QI(45,27, vlSelf->tb__DOT__dut__DOT__x_slopes))));
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__5__KET____DOT__prod_y 
        = (0x1fffffffffffULL & VL_MULS_QQQ(45, (0x1fffffffffffULL 
                                                & VL_EXTENDS_QI(45,18, 
                                                                ((0x527U 
                                                                  >= 
                                                                  (0x7ffU 
                                                                   & ((IData)(0x294U) 
                                                                      + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                                                                  ? 
                                                                 vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y
                                                                 [
                                                                 (0x7ffU 
                                                                  & ((IData)(0x294U) 
                                                                     + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                                                                  : 0U))), 
                                           (0x1fffffffffffULL 
                                            & VL_EXTENDS_QI(45,27, vlSelf->tb__DOT__dut__DOT__y_slopes))));
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__6__KET____DOT__prod_x 
        = (0x1fffffffffffULL & VL_MULS_QQQ(45, (0x1fffffffffffULL 
                                                & VL_EXTENDS_QI(45,18, 
                                                                ((0x527U 
                                                                  >= 
                                                                  (0x7ffU 
                                                                   & ((IData)(0x318U) 
                                                                      + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                                                                  ? 
                                                                 vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x
                                                                 [
                                                                 (0x7ffU 
                                                                  & ((IData)(0x318U) 
                                                                     + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                                                                  : 0U))), 
                                           (0x1fffffffffffULL 
                                            & VL_EXTENDS_QI(45,27, vlSelf->tb__DOT__dut__DOT__x_slopes))));
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__6__KET____DOT__prod_y 
        = (0x1fffffffffffULL & VL_MULS_QQQ(45, (0x1fffffffffffULL 
                                                & VL_EXTENDS_QI(45,18, 
                                                                ((0x527U 
                                                                  >= 
                                                                  (0x7ffU 
                                                                   & ((IData)(0x318U) 
                                                                      + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                                                                  ? 
                                                                 vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y
                                                                 [
                                                                 (0x7ffU 
                                                                  & ((IData)(0x318U) 
                                                                     + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                                                                  : 0U))), 
                                           (0x1fffffffffffULL 
                                            & VL_EXTENDS_QI(45,27, vlSelf->tb__DOT__dut__DOT__y_slopes))));
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__7__KET____DOT__prod_x 
        = (0x1fffffffffffULL & VL_MULS_QQQ(45, (0x1fffffffffffULL 
                                                & VL_EXTENDS_QI(45,18, 
                                                                ((0x527U 
                                                                  >= 
                                                                  (0x7ffU 
                                                                   & ((IData)(0x39cU) 
                                                                      + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                                                                  ? 
                                                                 vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x
                                                                 [
                                                                 (0x7ffU 
                                                                  & ((IData)(0x39cU) 
                                                                     + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                                                                  : 0U))), 
                                           (0x1fffffffffffULL 
                                            & VL_EXTENDS_QI(45,27, vlSelf->tb__DOT__dut__DOT__x_slopes))));
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__7__KET____DOT__prod_y 
        = (0x1fffffffffffULL & VL_MULS_QQQ(45, (0x1fffffffffffULL 
                                                & VL_EXTENDS_QI(45,18, 
                                                                ((0x527U 
                                                                  >= 
                                                                  (0x7ffU 
                                                                   & ((IData)(0x39cU) 
                                                                      + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                                                                  ? 
                                                                 vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y
                                                                 [
                                                                 (0x7ffU 
                                                                  & ((IData)(0x39cU) 
                                                                     + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                                                                  : 0U))), 
                                           (0x1fffffffffffULL 
                                            & VL_EXTENDS_QI(45,27, vlSelf->tb__DOT__dut__DOT__y_slopes))));
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__8__KET____DOT__prod_x 
        = (0x1fffffffffffULL & VL_MULS_QQQ(45, (0x1fffffffffffULL 
                                                & VL_EXTENDS_QI(45,18, 
                                                                ((0x527U 
                                                                  >= 
                                                                  (0x7ffU 
                                                                   & ((IData)(0x420U) 
                                                                      + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                                                                  ? 
                                                                 vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x
                                                                 [
                                                                 (0x7ffU 
                                                                  & ((IData)(0x420U) 
                                                                     + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                                                                  : 0U))), 
                                           (0x1fffffffffffULL 
                                            & VL_EXTENDS_QI(45,27, vlSelf->tb__DOT__dut__DOT__x_slopes))));
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__8__KET____DOT__prod_y 
        = (0x1fffffffffffULL & VL_MULS_QQQ(45, (0x1fffffffffffULL 
                                                & VL_EXTENDS_QI(45,18, 
                                                                ((0x527U 
                                                                  >= 
                                                                  (0x7ffU 
                                                                   & ((IData)(0x420U) 
                                                                      + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                                                                  ? 
                                                                 vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y
                                                                 [
                                                                 (0x7ffU 
                                                                  & ((IData)(0x420U) 
                                                                     + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                                                                  : 0U))), 
                                           (0x1fffffffffffULL 
                                            & VL_EXTENDS_QI(45,27, vlSelf->tb__DOT__dut__DOT__y_slopes))));
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__9__KET____DOT__prod_x 
        = (0x1fffffffffffULL & VL_MULS_QQQ(45, (0x1fffffffffffULL 
                                                & VL_EXTENDS_QI(45,18, 
                                                                ((0x527U 
                                                                  >= 
                                                                  (0x7ffU 
                                                                   & ((IData)(0x4a4U) 
                                                                      + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                                                                  ? 
                                                                 vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x
                                                                 [
                                                                 (0x7ffU 
                                                                  & ((IData)(0x4a4U) 
                                                                     + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                                                                  : 0U))), 
                                           (0x1fffffffffffULL 
                                            & VL_EXTENDS_QI(45,27, vlSelf->tb__DOT__dut__DOT__x_slopes))));
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__9__KET____DOT__prod_y 
        = (0x1fffffffffffULL & VL_MULS_QQQ(45, (0x1fffffffffffULL 
                                                & VL_EXTENDS_QI(45,18, 
                                                                ((0x527U 
                                                                  >= 
                                                                  (0x7ffU 
                                                                   & ((IData)(0x4a4U) 
                                                                      + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))))
                                                                  ? 
                                                                 vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y
                                                                 [
                                                                 (0x7ffU 
                                                                  & ((IData)(0x4a4U) 
                                                                     + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)))]
                                                                  : 0U))), 
                                           (0x1fffffffffffULL 
                                            & VL_EXTENDS_QI(45,27, vlSelf->tb__DOT__dut__DOT__y_slopes))));
    vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__x2_q0_27 
        = ((0x7ffffffU < (0x1fffffffU & (IData)((0x1fffffffULL 
                                                 & ((0x2000000ULL 
                                                     + vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__u_newton_step_1__DOT__prod_mul) 
                                                    >> 0x1aU)))))
            ? 0x7ffffffU : (0x7ffffffU & (IData)((0x1fffffffULL 
                                                  & ((0x2000000ULL 
                                                      + vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__u_newton_step_1__DOT__prod_mul) 
                                                     >> 0x1aU)))));
    __Vfunc_tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__clz16__0__x 
        = vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__v_safe;
    __Vfunc_tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__clz16__0__Vfuncout 
        = (((((((((0x8000U == (0x8000U & (IData)(__Vfunc_tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__clz16__0__x))) 
                  | (0x4000U == (0xc000U & (IData)(__Vfunc_tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))) 
                 | (0x2000U == (0xe000U & (IData)(__Vfunc_tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))) 
                | (0x1000U == (0xf000U & (IData)(__Vfunc_tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))) 
               | (0x800U == (0xf800U & (IData)(__Vfunc_tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))) 
              | (0x400U == (0xfc00U & (IData)(__Vfunc_tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))) 
             | (0x200U == (0xfe00U & (IData)(__Vfunc_tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))) 
            | (0x100U == (0xff00U & (IData)(__Vfunc_tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__clz16__0__x))))
            ? ((0x8000U == (0x8000U & (IData)(__Vfunc_tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))
                ? 0U : ((0x4000U == (0xc000U & (IData)(__Vfunc_tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))
                         ? 1U : ((0x2000U == (0xe000U 
                                              & (IData)(__Vfunc_tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))
                                  ? 2U : ((0x1000U 
                                           == (0xf000U 
                                               & (IData)(__Vfunc_tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))
                                           ? 3U : (
                                                   (0x800U 
                                                    == 
                                                    (0xf800U 
                                                     & (IData)(__Vfunc_tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))
                                                    ? 4U
                                                    : 
                                                   ((0x400U 
                                                     == 
                                                     (0xfc00U 
                                                      & (IData)(__Vfunc_tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))
                                                     ? 5U
                                                     : 
                                                    ((0x200U 
                                                      == 
                                                      (0xfe00U 
                                                       & (IData)(__Vfunc_tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))
                                                      ? 6U
                                                      : 7U)))))))
            : (((((((((0x80U == (0xff80U & (IData)(__Vfunc_tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__clz16__0__x))) 
                      | (0x40U == (0xffc0U & (IData)(__Vfunc_tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))) 
                     | (0x20U == (0xffe0U & (IData)(__Vfunc_tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))) 
                    | (0x10U == (0xfff0U & (IData)(__Vfunc_tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))) 
                   | (8U == (0xfff8U & (IData)(__Vfunc_tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))) 
                  | (4U == (0xfffcU & (IData)(__Vfunc_tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))) 
                 | (2U == (0xfffeU & (IData)(__Vfunc_tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))) 
                | (1U == (IData)(__Vfunc_tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))
                ? ((0x80U == (0xff80U & (IData)(__Vfunc_tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))
                    ? 8U : ((0x40U == (0xffc0U & (IData)(__Vfunc_tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))
                             ? 9U : ((0x20U == (0xffe0U 
                                                & (IData)(__Vfunc_tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))
                                      ? 0xaU : ((0x10U 
                                                 == 
                                                 (0xfff0U 
                                                  & (IData)(__Vfunc_tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))
                                                 ? 0xbU
                                                 : 
                                                ((8U 
                                                  == 
                                                  (0xfff8U 
                                                   & (IData)(__Vfunc_tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))
                                                  ? 0xcU
                                                  : 
                                                 ((4U 
                                                   == 
                                                   (0xfffcU 
                                                    & (IData)(__Vfunc_tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))
                                                   ? 0xdU
                                                   : 
                                                  ((2U 
                                                    == 
                                                    (0xfffeU 
                                                     & (IData)(__Vfunc_tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))
                                                    ? 0xeU
                                                    : 0xfU)))))))
                : 0x10U));
    vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__shift_left 
        = __Vfunc_tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__clz16__0__Vfuncout;
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_sum[0U] 
        = (0x7fffffffffffffULL & ((((QData)((IData)(
                                                    (0x3ffU 
                                                     & (- (IData)(
                                                                  (1U 
                                                                   & (IData)(
                                                                             (vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__0__KET____DOT__prod_x 
                                                                              >> 0x2cU)))))))) 
                                    << 0x2dU) | vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__0__KET____DOT__prod_x) 
                                  + (((QData)((IData)(
                                                      (0x3ffU 
                                                       & (- (IData)(
                                                                    (1U 
                                                                     & (IData)(
                                                                               (vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__0__KET____DOT__prod_y 
                                                                                >> 0x2cU)))))))) 
                                      << 0x2dU) | vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__0__KET____DOT__prod_y)));
    vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next[0U] 
        = (0x7fffffffffffffULL & (vlSelf->tb__DOT__dut__DOT__em__DOT__acc
                                  [0U] + ((((QData)((IData)(
                                                            (0x3ffU 
                                                             & (- (IData)(
                                                                          (1U 
                                                                           & (IData)(
                                                                                (vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__0__KET____DOT__prod_x 
                                                                                >> 0x2cU)))))))) 
                                            << 0x2dU) 
                                           | vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__0__KET____DOT__prod_x) 
                                          + (((QData)((IData)(
                                                              (0x3ffU 
                                                               & (- (IData)(
                                                                            (1U 
                                                                             & (IData)(
                                                                                (vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__0__KET____DOT__prod_y 
                                                                                >> 0x2cU)))))))) 
                                              << 0x2dU) 
                                             | vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__0__KET____DOT__prod_y))));
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_sum[1U] 
        = (0x7fffffffffffffULL & ((((QData)((IData)(
                                                    (0x3ffU 
                                                     & (- (IData)(
                                                                  (1U 
                                                                   & (IData)(
                                                                             (vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__1__KET____DOT__prod_x 
                                                                              >> 0x2cU)))))))) 
                                    << 0x2dU) | vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__1__KET____DOT__prod_x) 
                                  + (((QData)((IData)(
                                                      (0x3ffU 
                                                       & (- (IData)(
                                                                    (1U 
                                                                     & (IData)(
                                                                               (vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__1__KET____DOT__prod_y 
                                                                                >> 0x2cU)))))))) 
                                      << 0x2dU) | vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__1__KET____DOT__prod_y)));
    vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next[1U] 
        = (0x7fffffffffffffULL & (vlSelf->tb__DOT__dut__DOT__em__DOT__acc
                                  [1U] + ((((QData)((IData)(
                                                            (0x3ffU 
                                                             & (- (IData)(
                                                                          (1U 
                                                                           & (IData)(
                                                                                (vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__1__KET____DOT__prod_x 
                                                                                >> 0x2cU)))))))) 
                                            << 0x2dU) 
                                           | vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__1__KET____DOT__prod_x) 
                                          + (((QData)((IData)(
                                                              (0x3ffU 
                                                               & (- (IData)(
                                                                            (1U 
                                                                             & (IData)(
                                                                                (vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__1__KET____DOT__prod_y 
                                                                                >> 0x2cU)))))))) 
                                              << 0x2dU) 
                                             | vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__1__KET____DOT__prod_y))));
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_sum[2U] 
        = (0x7fffffffffffffULL & ((((QData)((IData)(
                                                    (0x3ffU 
                                                     & (- (IData)(
                                                                  (1U 
                                                                   & (IData)(
                                                                             (vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__2__KET____DOT__prod_x 
                                                                              >> 0x2cU)))))))) 
                                    << 0x2dU) | vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__2__KET____DOT__prod_x) 
                                  + (((QData)((IData)(
                                                      (0x3ffU 
                                                       & (- (IData)(
                                                                    (1U 
                                                                     & (IData)(
                                                                               (vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__2__KET____DOT__prod_y 
                                                                                >> 0x2cU)))))))) 
                                      << 0x2dU) | vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__2__KET____DOT__prod_y)));
    vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next[2U] 
        = (0x7fffffffffffffULL & (vlSelf->tb__DOT__dut__DOT__em__DOT__acc
                                  [2U] + ((((QData)((IData)(
                                                            (0x3ffU 
                                                             & (- (IData)(
                                                                          (1U 
                                                                           & (IData)(
                                                                                (vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__2__KET____DOT__prod_x 
                                                                                >> 0x2cU)))))))) 
                                            << 0x2dU) 
                                           | vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__2__KET____DOT__prod_x) 
                                          + (((QData)((IData)(
                                                              (0x3ffU 
                                                               & (- (IData)(
                                                                            (1U 
                                                                             & (IData)(
                                                                                (vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__2__KET____DOT__prod_y 
                                                                                >> 0x2cU)))))))) 
                                              << 0x2dU) 
                                             | vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__2__KET____DOT__prod_y))));
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_sum[3U] 
        = (0x7fffffffffffffULL & ((((QData)((IData)(
                                                    (0x3ffU 
                                                     & (- (IData)(
                                                                  (1U 
                                                                   & (IData)(
                                                                             (vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__3__KET____DOT__prod_x 
                                                                              >> 0x2cU)))))))) 
                                    << 0x2dU) | vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__3__KET____DOT__prod_x) 
                                  + (((QData)((IData)(
                                                      (0x3ffU 
                                                       & (- (IData)(
                                                                    (1U 
                                                                     & (IData)(
                                                                               (vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__3__KET____DOT__prod_y 
                                                                                >> 0x2cU)))))))) 
                                      << 0x2dU) | vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__3__KET____DOT__prod_y)));
    vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next[3U] 
        = (0x7fffffffffffffULL & (vlSelf->tb__DOT__dut__DOT__em__DOT__acc
                                  [3U] + ((((QData)((IData)(
                                                            (0x3ffU 
                                                             & (- (IData)(
                                                                          (1U 
                                                                           & (IData)(
                                                                                (vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__3__KET____DOT__prod_x 
                                                                                >> 0x2cU)))))))) 
                                            << 0x2dU) 
                                           | vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__3__KET____DOT__prod_x) 
                                          + (((QData)((IData)(
                                                              (0x3ffU 
                                                               & (- (IData)(
                                                                            (1U 
                                                                             & (IData)(
                                                                                (vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__3__KET____DOT__prod_y 
                                                                                >> 0x2cU)))))))) 
                                              << 0x2dU) 
                                             | vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__3__KET____DOT__prod_y))));
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_sum[4U] 
        = (0x7fffffffffffffULL & ((((QData)((IData)(
                                                    (0x3ffU 
                                                     & (- (IData)(
                                                                  (1U 
                                                                   & (IData)(
                                                                             (vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__4__KET____DOT__prod_x 
                                                                              >> 0x2cU)))))))) 
                                    << 0x2dU) | vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__4__KET____DOT__prod_x) 
                                  + (((QData)((IData)(
                                                      (0x3ffU 
                                                       & (- (IData)(
                                                                    (1U 
                                                                     & (IData)(
                                                                               (vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__4__KET____DOT__prod_y 
                                                                                >> 0x2cU)))))))) 
                                      << 0x2dU) | vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__4__KET____DOT__prod_y)));
    vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next[4U] 
        = (0x7fffffffffffffULL & (vlSelf->tb__DOT__dut__DOT__em__DOT__acc
                                  [4U] + ((((QData)((IData)(
                                                            (0x3ffU 
                                                             & (- (IData)(
                                                                          (1U 
                                                                           & (IData)(
                                                                                (vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__4__KET____DOT__prod_x 
                                                                                >> 0x2cU)))))))) 
                                            << 0x2dU) 
                                           | vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__4__KET____DOT__prod_x) 
                                          + (((QData)((IData)(
                                                              (0x3ffU 
                                                               & (- (IData)(
                                                                            (1U 
                                                                             & (IData)(
                                                                                (vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__4__KET____DOT__prod_y 
                                                                                >> 0x2cU)))))))) 
                                              << 0x2dU) 
                                             | vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__4__KET____DOT__prod_y))));
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_sum[5U] 
        = (0x7fffffffffffffULL & ((((QData)((IData)(
                                                    (0x3ffU 
                                                     & (- (IData)(
                                                                  (1U 
                                                                   & (IData)(
                                                                             (vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__5__KET____DOT__prod_x 
                                                                              >> 0x2cU)))))))) 
                                    << 0x2dU) | vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__5__KET____DOT__prod_x) 
                                  + (((QData)((IData)(
                                                      (0x3ffU 
                                                       & (- (IData)(
                                                                    (1U 
                                                                     & (IData)(
                                                                               (vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__5__KET____DOT__prod_y 
                                                                                >> 0x2cU)))))))) 
                                      << 0x2dU) | vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__5__KET____DOT__prod_y)));
    vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next[5U] 
        = (0x7fffffffffffffULL & (vlSelf->tb__DOT__dut__DOT__em__DOT__acc
                                  [5U] + ((((QData)((IData)(
                                                            (0x3ffU 
                                                             & (- (IData)(
                                                                          (1U 
                                                                           & (IData)(
                                                                                (vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__5__KET____DOT__prod_x 
                                                                                >> 0x2cU)))))))) 
                                            << 0x2dU) 
                                           | vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__5__KET____DOT__prod_x) 
                                          + (((QData)((IData)(
                                                              (0x3ffU 
                                                               & (- (IData)(
                                                                            (1U 
                                                                             & (IData)(
                                                                                (vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__5__KET____DOT__prod_y 
                                                                                >> 0x2cU)))))))) 
                                              << 0x2dU) 
                                             | vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__5__KET____DOT__prod_y))));
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_sum[6U] 
        = (0x7fffffffffffffULL & ((((QData)((IData)(
                                                    (0x3ffU 
                                                     & (- (IData)(
                                                                  (1U 
                                                                   & (IData)(
                                                                             (vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__6__KET____DOT__prod_x 
                                                                              >> 0x2cU)))))))) 
                                    << 0x2dU) | vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__6__KET____DOT__prod_x) 
                                  + (((QData)((IData)(
                                                      (0x3ffU 
                                                       & (- (IData)(
                                                                    (1U 
                                                                     & (IData)(
                                                                               (vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__6__KET____DOT__prod_y 
                                                                                >> 0x2cU)))))))) 
                                      << 0x2dU) | vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__6__KET____DOT__prod_y)));
    vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next[6U] 
        = (0x7fffffffffffffULL & (vlSelf->tb__DOT__dut__DOT__em__DOT__acc
                                  [6U] + ((((QData)((IData)(
                                                            (0x3ffU 
                                                             & (- (IData)(
                                                                          (1U 
                                                                           & (IData)(
                                                                                (vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__6__KET____DOT__prod_x 
                                                                                >> 0x2cU)))))))) 
                                            << 0x2dU) 
                                           | vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__6__KET____DOT__prod_x) 
                                          + (((QData)((IData)(
                                                              (0x3ffU 
                                                               & (- (IData)(
                                                                            (1U 
                                                                             & (IData)(
                                                                                (vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__6__KET____DOT__prod_y 
                                                                                >> 0x2cU)))))))) 
                                              << 0x2dU) 
                                             | vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__6__KET____DOT__prod_y))));
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_sum[7U] 
        = (0x7fffffffffffffULL & ((((QData)((IData)(
                                                    (0x3ffU 
                                                     & (- (IData)(
                                                                  (1U 
                                                                   & (IData)(
                                                                             (vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__7__KET____DOT__prod_x 
                                                                              >> 0x2cU)))))))) 
                                    << 0x2dU) | vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__7__KET____DOT__prod_x) 
                                  + (((QData)((IData)(
                                                      (0x3ffU 
                                                       & (- (IData)(
                                                                    (1U 
                                                                     & (IData)(
                                                                               (vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__7__KET____DOT__prod_y 
                                                                                >> 0x2cU)))))))) 
                                      << 0x2dU) | vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__7__KET____DOT__prod_y)));
    vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next[7U] 
        = (0x7fffffffffffffULL & (vlSelf->tb__DOT__dut__DOT__em__DOT__acc
                                  [7U] + ((((QData)((IData)(
                                                            (0x3ffU 
                                                             & (- (IData)(
                                                                          (1U 
                                                                           & (IData)(
                                                                                (vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__7__KET____DOT__prod_x 
                                                                                >> 0x2cU)))))))) 
                                            << 0x2dU) 
                                           | vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__7__KET____DOT__prod_x) 
                                          + (((QData)((IData)(
                                                              (0x3ffU 
                                                               & (- (IData)(
                                                                            (1U 
                                                                             & (IData)(
                                                                                (vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__7__KET____DOT__prod_y 
                                                                                >> 0x2cU)))))))) 
                                              << 0x2dU) 
                                             | vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__7__KET____DOT__prod_y))));
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_sum[8U] 
        = (0x7fffffffffffffULL & ((((QData)((IData)(
                                                    (0x3ffU 
                                                     & (- (IData)(
                                                                  (1U 
                                                                   & (IData)(
                                                                             (vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__8__KET____DOT__prod_x 
                                                                              >> 0x2cU)))))))) 
                                    << 0x2dU) | vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__8__KET____DOT__prod_x) 
                                  + (((QData)((IData)(
                                                      (0x3ffU 
                                                       & (- (IData)(
                                                                    (1U 
                                                                     & (IData)(
                                                                               (vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__8__KET____DOT__prod_y 
                                                                                >> 0x2cU)))))))) 
                                      << 0x2dU) | vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__8__KET____DOT__prod_y)));
    vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next[8U] 
        = (0x7fffffffffffffULL & (vlSelf->tb__DOT__dut__DOT__em__DOT__acc
                                  [8U] + ((((QData)((IData)(
                                                            (0x3ffU 
                                                             & (- (IData)(
                                                                          (1U 
                                                                           & (IData)(
                                                                                (vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__8__KET____DOT__prod_x 
                                                                                >> 0x2cU)))))))) 
                                            << 0x2dU) 
                                           | vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__8__KET____DOT__prod_x) 
                                          + (((QData)((IData)(
                                                              (0x3ffU 
                                                               & (- (IData)(
                                                                            (1U 
                                                                             & (IData)(
                                                                                (vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__8__KET____DOT__prod_y 
                                                                                >> 0x2cU)))))))) 
                                              << 0x2dU) 
                                             | vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__8__KET____DOT__prod_y))));
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_sum[9U] 
        = (0x7fffffffffffffULL & ((((QData)((IData)(
                                                    (0x3ffU 
                                                     & (- (IData)(
                                                                  (1U 
                                                                   & (IData)(
                                                                             (vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__9__KET____DOT__prod_x 
                                                                              >> 0x2cU)))))))) 
                                    << 0x2dU) | vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__9__KET____DOT__prod_x) 
                                  + (((QData)((IData)(
                                                      (0x3ffU 
                                                       & (- (IData)(
                                                                    (1U 
                                                                     & (IData)(
                                                                               (vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__9__KET____DOT__prod_y 
                                                                                >> 0x2cU)))))))) 
                                      << 0x2dU) | vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__9__KET____DOT__prod_y)));
    vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next[9U] 
        = (0x7fffffffffffffULL & (vlSelf->tb__DOT__dut__DOT__em__DOT__acc
                                  [9U] + ((((QData)((IData)(
                                                            (0x3ffU 
                                                             & (- (IData)(
                                                                          (1U 
                                                                           & (IData)(
                                                                                (vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__9__KET____DOT__prod_x 
                                                                                >> 0x2cU)))))))) 
                                            << 0x2dU) 
                                           | vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__9__KET____DOT__prod_x) 
                                          + (((QData)((IData)(
                                                              (0x3ffU 
                                                               & (- (IData)(
                                                                            (1U 
                                                                             & (IData)(
                                                                                (vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__9__KET____DOT__prod_y 
                                                                                >> 0x2cU)))))))) 
                                              << 0x2dU) 
                                             | vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__9__KET____DOT__prod_y))));
    vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__out_q0_27_ext 
        = (0xfffffffU & ((0U == (0x1fU & ((IData)(0xfU) 
                                          - (IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg3_shift_left))))
                          ? vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__x2_q0_27
                          : ((0xfffffffU & (vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__x2_q0_27 
                                            + ((0U 
                                                == 
                                                (0x1fU 
                                                 & ((IData)(0xfU) 
                                                    - (IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg3_shift_left))))
                                                ? 0U
                                                : (0x7ffffffU 
                                                   & ((IData)(1U) 
                                                      << 
                                                      (0x1fU 
                                                       & (((IData)(0xfU) 
                                                           - (IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg3_shift_left)) 
                                                          - (IData)(1U)))))))) 
                             >> (0x1fU & ((IData)(0xfU) 
                                          - (IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg3_shift_left))))));
    vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15 
        = (0xffffU & ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__v_safe) 
                      << (IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__shift_left)));
}

VL_ATTR_COLD void Vtb___024root___eval_stl(Vtb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb___024root___eval_stl\n"); );
    // Body
    if ((1ULL & vlSelf->__VstlTriggered.word(0U))) {
        Vtb___024root___stl_sequent__TOP__0(vlSelf);
        vlSelf->__Vm_traceActivity[2U] = 1U;
        vlSelf->__Vm_traceActivity[1U] = 1U;
        vlSelf->__Vm_traceActivity[0U] = 1U;
    }
}

VL_ATTR_COLD void Vtb___024root___eval_triggers__stl(Vtb___024root* vlSelf);

VL_ATTR_COLD bool Vtb___024root___eval_phase__stl(Vtb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb___024root___eval_phase__stl\n"); );
    // Init
    CData/*0:0*/ __VstlExecute;
    // Body
    Vtb___024root___eval_triggers__stl(vlSelf);
    __VstlExecute = vlSelf->__VstlTriggered.any();
    if (__VstlExecute) {
        Vtb___024root___eval_stl(vlSelf);
    }
    return (__VstlExecute);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb___024root___dump_triggers__act(Vtb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb___024root___dump_triggers__act\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VactTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelf->__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 0 is active: @(posedge tb.clk)\n");
    }
    if ((2ULL & vlSelf->__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 1 is active: @([true] __VdlySched.awaitingCurrentTime())\n");
    }
}
#endif  // VL_DEBUG

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb___024root___dump_triggers__nba(Vtb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb___024root___dump_triggers__nba\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VnbaTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelf->__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 0 is active: @(posedge tb.clk)\n");
    }
    if ((2ULL & vlSelf->__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 1 is active: @([true] __VdlySched.awaitingCurrentTime())\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD void Vtb___024root___ctor_var_reset(Vtb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb___024root___ctor_var_reset\n"); );
    // Body
    vlSelf->tb__DOT__clk = VL_RAND_RESET_I(1);
    vlSelf->tb__DOT__key_reset = VL_RAND_RESET_I(1);
    vlSelf->tb__DOT__hps_reset = VL_RAND_RESET_I(1);
    vlSelf->tb__DOT__i = VL_RAND_RESET_I(32);
    vlSelf->tb__DOT__dut__DOT__result_rdata = VL_RAND_RESET_I(32);
    vlSelf->tb__DOT__dut__DOT__result_wdata = VL_RAND_RESET_I(32);
    vlSelf->tb__DOT__dut__DOT__result_addr = VL_RAND_RESET_I(11);
    vlSelf->tb__DOT__dut__DOT__result_write = VL_RAND_RESET_I(1);
    vlSelf->tb__DOT__dut__DOT__streamer_data = VL_RAND_RESET_I(8);
    vlSelf->tb__DOT__dut__DOT__streamer_fv = VL_RAND_RESET_I(1);
    vlSelf->tb__DOT__dut__DOT__streamer_lv = VL_RAND_RESET_I(1);
    vlSelf->tb__DOT__dut__DOT__frame_complete = VL_RAND_RESET_I(1);
    vlSelf->tb__DOT__dut__DOT__streamer_valid = VL_RAND_RESET_I(1);
    vlSelf->tb__DOT__dut__DOT__ctrl_reg_h2f = VL_RAND_RESET_I(8);
    vlSelf->tb__DOT__dut__DOT__ctrl_reg_f2h = VL_RAND_RESET_I(8);
    vlSelf->tb__DOT__dut__DOT__reset = VL_RAND_RESET_I(1);
    vlSelf->tb__DOT__dut__DOT__start_sync = VL_RAND_RESET_I(2);
    vlSelf->tb__DOT__dut__DOT__full_frame_complete_accumulator = VL_RAND_RESET_I(1);
    vlSelf->tb__DOT__dut__DOT__subaps_done_accumulator = VL_RAND_RESET_I(8);
    vlSelf->tb__DOT__dut__DOT__intensity_accumulator = VL_RAND_RESET_I(16);
    vlSelf->tb__DOT__dut__DOT__xI_accumulator = VL_RAND_RESET_I(20);
    vlSelf->tb__DOT__dut__DOT__yI_accumulator = VL_RAND_RESET_I(20);
    vlSelf->tb__DOT__dut__DOT__rI_reciprocal = VL_RAND_RESET_I(27);
    vlSelf->tb__DOT__dut__DOT__x_centroid = VL_RAND_RESET_I(27);
    vlSelf->tb__DOT__dut__DOT__y_centroid = VL_RAND_RESET_I(27);
    vlSelf->tb__DOT__dut__DOT__x_slopes = VL_RAND_RESET_I(27);
    vlSelf->tb__DOT__dut__DOT__y_slopes = VL_RAND_RESET_I(27);
    vlSelf->tb__DOT__dut__DOT__new_subapeture = VL_RAND_RESET_I(1);
    vlSelf->tb__DOT__dut__DOT__subap_valid = VL_RAND_RESET_I(1);
    VL_RAND_RESET_W(270, vlSelf->tb__DOT__dut__DOT__zernike_out);
    vlSelf->tb__DOT__dut__DOT__done = VL_RAND_RESET_I(1);
    for (int __Vi0 = 0; __Vi0 < 10; ++__Vi0) {
        vlSelf->tb__DOT__dut__DOT__zernike_out_reg[__Vi0] = VL_RAND_RESET_I(27);
    }
    vlSelf->tb__DOT__dut__DOT__x_centroid_out = VL_RAND_RESET_I(27);
    vlSelf->tb__DOT__dut__DOT__y_centroid_out = VL_RAND_RESET_I(27);
    vlSelf->tb__DOT__dut__DOT__x_slopes_out = VL_RAND_RESET_I(27);
    vlSelf->tb__DOT__dut__DOT__y_slopes_out = VL_RAND_RESET_I(27);
    vlSelf->tb__DOT__dut__DOT__resw_write_idx = VL_RAND_RESET_I(11);
    vlSelf->tb__DOT__dut__DOT__resw_next_state = VL_RAND_RESET_I(5);
    vlSelf->tb__DOT__dut__DOT__resw_start = VL_RAND_RESET_I(1);
    vlSelf->tb__DOT__dut__DOT__resw_state = VL_RAND_RESET_I(5);
    vlSelf->tb__DOT__dut__DOT__write_zernike_latch = VL_RAND_RESET_I(1);
    vlSelf->tb__DOT__dut__DOT__write_zernike_count = VL_RAND_RESET_I(11);
    vlSelf->tb__DOT__dut__DOT__streamer__DOT__line_counter = VL_RAND_RESET_I(8);
    vlSelf->tb__DOT__dut__DOT__streamer__DOT__row_counter = VL_RAND_RESET_I(8);
    vlSelf->tb__DOT__dut__DOT__streamer__DOT__h_blank_counter = VL_RAND_RESET_I(2);
    vlSelf->tb__DOT__dut__DOT__streamer__DOT__v_blank_counter = VL_RAND_RESET_I(8);
    for (int __Vi0 = 0; __Vi0 < 65536; ++__Vi0) {
        vlSelf->tb__DOT__dut__DOT__streamer__DOT__mem[__Vi0] = VL_RAND_RESET_I(8);
    }
    vlSelf->tb__DOT__dut__DOT__streamer__DOT__state = VL_RAND_RESET_I(2);
    vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_col = VL_RAND_RESET_I(4);
    vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_row = VL_RAND_RESET_I(4);
    vlSelf->tb__DOT__dut__DOT__accumulator__DOT__count_pixel_h = VL_RAND_RESET_I(4);
    vlSelf->tb__DOT__dut__DOT__accumulator__DOT__count_pixel_v = VL_RAND_RESET_I(4);
    vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_idx = VL_RAND_RESET_I(8);
    vlSelf->tb__DOT__dut__DOT__accumulator__DOT__last_pixel_h = VL_RAND_RESET_I(1);
    vlSelf->tb__DOT__dut__DOT__accumulator__DOT__last_pixel_v = VL_RAND_RESET_I(1);
    vlSelf->tb__DOT__dut__DOT__accumulator__DOT__last_subap_col = VL_RAND_RESET_I(1);
    vlSelf->tb__DOT__dut__DOT__accumulator__DOT__last_subap_row = VL_RAND_RESET_I(1);
    for (int __Vi0 = 0; __Vi0 < 256; ++__Vi0) {
        vlSelf->tb__DOT__dut__DOT__accumulator__DOT__i[__Vi0] = VL_RAND_RESET_I(16);
    }
    for (int __Vi0 = 0; __Vi0 < 256; ++__Vi0) {
        vlSelf->tb__DOT__dut__DOT__accumulator__DOT__x_i[__Vi0] = VL_RAND_RESET_I(20);
    }
    for (int __Vi0 = 0; __Vi0 < 256; ++__Vi0) {
        vlSelf->tb__DOT__dut__DOT__accumulator__DOT__y_i[__Vi0] = VL_RAND_RESET_I(20);
    }
    vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_done_delay = VL_RAND_RESET_I(1);
    vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_idx_delay = VL_RAND_RESET_I(8);
    vlSelf->tb__DOT__dut__DOT__accumulator__DOT__j = VL_RAND_RESET_I(32);
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__xI_internal[__Vi0] = VL_RAND_RESET_I(20);
    }
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__yI_internal[__Vi0] = VL_RAND_RESET_I(20);
    }
    for (int __Vi0 = 0; __Vi0 < 4; ++__Vi0) {
        vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__centroids_done_internal[__Vi0] = VL_RAND_RESET_I(8);
    }
    vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__v_safe = VL_RAND_RESET_I(16);
    vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__shift_left = VL_RAND_RESET_I(5);
    vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15 = VL_RAND_RESET_I(16);
    vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__x2_q0_27 = VL_RAND_RESET_I(27);
    vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__out_q0_27_ext = VL_RAND_RESET_I(28);
    vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg1_v_u16 = VL_RAND_RESET_I(16);
    vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg2_x0_u27 = VL_RAND_RESET_I(27);
    vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg2_a_q1_26 = VL_RAND_RESET_I(27);
    vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg2_v_is_zero = VL_RAND_RESET_I(1);
    vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg2_shift_left = VL_RAND_RESET_I(5);
    vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg3_x1_q0_27 = VL_RAND_RESET_I(27);
    vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg3_a_q1_26 = VL_RAND_RESET_I(27);
    vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg3_v_is_zero = VL_RAND_RESET_I(1);
    vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg3_shift_left = VL_RAND_RESET_I(5);
    vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__u_newton_step_0__DOT__prod_mul = VL_RAND_RESET_Q(55);
    vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__u_newton_step_1__DOT__prod_mul = VL_RAND_RESET_Q(55);
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        VL_RAND_RESET_W(256, vlSelf->tb__DOT__dut__DOT__slopes__DOT__subap_bitmap_mem[__Vi0]);
    }
    vlSelf->tb__DOT__dut__DOT__slopes__DOT__current_subap = VL_RAND_RESET_I(8);
    vlSelf->tb__DOT__dut__DOT__slopes__DOT__x_mult__DOT__mult_out = VL_RAND_RESET_Q(47);
    vlSelf->tb__DOT__dut__DOT__slopes__DOT__y_mult__DOT__mult_out = VL_RAND_RESET_Q(47);
    for (int __Vi0 = 0; __Vi0 < 1320; ++__Vi0) {
        vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_x[__Vi0] = VL_RAND_RESET_I(18);
    }
    for (int __Vi0 = 0; __Vi0 < 1320; ++__Vi0) {
        vlSelf->tb__DOT__dut__DOT__em__DOT__e_rom_y[__Vi0] = VL_RAND_RESET_I(18);
    }
    vlSelf->tb__DOT__dut__DOT__em__DOT__state = VL_RAND_RESET_I(1);
    vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter = VL_RAND_RESET_I(8);
    for (int __Vi0 = 0; __Vi0 < 10; ++__Vi0) {
        vlSelf->tb__DOT__dut__DOT__em__DOT__mac_sum[__Vi0] = VL_RAND_RESET_Q(55);
    }
    for (int __Vi0 = 0; __Vi0 < 10; ++__Vi0) {
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc[__Vi0] = VL_RAND_RESET_Q(55);
    }
    for (int __Vi0 = 0; __Vi0 < 10; ++__Vi0) {
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next[__Vi0] = VL_RAND_RESET_Q(55);
    }
    vlSelf->tb__DOT__dut__DOT__em__DOT__i = VL_RAND_RESET_I(32);
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__0__KET____DOT__prod_x = VL_RAND_RESET_Q(45);
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__0__KET____DOT__prod_y = VL_RAND_RESET_Q(45);
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__1__KET____DOT__prod_x = VL_RAND_RESET_Q(45);
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__1__KET____DOT__prod_y = VL_RAND_RESET_Q(45);
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__2__KET____DOT__prod_x = VL_RAND_RESET_Q(45);
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__2__KET____DOT__prod_y = VL_RAND_RESET_Q(45);
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__3__KET____DOT__prod_x = VL_RAND_RESET_Q(45);
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__3__KET____DOT__prod_y = VL_RAND_RESET_Q(45);
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__4__KET____DOT__prod_x = VL_RAND_RESET_Q(45);
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__4__KET____DOT__prod_y = VL_RAND_RESET_Q(45);
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__5__KET____DOT__prod_x = VL_RAND_RESET_Q(45);
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__5__KET____DOT__prod_y = VL_RAND_RESET_Q(45);
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__6__KET____DOT__prod_x = VL_RAND_RESET_Q(45);
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__6__KET____DOT__prod_y = VL_RAND_RESET_Q(45);
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__7__KET____DOT__prod_x = VL_RAND_RESET_Q(45);
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__7__KET____DOT__prod_y = VL_RAND_RESET_Q(45);
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__8__KET____DOT__prod_x = VL_RAND_RESET_Q(45);
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__8__KET____DOT__prod_y = VL_RAND_RESET_Q(45);
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__9__KET____DOT__prod_x = VL_RAND_RESET_Q(45);
    vlSelf->tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__9__KET____DOT__prod_y = VL_RAND_RESET_Q(45);
    vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__1 = VL_RAND_RESET_Q(55);
    vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h017c4131__0 = VL_RAND_RESET_I(27);
    vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__2 = VL_RAND_RESET_Q(55);
    vlSelf->tb__DOT__dut__DOT__key_reset_sync__DOT__sync_ff = VL_RAND_RESET_I(2);
    vlSelf->tb__DOT__dut__DOT__key_reset_sync__DOT__sync_d = VL_RAND_RESET_I(1);
    vlSelf->tb__DOT__dut__DOT__key_reset_sync__DOT__reset_count = VL_RAND_RESET_I(4);
    vlSelf->tb__DOT__dut__DOT__hps_reset_sync__DOT__sync_ff = VL_RAND_RESET_I(2);
    vlSelf->tb__DOT__dut__DOT__hps_reset_sync__DOT__sync_d = VL_RAND_RESET_I(1);
    vlSelf->tb__DOT__dut__DOT__hps_reset_sync__DOT__reset_count = VL_RAND_RESET_I(4);
    vlSelf->__Vtrigprevexpr___TOP__tb__DOT__clk__0 = VL_RAND_RESET_I(1);
    for (int __Vi0 = 0; __Vi0 < 3; ++__Vi0) {
        vlSelf->__Vm_traceActivity[__Vi0] = 0;
    }
}

// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vshwfs_pipeline_tb.h for the primary calling header

#include "Vshwfs_pipeline_tb__pch.h"
#include "Vshwfs_pipeline_tb___024root.h"

VL_ATTR_COLD void Vshwfs_pipeline_tb___024root___eval_static(Vshwfs_pipeline_tb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vshwfs_pipeline_tb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vshwfs_pipeline_tb___024root___eval_static\n"); );
}

VL_ATTR_COLD void Vshwfs_pipeline_tb___024root___eval_final(Vshwfs_pipeline_tb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vshwfs_pipeline_tb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vshwfs_pipeline_tb___024root___eval_final\n"); );
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vshwfs_pipeline_tb___024root___dump_triggers__stl(Vshwfs_pipeline_tb___024root* vlSelf);
#endif  // VL_DEBUG
VL_ATTR_COLD bool Vshwfs_pipeline_tb___024root___eval_phase__stl(Vshwfs_pipeline_tb___024root* vlSelf);

VL_ATTR_COLD void Vshwfs_pipeline_tb___024root___eval_settle(Vshwfs_pipeline_tb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vshwfs_pipeline_tb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vshwfs_pipeline_tb___024root___eval_settle\n"); );
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
            Vshwfs_pipeline_tb___024root___dump_triggers__stl(vlSelf);
#endif
            VL_FATAL_MT("shwfs_pipeline_tb.v", 4, "", "Settle region did not converge.");
        }
        __VstlIterCount = ((IData)(1U) + __VstlIterCount);
        __VstlContinue = 0U;
        if (Vshwfs_pipeline_tb___024root___eval_phase__stl(vlSelf)) {
            __VstlContinue = 1U;
        }
        vlSelf->__VstlFirstIteration = 0U;
    }
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vshwfs_pipeline_tb___024root___dump_triggers__stl(Vshwfs_pipeline_tb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vshwfs_pipeline_tb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vshwfs_pipeline_tb___024root___dump_triggers__stl\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VstlTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelf->__VstlTriggered.word(0U))) {
        VL_DBG_MSGF("         'stl' region trigger index 0 is active: Internal 'stl' trigger - first iteration\n");
    }
}
#endif  // VL_DEBUG

extern const VlWide<128>/*4095:0*/ Vshwfs_pipeline_tb__ConstPool__CONST_h29f91db3_0;

VL_ATTR_COLD void Vshwfs_pipeline_tb___024root___stl_sequent__TOP__0(Vshwfs_pipeline_tb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vshwfs_pipeline_tb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vshwfs_pipeline_tb___024root___stl_sequent__TOP__0\n"); );
    // Init
    CData/*4:0*/ __Vfunc_shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__clz16__0__Vfuncout;
    __Vfunc_shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__clz16__0__Vfuncout = 0;
    SData/*15:0*/ __Vfunc_shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__clz16__0__x;
    __Vfunc_shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__clz16__0__x = 0;
    // Body
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__last_subap_col 
        = (0xfU == (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_col));
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__last_subap_row 
        = (0xfU == (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_row));
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__last_pixel_h 
        = (0xfU == (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__count_pixel_h));
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__last_pixel_v 
        = (0xfU == (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__count_pixel_v));
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer_valid 
        = ((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer_fv) 
           & (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer_lv));
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_idx 
        = (0xffU & (VL_SHIFTL_III(8,8,32, (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_row), 4U) 
                    + (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_col)));
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__v_safe 
        = ((0U == (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__intensity_accumulator))
            ? 1U : (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__intensity_accumulator));
    __Vfunc_shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__clz16__0__x 
        = vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__v_safe;
    __Vfunc_shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__clz16__0__Vfuncout 
        = (((((((((0x8000U == (0x8000U & (IData)(__Vfunc_shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__clz16__0__x))) 
                  | (0x4000U == (0xc000U & (IData)(__Vfunc_shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))) 
                 | (0x2000U == (0xe000U & (IData)(__Vfunc_shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))) 
                | (0x1000U == (0xf000U & (IData)(__Vfunc_shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))) 
               | (0x800U == (0xf800U & (IData)(__Vfunc_shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))) 
              | (0x400U == (0xfc00U & (IData)(__Vfunc_shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))) 
             | (0x200U == (0xfe00U & (IData)(__Vfunc_shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))) 
            | (0x100U == (0xff00U & (IData)(__Vfunc_shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__clz16__0__x))))
            ? ((0x8000U == (0x8000U & (IData)(__Vfunc_shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))
                ? 0U : ((0x4000U == (0xc000U & (IData)(__Vfunc_shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))
                         ? 1U : ((0x2000U == (0xe000U 
                                              & (IData)(__Vfunc_shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))
                                  ? 2U : ((0x1000U 
                                           == (0xf000U 
                                               & (IData)(__Vfunc_shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))
                                           ? 3U : (
                                                   (0x800U 
                                                    == 
                                                    (0xf800U 
                                                     & (IData)(__Vfunc_shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))
                                                    ? 4U
                                                    : 
                                                   ((0x400U 
                                                     == 
                                                     (0xfc00U 
                                                      & (IData)(__Vfunc_shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))
                                                     ? 5U
                                                     : 
                                                    ((0x200U 
                                                      == 
                                                      (0xfe00U 
                                                       & (IData)(__Vfunc_shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))
                                                      ? 6U
                                                      : 7U)))))))
            : (((((((((0x80U == (0xff80U & (IData)(__Vfunc_shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__clz16__0__x))) 
                      | (0x40U == (0xffc0U & (IData)(__Vfunc_shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))) 
                     | (0x20U == (0xffe0U & (IData)(__Vfunc_shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))) 
                    | (0x10U == (0xfff0U & (IData)(__Vfunc_shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))) 
                   | (8U == (0xfff8U & (IData)(__Vfunc_shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))) 
                  | (4U == (0xfffcU & (IData)(__Vfunc_shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))) 
                 | (2U == (0xfffeU & (IData)(__Vfunc_shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))) 
                | (1U == (IData)(__Vfunc_shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))
                ? ((0x80U == (0xff80U & (IData)(__Vfunc_shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))
                    ? 8U : ((0x40U == (0xffc0U & (IData)(__Vfunc_shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))
                             ? 9U : ((0x20U == (0xffe0U 
                                                & (IData)(__Vfunc_shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))
                                      ? 0xaU : ((0x10U 
                                                 == 
                                                 (0xfff0U 
                                                  & (IData)(__Vfunc_shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))
                                                 ? 0xbU
                                                 : 
                                                ((8U 
                                                  == 
                                                  (0xfff8U 
                                                   & (IData)(__Vfunc_shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))
                                                  ? 0xcU
                                                  : 
                                                 ((4U 
                                                   == 
                                                   (0xfffcU 
                                                    & (IData)(__Vfunc_shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))
                                                   ? 0xdU
                                                   : 
                                                  ((2U 
                                                    == 
                                                    (0xfffeU 
                                                     & (IData)(__Vfunc_shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__clz16__0__x)))
                                                    ? 0xeU
                                                    : 0xfU)))))))
                : 0x10U));
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__shift_left 
        = __Vfunc_shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__clz16__0__Vfuncout;
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__a_q1_15 
        = (0xffffU & ((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__v_safe) 
                      << (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__shift_left)));
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__u_newton_step_q16__DOT__prod_mul 
        = (0x1ffffffffULL & ((QData)((IData)((0xffffU 
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
                                                                         >> 7U)), 4U))))))) 
                             * (QData)((IData)((0x1ffffU 
                                                & ((IData)(0x10000U) 
                                                   - 
                                                   (((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
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
                                                    >> 0x10U)))))));
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__x1_q0_16 
        = ((0xffffU < (0x3ffffU & (IData)((0x3ffffULL 
                                           & ((0x4000ULL 
                                               + vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__u_newton_step_q16__DOT__prod_mul) 
                                              >> 0xfU)))))
            ? 0xffffU : (0xffffU & (IData)((0x3ffffULL 
                                            & ((0x4000ULL 
                                                + vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__u_newton_step_q16__DOT__prod_mul) 
                                               >> 0xfU)))));
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__out_q0_16_ext 
        = (0x1ffffU & ((0U == (0x1fU & ((IData)(0xfU) 
                                        - (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__shift_left))))
                        ? (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__x1_q0_16)
                        : ((0x1ffffU & ((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__x1_q0_16) 
                                        + ((0U == (0x1fU 
                                                   & ((IData)(0xfU) 
                                                      - (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__shift_left))))
                                            ? 0U : 
                                           (0xffffU 
                                            & ((IData)(1U) 
                                               << (0x1fU 
                                                   & (((IData)(0xfU) 
                                                       - (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__shift_left)) 
                                                      - (IData)(1U)))))))) 
                           >> (0x1fU & ((IData)(0xfU) 
                                        - (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__shift_left))))));
}

VL_ATTR_COLD void Vshwfs_pipeline_tb___024root___eval_stl(Vshwfs_pipeline_tb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vshwfs_pipeline_tb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vshwfs_pipeline_tb___024root___eval_stl\n"); );
    // Body
    if ((1ULL & vlSelf->__VstlTriggered.word(0U))) {
        Vshwfs_pipeline_tb___024root___stl_sequent__TOP__0(vlSelf);
        vlSelf->__Vm_traceActivity[1U] = 1U;
        vlSelf->__Vm_traceActivity[0U] = 1U;
    }
}

VL_ATTR_COLD void Vshwfs_pipeline_tb___024root___eval_triggers__stl(Vshwfs_pipeline_tb___024root* vlSelf);

VL_ATTR_COLD bool Vshwfs_pipeline_tb___024root___eval_phase__stl(Vshwfs_pipeline_tb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vshwfs_pipeline_tb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vshwfs_pipeline_tb___024root___eval_phase__stl\n"); );
    // Init
    CData/*0:0*/ __VstlExecute;
    // Body
    Vshwfs_pipeline_tb___024root___eval_triggers__stl(vlSelf);
    __VstlExecute = vlSelf->__VstlTriggered.any();
    if (__VstlExecute) {
        Vshwfs_pipeline_tb___024root___eval_stl(vlSelf);
    }
    return (__VstlExecute);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vshwfs_pipeline_tb___024root___dump_triggers__act(Vshwfs_pipeline_tb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vshwfs_pipeline_tb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vshwfs_pipeline_tb___024root___dump_triggers__act\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VactTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelf->__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 0 is active: @(posedge shwfs_pipeline_tb.clk_100)\n");
    }
    if ((2ULL & vlSelf->__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 1 is active: @([true] __VdlySched.awaitingCurrentTime())\n");
    }
}
#endif  // VL_DEBUG

#ifdef VL_DEBUG
VL_ATTR_COLD void Vshwfs_pipeline_tb___024root___dump_triggers__nba(Vshwfs_pipeline_tb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vshwfs_pipeline_tb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vshwfs_pipeline_tb___024root___dump_triggers__nba\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VnbaTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelf->__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 0 is active: @(posedge shwfs_pipeline_tb.clk_100)\n");
    }
    if ((2ULL & vlSelf->__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 1 is active: @([true] __VdlySched.awaitingCurrentTime())\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD void Vshwfs_pipeline_tb___024root___ctor_var_reset(Vshwfs_pipeline_tb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vshwfs_pipeline_tb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vshwfs_pipeline_tb___024root___ctor_var_reset\n"); );
    // Body
    vlSelf->shwfs_pipeline_tb__DOT__clk_100 = VL_RAND_RESET_I(1);
    vlSelf->shwfs_pipeline_tb__DOT__reset = VL_RAND_RESET_I(1);
    vlSelf->shwfs_pipeline_tb__DOT__xI_reciprocal = VL_RAND_RESET_I(20);
    vlSelf->shwfs_pipeline_tb__DOT__yI_reciprocal = VL_RAND_RESET_I(20);
    vlSelf->shwfs_pipeline_tb__DOT__rI_reciprocal = VL_RAND_RESET_I(16);
    vlSelf->shwfs_pipeline_tb__DOT__subaps_done_reciprocal = VL_RAND_RESET_I(8);
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer_data = VL_RAND_RESET_I(8);
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer_fv = VL_RAND_RESET_I(1);
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer_lv = VL_RAND_RESET_I(1);
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__frame_complete = VL_RAND_RESET_I(1);
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer_valid = VL_RAND_RESET_I(1);
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__full_frame_complete_accumulator = VL_RAND_RESET_I(1);
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__subaps_done_accumulator = VL_RAND_RESET_I(8);
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__intensity_accumulator = VL_RAND_RESET_I(16);
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__xI_accumulator = VL_RAND_RESET_I(20);
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__yI_accumulator = VL_RAND_RESET_I(20);
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__line_counter = VL_RAND_RESET_I(8);
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__row_counter = VL_RAND_RESET_I(8);
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__h_blank_counter = VL_RAND_RESET_I(2);
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__v_blank_counter = VL_RAND_RESET_I(8);
    for (int __Vi0 = 0; __Vi0 < 65536; ++__Vi0) {
        vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__mem[__Vi0] = VL_RAND_RESET_I(8);
    }
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__state = VL_RAND_RESET_I(2);
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_col = VL_RAND_RESET_I(4);
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_row = VL_RAND_RESET_I(4);
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__count_pixel_h = VL_RAND_RESET_I(4);
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__count_pixel_v = VL_RAND_RESET_I(4);
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_idx = VL_RAND_RESET_I(8);
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__last_pixel_h = VL_RAND_RESET_I(1);
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__last_pixel_v = VL_RAND_RESET_I(1);
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__last_subap_col = VL_RAND_RESET_I(1);
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__last_subap_row = VL_RAND_RESET_I(1);
    for (int __Vi0 = 0; __Vi0 < 256; ++__Vi0) {
        vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__i[__Vi0] = VL_RAND_RESET_I(16);
    }
    for (int __Vi0 = 0; __Vi0 < 256; ++__Vi0) {
        vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__x_i[__Vi0] = VL_RAND_RESET_I(20);
    }
    for (int __Vi0 = 0; __Vi0 < 256; ++__Vi0) {
        vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__y_i[__Vi0] = VL_RAND_RESET_I(20);
    }
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_done_delay = VL_RAND_RESET_I(1);
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_idx_delay = VL_RAND_RESET_I(8);
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__j = VL_RAND_RESET_I(32);
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__v_safe = VL_RAND_RESET_I(16);
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__shift_left = VL_RAND_RESET_I(5);
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__a_q1_15 = VL_RAND_RESET_I(16);
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__x1_q0_16 = VL_RAND_RESET_I(16);
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__out_q0_16_ext = VL_RAND_RESET_I(17);
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__u_newton_step_q16__DOT__prod_mul = VL_RAND_RESET_Q(33);
    vlSelf->__Vtrigprevexpr___TOP__shwfs_pipeline_tb__DOT__clk_100__0 = VL_RAND_RESET_I(1);
    for (int __Vi0 = 0; __Vi0 < 2; ++__Vi0) {
        vlSelf->__Vm_traceActivity[__Vi0] = 0;
    }
}

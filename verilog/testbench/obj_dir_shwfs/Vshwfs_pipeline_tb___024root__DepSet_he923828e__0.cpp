// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vshwfs_pipeline_tb.h for the primary calling header

#include "Vshwfs_pipeline_tb__pch.h"
#include "Vshwfs_pipeline_tb___024root.h"

VL_ATTR_COLD void Vshwfs_pipeline_tb___024root___eval_initial__TOP(Vshwfs_pipeline_tb___024root* vlSelf);
VlCoroutine Vshwfs_pipeline_tb___024root___eval_initial__TOP__Vtiming__0(Vshwfs_pipeline_tb___024root* vlSelf);
VlCoroutine Vshwfs_pipeline_tb___024root___eval_initial__TOP__Vtiming__1(Vshwfs_pipeline_tb___024root* vlSelf);
VlCoroutine Vshwfs_pipeline_tb___024root___eval_initial__TOP__Vtiming__2(Vshwfs_pipeline_tb___024root* vlSelf);

void Vshwfs_pipeline_tb___024root___eval_initial(Vshwfs_pipeline_tb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vshwfs_pipeline_tb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vshwfs_pipeline_tb___024root___eval_initial\n"); );
    // Body
    Vshwfs_pipeline_tb___024root___eval_initial__TOP(vlSelf);
    Vshwfs_pipeline_tb___024root___eval_initial__TOP__Vtiming__0(vlSelf);
    Vshwfs_pipeline_tb___024root___eval_initial__TOP__Vtiming__1(vlSelf);
    Vshwfs_pipeline_tb___024root___eval_initial__TOP__Vtiming__2(vlSelf);
    vlSelf->__Vtrigprevexpr___TOP__shwfs_pipeline_tb__DOT__clk_100__0 
        = vlSelf->shwfs_pipeline_tb__DOT__clk_100;
}

VL_INLINE_OPT VlCoroutine Vshwfs_pipeline_tb___024root___eval_initial__TOP__Vtiming__0(Vshwfs_pipeline_tb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vshwfs_pipeline_tb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vshwfs_pipeline_tb___024root___eval_initial__TOP__Vtiming__0\n"); );
    // Body
    vlSelf->shwfs_pipeline_tb__DOT__clk_100 = 0U;
    vlSelf->shwfs_pipeline_tb__DOT__reset = 1U;
    co_await vlSelf->__VdlySched.delay(0x2710ULL, nullptr, 
                                       "shwfs_pipeline_tb.v", 
                                       17);
    vlSelf->shwfs_pipeline_tb__DOT__reset = 0U;
}

VL_INLINE_OPT VlCoroutine Vshwfs_pipeline_tb___024root___eval_initial__TOP__Vtiming__1(Vshwfs_pipeline_tb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vshwfs_pipeline_tb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vshwfs_pipeline_tb___024root___eval_initial__TOP__Vtiming__1\n"); );
    // Body
    co_await vlSelf->__VdlySched.delay(0x989680ULL, 
                                       nullptr, "shwfs_pipeline_tb.v", 
                                       28);
    VL_FINISH_MT("shwfs_pipeline_tb.v", 29, "");
}

VL_INLINE_OPT VlCoroutine Vshwfs_pipeline_tb___024root___eval_initial__TOP__Vtiming__2(Vshwfs_pipeline_tb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vshwfs_pipeline_tb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vshwfs_pipeline_tb___024root___eval_initial__TOP__Vtiming__2\n"); );
    // Body
    while (1U) {
        co_await vlSelf->__VdlySched.delay(0x1388ULL, 
                                           nullptr, 
                                           "shwfs_pipeline_tb.v", 
                                           23);
        vlSelf->shwfs_pipeline_tb__DOT__clk_100 = (1U 
                                                   & (~ (IData)(vlSelf->shwfs_pipeline_tb__DOT__clk_100)));
    }
}

void Vshwfs_pipeline_tb___024root___eval_act(Vshwfs_pipeline_tb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vshwfs_pipeline_tb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vshwfs_pipeline_tb___024root___eval_act\n"); );
}

extern const VlWide<128>/*4095:0*/ Vshwfs_pipeline_tb__ConstPool__CONST_h29f91db3_0;

VL_INLINE_OPT void Vshwfs_pipeline_tb___024root___nba_sequent__TOP__0(Vshwfs_pipeline_tb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vshwfs_pipeline_tb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vshwfs_pipeline_tb___024root___nba_sequent__TOP__0\n"); );
    // Init
    CData/*4:0*/ __Vfunc_shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__clz16__0__Vfuncout;
    __Vfunc_shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__clz16__0__Vfuncout = 0;
    IData/*31:0*/ __Vilp;
    SData/*15:0*/ __Vfunc_shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__clz16__0__x;
    __Vfunc_shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__clz16__0__x = 0;
    CData/*1:0*/ __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__state;
    __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__state = 0;
    CData/*7:0*/ __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__line_counter;
    __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__line_counter = 0;
    CData/*7:0*/ __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__row_counter;
    __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__row_counter = 0;
    CData/*1:0*/ __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__h_blank_counter;
    __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__h_blank_counter = 0;
    CData/*7:0*/ __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__v_blank_counter;
    __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__v_blank_counter = 0;
    CData/*3:0*/ __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_col;
    __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_col = 0;
    CData/*3:0*/ __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_row;
    __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_row = 0;
    CData/*3:0*/ __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__count_pixel_h;
    __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__count_pixel_h = 0;
    CData/*3:0*/ __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__count_pixel_v;
    __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__count_pixel_v = 0;
    CData/*7:0*/ __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__subaps_done_accumulator;
    __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__subaps_done_accumulator = 0;
    CData/*0:0*/ __Vdlyvset__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__i__v0;
    __Vdlyvset__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__i__v0 = 0;
    CData/*7:0*/ __Vdlyvdim0__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__i__v256;
    __Vdlyvdim0__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__i__v256 = 0;
    SData/*15:0*/ __Vdlyvval__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__i__v256;
    __Vdlyvval__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__i__v256 = 0;
    CData/*0:0*/ __Vdlyvset__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__i__v256;
    __Vdlyvset__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__i__v256 = 0;
    CData/*0:0*/ __Vdlyvset__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__x_i__v0;
    __Vdlyvset__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__x_i__v0 = 0;
    CData/*7:0*/ __Vdlyvdim0__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__x_i__v256;
    __Vdlyvdim0__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__x_i__v256 = 0;
    IData/*19:0*/ __Vdlyvval__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__x_i__v256;
    __Vdlyvval__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__x_i__v256 = 0;
    CData/*0:0*/ __Vdlyvset__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__x_i__v256;
    __Vdlyvset__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__x_i__v256 = 0;
    CData/*0:0*/ __Vdlyvset__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__y_i__v0;
    __Vdlyvset__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__y_i__v0 = 0;
    CData/*7:0*/ __Vdlyvdim0__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__y_i__v256;
    __Vdlyvdim0__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__y_i__v256 = 0;
    IData/*19:0*/ __Vdlyvval__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__y_i__v256;
    __Vdlyvval__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__y_i__v256 = 0;
    CData/*0:0*/ __Vdlyvset__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__y_i__v256;
    __Vdlyvset__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__y_i__v256 = 0;
    // Body
    __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__count_pixel_h 
        = vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__count_pixel_h;
    __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_col 
        = vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_col;
    __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__v_blank_counter 
        = vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__v_blank_counter;
    __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__h_blank_counter 
        = vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__h_blank_counter;
    __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__row_counter 
        = vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__row_counter;
    __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__line_counter 
        = vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__line_counter;
    __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__state 
        = vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__state;
    __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__count_pixel_v 
        = vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__count_pixel_v;
    __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_row 
        = vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_row;
    __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__subaps_done_accumulator 
        = vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__subaps_done_accumulator;
    __Vdlyvset__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__i__v0 = 0U;
    __Vdlyvset__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__i__v256 = 0U;
    __Vdlyvset__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__y_i__v0 = 0U;
    __Vdlyvset__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__y_i__v256 = 0U;
    __Vdlyvset__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__x_i__v0 = 0U;
    __Vdlyvset__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__x_i__v256 = 0U;
    if (vlSelf->shwfs_pipeline_tb__DOT__reset) {
        vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__j = 0x100U;
        __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__count_pixel_h = 0U;
        __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_col = 0U;
        __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__count_pixel_v = 0U;
        __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_row = 0U;
        __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__subaps_done_accumulator = 0U;
        __Vdlyvset__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__i__v0 = 1U;
        __Vdlyvset__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__y_i__v0 = 1U;
        __Vdlyvset__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__x_i__v0 = 1U;
        vlSelf->shwfs_pipeline_tb__DOT__subaps_done_reciprocal = 0U;
        vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__full_frame_complete_accumulator = 0U;
        vlSelf->shwfs_pipeline_tb__DOT__yI_reciprocal = 0U;
        vlSelf->shwfs_pipeline_tb__DOT__xI_reciprocal = 0U;
        vlSelf->shwfs_pipeline_tb__DOT__rI_reciprocal = 0U;
        __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__state = 0U;
        __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__line_counter = 0U;
        __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__row_counter = 0U;
        __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__h_blank_counter = 0U;
        __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__v_blank_counter = 0U;
        vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer_lv = 0U;
        vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer_fv = 0U;
        vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__frame_complete = 0U;
        vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__yI_accumulator = 0U;
        vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__xI_accumulator = 0U;
        vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__intensity_accumulator = 0U;
        vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_idx_delay = 0U;
    } else {
        if (vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer_valid) {
            __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__count_pixel_h 
                = ((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__last_pixel_h)
                    ? 0U : (0xfU & ((IData)(1U) + (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__count_pixel_h))));
            if ((0xfU == (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__count_pixel_h))) {
                __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_col 
                    = ((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__last_subap_col)
                        ? 0U : (0xfU & ((IData)(1U) 
                                        + (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_col))));
            }
            __Vdlyvval__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__i__v256 
                = (0xffffU & (vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__i
                              [vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_idx] 
                              + (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer_data)));
            __Vdlyvset__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__i__v256 = 1U;
            __Vdlyvdim0__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__i__v256 
                = vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_idx;
            __Vdlyvval__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__y_i__v256 
                = (0xfffffU & (vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__y_i
                               [vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_idx] 
                               + ((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer_data) 
                                  * (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__count_pixel_v))));
            __Vdlyvset__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__y_i__v256 = 1U;
            __Vdlyvdim0__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__y_i__v256 
                = vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_idx;
            __Vdlyvval__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__x_i__v256 
                = (0xfffffU & (vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__x_i
                               [vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_idx] 
                               + ((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer_data) 
                                  * (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__count_pixel_h))));
            __Vdlyvset__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__x_i__v256 = 1U;
            __Vdlyvdim0__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__x_i__v256 
                = vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_idx;
            vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__full_frame_complete_accumulator = 0U;
            if (((0xfU == (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__count_pixel_h)) 
                 & (0xfU == (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_col)))) {
                __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__count_pixel_v 
                    = ((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__last_pixel_v)
                        ? 0U : (0xfU & ((IData)(1U) 
                                        + (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__count_pixel_v))));
                if ((0xfU == (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__count_pixel_v))) {
                    __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_row 
                        = ((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__last_subap_row)
                            ? 0U : (0xfU & ((IData)(1U) 
                                            + (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_row))));
                    if ((0xfU == (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_row))) {
                        vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__full_frame_complete_accumulator = 1U;
                    }
                }
            }
        }
        vlSelf->shwfs_pipeline_tb__DOT__subaps_done_reciprocal 
            = vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__subaps_done_accumulator;
        vlSelf->shwfs_pipeline_tb__DOT__yI_reciprocal 
            = vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__yI_accumulator;
        vlSelf->shwfs_pipeline_tb__DOT__xI_reciprocal 
            = vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__xI_accumulator;
        vlSelf->shwfs_pipeline_tb__DOT__rI_reciprocal 
            = ((0U == (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__intensity_accumulator))
                ? 0xffffU : ((0xffffU < vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__out_q0_16_ext)
                              ? 0xffffU : (0xffffU 
                                           & vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__out_q0_16_ext)));
        if (vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_done_delay) {
            __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__subaps_done_accumulator 
                = (0xffU & ((IData)(1U) + (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__subaps_done_accumulator)));
            vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__yI_accumulator 
                = vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__y_i
                [vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_idx_delay];
            vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__xI_accumulator 
                = vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__x_i
                [vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_idx_delay];
            vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__intensity_accumulator 
                = vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__i
                [vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_idx_delay];
        }
        if ((2U & (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__state))) {
            if ((1U & (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__state))) {
                vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer_fv = 0U;
                vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer_lv = 0U;
                __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__line_counter = 0U;
                __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__row_counter = 0U;
                __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__v_blank_counter 
                    = (0xffU & ((IData)(1U) + (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__v_blank_counter)));
                if ((0x97U == (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__v_blank_counter))) {
                    __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__v_blank_counter = 0U;
                    __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__state = 0U;
                }
            } else {
                vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer_lv = 0U;
                __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__line_counter = 0U;
                __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__h_blank_counter 
                    = (3U & ((IData)(1U) + (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__h_blank_counter)));
                if ((3U == (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__h_blank_counter))) {
                    __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__row_counter 
                        = (0xffU & ((IData)(1U) + (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__row_counter)));
                    __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__h_blank_counter = 0U;
                    __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__state = 1U;
                }
            }
        } else if ((1U & (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__state))) {
            vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer_data 
                = vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__mem
                [(0xffffU & (VL_SHIFTL_III(16,32,32, (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__row_counter), 8U) 
                             + (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__line_counter)))];
            vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer_lv = 1U;
            __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__line_counter 
                = (0xffU & ((IData)(1U) + (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__line_counter)));
            if (((0xffU == (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__line_counter)) 
                 & (0xffU == (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__row_counter)))) {
                __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__state = 3U;
                vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__frame_complete = 1U;
            } else if ((0xffU == (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__line_counter))) {
                __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__state = 2U;
            }
        } else {
            vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer_fv = 1U;
            __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__line_counter = 0U;
            __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__row_counter = 0U;
            __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__state = 1U;
        }
        vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_idx_delay 
            = vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_idx;
    }
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__subaps_done_accumulator 
        = __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__subaps_done_accumulator;
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_row 
        = __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_row;
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_col 
        = __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_col;
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__last_subap_row 
        = (0xfU == (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_row));
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__last_subap_col 
        = (0xfU == (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_col));
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__state 
        = __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__state;
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__line_counter 
        = __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__line_counter;
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__row_counter 
        = __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__row_counter;
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__h_blank_counter 
        = __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__h_blank_counter;
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__v_blank_counter 
        = __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__v_blank_counter;
    if (__Vdlyvset__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__y_i__v0) {
        __Vilp = 0U;
        while ((__Vilp <= 0xffU)) {
            vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__y_i[__Vilp] = 0U;
            __Vilp = ((IData)(1U) + __Vilp);
        }
    }
    if (__Vdlyvset__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__y_i__v256) {
        vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__y_i[__Vdlyvdim0__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__y_i__v256] 
            = __Vdlyvval__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__y_i__v256;
    }
    if (__Vdlyvset__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__x_i__v0) {
        __Vilp = 0U;
        while ((__Vilp <= 0xffU)) {
            vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__x_i[__Vilp] = 0U;
            __Vilp = ((IData)(1U) + __Vilp);
        }
    }
    if (__Vdlyvset__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__x_i__v256) {
        vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__x_i[__Vdlyvdim0__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__x_i__v256] 
            = __Vdlyvval__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__x_i__v256;
    }
    if (__Vdlyvset__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__i__v0) {
        __Vilp = 0U;
        while ((__Vilp <= 0xffU)) {
            vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__i[__Vilp] = 0U;
            __Vilp = ((IData)(1U) + __Vilp);
        }
    }
    if (__Vdlyvset__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__i__v256) {
        vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__i[__Vdlyvdim0__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__i__v256] 
            = __Vdlyvval__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__i__v256;
    }
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__v_safe 
        = ((0U == (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__intensity_accumulator))
            ? 1U : (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__intensity_accumulator));
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_done_delay 
        = ((1U & (~ (IData)(vlSelf->shwfs_pipeline_tb__DOT__reset))) 
           && (((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer_valid) 
                & (0xfU == (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__count_pixel_h))) 
               & (0xfU == (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__count_pixel_v))));
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
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_idx 
        = (0xffU & (VL_SHIFTL_III(8,8,32, (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_row), 4U) 
                    + (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_col)));
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__count_pixel_v 
        = __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__count_pixel_v;
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer_valid 
        = ((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer_fv) 
           & (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer_lv));
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__count_pixel_h 
        = __Vdly__shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__count_pixel_h;
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__a_q1_15 
        = (0xffffU & ((IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__v_safe) 
                      << (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__shift_left)));
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__last_pixel_v 
        = (0xfU == (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__count_pixel_v));
    vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__last_pixel_h 
        = (0xfU == (IData)(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__count_pixel_h));
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

void Vshwfs_pipeline_tb___024root___eval_nba(Vshwfs_pipeline_tb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vshwfs_pipeline_tb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vshwfs_pipeline_tb___024root___eval_nba\n"); );
    // Body
    if ((1ULL & vlSelf->__VnbaTriggered.word(0U))) {
        Vshwfs_pipeline_tb___024root___nba_sequent__TOP__0(vlSelf);
        vlSelf->__Vm_traceActivity[1U] = 1U;
    }
}

void Vshwfs_pipeline_tb___024root___timing_resume(Vshwfs_pipeline_tb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vshwfs_pipeline_tb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vshwfs_pipeline_tb___024root___timing_resume\n"); );
    // Body
    if ((2ULL & vlSelf->__VactTriggered.word(0U))) {
        vlSelf->__VdlySched.resume();
    }
}

void Vshwfs_pipeline_tb___024root___eval_triggers__act(Vshwfs_pipeline_tb___024root* vlSelf);

bool Vshwfs_pipeline_tb___024root___eval_phase__act(Vshwfs_pipeline_tb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vshwfs_pipeline_tb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vshwfs_pipeline_tb___024root___eval_phase__act\n"); );
    // Init
    VlTriggerVec<2> __VpreTriggered;
    CData/*0:0*/ __VactExecute;
    // Body
    Vshwfs_pipeline_tb___024root___eval_triggers__act(vlSelf);
    __VactExecute = vlSelf->__VactTriggered.any();
    if (__VactExecute) {
        __VpreTriggered.andNot(vlSelf->__VactTriggered, vlSelf->__VnbaTriggered);
        vlSelf->__VnbaTriggered.thisOr(vlSelf->__VactTriggered);
        Vshwfs_pipeline_tb___024root___timing_resume(vlSelf);
        Vshwfs_pipeline_tb___024root___eval_act(vlSelf);
    }
    return (__VactExecute);
}

bool Vshwfs_pipeline_tb___024root___eval_phase__nba(Vshwfs_pipeline_tb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vshwfs_pipeline_tb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vshwfs_pipeline_tb___024root___eval_phase__nba\n"); );
    // Init
    CData/*0:0*/ __VnbaExecute;
    // Body
    __VnbaExecute = vlSelf->__VnbaTriggered.any();
    if (__VnbaExecute) {
        Vshwfs_pipeline_tb___024root___eval_nba(vlSelf);
        vlSelf->__VnbaTriggered.clear();
    }
    return (__VnbaExecute);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vshwfs_pipeline_tb___024root___dump_triggers__nba(Vshwfs_pipeline_tb___024root* vlSelf);
#endif  // VL_DEBUG
#ifdef VL_DEBUG
VL_ATTR_COLD void Vshwfs_pipeline_tb___024root___dump_triggers__act(Vshwfs_pipeline_tb___024root* vlSelf);
#endif  // VL_DEBUG

void Vshwfs_pipeline_tb___024root___eval(Vshwfs_pipeline_tb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vshwfs_pipeline_tb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vshwfs_pipeline_tb___024root___eval\n"); );
    // Init
    IData/*31:0*/ __VnbaIterCount;
    CData/*0:0*/ __VnbaContinue;
    // Body
    __VnbaIterCount = 0U;
    __VnbaContinue = 1U;
    while (__VnbaContinue) {
        if (VL_UNLIKELY((0x64U < __VnbaIterCount))) {
#ifdef VL_DEBUG
            Vshwfs_pipeline_tb___024root___dump_triggers__nba(vlSelf);
#endif
            VL_FATAL_MT("shwfs_pipeline_tb.v", 4, "", "NBA region did not converge.");
        }
        __VnbaIterCount = ((IData)(1U) + __VnbaIterCount);
        __VnbaContinue = 0U;
        vlSelf->__VactIterCount = 0U;
        vlSelf->__VactContinue = 1U;
        while (vlSelf->__VactContinue) {
            if (VL_UNLIKELY((0x64U < vlSelf->__VactIterCount))) {
#ifdef VL_DEBUG
                Vshwfs_pipeline_tb___024root___dump_triggers__act(vlSelf);
#endif
                VL_FATAL_MT("shwfs_pipeline_tb.v", 4, "", "Active region did not converge.");
            }
            vlSelf->__VactIterCount = ((IData)(1U) 
                                       + vlSelf->__VactIterCount);
            vlSelf->__VactContinue = 0U;
            if (Vshwfs_pipeline_tb___024root___eval_phase__act(vlSelf)) {
                vlSelf->__VactContinue = 1U;
            }
        }
        if (Vshwfs_pipeline_tb___024root___eval_phase__nba(vlSelf)) {
            __VnbaContinue = 1U;
        }
    }
}

#ifdef VL_DEBUG
void Vshwfs_pipeline_tb___024root___eval_debug_assertions(Vshwfs_pipeline_tb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vshwfs_pipeline_tb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vshwfs_pipeline_tb___024root___eval_debug_assertions\n"); );
}
#endif  // VL_DEBUG

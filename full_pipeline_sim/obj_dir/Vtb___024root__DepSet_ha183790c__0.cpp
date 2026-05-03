// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtb.h for the primary calling header

#include "Vtb__pch.h"
#include "Vtb___024root.h"

VL_ATTR_COLD void Vtb___024root___eval_initial__TOP(Vtb___024root* vlSelf);
VlCoroutine Vtb___024root___eval_initial__TOP__Vtiming__0(Vtb___024root* vlSelf);
VlCoroutine Vtb___024root___eval_initial__TOP__Vtiming__1(Vtb___024root* vlSelf);

void Vtb___024root___eval_initial(Vtb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb___024root___eval_initial\n"); );
    // Body
    Vtb___024root___eval_initial__TOP(vlSelf);
    vlSelf->__Vm_traceActivity[1U] = 1U;
    Vtb___024root___eval_initial__TOP__Vtiming__0(vlSelf);
    Vtb___024root___eval_initial__TOP__Vtiming__1(vlSelf);
    vlSelf->__Vtrigprevexpr___TOP__tb__DOT__clk__0 
        = vlSelf->tb__DOT__clk;
}

VL_INLINE_OPT VlCoroutine Vtb___024root___eval_initial__TOP__Vtiming__1(Vtb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb___024root___eval_initial__TOP__Vtiming__1\n"); );
    // Body
    vlSelf->tb__DOT__i = 0U;
    while (VL_GTS_III(32, 0x105b8U, vlSelf->tb__DOT__i)) {
        vlSelf->tb__DOT__clk = 1U;
        co_await vlSelf->__VdlySched.delay(0x3e8ULL, 
                                           nullptr, 
                                           "tb.v", 
                                           29);
        vlSelf->tb__DOT__clk = 0U;
        co_await vlSelf->__VdlySched.delay(0x3e8ULL, 
                                           nullptr, 
                                           "tb.v", 
                                           31);
        vlSelf->tb__DOT__i = ((IData)(1U) + vlSelf->tb__DOT__i);
    }
    vlSelf->tb__DOT__key_reset = 1U;
    co_await vlSelf->__VdlySched.delay(0x3e8ULL, nullptr, 
                                       "tb.v", 35);
    vlSelf->tb__DOT__clk = 1U;
    co_await vlSelf->__VdlySched.delay(0x3e8ULL, nullptr, 
                                       "tb.v", 37);
    vlSelf->tb__DOT__clk = 0U;
    vlSelf->tb__DOT__key_reset = 0U;
    vlSelf->tb__DOT__i = 0U;
    while (VL_GTS_III(32, 0x105b8U, vlSelf->tb__DOT__i)) {
        vlSelf->tb__DOT__clk = 1U;
        co_await vlSelf->__VdlySched.delay(0x3e8ULL, 
                                           nullptr, 
                                           "tb.v", 
                                           43);
        vlSelf->tb__DOT__clk = 0U;
        co_await vlSelf->__VdlySched.delay(0x3e8ULL, 
                                           nullptr, 
                                           "tb.v", 
                                           45);
        vlSelf->tb__DOT__i = ((IData)(1U) + vlSelf->tb__DOT__i);
    }
}

void Vtb___024root___eval_act(Vtb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb___024root___eval_act\n"); );
}

extern const VlWide<128>/*4095:0*/ Vtb__ConstPool__CONST_h29f91db3_0;
extern const VlWide<9>/*287:0*/ Vtb__ConstPool__CONST_h82fc43e2_0;

VL_INLINE_OPT void Vtb___024root___nba_sequent__TOP__0(Vtb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb___024root___nba_sequent__TOP__0\n"); );
    // Init
    CData/*4:0*/ __Vfunc_tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__clz16__0__Vfuncout;
    __Vfunc_tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__clz16__0__Vfuncout = 0;
    SData/*15:0*/ __Vfunc_tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__clz16__0__x;
    __Vfunc_tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__clz16__0__x = 0;
    IData/*26:0*/ __Vdlyvval__tb__DOT__dut__DOT__zernike_out_reg__v0;
    __Vdlyvval__tb__DOT__dut__DOT__zernike_out_reg__v0 = 0;
    CData/*0:0*/ __Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v0;
    __Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v0 = 0;
    IData/*26:0*/ __Vdlyvval__tb__DOT__dut__DOT__zernike_out_reg__v1;
    __Vdlyvval__tb__DOT__dut__DOT__zernike_out_reg__v1 = 0;
    CData/*0:0*/ __Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v1;
    __Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v1 = 0;
    IData/*26:0*/ __Vdlyvval__tb__DOT__dut__DOT__zernike_out_reg__v2;
    __Vdlyvval__tb__DOT__dut__DOT__zernike_out_reg__v2 = 0;
    CData/*0:0*/ __Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v2;
    __Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v2 = 0;
    IData/*26:0*/ __Vdlyvval__tb__DOT__dut__DOT__zernike_out_reg__v3;
    __Vdlyvval__tb__DOT__dut__DOT__zernike_out_reg__v3 = 0;
    CData/*0:0*/ __Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v3;
    __Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v3 = 0;
    IData/*26:0*/ __Vdlyvval__tb__DOT__dut__DOT__zernike_out_reg__v4;
    __Vdlyvval__tb__DOT__dut__DOT__zernike_out_reg__v4 = 0;
    CData/*0:0*/ __Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v4;
    __Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v4 = 0;
    IData/*26:0*/ __Vdlyvval__tb__DOT__dut__DOT__zernike_out_reg__v5;
    __Vdlyvval__tb__DOT__dut__DOT__zernike_out_reg__v5 = 0;
    CData/*0:0*/ __Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v5;
    __Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v5 = 0;
    IData/*26:0*/ __Vdlyvval__tb__DOT__dut__DOT__zernike_out_reg__v6;
    __Vdlyvval__tb__DOT__dut__DOT__zernike_out_reg__v6 = 0;
    CData/*0:0*/ __Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v6;
    __Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v6 = 0;
    IData/*26:0*/ __Vdlyvval__tb__DOT__dut__DOT__zernike_out_reg__v7;
    __Vdlyvval__tb__DOT__dut__DOT__zernike_out_reg__v7 = 0;
    CData/*0:0*/ __Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v7;
    __Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v7 = 0;
    IData/*26:0*/ __Vdlyvval__tb__DOT__dut__DOT__zernike_out_reg__v8;
    __Vdlyvval__tb__DOT__dut__DOT__zernike_out_reg__v8 = 0;
    CData/*0:0*/ __Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v8;
    __Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v8 = 0;
    IData/*26:0*/ __Vdlyvval__tb__DOT__dut__DOT__zernike_out_reg__v9;
    __Vdlyvval__tb__DOT__dut__DOT__zernike_out_reg__v9 = 0;
    CData/*0:0*/ __Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v9;
    __Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v9 = 0;
    SData/*10:0*/ __Vdly__tb__DOT__dut__DOT__resw_write_idx;
    __Vdly__tb__DOT__dut__DOT__resw_write_idx = 0;
    SData/*10:0*/ __Vdly__tb__DOT__dut__DOT__write_zernike_count;
    __Vdly__tb__DOT__dut__DOT__write_zernike_count = 0;
    CData/*1:0*/ __Vdly__tb__DOT__dut__DOT__streamer__DOT__state;
    __Vdly__tb__DOT__dut__DOT__streamer__DOT__state = 0;
    CData/*7:0*/ __Vdly__tb__DOT__dut__DOT__streamer__DOT__line_counter;
    __Vdly__tb__DOT__dut__DOT__streamer__DOT__line_counter = 0;
    CData/*7:0*/ __Vdly__tb__DOT__dut__DOT__streamer__DOT__row_counter;
    __Vdly__tb__DOT__dut__DOT__streamer__DOT__row_counter = 0;
    CData/*1:0*/ __Vdly__tb__DOT__dut__DOT__streamer__DOT__h_blank_counter;
    __Vdly__tb__DOT__dut__DOT__streamer__DOT__h_blank_counter = 0;
    CData/*7:0*/ __Vdly__tb__DOT__dut__DOT__streamer__DOT__v_blank_counter;
    __Vdly__tb__DOT__dut__DOT__streamer__DOT__v_blank_counter = 0;
    CData/*3:0*/ __Vdly__tb__DOT__dut__DOT__accumulator__DOT__subap_col;
    __Vdly__tb__DOT__dut__DOT__accumulator__DOT__subap_col = 0;
    CData/*3:0*/ __Vdly__tb__DOT__dut__DOT__accumulator__DOT__subap_row;
    __Vdly__tb__DOT__dut__DOT__accumulator__DOT__subap_row = 0;
    CData/*3:0*/ __Vdly__tb__DOT__dut__DOT__accumulator__DOT__count_pixel_h;
    __Vdly__tb__DOT__dut__DOT__accumulator__DOT__count_pixel_h = 0;
    CData/*3:0*/ __Vdly__tb__DOT__dut__DOT__accumulator__DOT__count_pixel_v;
    __Vdly__tb__DOT__dut__DOT__accumulator__DOT__count_pixel_v = 0;
    CData/*7:0*/ __Vdly__tb__DOT__dut__DOT__subaps_done_accumulator;
    __Vdly__tb__DOT__dut__DOT__subaps_done_accumulator = 0;
    CData/*7:0*/ __Vdlyvdim0__tb__DOT__dut__DOT__accumulator__DOT__i__v0;
    __Vdlyvdim0__tb__DOT__dut__DOT__accumulator__DOT__i__v0 = 0;
    SData/*15:0*/ __Vdlyvval__tb__DOT__dut__DOT__accumulator__DOT__i__v0;
    __Vdlyvval__tb__DOT__dut__DOT__accumulator__DOT__i__v0 = 0;
    CData/*0:0*/ __Vdlyvset__tb__DOT__dut__DOT__accumulator__DOT__i__v0;
    __Vdlyvset__tb__DOT__dut__DOT__accumulator__DOT__i__v0 = 0;
    CData/*7:0*/ __Vdlyvdim0__tb__DOT__dut__DOT__accumulator__DOT__x_i__v0;
    __Vdlyvdim0__tb__DOT__dut__DOT__accumulator__DOT__x_i__v0 = 0;
    IData/*19:0*/ __Vdlyvval__tb__DOT__dut__DOT__accumulator__DOT__x_i__v0;
    __Vdlyvval__tb__DOT__dut__DOT__accumulator__DOT__x_i__v0 = 0;
    CData/*7:0*/ __Vdlyvdim0__tb__DOT__dut__DOT__accumulator__DOT__y_i__v0;
    __Vdlyvdim0__tb__DOT__dut__DOT__accumulator__DOT__y_i__v0 = 0;
    IData/*19:0*/ __Vdlyvval__tb__DOT__dut__DOT__accumulator__DOT__y_i__v0;
    __Vdlyvval__tb__DOT__dut__DOT__accumulator__DOT__y_i__v0 = 0;
    CData/*0:0*/ __Vdlyvset__tb__DOT__dut__DOT__reciprocal__DOT__xI_internal__v0;
    __Vdlyvset__tb__DOT__dut__DOT__reciprocal__DOT__xI_internal__v0 = 0;
    IData/*19:0*/ __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__xI_internal__v4;
    __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__xI_internal__v4 = 0;
    CData/*0:0*/ __Vdlyvset__tb__DOT__dut__DOT__reciprocal__DOT__xI_internal__v4;
    __Vdlyvset__tb__DOT__dut__DOT__reciprocal__DOT__xI_internal__v4 = 0;
    IData/*19:0*/ __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__xI_internal__v5;
    __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__xI_internal__v5 = 0;
    IData/*19:0*/ __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__xI_internal__v6;
    __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__xI_internal__v6 = 0;
    IData/*19:0*/ __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__xI_internal__v7;
    __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__xI_internal__v7 = 0;
    CData/*0:0*/ __Vdlyvset__tb__DOT__dut__DOT__reciprocal__DOT__yI_internal__v0;
    __Vdlyvset__tb__DOT__dut__DOT__reciprocal__DOT__yI_internal__v0 = 0;
    IData/*19:0*/ __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__yI_internal__v4;
    __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__yI_internal__v4 = 0;
    CData/*0:0*/ __Vdlyvset__tb__DOT__dut__DOT__reciprocal__DOT__yI_internal__v4;
    __Vdlyvset__tb__DOT__dut__DOT__reciprocal__DOT__yI_internal__v4 = 0;
    IData/*19:0*/ __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__yI_internal__v5;
    __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__yI_internal__v5 = 0;
    IData/*19:0*/ __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__yI_internal__v6;
    __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__yI_internal__v6 = 0;
    IData/*19:0*/ __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__yI_internal__v7;
    __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__yI_internal__v7 = 0;
    CData/*0:0*/ __Vdlyvset__tb__DOT__dut__DOT__reciprocal__DOT__centroids_done_internal__v0;
    __Vdlyvset__tb__DOT__dut__DOT__reciprocal__DOT__centroids_done_internal__v0 = 0;
    CData/*7:0*/ __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__centroids_done_internal__v4;
    __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__centroids_done_internal__v4 = 0;
    CData/*0:0*/ __Vdlyvset__tb__DOT__dut__DOT__reciprocal__DOT__centroids_done_internal__v4;
    __Vdlyvset__tb__DOT__dut__DOT__reciprocal__DOT__centroids_done_internal__v4 = 0;
    CData/*7:0*/ __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__centroids_done_internal__v5;
    __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__centroids_done_internal__v5 = 0;
    CData/*7:0*/ __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__centroids_done_internal__v6;
    __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__centroids_done_internal__v6 = 0;
    CData/*7:0*/ __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__centroids_done_internal__v7;
    __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__centroids_done_internal__v7 = 0;
    CData/*7:0*/ __Vdly__tb__DOT__dut__DOT__slopes__DOT__current_subap;
    __Vdly__tb__DOT__dut__DOT__slopes__DOT__current_subap = 0;
    CData/*0:0*/ __Vdly__tb__DOT__dut__DOT__em__DOT__state;
    __Vdly__tb__DOT__dut__DOT__em__DOT__state = 0;
    CData/*7:0*/ __Vdly__tb__DOT__dut__DOT__em__DOT__sub_counter;
    __Vdly__tb__DOT__dut__DOT__em__DOT__sub_counter = 0;
    CData/*0:0*/ __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v0;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v0 = 0;
    CData/*0:0*/ __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v10;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v10 = 0;
    CData/*0:0*/ __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v11;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v11 = 0;
    QData/*54:0*/ __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v20;
    __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v20 = 0;
    CData/*0:0*/ __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v20;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v20 = 0;
    QData/*54:0*/ __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v21;
    __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v21 = 0;
    CData/*0:0*/ __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v21;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v21 = 0;
    QData/*54:0*/ __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v22;
    __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v22 = 0;
    CData/*0:0*/ __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v22;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v22 = 0;
    QData/*54:0*/ __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v23;
    __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v23 = 0;
    CData/*0:0*/ __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v23;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v23 = 0;
    QData/*54:0*/ __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v24;
    __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v24 = 0;
    CData/*0:0*/ __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v24;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v24 = 0;
    QData/*54:0*/ __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v25;
    __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v25 = 0;
    CData/*0:0*/ __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v25;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v25 = 0;
    QData/*54:0*/ __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v26;
    __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v26 = 0;
    CData/*0:0*/ __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v26;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v26 = 0;
    QData/*54:0*/ __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v27;
    __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v27 = 0;
    CData/*0:0*/ __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v27;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v27 = 0;
    QData/*54:0*/ __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v28;
    __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v28 = 0;
    CData/*0:0*/ __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v28;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v28 = 0;
    QData/*54:0*/ __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v29;
    __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v29 = 0;
    CData/*0:0*/ __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v29;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v29 = 0;
    QData/*54:0*/ __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v30;
    __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v30 = 0;
    CData/*0:0*/ __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v30;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v30 = 0;
    QData/*54:0*/ __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v31;
    __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v31 = 0;
    CData/*0:0*/ __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v31;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v31 = 0;
    QData/*54:0*/ __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v32;
    __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v32 = 0;
    CData/*0:0*/ __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v32;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v32 = 0;
    QData/*54:0*/ __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v33;
    __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v33 = 0;
    CData/*0:0*/ __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v33;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v33 = 0;
    QData/*54:0*/ __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v34;
    __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v34 = 0;
    CData/*0:0*/ __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v34;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v34 = 0;
    QData/*54:0*/ __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v35;
    __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v35 = 0;
    CData/*0:0*/ __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v35;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v35 = 0;
    QData/*54:0*/ __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v36;
    __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v36 = 0;
    CData/*0:0*/ __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v36;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v36 = 0;
    QData/*54:0*/ __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v37;
    __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v37 = 0;
    CData/*0:0*/ __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v37;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v37 = 0;
    QData/*54:0*/ __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v38;
    __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v38 = 0;
    CData/*0:0*/ __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v38;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v38 = 0;
    QData/*54:0*/ __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v39;
    __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v39 = 0;
    CData/*0:0*/ __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v39;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v39 = 0;
    CData/*1:0*/ __Vdly__tb__DOT__dut__DOT__key_reset_sync__DOT__sync_ff;
    __Vdly__tb__DOT__dut__DOT__key_reset_sync__DOT__sync_ff = 0;
    CData/*3:0*/ __Vdly__tb__DOT__dut__DOT__key_reset_sync__DOT__reset_count;
    __Vdly__tb__DOT__dut__DOT__key_reset_sync__DOT__reset_count = 0;
    CData/*1:0*/ __Vdly__tb__DOT__dut__DOT__hps_reset_sync__DOT__sync_ff;
    __Vdly__tb__DOT__dut__DOT__hps_reset_sync__DOT__sync_ff = 0;
    CData/*3:0*/ __Vdly__tb__DOT__dut__DOT__hps_reset_sync__DOT__reset_count;
    __Vdly__tb__DOT__dut__DOT__hps_reset_sync__DOT__reset_count = 0;
    // Body
    __Vdly__tb__DOT__dut__DOT__hps_reset_sync__DOT__sync_ff 
        = vlSelf->tb__DOT__dut__DOT__hps_reset_sync__DOT__sync_ff;
    __Vdly__tb__DOT__dut__DOT__key_reset_sync__DOT__sync_ff 
        = vlSelf->tb__DOT__dut__DOT__key_reset_sync__DOT__sync_ff;
    __Vdly__tb__DOT__dut__DOT__hps_reset_sync__DOT__reset_count 
        = vlSelf->tb__DOT__dut__DOT__hps_reset_sync__DOT__reset_count;
    __Vdly__tb__DOT__dut__DOT__key_reset_sync__DOT__reset_count 
        = vlSelf->tb__DOT__dut__DOT__key_reset_sync__DOT__reset_count;
    __Vdly__tb__DOT__dut__DOT__slopes__DOT__current_subap 
        = vlSelf->tb__DOT__dut__DOT__slopes__DOT__current_subap;
    __Vdly__tb__DOT__dut__DOT__accumulator__DOT__count_pixel_h 
        = vlSelf->tb__DOT__dut__DOT__accumulator__DOT__count_pixel_h;
    __Vdly__tb__DOT__dut__DOT__accumulator__DOT__subap_col 
        = vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_col;
    __Vdlyvset__tb__DOT__dut__DOT__reciprocal__DOT__centroids_done_internal__v0 = 0U;
    __Vdlyvset__tb__DOT__dut__DOT__reciprocal__DOT__centroids_done_internal__v4 = 0U;
    __Vdly__tb__DOT__dut__DOT__accumulator__DOT__count_pixel_v 
        = vlSelf->tb__DOT__dut__DOT__accumulator__DOT__count_pixel_v;
    __Vdly__tb__DOT__dut__DOT__streamer__DOT__v_blank_counter 
        = vlSelf->tb__DOT__dut__DOT__streamer__DOT__v_blank_counter;
    __Vdly__tb__DOT__dut__DOT__streamer__DOT__h_blank_counter 
        = vlSelf->tb__DOT__dut__DOT__streamer__DOT__h_blank_counter;
    __Vdly__tb__DOT__dut__DOT__streamer__DOT__row_counter 
        = vlSelf->tb__DOT__dut__DOT__streamer__DOT__row_counter;
    __Vdly__tb__DOT__dut__DOT__streamer__DOT__line_counter 
        = vlSelf->tb__DOT__dut__DOT__streamer__DOT__line_counter;
    __Vdly__tb__DOT__dut__DOT__streamer__DOT__state 
        = vlSelf->tb__DOT__dut__DOT__streamer__DOT__state;
    __Vdly__tb__DOT__dut__DOT__accumulator__DOT__subap_row 
        = vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_row;
    __Vdly__tb__DOT__dut__DOT__subaps_done_accumulator 
        = vlSelf->tb__DOT__dut__DOT__subaps_done_accumulator;
    __Vdlyvset__tb__DOT__dut__DOT__accumulator__DOT__i__v0 = 0U;
    __Vdlyvset__tb__DOT__dut__DOT__reciprocal__DOT__yI_internal__v0 = 0U;
    __Vdlyvset__tb__DOT__dut__DOT__reciprocal__DOT__yI_internal__v4 = 0U;
    __Vdlyvset__tb__DOT__dut__DOT__reciprocal__DOT__xI_internal__v0 = 0U;
    __Vdlyvset__tb__DOT__dut__DOT__reciprocal__DOT__xI_internal__v4 = 0U;
    __Vdly__tb__DOT__dut__DOT__em__DOT__state = vlSelf->tb__DOT__dut__DOT__em__DOT__state;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v0 = 0U;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v10 = 0U;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v11 = 0U;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v20 = 0U;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v21 = 0U;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v22 = 0U;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v23 = 0U;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v24 = 0U;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v25 = 0U;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v26 = 0U;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v27 = 0U;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v28 = 0U;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v29 = 0U;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v30 = 0U;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v31 = 0U;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v32 = 0U;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v33 = 0U;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v34 = 0U;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v35 = 0U;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v36 = 0U;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v37 = 0U;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v38 = 0U;
    __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v39 = 0U;
    __Vdly__tb__DOT__dut__DOT__em__DOT__sub_counter 
        = vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter;
    __Vdly__tb__DOT__dut__DOT__resw_write_idx = vlSelf->tb__DOT__dut__DOT__resw_write_idx;
    __Vdly__tb__DOT__dut__DOT__write_zernike_count 
        = vlSelf->tb__DOT__dut__DOT__write_zernike_count;
    __Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v0 = 0U;
    __Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v1 = 0U;
    __Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v2 = 0U;
    __Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v3 = 0U;
    __Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v4 = 0U;
    __Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v5 = 0U;
    __Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v6 = 0U;
    __Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v7 = 0U;
    __Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v8 = 0U;
    __Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v9 = 0U;
    vlSelf->tb__DOT__dut__DOT__start_sync = ((2U & 
                                              ((IData)(vlSelf->tb__DOT__dut__DOT__start_sync) 
                                               << 1U)) 
                                             | (1U 
                                                & (IData)(vlSelf->tb__DOT__dut__DOT__ctrl_reg_h2f)));
    __Vdly__tb__DOT__dut__DOT__hps_reset_sync__DOT__sync_ff 
        = ((2U & ((IData)(vlSelf->tb__DOT__dut__DOT__hps_reset_sync__DOT__sync_ff) 
                  << 1U)) | (IData)(vlSelf->tb__DOT__hps_reset));
    __Vdly__tb__DOT__dut__DOT__key_reset_sync__DOT__sync_ff 
        = ((2U & ((IData)(vlSelf->tb__DOT__dut__DOT__key_reset_sync__DOT__sync_ff) 
                  << 1U)) | (IData)(vlSelf->tb__DOT__key_reset));
    if (((~ (IData)(vlSelf->tb__DOT__dut__DOT__hps_reset_sync__DOT__sync_d)) 
         & ((IData)(vlSelf->tb__DOT__dut__DOT__hps_reset_sync__DOT__sync_ff) 
            >> 1U))) {
        __Vdly__tb__DOT__dut__DOT__hps_reset_sync__DOT__reset_count = 8U;
    } else if ((0U != (IData)(vlSelf->tb__DOT__dut__DOT__hps_reset_sync__DOT__reset_count))) {
        __Vdly__tb__DOT__dut__DOT__hps_reset_sync__DOT__reset_count 
            = (0xfU & ((IData)(vlSelf->tb__DOT__dut__DOT__hps_reset_sync__DOT__reset_count) 
                       - (IData)(1U)));
    }
    if (((~ (IData)(vlSelf->tb__DOT__dut__DOT__key_reset_sync__DOT__sync_d)) 
         & ((IData)(vlSelf->tb__DOT__dut__DOT__key_reset_sync__DOT__sync_ff) 
            >> 1U))) {
        __Vdly__tb__DOT__dut__DOT__key_reset_sync__DOT__reset_count = 8U;
    } else if ((0U != (IData)(vlSelf->tb__DOT__dut__DOT__key_reset_sync__DOT__reset_count))) {
        __Vdly__tb__DOT__dut__DOT__key_reset_sync__DOT__reset_count 
            = (0xfU & ((IData)(vlSelf->tb__DOT__dut__DOT__key_reset_sync__DOT__reset_count) 
                       - (IData)(1U)));
    }
    if (vlSelf->tb__DOT__dut__DOT__done) {
        __Vdlyvval__tb__DOT__dut__DOT__zernike_out_reg__v0 
            = (0x7ffffffU & vlSelf->tb__DOT__dut__DOT__zernike_out[0U]);
        __Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v0 = 1U;
        __Vdlyvval__tb__DOT__dut__DOT__zernike_out_reg__v1 
            = (0x7ffffffU & ((vlSelf->tb__DOT__dut__DOT__zernike_out[1U] 
                              << 5U) | (vlSelf->tb__DOT__dut__DOT__zernike_out[0U] 
                                        >> 0x1bU)));
        __Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v1 = 1U;
        __Vdlyvval__tb__DOT__dut__DOT__zernike_out_reg__v2 
            = (0x7ffffffU & ((vlSelf->tb__DOT__dut__DOT__zernike_out[2U] 
                              << 0xaU) | (vlSelf->tb__DOT__dut__DOT__zernike_out[1U] 
                                          >> 0x16U)));
        __Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v2 = 1U;
        __Vdlyvval__tb__DOT__dut__DOT__zernike_out_reg__v3 
            = (0x7ffffffU & ((vlSelf->tb__DOT__dut__DOT__zernike_out[3U] 
                              << 0xfU) | (vlSelf->tb__DOT__dut__DOT__zernike_out[2U] 
                                          >> 0x11U)));
        __Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v3 = 1U;
        __Vdlyvval__tb__DOT__dut__DOT__zernike_out_reg__v4 
            = (0x7ffffffU & ((vlSelf->tb__DOT__dut__DOT__zernike_out[4U] 
                              << 0x14U) | (vlSelf->tb__DOT__dut__DOT__zernike_out[3U] 
                                           >> 0xcU)));
        __Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v4 = 1U;
        __Vdlyvval__tb__DOT__dut__DOT__zernike_out_reg__v5 
            = (0x7ffffffU & ((vlSelf->tb__DOT__dut__DOT__zernike_out[5U] 
                              << 0x19U) | (vlSelf->tb__DOT__dut__DOT__zernike_out[4U] 
                                           >> 7U)));
        __Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v5 = 1U;
        __Vdlyvval__tb__DOT__dut__DOT__zernike_out_reg__v6 
            = (0x7ffffffU & (vlSelf->tb__DOT__dut__DOT__zernike_out[5U] 
                             >> 2U));
        __Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v6 = 1U;
        __Vdlyvval__tb__DOT__dut__DOT__zernike_out_reg__v7 
            = (0x7ffffffU & ((vlSelf->tb__DOT__dut__DOT__zernike_out[6U] 
                              << 3U) | (vlSelf->tb__DOT__dut__DOT__zernike_out[5U] 
                                        >> 0x1dU)));
        __Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v7 = 1U;
        __Vdlyvval__tb__DOT__dut__DOT__zernike_out_reg__v8 
            = (0x7ffffffU & ((vlSelf->tb__DOT__dut__DOT__zernike_out[7U] 
                              << 8U) | (vlSelf->tb__DOT__dut__DOT__zernike_out[6U] 
                                        >> 0x18U)));
        __Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v8 = 1U;
        __Vdlyvval__tb__DOT__dut__DOT__zernike_out_reg__v9 
            = (0x7ffffffU & ((vlSelf->tb__DOT__dut__DOT__zernike_out[8U] 
                              << 0xdU) | (vlSelf->tb__DOT__dut__DOT__zernike_out[7U] 
                                          >> 0x13U)));
        __Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v9 = 1U;
    }
    if ((1U & (~ (IData)(vlSelf->tb__DOT__dut__DOT__reset)))) {
        vlSelf->tb__DOT__dut__DOT__y_centroid_out = vlSelf->tb__DOT__dut__DOT__y_centroid;
        vlSelf->tb__DOT__dut__DOT__x_centroid_out = vlSelf->tb__DOT__dut__DOT__x_centroid;
        vlSelf->tb__DOT__dut__DOT__x_slopes_out = vlSelf->tb__DOT__dut__DOT__x_slopes;
        vlSelf->tb__DOT__dut__DOT__y_slopes_out = vlSelf->tb__DOT__dut__DOT__y_slopes;
    }
    vlSelf->tb__DOT__dut__DOT__resw_start = ((1U & 
                                              (~ (IData)(vlSelf->tb__DOT__dut__DOT__reset))) 
                                             && (IData)(vlSelf->tb__DOT__dut__DOT__new_subapeture));
    vlSelf->tb__DOT__dut__DOT__hps_reset_sync__DOT__reset_count 
        = __Vdly__tb__DOT__dut__DOT__hps_reset_sync__DOT__reset_count;
    vlSelf->tb__DOT__dut__DOT__key_reset_sync__DOT__reset_count 
        = __Vdly__tb__DOT__dut__DOT__key_reset_sync__DOT__reset_count;
    if (vlSelf->tb__DOT__dut__DOT__reset) {
        __Vdly__tb__DOT__dut__DOT__accumulator__DOT__count_pixel_h = 0U;
        __Vdly__tb__DOT__dut__DOT__accumulator__DOT__subap_col = 0U;
        __Vdlyvset__tb__DOT__dut__DOT__reciprocal__DOT__centroids_done_internal__v0 = 1U;
        __Vdly__tb__DOT__dut__DOT__accumulator__DOT__count_pixel_v = 0U;
        __Vdly__tb__DOT__dut__DOT__accumulator__DOT__subap_row = 0U;
        __Vdly__tb__DOT__dut__DOT__subaps_done_accumulator = 0U;
        vlSelf->tb__DOT__dut__DOT__subaps_done_accumulator 
            = __Vdly__tb__DOT__dut__DOT__subaps_done_accumulator;
        vlSelf->tb__DOT__dut__DOT__accumulator__DOT__j = 0U;
        while (VL_GTS_III(32, 0x100U, vlSelf->tb__DOT__dut__DOT__accumulator__DOT__j)) {
            vlSelf->tb__DOT__dut__DOT__accumulator__DOT__i[(0xffU 
                                                            & vlSelf->tb__DOT__dut__DOT__accumulator__DOT__j)] = 0U;
            vlSelf->tb__DOT__dut__DOT__accumulator__DOT__x_i[(0xffU 
                                                              & vlSelf->tb__DOT__dut__DOT__accumulator__DOT__j)] = 0U;
            vlSelf->tb__DOT__dut__DOT__accumulator__DOT__y_i[(0xffU 
                                                              & vlSelf->tb__DOT__dut__DOT__accumulator__DOT__j)] = 0U;
            vlSelf->tb__DOT__dut__DOT__accumulator__DOT__j 
                = ((IData)(1U) + vlSelf->tb__DOT__dut__DOT__accumulator__DOT__j);
        }
        __Vdlyvset__tb__DOT__dut__DOT__reciprocal__DOT__yI_internal__v0 = 1U;
    } else {
        if (vlSelf->tb__DOT__dut__DOT__streamer_valid) {
            __Vdly__tb__DOT__dut__DOT__accumulator__DOT__count_pixel_h 
                = ((IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__last_pixel_h)
                    ? 0U : (0xfU & ((IData)(1U) + (IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__count_pixel_h))));
            if ((0xfU == (IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__count_pixel_h))) {
                __Vdly__tb__DOT__dut__DOT__accumulator__DOT__subap_col 
                    = ((IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__last_subap_col)
                        ? 0U : (0xfU & ((IData)(1U) 
                                        + (IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_col))));
            }
            if (((0xfU == (IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__count_pixel_h)) 
                 & (0xfU == (IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_col)))) {
                __Vdly__tb__DOT__dut__DOT__accumulator__DOT__count_pixel_v 
                    = ((IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__last_pixel_v)
                        ? 0U : (0xfU & ((IData)(1U) 
                                        + (IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__count_pixel_v))));
                if ((0xfU == (IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__count_pixel_v))) {
                    __Vdly__tb__DOT__dut__DOT__accumulator__DOT__subap_row 
                        = ((IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__last_subap_row)
                            ? 0U : (0xfU & ((IData)(1U) 
                                            + (IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_row))));
                }
            }
            __Vdlyvval__tb__DOT__dut__DOT__accumulator__DOT__i__v0 
                = (0xffffU & (vlSelf->tb__DOT__dut__DOT__accumulator__DOT__i
                              [vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_idx] 
                              + (IData)(vlSelf->tb__DOT__dut__DOT__streamer_data)));
            __Vdlyvset__tb__DOT__dut__DOT__accumulator__DOT__i__v0 = 1U;
            __Vdlyvdim0__tb__DOT__dut__DOT__accumulator__DOT__i__v0 
                = vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_idx;
            __Vdlyvval__tb__DOT__dut__DOT__accumulator__DOT__x_i__v0 
                = (0xfffffU & (vlSelf->tb__DOT__dut__DOT__accumulator__DOT__x_i
                               [vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_idx] 
                               + ((IData)(vlSelf->tb__DOT__dut__DOT__streamer_data) 
                                  * (IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__count_pixel_h))));
            __Vdlyvdim0__tb__DOT__dut__DOT__accumulator__DOT__x_i__v0 
                = vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_idx;
            __Vdlyvval__tb__DOT__dut__DOT__accumulator__DOT__y_i__v0 
                = (0xfffffU & (vlSelf->tb__DOT__dut__DOT__accumulator__DOT__y_i
                               [vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_idx] 
                               + ((IData)(vlSelf->tb__DOT__dut__DOT__streamer_data) 
                                  * (IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__count_pixel_v))));
            __Vdlyvdim0__tb__DOT__dut__DOT__accumulator__DOT__y_i__v0 
                = vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_idx;
        }
        __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__centroids_done_internal__v4 
            = vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__centroids_done_internal
            [2U];
        __Vdlyvset__tb__DOT__dut__DOT__reciprocal__DOT__centroids_done_internal__v4 = 1U;
        __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__centroids_done_internal__v5 
            = vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__centroids_done_internal
            [1U];
        __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__centroids_done_internal__v6 
            = vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__centroids_done_internal
            [0U];
        __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__centroids_done_internal__v7 
            = vlSelf->tb__DOT__dut__DOT__subaps_done_accumulator;
        if (vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_done_delay) {
            __Vdly__tb__DOT__dut__DOT__subaps_done_accumulator 
                = (0xffU & ((IData)(1U) + (IData)(vlSelf->tb__DOT__dut__DOT__subaps_done_accumulator)));
        }
        vlSelf->tb__DOT__dut__DOT__subaps_done_accumulator 
            = __Vdly__tb__DOT__dut__DOT__subaps_done_accumulator;
        __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__yI_internal__v4 
            = vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__yI_internal
            [2U];
        __Vdlyvset__tb__DOT__dut__DOT__reciprocal__DOT__yI_internal__v4 = 1U;
        __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__yI_internal__v5 
            = vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__yI_internal
            [1U];
        __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__yI_internal__v6 
            = vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__yI_internal
            [0U];
        __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__yI_internal__v7 
            = vlSelf->tb__DOT__dut__DOT__yI_accumulator;
    }
    if (__Vdlyvset__tb__DOT__dut__DOT__reciprocal__DOT__yI_internal__v0) {
        vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__yI_internal[3U] = 0U;
        vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__yI_internal[2U] = 0U;
        vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__yI_internal[1U] = 0U;
        vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__yI_internal[0U] = 0U;
    }
    if (__Vdlyvset__tb__DOT__dut__DOT__reciprocal__DOT__yI_internal__v4) {
        vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__yI_internal[3U] 
            = __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__yI_internal__v4;
        vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__yI_internal[2U] 
            = __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__yI_internal__v5;
        vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__yI_internal[1U] 
            = __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__yI_internal__v6;
        vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__yI_internal[0U] 
            = __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__yI_internal__v7;
    }
    if (vlSelf->tb__DOT__dut__DOT__reset) {
        __Vdlyvset__tb__DOT__dut__DOT__reciprocal__DOT__xI_internal__v0 = 1U;
    } else {
        __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__xI_internal__v4 
            = vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__xI_internal
            [2U];
        __Vdlyvset__tb__DOT__dut__DOT__reciprocal__DOT__xI_internal__v4 = 1U;
        __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__xI_internal__v5 
            = vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__xI_internal
            [1U];
        __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__xI_internal__v6 
            = vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__xI_internal
            [0U];
        __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__xI_internal__v7 
            = vlSelf->tb__DOT__dut__DOT__xI_accumulator;
    }
    if (__Vdlyvset__tb__DOT__dut__DOT__reciprocal__DOT__xI_internal__v0) {
        vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__xI_internal[3U] = 0U;
        vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__xI_internal[2U] = 0U;
        vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__xI_internal[1U] = 0U;
        vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__xI_internal[0U] = 0U;
    }
    if (__Vdlyvset__tb__DOT__dut__DOT__reciprocal__DOT__xI_internal__v4) {
        vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__xI_internal[3U] 
            = __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__xI_internal__v4;
        vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__xI_internal[2U] 
            = __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__xI_internal__v5;
        vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__xI_internal[1U] 
            = __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__xI_internal__v6;
        vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__xI_internal[0U] 
            = __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__xI_internal__v7;
    }
    if (vlSelf->tb__DOT__dut__DOT__reset) {
        vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg2_x0_u27 = 0U;
        vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg3_x1_q0_27 = 0U;
        vlSelf->tb__DOT__dut__DOT__full_frame_complete_accumulator = 0U;
        vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_row 
            = __Vdly__tb__DOT__dut__DOT__accumulator__DOT__subap_row;
        vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_col 
            = __Vdly__tb__DOT__dut__DOT__accumulator__DOT__subap_col;
        vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg3_shift_left = 0U;
        vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg3_a_q1_26 = 0U;
        vlSelf->tb__DOT__dut__DOT__rI_reciprocal = 0U;
        __Vdly__tb__DOT__dut__DOT__resw_write_idx = 0U;
        vlSelf->tb__DOT__dut__DOT__result_write = 0U;
        vlSelf->tb__DOT__dut__DOT__result_wdata = 0U;
        vlSelf->tb__DOT__dut__DOT__result_addr = 0U;
        vlSelf->tb__DOT__dut__DOT__write_zernike_latch = 0U;
    } else {
        vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg2_x0_u27 
            = (0x7fff800U & ((((0U == (0x1fU & VL_SHIFTL_III(12,12,32, 
                                                             (0xffU 
                                                              & ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                 >> 7U)), 4U)))
                                ? 0U : (Vtb__ConstPool__CONST_h29f91db3_0[
                                        (((IData)(0xfU) 
                                          + (0xfffU 
                                             & VL_SHIFTL_III(12,12,32, 
                                                             (0xffU 
                                                              & ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                                 >> 7U)), 4U))) 
                                         >> 5U)] << 
                                        ((IData)(0x20U) 
                                         - (0x1fU & 
                                            VL_SHIFTL_III(12,12,32, 
                                                          (0xffU 
                                                           & ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                              >> 7U)), 4U))))) 
                              | (Vtb__ConstPool__CONST_h29f91db3_0[
                                 (0x7fU & (VL_SHIFTL_III(12,12,32, 
                                                         (0xffU 
                                                          & ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                             >> 7U)), 4U) 
                                           >> 5U))] 
                                 >> (0x1fU & VL_SHIFTL_III(12,12,32, 
                                                           (0xffU 
                                                            & ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
                                                               >> 7U)), 4U)))) 
                             << 0xbU));
        vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg3_x1_q0_27 
            = ((0x7ffffffU < (0x1fffffffU & (IData)(
                                                    (0x1fffffffULL 
                                                     & ((0x2000000ULL 
                                                         + vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__u_newton_step_0__DOT__prod_mul) 
                                                        >> 0x1aU)))))
                ? 0x7ffffffU : (0x7ffffffU & (IData)(
                                                     (0x1fffffffULL 
                                                      & ((0x2000000ULL 
                                                          + vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__u_newton_step_0__DOT__prod_mul) 
                                                         >> 0x1aU)))));
        if (vlSelf->tb__DOT__dut__DOT__streamer_valid) {
            vlSelf->tb__DOT__dut__DOT__full_frame_complete_accumulator = 0U;
            if (((0xfU == (IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__count_pixel_h)) 
                 & (0xfU == (IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_col)))) {
                if ((0xfU == (IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__count_pixel_v))) {
                    if ((0xfU == (IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_row))) {
                        vlSelf->tb__DOT__dut__DOT__full_frame_complete_accumulator = 1U;
                    }
                }
            }
        }
        vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_row 
            = __Vdly__tb__DOT__dut__DOT__accumulator__DOT__subap_row;
        vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_col 
            = __Vdly__tb__DOT__dut__DOT__accumulator__DOT__subap_col;
        vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg3_shift_left 
            = vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg2_shift_left;
        vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg3_a_q1_26 
            = vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg2_a_q1_26;
        vlSelf->tb__DOT__dut__DOT__rI_reciprocal = 
            ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg3_v_is_zero)
              ? 0x7ffffffU : ((0x7ffffffU < vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__out_q0_27_ext)
                               ? 0x7ffffffU : (0x7ffffffU 
                                               & vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__out_q0_27_ext)));
        if (vlSelf->tb__DOT__dut__DOT__done) {
            vlSelf->tb__DOT__dut__DOT__write_zernike_latch = 1U;
        } else if ((5U == (IData)(vlSelf->tb__DOT__dut__DOT__resw_state))) {
            vlSelf->tb__DOT__dut__DOT__write_zernike_latch = 0U;
        }
        if ((0x10U & (IData)(vlSelf->tb__DOT__dut__DOT__resw_state))) {
            vlSelf->tb__DOT__dut__DOT__result_wdata = 0U;
            vlSelf->tb__DOT__dut__DOT__result_addr = 0U;
            vlSelf->tb__DOT__dut__DOT__result_write = 0U;
            __Vdly__tb__DOT__dut__DOT__write_zernike_count = 0U;
        } else if ((8U & (IData)(vlSelf->tb__DOT__dut__DOT__resw_state))) {
            vlSelf->tb__DOT__dut__DOT__result_wdata = 0U;
            vlSelf->tb__DOT__dut__DOT__result_addr = 0U;
            vlSelf->tb__DOT__dut__DOT__result_write = 0U;
            __Vdly__tb__DOT__dut__DOT__write_zernike_count = 0U;
        } else if ((4U & (IData)(vlSelf->tb__DOT__dut__DOT__resw_state))) {
            if ((2U & (IData)(vlSelf->tb__DOT__dut__DOT__resw_state))) {
                if ((1U & (IData)(vlSelf->tb__DOT__dut__DOT__resw_state))) {
                    vlSelf->tb__DOT__dut__DOT__result_wdata = 0U;
                    vlSelf->tb__DOT__dut__DOT__result_addr = 0U;
                    vlSelf->tb__DOT__dut__DOT__result_write = 0U;
                    __Vdly__tb__DOT__dut__DOT__write_zernike_count = 0U;
                } else {
                    __Vdly__tb__DOT__dut__DOT__resw_write_idx 
                        = (0x7ffU & ((IData)(1U) + (IData)(vlSelf->tb__DOT__dut__DOT__resw_write_idx)));
                    vlSelf->tb__DOT__dut__DOT__result_write = 0U;
                }
            } else if ((1U & (IData)(vlSelf->tb__DOT__dut__DOT__resw_state))) {
                __Vdly__tb__DOT__dut__DOT__write_zernike_count 
                    = (0x7ffU & ((IData)(1U) + (IData)(vlSelf->tb__DOT__dut__DOT__write_zernike_count)));
                vlSelf->tb__DOT__dut__DOT__result_wdata 
                    = ((9U >= (0xfU & (IData)(vlSelf->tb__DOT__dut__DOT__write_zernike_count)))
                        ? vlSelf->tb__DOT__dut__DOT__zernike_out_reg
                       [(0xfU & (IData)(vlSelf->tb__DOT__dut__DOT__write_zernike_count))]
                        : 0U);
                vlSelf->tb__DOT__dut__DOT__result_write = 1U;
                vlSelf->tb__DOT__dut__DOT__result_addr 
                    = (0x7ffU & ((IData)(0x400U) + (IData)(vlSelf->tb__DOT__dut__DOT__write_zernike_count)));
            } else {
                vlSelf->tb__DOT__dut__DOT__result_wdata 
                    = vlSelf->tb__DOT__dut__DOT__y_centroid;
                vlSelf->tb__DOT__dut__DOT__result_addr 
                    = (0x7ffU & ((IData)(0x300U) + (IData)(vlSelf->tb__DOT__dut__DOT__resw_write_idx)));
                vlSelf->tb__DOT__dut__DOT__result_write = 1U;
            }
        } else if ((2U & (IData)(vlSelf->tb__DOT__dut__DOT__resw_state))) {
            if ((1U & (IData)(vlSelf->tb__DOT__dut__DOT__resw_state))) {
                vlSelf->tb__DOT__dut__DOT__result_wdata 
                    = vlSelf->tb__DOT__dut__DOT__x_centroid;
                vlSelf->tb__DOT__dut__DOT__result_addr 
                    = (0x7ffU & ((IData)(0x200U) + (IData)(vlSelf->tb__DOT__dut__DOT__resw_write_idx)));
                vlSelf->tb__DOT__dut__DOT__result_write = 1U;
            } else {
                vlSelf->tb__DOT__dut__DOT__result_wdata 
                    = vlSelf->tb__DOT__dut__DOT__y_slopes;
                vlSelf->tb__DOT__dut__DOT__result_addr 
                    = (0x7ffU & ((IData)(0x100U) + (IData)(vlSelf->tb__DOT__dut__DOT__resw_write_idx)));
                vlSelf->tb__DOT__dut__DOT__result_write = 1U;
            }
        } else if ((1U & (IData)(vlSelf->tb__DOT__dut__DOT__resw_state))) {
            vlSelf->tb__DOT__dut__DOT__result_wdata 
                = vlSelf->tb__DOT__dut__DOT__x_slopes;
            vlSelf->tb__DOT__dut__DOT__result_addr 
                = vlSelf->tb__DOT__dut__DOT__resw_write_idx;
            vlSelf->tb__DOT__dut__DOT__result_write = 1U;
        } else {
            vlSelf->tb__DOT__dut__DOT__result_wdata = 0U;
            vlSelf->tb__DOT__dut__DOT__result_addr = 0U;
            vlSelf->tb__DOT__dut__DOT__result_write = 0U;
            __Vdly__tb__DOT__dut__DOT__write_zernike_count = 0U;
        }
    }
    vlSelf->tb__DOT__dut__DOT__resw_write_idx = __Vdly__tb__DOT__dut__DOT__resw_write_idx;
    vlSelf->tb__DOT__dut__DOT__write_zernike_count 
        = __Vdly__tb__DOT__dut__DOT__write_zernike_count;
    if (__Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v0) {
        vlSelf->tb__DOT__dut__DOT__zernike_out_reg[0U] 
            = __Vdlyvval__tb__DOT__dut__DOT__zernike_out_reg__v0;
    }
    if (__Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v1) {
        vlSelf->tb__DOT__dut__DOT__zernike_out_reg[1U] 
            = __Vdlyvval__tb__DOT__dut__DOT__zernike_out_reg__v1;
    }
    if (__Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v2) {
        vlSelf->tb__DOT__dut__DOT__zernike_out_reg[2U] 
            = __Vdlyvval__tb__DOT__dut__DOT__zernike_out_reg__v2;
    }
    if (__Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v3) {
        vlSelf->tb__DOT__dut__DOT__zernike_out_reg[3U] 
            = __Vdlyvval__tb__DOT__dut__DOT__zernike_out_reg__v3;
    }
    if (__Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v4) {
        vlSelf->tb__DOT__dut__DOT__zernike_out_reg[4U] 
            = __Vdlyvval__tb__DOT__dut__DOT__zernike_out_reg__v4;
    }
    if (__Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v5) {
        vlSelf->tb__DOT__dut__DOT__zernike_out_reg[5U] 
            = __Vdlyvval__tb__DOT__dut__DOT__zernike_out_reg__v5;
    }
    if (__Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v6) {
        vlSelf->tb__DOT__dut__DOT__zernike_out_reg[6U] 
            = __Vdlyvval__tb__DOT__dut__DOT__zernike_out_reg__v6;
    }
    if (__Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v7) {
        vlSelf->tb__DOT__dut__DOT__zernike_out_reg[7U] 
            = __Vdlyvval__tb__DOT__dut__DOT__zernike_out_reg__v7;
    }
    if (__Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v8) {
        vlSelf->tb__DOT__dut__DOT__zernike_out_reg[8U] 
            = __Vdlyvval__tb__DOT__dut__DOT__zernike_out_reg__v8;
    }
    if (__Vdlyvset__tb__DOT__dut__DOT__zernike_out_reg__v9) {
        vlSelf->tb__DOT__dut__DOT__zernike_out_reg[9U] 
            = __Vdlyvval__tb__DOT__dut__DOT__zernike_out_reg__v9;
    }
    vlSelf->tb__DOT__dut__DOT__hps_reset_sync__DOT__sync_d 
        = (1U & ((IData)(vlSelf->tb__DOT__dut__DOT__hps_reset_sync__DOT__sync_ff) 
                 >> 1U));
    vlSelf->tb__DOT__dut__DOT__key_reset_sync__DOT__sync_d 
        = (1U & ((IData)(vlSelf->tb__DOT__dut__DOT__key_reset_sync__DOT__sync_ff) 
                 >> 1U));
    vlSelf->tb__DOT__dut__DOT__accumulator__DOT__last_subap_row 
        = (0xfU == (IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_row));
    vlSelf->tb__DOT__dut__DOT__accumulator__DOT__last_subap_col 
        = (0xfU == (IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_col));
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
    vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg3_v_is_zero 
        = ((1U & (~ (IData)(vlSelf->tb__DOT__dut__DOT__reset))) 
           && (IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg2_v_is_zero));
    vlSelf->tb__DOT__dut__DOT__hps_reset_sync__DOT__sync_ff 
        = __Vdly__tb__DOT__dut__DOT__hps_reset_sync__DOT__sync_ff;
    vlSelf->tb__DOT__dut__DOT__key_reset_sync__DOT__sync_ff 
        = __Vdly__tb__DOT__dut__DOT__key_reset_sync__DOT__sync_ff;
    if (vlSelf->tb__DOT__dut__DOT__reset) {
        __Vdly__tb__DOT__dut__DOT__streamer__DOT__state = 0U;
        __Vdly__tb__DOT__dut__DOT__streamer__DOT__line_counter = 0U;
        __Vdly__tb__DOT__dut__DOT__streamer__DOT__row_counter = 0U;
        __Vdly__tb__DOT__dut__DOT__streamer__DOT__h_blank_counter = 0U;
        __Vdly__tb__DOT__dut__DOT__streamer__DOT__v_blank_counter = 0U;
        vlSelf->tb__DOT__dut__DOT__streamer_lv = 0U;
        vlSelf->tb__DOT__dut__DOT__streamer_fv = 0U;
        vlSelf->tb__DOT__dut__DOT__frame_complete = 0U;
        vlSelf->tb__DOT__dut__DOT__streamer__DOT__state 
            = __Vdly__tb__DOT__dut__DOT__streamer__DOT__state;
        vlSelf->tb__DOT__dut__DOT__streamer__DOT__line_counter 
            = __Vdly__tb__DOT__dut__DOT__streamer__DOT__line_counter;
        vlSelf->tb__DOT__dut__DOT__streamer__DOT__row_counter 
            = __Vdly__tb__DOT__dut__DOT__streamer__DOT__row_counter;
        vlSelf->tb__DOT__dut__DOT__streamer__DOT__h_blank_counter 
            = __Vdly__tb__DOT__dut__DOT__streamer__DOT__h_blank_counter;
        vlSelf->tb__DOT__dut__DOT__streamer__DOT__v_blank_counter 
            = __Vdly__tb__DOT__dut__DOT__streamer__DOT__v_blank_counter;
        vlSelf->tb__DOT__dut__DOT__yI_accumulator = 0U;
        vlSelf->tb__DOT__dut__DOT__xI_accumulator = 0U;
    } else {
        if ((2U & (IData)(vlSelf->tb__DOT__dut__DOT__streamer__DOT__state))) {
            if ((1U & (IData)(vlSelf->tb__DOT__dut__DOT__streamer__DOT__state))) {
                vlSelf->tb__DOT__dut__DOT__streamer_fv = 0U;
                vlSelf->tb__DOT__dut__DOT__streamer_lv = 0U;
                __Vdly__tb__DOT__dut__DOT__streamer__DOT__line_counter = 0U;
                __Vdly__tb__DOT__dut__DOT__streamer__DOT__row_counter = 0U;
                __Vdly__tb__DOT__dut__DOT__streamer__DOT__v_blank_counter 
                    = (0xffU & ((IData)(1U) + (IData)(vlSelf->tb__DOT__dut__DOT__streamer__DOT__v_blank_counter)));
                if ((0x97U == (IData)(vlSelf->tb__DOT__dut__DOT__streamer__DOT__v_blank_counter))) {
                    __Vdly__tb__DOT__dut__DOT__streamer__DOT__v_blank_counter = 0U;
                    __Vdly__tb__DOT__dut__DOT__streamer__DOT__state = 0U;
                }
            } else {
                vlSelf->tb__DOT__dut__DOT__streamer_lv = 0U;
                __Vdly__tb__DOT__dut__DOT__streamer__DOT__line_counter = 0U;
                __Vdly__tb__DOT__dut__DOT__streamer__DOT__h_blank_counter 
                    = (3U & ((IData)(1U) + (IData)(vlSelf->tb__DOT__dut__DOT__streamer__DOT__h_blank_counter)));
                if ((3U == (IData)(vlSelf->tb__DOT__dut__DOT__streamer__DOT__h_blank_counter))) {
                    __Vdly__tb__DOT__dut__DOT__streamer__DOT__row_counter 
                        = (0xffU & ((IData)(1U) + (IData)(vlSelf->tb__DOT__dut__DOT__streamer__DOT__row_counter)));
                    __Vdly__tb__DOT__dut__DOT__streamer__DOT__h_blank_counter = 0U;
                    __Vdly__tb__DOT__dut__DOT__streamer__DOT__state = 1U;
                }
            }
        } else if ((1U & (IData)(vlSelf->tb__DOT__dut__DOT__streamer__DOT__state))) {
            vlSelf->tb__DOT__dut__DOT__streamer_data 
                = vlSelf->tb__DOT__dut__DOT__streamer__DOT__mem
                [(0xffffU & (VL_SHIFTL_III(16,32,32, (IData)(vlSelf->tb__DOT__dut__DOT__streamer__DOT__row_counter), 8U) 
                             + (IData)(vlSelf->tb__DOT__dut__DOT__streamer__DOT__line_counter)))];
            vlSelf->tb__DOT__dut__DOT__streamer_lv = 1U;
            __Vdly__tb__DOT__dut__DOT__streamer__DOT__line_counter 
                = (0xffU & ((IData)(1U) + (IData)(vlSelf->tb__DOT__dut__DOT__streamer__DOT__line_counter)));
            if (((0xffU == (IData)(vlSelf->tb__DOT__dut__DOT__streamer__DOT__line_counter)) 
                 & (0xffU == (IData)(vlSelf->tb__DOT__dut__DOT__streamer__DOT__row_counter)))) {
                __Vdly__tb__DOT__dut__DOT__streamer__DOT__state = 3U;
                vlSelf->tb__DOT__dut__DOT__frame_complete = 1U;
            } else if ((0xffU == (IData)(vlSelf->tb__DOT__dut__DOT__streamer__DOT__line_counter))) {
                __Vdly__tb__DOT__dut__DOT__streamer__DOT__state = 2U;
            }
        } else {
            vlSelf->tb__DOT__dut__DOT__streamer_fv = 1U;
            __Vdly__tb__DOT__dut__DOT__streamer__DOT__line_counter = 0U;
            __Vdly__tb__DOT__dut__DOT__streamer__DOT__row_counter = 0U;
            __Vdly__tb__DOT__dut__DOT__streamer__DOT__state = 1U;
        }
        vlSelf->tb__DOT__dut__DOT__streamer__DOT__state 
            = __Vdly__tb__DOT__dut__DOT__streamer__DOT__state;
        vlSelf->tb__DOT__dut__DOT__streamer__DOT__line_counter 
            = __Vdly__tb__DOT__dut__DOT__streamer__DOT__line_counter;
        vlSelf->tb__DOT__dut__DOT__streamer__DOT__row_counter 
            = __Vdly__tb__DOT__dut__DOT__streamer__DOT__row_counter;
        vlSelf->tb__DOT__dut__DOT__streamer__DOT__h_blank_counter 
            = __Vdly__tb__DOT__dut__DOT__streamer__DOT__h_blank_counter;
        vlSelf->tb__DOT__dut__DOT__streamer__DOT__v_blank_counter 
            = __Vdly__tb__DOT__dut__DOT__streamer__DOT__v_blank_counter;
        if (vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_done_delay) {
            vlSelf->tb__DOT__dut__DOT__yI_accumulator 
                = vlSelf->tb__DOT__dut__DOT__accumulator__DOT__y_i
                [vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_idx_delay];
            vlSelf->tb__DOT__dut__DOT__xI_accumulator 
                = vlSelf->tb__DOT__dut__DOT__accumulator__DOT__x_i
                [vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_idx_delay];
        }
    }
    vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__x2_q0_27 
        = ((0x7ffffffU < (0x1fffffffU & (IData)((0x1fffffffULL 
                                                 & ((0x2000000ULL 
                                                     + vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__u_newton_step_1__DOT__prod_mul) 
                                                    >> 0x1aU)))))
            ? 0x7ffffffU : (0x7ffffffU & (IData)((0x1fffffffULL 
                                                  & ((0x2000000ULL 
                                                      + vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__u_newton_step_1__DOT__prod_mul) 
                                                     >> 0x1aU)))));
    if (vlSelf->tb__DOT__dut__DOT__reset) {
        vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg2_shift_left = 0U;
        vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg2_a_q1_26 = 0U;
        vlSelf->tb__DOT__dut__DOT__y_centroid = 0U;
        vlSelf->tb__DOT__dut__DOT__x_centroid = 0U;
        vlSelf->tb__DOT__dut__DOT__resw_state = 0U;
        vlSelf->tb__DOT__dut__DOT__y_slopes = 0U;
    } else {
        vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg2_shift_left 
            = vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__shift_left;
        vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg2_a_q1_26 
            = ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15) 
               << 0xbU);
        vlSelf->tb__DOT__dut__DOT__y_centroid = (0x7ffffffU 
                                                 & (IData)(
                                                           (vlSelf->tb__DOT__dut__DOT__slopes__DOT__y_mult__DOT__mult_out 
                                                            >> 4U)));
        vlSelf->tb__DOT__dut__DOT__x_centroid = (0x7ffffffU 
                                                 & (IData)(
                                                           (vlSelf->tb__DOT__dut__DOT__slopes__DOT__x_mult__DOT__mult_out 
                                                            >> 4U)));
        vlSelf->tb__DOT__dut__DOT__resw_state = vlSelf->tb__DOT__dut__DOT__resw_next_state;
        vlSelf->tb__DOT__dut__DOT__y_slopes = (0x7ffffffU 
                                               & (VL_EXTENDS_II(27,27, 
                                                                (0x7ffffffU 
                                                                 & (IData)(
                                                                           (vlSelf->tb__DOT__dut__DOT__slopes__DOT__y_mult__DOT__mult_out 
                                                                            >> 4U)))) 
                                                  - (IData)(0x3c00000U)));
    }
    vlSelf->tb__DOT__dut__DOT__slopes__DOT__y_mult__DOT__mult_out 
        = (0x7fffffffffffULL & VL_MULS_QQQ(47, (0x7fffffffffffULL 
                                                & VL_EXTENDS_QI(47,20, 
                                                                vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__yI_internal
                                                                [3U])), 
                                           (0x7fffffffffffULL 
                                            & VL_EXTENDS_QI(47,27, vlSelf->tb__DOT__dut__DOT__rI_reciprocal))));
    vlSelf->tb__DOT__dut__DOT__x_slopes = ((IData)(vlSelf->tb__DOT__dut__DOT__reset)
                                            ? 0U : 
                                           (0x7ffffffU 
                                            & (VL_EXTENDS_II(27,27, 
                                                             (0x7ffffffU 
                                                              & (IData)(
                                                                        (vlSelf->tb__DOT__dut__DOT__slopes__DOT__x_mult__DOT__mult_out 
                                                                         >> 4U)))) 
                                               - (IData)(0x3c00000U))));
    vlSelf->tb__DOT__dut__DOT__slopes__DOT__x_mult__DOT__mult_out 
        = (0x7fffffffffffULL & VL_MULS_QQQ(47, (0x7fffffffffffULL 
                                                & VL_EXTENDS_QI(47,20, 
                                                                vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__xI_internal
                                                                [3U])), 
                                           (0x7fffffffffffULL 
                                            & VL_EXTENDS_QI(47,27, vlSelf->tb__DOT__dut__DOT__rI_reciprocal))));
    if (vlSelf->tb__DOT__dut__DOT__reset) {
        vlSelf->tb__DOT__dut__DOT__em__DOT__i = 0xaU;
        __Vdly__tb__DOT__dut__DOT__em__DOT__state = 0U;
        __Vdly__tb__DOT__dut__DOT__em__DOT__sub_counter = 0U;
        vlSelf->tb__DOT__dut__DOT__done = 0U;
        vlSelf->tb__DOT__dut__DOT__zernike_out[0U] 
            = Vtb__ConstPool__CONST_h82fc43e2_0[0U];
        vlSelf->tb__DOT__dut__DOT__zernike_out[1U] 
            = Vtb__ConstPool__CONST_h82fc43e2_0[1U];
        vlSelf->tb__DOT__dut__DOT__zernike_out[2U] 
            = Vtb__ConstPool__CONST_h82fc43e2_0[2U];
        vlSelf->tb__DOT__dut__DOT__zernike_out[3U] 
            = Vtb__ConstPool__CONST_h82fc43e2_0[3U];
        vlSelf->tb__DOT__dut__DOT__zernike_out[4U] 
            = Vtb__ConstPool__CONST_h82fc43e2_0[4U];
        vlSelf->tb__DOT__dut__DOT__zernike_out[5U] 
            = Vtb__ConstPool__CONST_h82fc43e2_0[5U];
        vlSelf->tb__DOT__dut__DOT__zernike_out[6U] 
            = Vtb__ConstPool__CONST_h82fc43e2_0[6U];
        vlSelf->tb__DOT__dut__DOT__zernike_out[7U] 
            = Vtb__ConstPool__CONST_h82fc43e2_0[7U];
        vlSelf->tb__DOT__dut__DOT__zernike_out[8U] 
            = Vtb__ConstPool__CONST_h82fc43e2_0[8U];
        __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v0 = 1U;
    } else {
        vlSelf->tb__DOT__dut__DOT__done = 0U;
        if (vlSelf->tb__DOT__dut__DOT__em__DOT__state) {
            if (vlSelf->tb__DOT__dut__DOT__em__DOT__state) {
                vlSelf->tb__DOT__dut__DOT__em__DOT__i = 0xaU;
                vlSelf->tb__DOT__dut__DOT__done = 1U;
                __Vdly__tb__DOT__dut__DOT__em__DOT__sub_counter = 0U;
                __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v10 = 1U;
                __Vdly__tb__DOT__dut__DOT__em__DOT__state = 0U;
                __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v11 = 1U;
            } else {
                __Vdly__tb__DOT__dut__DOT__em__DOT__state = 0U;
            }
        } else if (vlSelf->tb__DOT__dut__DOT__subap_valid) {
            if ((0x83U == (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter))) {
                vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__1 
                    = vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next
                    [0U];
                vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h017c4131__0 
                    = (0x7ffffffU & (IData)((vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next
                                             [0U] >> 0x11U)));
                vlSelf->tb__DOT__dut__DOT__em__DOT__i = 0xaU;
                __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v20 
                    = vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__1;
                __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v20 = 1U;
                vlSelf->tb__DOT__dut__DOT__zernike_out[0U] 
                    = ((0xf8000000U & vlSelf->tb__DOT__dut__DOT__zernike_out[0U]) 
                       | vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h017c4131__0);
                __Vdly__tb__DOT__dut__DOT__em__DOT__state = 1U;
                vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__1 
                    = vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next
                    [1U];
                vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h017c4131__0 
                    = (0x7ffffffU & (IData)((vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next
                                             [1U] >> 0x11U)));
                __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v21 
                    = vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__1;
                __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v21 = 1U;
                vlSelf->tb__DOT__dut__DOT__zernike_out[0U] 
                    = ((0x7ffffffU & vlSelf->tb__DOT__dut__DOT__zernike_out[0U]) 
                       | (vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h017c4131__0 
                          << 0x1bU));
                vlSelf->tb__DOT__dut__DOT__zernike_out[1U] 
                    = ((0xffc00000U & vlSelf->tb__DOT__dut__DOT__zernike_out[1U]) 
                       | (vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h017c4131__0 
                          >> 5U));
                vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__1 
                    = vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next
                    [2U];
                vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h017c4131__0 
                    = (0x7ffffffU & (IData)((vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next
                                             [2U] >> 0x11U)));
                __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v22 
                    = vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__1;
                __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v22 = 1U;
                vlSelf->tb__DOT__dut__DOT__zernike_out[1U] 
                    = ((0x3fffffU & vlSelf->tb__DOT__dut__DOT__zernike_out[1U]) 
                       | (vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h017c4131__0 
                          << 0x16U));
                vlSelf->tb__DOT__dut__DOT__zernike_out[2U] 
                    = ((0xfffe0000U & vlSelf->tb__DOT__dut__DOT__zernike_out[2U]) 
                       | (vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h017c4131__0 
                          >> 0xaU));
                vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__1 
                    = vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next
                    [3U];
                vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h017c4131__0 
                    = (0x7ffffffU & (IData)((vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next
                                             [3U] >> 0x11U)));
                __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v23 
                    = vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__1;
                __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v23 = 1U;
                vlSelf->tb__DOT__dut__DOT__zernike_out[2U] 
                    = ((0x1ffffU & vlSelf->tb__DOT__dut__DOT__zernike_out[2U]) 
                       | (vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h017c4131__0 
                          << 0x11U));
                vlSelf->tb__DOT__dut__DOT__zernike_out[3U] 
                    = ((0xfffff000U & vlSelf->tb__DOT__dut__DOT__zernike_out[3U]) 
                       | (vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h017c4131__0 
                          >> 0xfU));
                vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__1 
                    = vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next
                    [4U];
                vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h017c4131__0 
                    = (0x7ffffffU & (IData)((vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next
                                             [4U] >> 0x11U)));
                __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v24 
                    = vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__1;
                __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v24 = 1U;
                vlSelf->tb__DOT__dut__DOT__zernike_out[3U] 
                    = ((0xfffU & vlSelf->tb__DOT__dut__DOT__zernike_out[3U]) 
                       | (vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h017c4131__0 
                          << 0xcU));
                vlSelf->tb__DOT__dut__DOT__zernike_out[4U] 
                    = ((0xffffff80U & vlSelf->tb__DOT__dut__DOT__zernike_out[4U]) 
                       | (vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h017c4131__0 
                          >> 0x14U));
                vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__1 
                    = vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next
                    [5U];
                vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h017c4131__0 
                    = (0x7ffffffU & (IData)((vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next
                                             [5U] >> 0x11U)));
                __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v25 
                    = vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__1;
                __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v25 = 1U;
                vlSelf->tb__DOT__dut__DOT__zernike_out[4U] 
                    = ((0x7fU & vlSelf->tb__DOT__dut__DOT__zernike_out[4U]) 
                       | (vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h017c4131__0 
                          << 7U));
                vlSelf->tb__DOT__dut__DOT__zernike_out[5U] 
                    = ((0xfffffffcU & vlSelf->tb__DOT__dut__DOT__zernike_out[5U]) 
                       | (vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h017c4131__0 
                          >> 0x19U));
                vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__1 
                    = vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next
                    [6U];
                vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h017c4131__0 
                    = (0x7ffffffU & (IData)((vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next
                                             [6U] >> 0x11U)));
                __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v26 
                    = vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__1;
                __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v26 = 1U;
                vlSelf->tb__DOT__dut__DOT__zernike_out[5U] 
                    = ((0xe0000003U & vlSelf->tb__DOT__dut__DOT__zernike_out[5U]) 
                       | (vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h017c4131__0 
                          << 2U));
                vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__1 
                    = vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next
                    [7U];
                vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h017c4131__0 
                    = (0x7ffffffU & (IData)((vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next
                                             [7U] >> 0x11U)));
                __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v27 
                    = vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__1;
                __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v27 = 1U;
                vlSelf->tb__DOT__dut__DOT__zernike_out[5U] 
                    = ((0x1fffffffU & vlSelf->tb__DOT__dut__DOT__zernike_out[5U]) 
                       | (vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h017c4131__0 
                          << 0x1dU));
                vlSelf->tb__DOT__dut__DOT__zernike_out[6U] 
                    = ((0xff000000U & vlSelf->tb__DOT__dut__DOT__zernike_out[6U]) 
                       | (vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h017c4131__0 
                          >> 3U));
                vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__1 
                    = vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next
                    [8U];
                vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h017c4131__0 
                    = (0x7ffffffU & (IData)((vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next
                                             [8U] >> 0x11U)));
                __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v28 
                    = vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__1;
                __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v28 = 1U;
                vlSelf->tb__DOT__dut__DOT__zernike_out[6U] 
                    = ((0xffffffU & vlSelf->tb__DOT__dut__DOT__zernike_out[6U]) 
                       | (vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h017c4131__0 
                          << 0x18U));
                vlSelf->tb__DOT__dut__DOT__zernike_out[7U] 
                    = ((0xfff80000U & vlSelf->tb__DOT__dut__DOT__zernike_out[7U]) 
                       | (vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h017c4131__0 
                          >> 8U));
                vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__1 
                    = vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next
                    [9U];
                vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h017c4131__0 
                    = (0x7ffffffU & (IData)((vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next
                                             [9U] >> 0x11U)));
                __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v29 
                    = vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__1;
                __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v29 = 1U;
                vlSelf->tb__DOT__dut__DOT__zernike_out[7U] 
                    = ((0x7ffffU & vlSelf->tb__DOT__dut__DOT__zernike_out[7U]) 
                       | (vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h017c4131__0 
                          << 0x13U));
                vlSelf->tb__DOT__dut__DOT__zernike_out[8U] 
                    = (0x3fffU & (vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h017c4131__0 
                                  >> 0xdU));
            } else {
                __Vdly__tb__DOT__dut__DOT__em__DOT__sub_counter 
                    = (0xffU & ((IData)(1U) + (IData)(vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter)));
                vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__2 
                    = vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next
                    [0U];
                vlSelf->tb__DOT__dut__DOT__em__DOT__i = 0xaU;
                __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v30 
                    = vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__2;
                __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v30 = 1U;
                vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__2 
                    = vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next
                    [1U];
                __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v31 
                    = vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__2;
                __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v31 = 1U;
                vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__2 
                    = vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next
                    [2U];
                __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v32 
                    = vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__2;
                __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v32 = 1U;
                vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__2 
                    = vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next
                    [3U];
                __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v33 
                    = vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__2;
                __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v33 = 1U;
                vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__2 
                    = vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next
                    [4U];
                __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v34 
                    = vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__2;
                __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v34 = 1U;
                vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__2 
                    = vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next
                    [5U];
                __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v35 
                    = vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__2;
                __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v35 = 1U;
                vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__2 
                    = vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next
                    [6U];
                __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v36 
                    = vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__2;
                __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v36 = 1U;
                vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__2 
                    = vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next
                    [7U];
                __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v37 
                    = vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__2;
                __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v37 = 1U;
                vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__2 
                    = vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next
                    [8U];
                __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v38 
                    = vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__2;
                __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v38 = 1U;
                vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__2 
                    = vlSelf->tb__DOT__dut__DOT__em__DOT__acc_next
                    [9U];
                __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v39 
                    = vlSelf->tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__2;
                __Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v39 = 1U;
            }
        }
    }
    vlSelf->tb__DOT__dut__DOT__em__DOT__state = __Vdly__tb__DOT__dut__DOT__em__DOT__state;
    if (__Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v0) {
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc[0U] = 0ULL;
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc[1U] = 0ULL;
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc[2U] = 0ULL;
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc[3U] = 0ULL;
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc[4U] = 0ULL;
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc[5U] = 0ULL;
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc[6U] = 0ULL;
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc[7U] = 0ULL;
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc[8U] = 0ULL;
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc[9U] = 0ULL;
    }
    if (__Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v10) {
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc[0U] = 0ULL;
    }
    if (__Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v11) {
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc[1U] = 0ULL;
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc[2U] = 0ULL;
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc[3U] = 0ULL;
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc[4U] = 0ULL;
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc[5U] = 0ULL;
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc[6U] = 0ULL;
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc[7U] = 0ULL;
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc[8U] = 0ULL;
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc[9U] = 0ULL;
    }
    if (__Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v20) {
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc[0U] 
            = __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v20;
    }
    if (__Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v21) {
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc[1U] 
            = __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v21;
    }
    if (__Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v22) {
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc[2U] 
            = __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v22;
    }
    if (__Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v23) {
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc[3U] 
            = __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v23;
    }
    if (__Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v24) {
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc[4U] 
            = __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v24;
    }
    if (__Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v25) {
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc[5U] 
            = __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v25;
    }
    if (__Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v26) {
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc[6U] 
            = __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v26;
    }
    if (__Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v27) {
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc[7U] 
            = __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v27;
    }
    if (__Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v28) {
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc[8U] 
            = __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v28;
    }
    if (__Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v29) {
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc[9U] 
            = __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v29;
    }
    if (__Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v30) {
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc[0U] 
            = __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v30;
    }
    if (__Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v31) {
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc[1U] 
            = __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v31;
    }
    if (__Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v32) {
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc[2U] 
            = __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v32;
    }
    if (__Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v33) {
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc[3U] 
            = __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v33;
    }
    if (__Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v34) {
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc[4U] 
            = __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v34;
    }
    if (__Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v35) {
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc[5U] 
            = __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v35;
    }
    if (__Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v36) {
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc[6U] 
            = __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v36;
    }
    if (__Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v37) {
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc[7U] 
            = __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v37;
    }
    if (__Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v38) {
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc[8U] 
            = __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v38;
    }
    if (__Vdlyvset__tb__DOT__dut__DOT__em__DOT__acc__v39) {
        vlSelf->tb__DOT__dut__DOT__em__DOT__acc[9U] 
            = __Vdlyvval__tb__DOT__dut__DOT__em__DOT__acc__v39;
    }
    vlSelf->tb__DOT__dut__DOT__em__DOT__sub_counter 
        = __Vdly__tb__DOT__dut__DOT__em__DOT__sub_counter;
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
    vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg2_v_is_zero 
        = ((1U & (~ (IData)(vlSelf->tb__DOT__dut__DOT__reset))) 
           && (0U == (IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg1_v_u16)));
    if (vlSelf->tb__DOT__dut__DOT__reset) {
        __Vdly__tb__DOT__dut__DOT__slopes__DOT__current_subap = 0U;
        vlSelf->tb__DOT__dut__DOT__new_subapeture = 0U;
        vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg1_v_u16 = 0U;
        vlSelf->tb__DOT__dut__DOT__intensity_accumulator = 0U;
        vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_idx_delay = 0U;
    } else {
        if ((vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__centroids_done_internal
             [3U] > (IData)(vlSelf->tb__DOT__dut__DOT__slopes__DOT__current_subap))) {
            __Vdly__tb__DOT__dut__DOT__slopes__DOT__current_subap 
                = (0xffU & ((IData)(1U) + (IData)(vlSelf->tb__DOT__dut__DOT__slopes__DOT__current_subap)));
            vlSelf->tb__DOT__dut__DOT__new_subapeture = 1U;
            vlSelf->tb__DOT__dut__DOT__subap_valid 
                = (1U & (vlSelf->tb__DOT__dut__DOT__slopes__DOT__subap_bitmap_mem
                         [0U][(7U & ((vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__centroids_done_internal
                                      [3U] - (IData)(1U)) 
                                     >> 5U))] >> (0x1fU 
                                                  & (vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__centroids_done_internal
                                                     [3U] 
                                                     - (IData)(1U)))));
        } else {
            vlSelf->tb__DOT__dut__DOT__new_subapeture = 0U;
            vlSelf->tb__DOT__dut__DOT__subap_valid = 0U;
        }
        vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg1_v_u16 
            = vlSelf->tb__DOT__dut__DOT__intensity_accumulator;
        if (vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_done_delay) {
            vlSelf->tb__DOT__dut__DOT__intensity_accumulator 
                = vlSelf->tb__DOT__dut__DOT__accumulator__DOT__i
                [vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_idx_delay];
        }
        vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_idx_delay 
            = vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_idx;
    }
    if (__Vdlyvset__tb__DOT__dut__DOT__accumulator__DOT__i__v0) {
        vlSelf->tb__DOT__dut__DOT__accumulator__DOT__y_i[__Vdlyvdim0__tb__DOT__dut__DOT__accumulator__DOT__y_i__v0] 
            = __Vdlyvval__tb__DOT__dut__DOT__accumulator__DOT__y_i__v0;
        vlSelf->tb__DOT__dut__DOT__accumulator__DOT__x_i[__Vdlyvdim0__tb__DOT__dut__DOT__accumulator__DOT__x_i__v0] 
            = __Vdlyvval__tb__DOT__dut__DOT__accumulator__DOT__x_i__v0;
        vlSelf->tb__DOT__dut__DOT__accumulator__DOT__i[__Vdlyvdim0__tb__DOT__dut__DOT__accumulator__DOT__i__v0] 
            = __Vdlyvval__tb__DOT__dut__DOT__accumulator__DOT__i__v0;
    }
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
    vlSelf->tb__DOT__dut__DOT__ctrl_reg_f2h = ((0xfeU 
                                                & (IData)(vlSelf->tb__DOT__dut__DOT__ctrl_reg_f2h)) 
                                               | (IData)(vlSelf->tb__DOT__dut__DOT__done));
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
    vlSelf->tb__DOT__dut__DOT__slopes__DOT__current_subap 
        = __Vdly__tb__DOT__dut__DOT__slopes__DOT__current_subap;
    if (__Vdlyvset__tb__DOT__dut__DOT__reciprocal__DOT__centroids_done_internal__v0) {
        vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__centroids_done_internal[3U] = 0U;
        vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__centroids_done_internal[2U] = 0U;
        vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__centroids_done_internal[1U] = 0U;
        vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__centroids_done_internal[0U] = 0U;
    }
    if (__Vdlyvset__tb__DOT__dut__DOT__reciprocal__DOT__centroids_done_internal__v4) {
        vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__centroids_done_internal[3U] 
            = __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__centroids_done_internal__v4;
        vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__centroids_done_internal[2U] 
            = __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__centroids_done_internal__v5;
        vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__centroids_done_internal[1U] 
            = __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__centroids_done_internal__v6;
        vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__centroids_done_internal[0U] 
            = __Vdlyvval__tb__DOT__dut__DOT__reciprocal__DOT__centroids_done_internal__v7;
    }
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
    vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__v_safe 
        = ((0U == (IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg1_v_u16))
            ? 1U : (IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__stg1_v_u16));
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
    vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15 
        = (0xffffU & ((IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__v_safe) 
                      << (IData)(vlSelf->tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__shift_left)));
    vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_done_delay 
        = ((1U & (~ (IData)(vlSelf->tb__DOT__dut__DOT__reset))) 
           && (((IData)(vlSelf->tb__DOT__dut__DOT__streamer_valid) 
                & (0xfU == (IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__count_pixel_h))) 
               & (0xfU == (IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__count_pixel_v))));
    vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_idx 
        = (0xffU & (VL_SHIFTL_III(8,8,32, (IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_row), 4U) 
                    + (IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__subap_col)));
    vlSelf->tb__DOT__dut__DOT__streamer_valid = ((IData)(vlSelf->tb__DOT__dut__DOT__streamer_fv) 
                                                 & (IData)(vlSelf->tb__DOT__dut__DOT__streamer_lv));
    vlSelf->tb__DOT__dut__DOT__accumulator__DOT__count_pixel_v 
        = __Vdly__tb__DOT__dut__DOT__accumulator__DOT__count_pixel_v;
    vlSelf->tb__DOT__dut__DOT__accumulator__DOT__count_pixel_h 
        = __Vdly__tb__DOT__dut__DOT__accumulator__DOT__count_pixel_h;
    vlSelf->tb__DOT__dut__DOT__reset = ((0U != (IData)(vlSelf->tb__DOT__dut__DOT__hps_reset_sync__DOT__reset_count)) 
                                        | (0U != (IData)(vlSelf->tb__DOT__dut__DOT__key_reset_sync__DOT__reset_count)));
    vlSelf->tb__DOT__dut__DOT__accumulator__DOT__last_pixel_v 
        = (0xfU == (IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__count_pixel_v));
    vlSelf->tb__DOT__dut__DOT__accumulator__DOT__last_pixel_h 
        = (0xfU == (IData)(vlSelf->tb__DOT__dut__DOT__accumulator__DOT__count_pixel_h));
}

void Vtb___024root___eval_nba(Vtb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb___024root___eval_nba\n"); );
    // Body
    if ((1ULL & vlSelf->__VnbaTriggered.word(0U))) {
        Vtb___024root___nba_sequent__TOP__0(vlSelf);
        vlSelf->__Vm_traceActivity[2U] = 1U;
    }
}

void Vtb___024root___timing_resume(Vtb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb___024root___timing_resume\n"); );
    // Body
    if ((2ULL & vlSelf->__VactTriggered.word(0U))) {
        vlSelf->__VdlySched.resume();
    }
}

void Vtb___024root___eval_triggers__act(Vtb___024root* vlSelf);

bool Vtb___024root___eval_phase__act(Vtb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb___024root___eval_phase__act\n"); );
    // Init
    VlTriggerVec<2> __VpreTriggered;
    CData/*0:0*/ __VactExecute;
    // Body
    Vtb___024root___eval_triggers__act(vlSelf);
    __VactExecute = vlSelf->__VactTriggered.any();
    if (__VactExecute) {
        __VpreTriggered.andNot(vlSelf->__VactTriggered, vlSelf->__VnbaTriggered);
        vlSelf->__VnbaTriggered.thisOr(vlSelf->__VactTriggered);
        Vtb___024root___timing_resume(vlSelf);
        Vtb___024root___eval_act(vlSelf);
    }
    return (__VactExecute);
}

bool Vtb___024root___eval_phase__nba(Vtb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb___024root___eval_phase__nba\n"); );
    // Init
    CData/*0:0*/ __VnbaExecute;
    // Body
    __VnbaExecute = vlSelf->__VnbaTriggered.any();
    if (__VnbaExecute) {
        Vtb___024root___eval_nba(vlSelf);
        vlSelf->__VnbaTriggered.clear();
    }
    return (__VnbaExecute);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb___024root___dump_triggers__nba(Vtb___024root* vlSelf);
#endif  // VL_DEBUG
#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb___024root___dump_triggers__act(Vtb___024root* vlSelf);
#endif  // VL_DEBUG

void Vtb___024root___eval(Vtb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb___024root___eval\n"); );
    // Init
    IData/*31:0*/ __VnbaIterCount;
    CData/*0:0*/ __VnbaContinue;
    // Body
    __VnbaIterCount = 0U;
    __VnbaContinue = 1U;
    while (__VnbaContinue) {
        if (VL_UNLIKELY((0x64U < __VnbaIterCount))) {
#ifdef VL_DEBUG
            Vtb___024root___dump_triggers__nba(vlSelf);
#endif
            VL_FATAL_MT("tb.v", 3, "", "NBA region did not converge.");
        }
        __VnbaIterCount = ((IData)(1U) + __VnbaIterCount);
        __VnbaContinue = 0U;
        vlSelf->__VactIterCount = 0U;
        vlSelf->__VactContinue = 1U;
        while (vlSelf->__VactContinue) {
            if (VL_UNLIKELY((0x64U < vlSelf->__VactIterCount))) {
#ifdef VL_DEBUG
                Vtb___024root___dump_triggers__act(vlSelf);
#endif
                VL_FATAL_MT("tb.v", 3, "", "Active region did not converge.");
            }
            vlSelf->__VactIterCount = ((IData)(1U) 
                                       + vlSelf->__VactIterCount);
            vlSelf->__VactContinue = 0U;
            if (Vtb___024root___eval_phase__act(vlSelf)) {
                vlSelf->__VactContinue = 1U;
            }
        }
        if (Vtb___024root___eval_phase__nba(vlSelf)) {
            __VnbaContinue = 1U;
        }
    }
}

#ifdef VL_DEBUG
void Vtb___024root___eval_debug_assertions(Vtb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb___024root___eval_debug_assertions\n"); );
}
#endif  // VL_DEBUG

// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vtb.h for the primary calling header

#ifndef VERILATED_VTB___024ROOT_H_
#define VERILATED_VTB___024ROOT_H_  // guard

#include "verilated.h"
#include "verilated_timing.h"


class Vtb__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vtb___024root final : public VerilatedModule {
  public:

    // DESIGN SPECIFIC STATE
    // Anonymous structures to workaround compiler member-count bugs
    struct {
        CData/*0:0*/ tb__DOT__clk;
        CData/*0:0*/ tb__DOT__key_reset;
        CData/*0:0*/ tb__DOT__hps_reset;
        CData/*0:0*/ tb__DOT__dut__DOT__result_write;
        CData/*7:0*/ tb__DOT__dut__DOT__streamer_data;
        CData/*0:0*/ tb__DOT__dut__DOT__streamer_fv;
        CData/*0:0*/ tb__DOT__dut__DOT__streamer_lv;
        CData/*0:0*/ tb__DOT__dut__DOT__frame_complete;
        CData/*0:0*/ tb__DOT__dut__DOT__streamer_valid;
        CData/*7:0*/ tb__DOT__dut__DOT__ctrl_reg_h2f;
        CData/*7:0*/ tb__DOT__dut__DOT__ctrl_reg_f2h;
        CData/*0:0*/ tb__DOT__dut__DOT__reset;
        CData/*1:0*/ tb__DOT__dut__DOT__start_sync;
        CData/*0:0*/ tb__DOT__dut__DOT__full_frame_complete_accumulator;
        CData/*7:0*/ tb__DOT__dut__DOT__subaps_done_accumulator;
        CData/*7:0*/ tb__DOT__dut__DOT__subaps_done_reciprocal;
        CData/*0:0*/ tb__DOT__dut__DOT__new_subapeture;
        CData/*0:0*/ tb__DOT__dut__DOT__done;
        CData/*4:0*/ tb__DOT__dut__DOT__resw_next_state;
        CData/*0:0*/ tb__DOT__dut__DOT__resw_start;
        CData/*4:0*/ tb__DOT__dut__DOT__resw_state;
        CData/*0:0*/ tb__DOT__dut__DOT__write_zernike_latch;
        CData/*7:0*/ tb__DOT__dut__DOT__streamer__DOT__line_counter;
        CData/*7:0*/ tb__DOT__dut__DOT__streamer__DOT__row_counter;
        CData/*1:0*/ tb__DOT__dut__DOT__streamer__DOT__h_blank_counter;
        CData/*7:0*/ tb__DOT__dut__DOT__streamer__DOT__v_blank_counter;
        CData/*1:0*/ tb__DOT__dut__DOT__streamer__DOT__state;
        CData/*3:0*/ tb__DOT__dut__DOT__accumulator__DOT__subap_col;
        CData/*3:0*/ tb__DOT__dut__DOT__accumulator__DOT__subap_row;
        CData/*3:0*/ tb__DOT__dut__DOT__accumulator__DOT__count_pixel_h;
        CData/*3:0*/ tb__DOT__dut__DOT__accumulator__DOT__count_pixel_v;
        CData/*7:0*/ tb__DOT__dut__DOT__accumulator__DOT__subap_idx;
        CData/*0:0*/ tb__DOT__dut__DOT__accumulator__DOT__last_pixel_h;
        CData/*0:0*/ tb__DOT__dut__DOT__accumulator__DOT__last_pixel_v;
        CData/*0:0*/ tb__DOT__dut__DOT__accumulator__DOT__last_subap_col;
        CData/*0:0*/ tb__DOT__dut__DOT__accumulator__DOT__last_subap_row;
        CData/*0:0*/ tb__DOT__dut__DOT__accumulator__DOT__subap_done_delay;
        CData/*7:0*/ tb__DOT__dut__DOT__accumulator__DOT__subap_idx_delay;
        CData/*4:0*/ tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__shift_left;
        CData/*7:0*/ tb__DOT__dut__DOT__slopes__DOT__current_subap;
        CData/*0:0*/ tb__DOT__dut__DOT__em__DOT__state;
        CData/*7:0*/ tb__DOT__dut__DOT__em__DOT__sub_counter;
        CData/*1:0*/ tb__DOT__dut__DOT__key_reset_sync__DOT__sync_ff;
        CData/*0:0*/ tb__DOT__dut__DOT__key_reset_sync__DOT__sync_d;
        CData/*3:0*/ tb__DOT__dut__DOT__key_reset_sync__DOT__reset_count;
        CData/*1:0*/ tb__DOT__dut__DOT__hps_reset_sync__DOT__sync_ff;
        CData/*0:0*/ tb__DOT__dut__DOT__hps_reset_sync__DOT__sync_d;
        CData/*3:0*/ tb__DOT__dut__DOT__hps_reset_sync__DOT__reset_count;
        CData/*0:0*/ __VstlFirstIteration;
        CData/*0:0*/ __Vtrigprevexpr___TOP__tb__DOT__clk__0;
        CData/*0:0*/ __VactContinue;
        SData/*10:0*/ tb__DOT__dut__DOT__result_addr;
        SData/*15:0*/ tb__DOT__dut__DOT__intensity_accumulator;
        SData/*10:0*/ tb__DOT__dut__DOT__resw_write_idx;
        SData/*10:0*/ tb__DOT__dut__DOT__write_zernike_count;
        SData/*15:0*/ tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__v_safe;
        SData/*15:0*/ tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__a_q1_15;
        IData/*31:0*/ tb__DOT__i;
        IData/*31:0*/ tb__DOT__dut__DOT__result_rdata;
        IData/*31:0*/ tb__DOT__dut__DOT__result_wdata;
        IData/*19:0*/ tb__DOT__dut__DOT__xI_accumulator;
        IData/*19:0*/ tb__DOT__dut__DOT__yI_accumulator;
        IData/*19:0*/ tb__DOT__dut__DOT__xI_reciprocal;
        IData/*19:0*/ tb__DOT__dut__DOT__yI_reciprocal;
    };
    struct {
        IData/*26:0*/ tb__DOT__dut__DOT__rI_reciprocal;
        IData/*26:0*/ tb__DOT__dut__DOT__x_centroid;
        IData/*26:0*/ tb__DOT__dut__DOT__y_centroid;
        IData/*26:0*/ tb__DOT__dut__DOT__x_slopes;
        IData/*26:0*/ tb__DOT__dut__DOT__y_slopes;
        VlWide<9>/*269:0*/ tb__DOT__dut__DOT__zernike_out;
        IData/*26:0*/ tb__DOT__dut__DOT__x_centroid_out;
        IData/*26:0*/ tb__DOT__dut__DOT__y_centroid_out;
        IData/*26:0*/ tb__DOT__dut__DOT__x_slopes_out;
        IData/*26:0*/ tb__DOT__dut__DOT__y_slopes_out;
        IData/*31:0*/ tb__DOT__dut__DOT__accumulator__DOT__j;
        IData/*26:0*/ tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__x1_q0_27;
        IData/*26:0*/ tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__x2_q0_27;
        IData/*27:0*/ tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__out_q0_27_ext;
        IData/*31:0*/ tb__DOT__dut__DOT__em__DOT__i;
        IData/*26:0*/ tb__DOT__dut__DOT__em__DOT____Vlvbound_h017c4131__0;
        IData/*31:0*/ __VactIterCount;
        QData/*54:0*/ tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__u_newton_step_0__DOT__prod_mul;
        QData/*54:0*/ tb__DOT__dut__DOT__reciprocal__DOT__recip__DOT__u_newton_step_1__DOT__prod_mul;
        QData/*46:0*/ tb__DOT__dut__DOT__slopes__DOT__x_mult__DOT__mult_out;
        QData/*46:0*/ tb__DOT__dut__DOT__slopes__DOT__y_mult__DOT__mult_out;
        QData/*44:0*/ tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__0__KET____DOT__prod_x;
        QData/*44:0*/ tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__0__KET____DOT__prod_y;
        QData/*44:0*/ tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__1__KET____DOT__prod_x;
        QData/*44:0*/ tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__1__KET____DOT__prod_y;
        QData/*44:0*/ tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__2__KET____DOT__prod_x;
        QData/*44:0*/ tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__2__KET____DOT__prod_y;
        QData/*44:0*/ tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__3__KET____DOT__prod_x;
        QData/*44:0*/ tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__3__KET____DOT__prod_y;
        QData/*44:0*/ tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__4__KET____DOT__prod_x;
        QData/*44:0*/ tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__4__KET____DOT__prod_y;
        QData/*44:0*/ tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__5__KET____DOT__prod_x;
        QData/*44:0*/ tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__5__KET____DOT__prod_y;
        QData/*44:0*/ tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__6__KET____DOT__prod_x;
        QData/*44:0*/ tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__6__KET____DOT__prod_y;
        QData/*44:0*/ tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__7__KET____DOT__prod_x;
        QData/*44:0*/ tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__7__KET____DOT__prod_y;
        QData/*44:0*/ tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__8__KET____DOT__prod_x;
        QData/*44:0*/ tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__8__KET____DOT__prod_y;
        QData/*44:0*/ tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__9__KET____DOT__prod_x;
        QData/*44:0*/ tb__DOT__dut__DOT__em__DOT__mac_gen__BRA__9__KET____DOT__prod_y;
        QData/*54:0*/ tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__1;
        QData/*54:0*/ tb__DOT__dut__DOT__em__DOT____Vlvbound_h252efdf3__2;
        VlUnpacked<IData/*26:0*/, 10> tb__DOT__dut__DOT__zernike_out_reg;
        VlUnpacked<CData/*7:0*/, 65536> tb__DOT__dut__DOT__streamer__DOT__mem;
        VlUnpacked<SData/*15:0*/, 256> tb__DOT__dut__DOT__accumulator__DOT__i;
        VlUnpacked<IData/*19:0*/, 256> tb__DOT__dut__DOT__accumulator__DOT__x_i;
        VlUnpacked<IData/*19:0*/, 256> tb__DOT__dut__DOT__accumulator__DOT__y_i;
        VlUnpacked<IData/*17:0*/, 1680> tb__DOT__dut__DOT__em__DOT__e_rom_x;
        VlUnpacked<IData/*17:0*/, 1680> tb__DOT__dut__DOT__em__DOT__e_rom_y;
        VlUnpacked<QData/*54:0*/, 10> tb__DOT__dut__DOT__em__DOT__mac_sum;
        VlUnpacked<QData/*54:0*/, 10> tb__DOT__dut__DOT__em__DOT__acc;
        VlUnpacked<QData/*54:0*/, 10> tb__DOT__dut__DOT__em__DOT__acc_next;
        VlUnpacked<CData/*0:0*/, 3> __Vm_traceActivity;
    };
    VlDelayScheduler __VdlySched;
    VlTriggerVec<1> __VstlTriggered;
    VlTriggerVec<2> __VactTriggered;
    VlTriggerVec<2> __VnbaTriggered;

    // INTERNAL VARIABLES
    Vtb__Syms* const vlSymsp;

    // CONSTRUCTORS
    Vtb___024root(Vtb__Syms* symsp, const char* v__name);
    ~Vtb___024root();
    VL_UNCOPYABLE(Vtb___024root);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard

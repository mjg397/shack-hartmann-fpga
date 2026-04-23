// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vshwfs_pipeline_tb.h for the primary calling header

#ifndef VERILATED_VSHWFS_PIPELINE_TB___024ROOT_H_
#define VERILATED_VSHWFS_PIPELINE_TB___024ROOT_H_  // guard

#include "verilated.h"
#include "verilated_timing.h"


class Vshwfs_pipeline_tb__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vshwfs_pipeline_tb___024root final : public VerilatedModule {
  public:

    // DESIGN SPECIFIC STATE
    CData/*0:0*/ shwfs_pipeline_tb__DOT__clk_100;
    CData/*0:0*/ shwfs_pipeline_tb__DOT__reset;
    CData/*7:0*/ shwfs_pipeline_tb__DOT__subaps_done_reciprocal;
    CData/*7:0*/ shwfs_pipeline_tb__DOT__DUT__DOT__streamer_data;
    CData/*0:0*/ shwfs_pipeline_tb__DOT__DUT__DOT__streamer_fv;
    CData/*0:0*/ shwfs_pipeline_tb__DOT__DUT__DOT__streamer_lv;
    CData/*0:0*/ shwfs_pipeline_tb__DOT__DUT__DOT__frame_complete;
    CData/*0:0*/ shwfs_pipeline_tb__DOT__DUT__DOT__streamer_valid;
    CData/*0:0*/ shwfs_pipeline_tb__DOT__DUT__DOT__full_frame_complete_accumulator;
    CData/*7:0*/ shwfs_pipeline_tb__DOT__DUT__DOT__subaps_done_accumulator;
    CData/*7:0*/ shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__line_counter;
    CData/*7:0*/ shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__row_counter;
    CData/*1:0*/ shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__h_blank_counter;
    CData/*7:0*/ shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__v_blank_counter;
    CData/*1:0*/ shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__state;
    CData/*3:0*/ shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_col;
    CData/*3:0*/ shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_row;
    CData/*3:0*/ shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__count_pixel_h;
    CData/*3:0*/ shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__count_pixel_v;
    CData/*7:0*/ shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_idx;
    CData/*0:0*/ shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__last_pixel_h;
    CData/*0:0*/ shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__last_pixel_v;
    CData/*0:0*/ shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__last_subap_col;
    CData/*0:0*/ shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__last_subap_row;
    CData/*0:0*/ shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_done_delay;
    CData/*7:0*/ shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__subap_idx_delay;
    CData/*4:0*/ shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__shift_left;
    CData/*0:0*/ __VstlFirstIteration;
    CData/*0:0*/ __Vtrigprevexpr___TOP__shwfs_pipeline_tb__DOT__clk_100__0;
    CData/*0:0*/ __VactContinue;
    SData/*15:0*/ shwfs_pipeline_tb__DOT__rI_reciprocal;
    SData/*15:0*/ shwfs_pipeline_tb__DOT__DUT__DOT__intensity_accumulator;
    SData/*15:0*/ shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__v_safe;
    SData/*15:0*/ shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__a_q1_15;
    SData/*15:0*/ shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__x1_q0_16;
    IData/*19:0*/ shwfs_pipeline_tb__DOT__xI_reciprocal;
    IData/*19:0*/ shwfs_pipeline_tb__DOT__yI_reciprocal;
    IData/*19:0*/ shwfs_pipeline_tb__DOT__DUT__DOT__xI_accumulator;
    IData/*19:0*/ shwfs_pipeline_tb__DOT__DUT__DOT__yI_accumulator;
    IData/*31:0*/ shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__j;
    IData/*16:0*/ shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__out_q0_16_ext;
    IData/*31:0*/ __VactIterCount;
    QData/*32:0*/ shwfs_pipeline_tb__DOT__DUT__DOT__reciprocal__DOT__recip__DOT__u_newton_step_q16__DOT__prod_mul;
    VlUnpacked<CData/*7:0*/, 65536> shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__mem;
    VlUnpacked<SData/*15:0*/, 256> shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__i;
    VlUnpacked<IData/*19:0*/, 256> shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__x_i;
    VlUnpacked<IData/*19:0*/, 256> shwfs_pipeline_tb__DOT__DUT__DOT__accumulator__DOT__y_i;
    VlUnpacked<CData/*0:0*/, 2> __Vm_traceActivity;
    VlDelayScheduler __VdlySched;
    VlTriggerVec<1> __VstlTriggered;
    VlTriggerVec<2> __VactTriggered;
    VlTriggerVec<2> __VnbaTriggered;

    // INTERNAL VARIABLES
    Vshwfs_pipeline_tb__Syms* const vlSymsp;

    // CONSTRUCTORS
    Vshwfs_pipeline_tb___024root(Vshwfs_pipeline_tb__Syms* symsp, const char* v__name);
    ~Vshwfs_pipeline_tb___024root();
    VL_UNCOPYABLE(Vshwfs_pipeline_tb___024root);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard

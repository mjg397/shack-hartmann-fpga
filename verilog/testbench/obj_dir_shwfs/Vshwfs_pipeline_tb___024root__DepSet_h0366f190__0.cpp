// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vshwfs_pipeline_tb.h for the primary calling header

#include "Vshwfs_pipeline_tb__pch.h"
#include "Vshwfs_pipeline_tb__Syms.h"
#include "Vshwfs_pipeline_tb___024root.h"

#ifdef VL_DEBUG
VL_ATTR_COLD void Vshwfs_pipeline_tb___024root___dump_triggers__act(Vshwfs_pipeline_tb___024root* vlSelf);
#endif  // VL_DEBUG

void Vshwfs_pipeline_tb___024root___eval_triggers__act(Vshwfs_pipeline_tb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vshwfs_pipeline_tb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vshwfs_pipeline_tb___024root___eval_triggers__act\n"); );
    // Body
    vlSelf->__VactTriggered.set(0U, ((IData)(vlSelf->shwfs_pipeline_tb__DOT__clk_100) 
                                     & (~ (IData)(vlSelf->__Vtrigprevexpr___TOP__shwfs_pipeline_tb__DOT__clk_100__0))));
    vlSelf->__VactTriggered.set(1U, vlSelf->__VdlySched.awaitingCurrentTime());
    vlSelf->__Vtrigprevexpr___TOP__shwfs_pipeline_tb__DOT__clk_100__0 
        = vlSelf->shwfs_pipeline_tb__DOT__clk_100;
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vshwfs_pipeline_tb___024root___dump_triggers__act(vlSelf);
    }
#endif
}

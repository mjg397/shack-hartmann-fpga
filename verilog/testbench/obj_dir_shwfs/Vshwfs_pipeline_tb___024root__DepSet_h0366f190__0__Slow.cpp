// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vshwfs_pipeline_tb.h for the primary calling header

#include "Vshwfs_pipeline_tb__pch.h"
#include "Vshwfs_pipeline_tb__Syms.h"
#include "Vshwfs_pipeline_tb___024root.h"

extern const VlWide<9>/*287:0*/ Vshwfs_pipeline_tb__ConstPool__CONST_h5416689c_0;

VL_ATTR_COLD void Vshwfs_pipeline_tb___024root___eval_initial__TOP(Vshwfs_pipeline_tb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vshwfs_pipeline_tb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vshwfs_pipeline_tb___024root___eval_initial__TOP\n"); );
    // Init
    VlWide<6>/*191:0*/ __Vtemp_1;
    // Body
    __Vtemp_1[0U] = 0x2e766364U;
    __Vtemp_1[1U] = 0x655f7462U;
    __Vtemp_1[2U] = 0x656c696eU;
    __Vtemp_1[3U] = 0x5f706970U;
    __Vtemp_1[4U] = 0x68776673U;
    __Vtemp_1[5U] = 0x73U;
    vlSymsp->_vm_contextp__->dumpfile(VL_CVT_PACK_STR_NW(6, __Vtemp_1));
    vlSymsp->_traceDumpOpen();
    VL_READMEM_N(true, 8, 65536, 0, VL_CVT_PACK_STR_NW(9, Vshwfs_pipeline_tb__ConstPool__CONST_h5416689c_0)
                 ,  &(vlSelf->shwfs_pipeline_tb__DOT__DUT__DOT__streamer__DOT__mem)
                 , 0, ~0ULL);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vshwfs_pipeline_tb___024root___dump_triggers__stl(Vshwfs_pipeline_tb___024root* vlSelf);
#endif  // VL_DEBUG

VL_ATTR_COLD void Vshwfs_pipeline_tb___024root___eval_triggers__stl(Vshwfs_pipeline_tb___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vshwfs_pipeline_tb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vshwfs_pipeline_tb___024root___eval_triggers__stl\n"); );
    // Body
    vlSelf->__VstlTriggered.set(0U, (IData)(vlSelf->__VstlFirstIteration));
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vshwfs_pipeline_tb___024root___dump_triggers__stl(vlSelf);
    }
#endif
}

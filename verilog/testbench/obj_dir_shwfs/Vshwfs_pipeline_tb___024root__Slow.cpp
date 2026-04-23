// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vshwfs_pipeline_tb.h for the primary calling header

#include "Vshwfs_pipeline_tb__pch.h"
#include "Vshwfs_pipeline_tb__Syms.h"
#include "Vshwfs_pipeline_tb___024root.h"

void Vshwfs_pipeline_tb___024root___ctor_var_reset(Vshwfs_pipeline_tb___024root* vlSelf);

Vshwfs_pipeline_tb___024root::Vshwfs_pipeline_tb___024root(Vshwfs_pipeline_tb__Syms* symsp, const char* v__name)
    : VerilatedModule{v__name}
    , __VdlySched{*symsp->_vm_contextp__}
    , vlSymsp{symsp}
 {
    // Reset structure values
    Vshwfs_pipeline_tb___024root___ctor_var_reset(this);
}

void Vshwfs_pipeline_tb___024root::__Vconfigure(bool first) {
    if (false && first) {}  // Prevent unused
}

Vshwfs_pipeline_tb___024root::~Vshwfs_pipeline_tb___024root() {
}

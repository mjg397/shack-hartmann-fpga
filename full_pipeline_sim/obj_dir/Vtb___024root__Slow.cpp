// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtb.h for the primary calling header

#include "Vtb__pch.h"
#include "Vtb__Syms.h"
#include "Vtb___024root.h"

void Vtb___024root___ctor_var_reset(Vtb___024root* vlSelf);

Vtb___024root::Vtb___024root(Vtb__Syms* symsp, const char* v__name)
    : VerilatedModule{v__name}
    , __VdlySched{*symsp->_vm_contextp__}
    , vlSymsp{symsp}
 {
    // Reset structure values
    Vtb___024root___ctor_var_reset(this);
}

void Vtb___024root::__Vconfigure(bool first) {
    if (false && first) {}  // Prevent unused
}

Vtb___024root::~Vtb___024root() {
}

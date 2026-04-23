// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Model implementation (design independent parts)

#include "Vshwfs_pipeline_tb__pch.h"
#include "verilated_vcd_c.h"

//============================================================
// Constructors

Vshwfs_pipeline_tb::Vshwfs_pipeline_tb(VerilatedContext* _vcontextp__, const char* _vcname__)
    : VerilatedModel{*_vcontextp__}
    , vlSymsp{new Vshwfs_pipeline_tb__Syms(contextp(), _vcname__, this)}
    , rootp{&(vlSymsp->TOP)}
{
    // Register model with the context
    contextp()->addModel(this);
}

Vshwfs_pipeline_tb::Vshwfs_pipeline_tb(const char* _vcname__)
    : Vshwfs_pipeline_tb(Verilated::threadContextp(), _vcname__)
{
}

//============================================================
// Destructor

Vshwfs_pipeline_tb::~Vshwfs_pipeline_tb() {
    delete vlSymsp;
}

//============================================================
// Evaluation function

#ifdef VL_DEBUG
void Vshwfs_pipeline_tb___024root___eval_debug_assertions(Vshwfs_pipeline_tb___024root* vlSelf);
#endif  // VL_DEBUG
void Vshwfs_pipeline_tb___024root___eval_static(Vshwfs_pipeline_tb___024root* vlSelf);
void Vshwfs_pipeline_tb___024root___eval_initial(Vshwfs_pipeline_tb___024root* vlSelf);
void Vshwfs_pipeline_tb___024root___eval_settle(Vshwfs_pipeline_tb___024root* vlSelf);
void Vshwfs_pipeline_tb___024root___eval(Vshwfs_pipeline_tb___024root* vlSelf);

void Vshwfs_pipeline_tb::eval_step() {
    VL_DEBUG_IF(VL_DBG_MSGF("+++++TOP Evaluate Vshwfs_pipeline_tb::eval_step\n"); );
#ifdef VL_DEBUG
    // Debug assertions
    Vshwfs_pipeline_tb___024root___eval_debug_assertions(&(vlSymsp->TOP));
#endif  // VL_DEBUG
    vlSymsp->__Vm_activity = true;
    vlSymsp->__Vm_deleter.deleteAll();
    if (VL_UNLIKELY(!vlSymsp->__Vm_didInit)) {
        vlSymsp->__Vm_didInit = true;
        VL_DEBUG_IF(VL_DBG_MSGF("+ Initial\n"););
        Vshwfs_pipeline_tb___024root___eval_static(&(vlSymsp->TOP));
        Vshwfs_pipeline_tb___024root___eval_initial(&(vlSymsp->TOP));
        Vshwfs_pipeline_tb___024root___eval_settle(&(vlSymsp->TOP));
    }
    VL_DEBUG_IF(VL_DBG_MSGF("+ Eval\n"););
    Vshwfs_pipeline_tb___024root___eval(&(vlSymsp->TOP));
    // Evaluate cleanup
    Verilated::endOfEval(vlSymsp->__Vm_evalMsgQp);
}

void Vshwfs_pipeline_tb::eval_end_step() {
    VL_DEBUG_IF(VL_DBG_MSGF("+eval_end_step Vshwfs_pipeline_tb::eval_end_step\n"); );
#ifdef VM_TRACE
    // Tracing
    if (VL_UNLIKELY(vlSymsp->__Vm_dumping)) vlSymsp->_traceDump();
#endif  // VM_TRACE
}

//============================================================
// Events and timing
bool Vshwfs_pipeline_tb::eventsPending() { return !vlSymsp->TOP.__VdlySched.empty(); }

uint64_t Vshwfs_pipeline_tb::nextTimeSlot() { return vlSymsp->TOP.__VdlySched.nextTimeSlot(); }

//============================================================
// Utilities

const char* Vshwfs_pipeline_tb::name() const {
    return vlSymsp->name();
}

//============================================================
// Invoke final blocks

void Vshwfs_pipeline_tb___024root___eval_final(Vshwfs_pipeline_tb___024root* vlSelf);

VL_ATTR_COLD void Vshwfs_pipeline_tb::final() {
    Vshwfs_pipeline_tb___024root___eval_final(&(vlSymsp->TOP));
}

//============================================================
// Implementations of abstract methods from VerilatedModel

const char* Vshwfs_pipeline_tb::hierName() const { return vlSymsp->name(); }
const char* Vshwfs_pipeline_tb::modelName() const { return "Vshwfs_pipeline_tb"; }
unsigned Vshwfs_pipeline_tb::threads() const { return 1; }
void Vshwfs_pipeline_tb::prepareClone() const { contextp()->prepareClone(); }
void Vshwfs_pipeline_tb::atClone() const {
    contextp()->threadPoolpOnClone();
}
std::unique_ptr<VerilatedTraceConfig> Vshwfs_pipeline_tb::traceConfig() const {
    return std::unique_ptr<VerilatedTraceConfig>{new VerilatedTraceConfig{false, false, false}};
};

//============================================================
// Trace configuration

void Vshwfs_pipeline_tb___024root__trace_decl_types(VerilatedVcd* tracep);

void Vshwfs_pipeline_tb___024root__trace_init_top(Vshwfs_pipeline_tb___024root* vlSelf, VerilatedVcd* tracep);

VL_ATTR_COLD static void trace_init(void* voidSelf, VerilatedVcd* tracep, uint32_t code) {
    // Callback from tracep->open()
    Vshwfs_pipeline_tb___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vshwfs_pipeline_tb___024root*>(voidSelf);
    Vshwfs_pipeline_tb__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    if (!vlSymsp->_vm_contextp__->calcUnusedSigs()) {
        VL_FATAL_MT(__FILE__, __LINE__, __FILE__,
            "Turning on wave traces requires Verilated::traceEverOn(true) call before time 0.");
    }
    vlSymsp->__Vm_baseCode = code;
    tracep->pushPrefix(std::string{vlSymsp->name()}, VerilatedTracePrefixType::SCOPE_MODULE);
    Vshwfs_pipeline_tb___024root__trace_decl_types(tracep);
    Vshwfs_pipeline_tb___024root__trace_init_top(vlSelf, tracep);
    tracep->popPrefix();
}

VL_ATTR_COLD void Vshwfs_pipeline_tb___024root__trace_register(Vshwfs_pipeline_tb___024root* vlSelf, VerilatedVcd* tracep);

VL_ATTR_COLD void Vshwfs_pipeline_tb::trace(VerilatedVcdC* tfp, int levels, int options) {
    if (tfp->isOpen()) {
        vl_fatal(__FILE__, __LINE__, __FILE__,"'Vshwfs_pipeline_tb::trace()' shall not be called after 'VerilatedVcdC::open()'.");
    }
    if (false && levels && options) {}  // Prevent unused
    tfp->spTrace()->addModel(this);
    tfp->spTrace()->addInitCb(&trace_init, &(vlSymsp->TOP));
    Vshwfs_pipeline_tb___024root__trace_register(&(vlSymsp->TOP), tfp->spTrace());
}

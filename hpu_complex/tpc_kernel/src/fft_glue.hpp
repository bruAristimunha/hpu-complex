#pragma once

#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"

// One glue class for both FFT kernels (probe + r2c), selected by name.
class FftGlue
{
public:
    enum Variant { PROBE, R2C };
    explicit FftGlue(Variant v) : variant_(v) {}
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams* in_defs,
        tpc_lib_api::HabanaKernelInstantiation* out_defs);
    tpc_lib_api::GlueCodeReturn GetKernelName(
        char kernelName[tpc_lib_api::MAX_NODE_NAME]);

private:
    Variant variant_;
};

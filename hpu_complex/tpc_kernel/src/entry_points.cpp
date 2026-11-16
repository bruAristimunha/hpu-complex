#include <cstring>

#include "fft_glue.hpp"

extern "C" {

tpc_lib_api::GlueCodeReturn GetKernelGuids(
    tpc_lib_api::DeviceId deviceId,
    uint32_t* kernelCount,
    tpc_lib_api::GuidInfo* guids)
{
    if (deviceId != tpc_lib_api::DEVICE_ID_GAUDI2)
    {
        if (kernelCount != nullptr)
        {
            *kernelCount = 0;
        }
        return tpc_lib_api::GLUE_SUCCESS;
    }
    if (guids == nullptr)
    {
        if (kernelCount != nullptr)
        {
            *kernelCount = 2;
        }
        return tpc_lib_api::GLUE_SUCCESS;
    }
    FftGlue probe(FftGlue::PROBE);
    probe.GetKernelName(guids[0].name);
    FftGlue r2c(FftGlue::R2C);
    r2c.GetKernelName(guids[1].name);
    if (kernelCount != nullptr)
    {
        *kernelCount = 2;
    }
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(
    tpc_lib_api::HabanaKernelParams* params,
    tpc_lib_api::HabanaKernelInstantiation* instance)
{
    char kernelName[tpc_lib_api::MAX_NODE_NAME];
    FftGlue probe(FftGlue::PROBE);
    probe.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return probe.GetGcDefinitions(params, instance);
    }
    FftGlue r2c(FftGlue::R2C);
    r2c.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return r2c.GetGcDefinitions(params, instance);
    }
    return tpc_lib_api::GLUE_NODE_NOT_FOUND;
}

tpc_lib_api::GlueCodeReturn GetShapeInference(
    tpc_lib_api::DeviceId /*deviceId*/,
    const tpc_lib_api::ShapeInferenceParams* /*inputParams*/,
    tpc_lib_api::ShapeInferenceOutput* /*outputData*/)
{
    return tpc_lib_api::GLUE_SUCCESS;
}

} // extern "C"

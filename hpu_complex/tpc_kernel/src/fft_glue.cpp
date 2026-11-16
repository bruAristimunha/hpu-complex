#include "fft_glue.hpp"

#include <cstring>

extern "C" {
extern const char _binary___fft_probe_f32_gaudi2_o_start[];
extern const char _binary___fft_probe_f32_gaudi2_o_end[];
extern const char _binary___fft_r2c_f32_gaudi2_o_start[];
extern const char _binary___fft_r2c_f32_gaudi2_o_end[];
}

tpc_lib_api::GlueCodeReturn FftGlue::GetKernelName(
    char kernelName[tpc_lib_api::MAX_NODE_NAME])
{
    strcpy(
        kernelName,
        variant_ == PROBE ? "fft_probe_f32_gaudi2"
                            : "fft_r2c_f32_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

// coord = indexSpaceCoord[dim] * a + b, b in [start_b, end_b].
// a=0 + end_b=size-1 means: this instance accesses the whole dim.
static void full_dim(
    tpc_lib_api::TensorAccessPattern& pattern,
    const int tensor_dim,
    const uint64_t size)
{
    pattern.mapping[tensor_dim].indexSpaceDim = tensor_dim;
    pattern.mapping[tensor_dim].allRequired = true; // whole dim may be read
    pattern.mapping[tensor_dim].a = 0;
    pattern.mapping[tensor_dim].start_b = 0;
    pattern.mapping[tensor_dim].end_b = size - 1;
}

static void row_dim(
    tpc_lib_api::TensorAccessPattern& pattern,
    const int tensor_dim)
{
    pattern.mapping[tensor_dim].indexSpaceDim = 0; // index-space dim 0 = rows
    pattern.mapping[tensor_dim].allRequired = true; // required for split dim
    pattern.mapping[tensor_dim].a = 1;
    pattern.mapping[tensor_dim].start_b = 0;
    pattern.mapping[tensor_dim].end_b = 0;
}

static void base_pattern(tpc_lib_api::TensorAccessPattern& pattern)
{
    for (int d = 0; d < 5; d++)
    {
        pattern.mapping[d].indexSpaceDim = d;
        pattern.mapping[d].a = 1;
        pattern.mapping[d].start_b = 0;
        pattern.mapping[d].end_b = 0;
    }
}

tpc_lib_api::GlueCodeReturn FftGlue::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in_defs,
    tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    const bool probe = variant_ == PROBE;
    const unsigned expected_in = probe ? 2u : 2u; // probe: signal + lut; r2c: signal + twiddles
    if (in_defs->inputTensorNr != expected_in)
    {
        in_defs->inputTensorNr = expected_in;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (in_defs->outputTensorNr != 1)
    {
        in_defs->outputTensorNr = 1;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    for (unsigned i = 0; i < expected_in; i++)
    {
        const auto want = (probe && i == 1) ? tpc_lib_api::DATA_U8
                                            : tpc_lib_api::DATA_F32;
        if (in_defs->inputTensors[i].geometry.dataType != want)
        {
            return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
        }
    }
    if (in_defs->outputTensors[0].geometry.dataType != tpc_lib_api::DATA_F32)
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }


    const uint64_t n = in_defs->inputTensors[0].geometry.maxSizes[0];
    const uint64_t rows = in_defs->inputTensors[0].geometry.maxSizes[1];
    out_defs->indexSpaceRank = 4;
    out_defs->preferredSplitDim = probe ? 0 : 1; // split along rows (dim 0)
    out_defs->indexSpaceGeometry[0] = probe ? 1 : rows;
    out_defs->indexSpaceGeometry[1] = 1;
    out_defs->indexSpaceGeometry[3] = 1;
    out_defs->indexSpaceGeometry[2] = 1;
    out_defs->indexSpaceGeometry[3] = 1;
    out_defs->indexSpaceGeometry[4] = 1;

    if (probe)
    {
        base_pattern(out_defs->inputTensorAccessPattern[0]);
        full_dim(out_defs->inputTensorAccessPattern[0], 0, n);
        base_pattern(out_defs->inputTensorAccessPattern[1]);
        full_dim(out_defs->inputTensorAccessPattern[1], 0, 256);
        full_dim(out_defs->inputTensorAccessPattern[1], 1, 64);
        base_pattern(out_defs->outputTensorAccessPattern[0]);
        full_dim(out_defs->outputTensorAccessPattern[0], 0, n);
        (void)rows;
    }
    else
    {
        // signal [n, rows]: whole n per instance, rows partitioned (dim 1)
        base_pattern(out_defs->inputTensorAccessPattern[0]);
        full_dim(out_defs->inputTensorAccessPattern[0], 0, n);
        row_dim(out_defs->inputTensorAccessPattern[0], 1);
        // twiddles [n, n, 2] geometry (j, k, plane): read-only, whole
        base_pattern(out_defs->inputTensorAccessPattern[1]);
        full_dim(out_defs->inputTensorAccessPattern[1], 0, n);
        full_dim(out_defs->inputTensorAccessPattern[1], 1, n);
        full_dim(out_defs->inputTensorAccessPattern[1], 2, 2);
        // output [n, 2, rows] geometry (k, plane, row)
        base_pattern(out_defs->outputTensorAccessPattern[0]);
        full_dim(out_defs->outputTensorAccessPattern[0], 0, n);
        full_dim(out_defs->outputTensorAccessPattern[0], 1, 2);
        row_dim(out_defs->outputTensorAccessPattern[0], 2);
    }

    const char* elf_start = probe ? _binary___fft_probe_f32_gaudi2_o_start
                                  : _binary___fft_r2c_f32_gaudi2_o_start;
    const char* elf_end = probe ? _binary___fft_probe_f32_gaudi2_o_end
                                : _binary___fft_r2c_f32_gaudi2_o_end;
    const unsigned isa_size =
        static_cast<unsigned>(elf_end - elf_start);
    if (out_defs->kernel.elfSize < isa_size)
    {
        out_defs->kernel.elfSize = isa_size;
        return tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
    }
    out_defs->kernel.elfSize = isa_size;
    memcpy(out_defs->kernel.kernelElf, elf_start, isa_size);
    return tpc_lib_api::GLUE_SUCCESS;
}

"""Shuffle splat probe: output chunks should be constant P[c]."""
import pathlib

import torch
from torch.utils.cpp_extension import load

CPP = r"""
#include <torch/extension.h>
#include "hpu_custom_op_pt2.h"

static habana::PartialOutputMetaDataVector output_meta(const at::Stack& inputs) {
  auto self = inputs[0].toTensor();
  habana::PartialOutputMetaData meta{self.scalar_type(), {384}};
  return {meta};
}

bool register_it() {
  habana::custom_op::registerUserCustomOp(
      "custom_op::fft_probe", "fft_probe_f32_gaudi2", output_meta, nullptr);
  return true;
}
static bool registered = register_it();

at::Tensor fft_probe(at::Tensor a, at::Tensor lut) {
  std::vector<c10::IValue> inputs{a, lut};
  auto desc = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
      "custom_op::fft_probe");
  auto out = desc.execute(inputs);
  return out[0];
}

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
  m.def("fft_probe(Tensor self, Tensor lut) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl("fft_probe", fft_probe);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
  m.impl("fft_probe", [](at::Tensor a, at::Tensor l) { return a.new_empty({384}); });
}
"""

from habana_frameworks.torch.utils.lib_utils import get_include_dir, get_lib_dir
build = pathlib.Path("/tmp/tpc_shuffle_plugin_build")
build.mkdir(exist_ok=True)
src = build / "shuffle_plugin.cpp"
src.write_text(CPP)
load(name="fft_shuffle_plugin", sources=[str(src)],
     build_directory=str(build), verbose=False, is_python_module=False,
     extra_include_paths=[get_include_dir(), "/usr/include/habanalabs"],
     extra_ldflags=[f"-L{get_lib_dir()}", "-lhabana_pytorch2_plugin"])
print("PLUGIN LOADED", flush=True)

lut = torch.zeros(64, 256, dtype=torch.uint8)
for p in range(64):
    lut[p, :] = 0x80 | ((p & 31) << 2) | (p >> 5)
x = torch.arange(64, dtype=torch.float32, device="hpu")
out = torch.ops.custom_op.fft_probe(x, lut.to("hpu"))
torch.hpu.synchronize()
o = out.cpu()
for c, p in enumerate((0, 1, 31, 32, 33, 63)):
    seg = o[c * 64:(c + 1) * 64]
    print(f"splat lane {p}: first={seg[0].item()} all_same={bool((seg == seg[0]).all())}", flush=True)
print("SHUFFLE_DONE")

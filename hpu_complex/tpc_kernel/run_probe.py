"""Run the local-memory addressing probe on Gaudi."""
import pathlib

import torch
from torch.utils.cpp_extension import load

CPP = r"""
#include <torch/extension.h>
#include "hpu_custom_op_pt2.h"

static habana::PartialOutputMetaDataVector output_meta(const at::Stack& inputs) {
  auto self = inputs[0].toTensor();
  habana::PartialOutputMetaData meta{self.scalar_type(), self.sizes().vec()};
  return {meta};
}

bool register_it() {
  habana::custom_op::registerUserCustomOp(
      "custom_op::fft_probe", "fft_probe_f32_gaudi2", output_meta, nullptr);
  return true;
}
static bool registered = register_it();

at::Tensor fft_probe(at::Tensor a) {
  std::vector<c10::IValue> inputs{a};
  auto desc = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
      "custom_op::fft_probe");
  auto out = desc.execute(inputs);
  return out[0];
}

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
  m.def("fft_probe(Tensor self) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl("fft_probe", fft_probe);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
  m.impl("fft_probe", [](at::Tensor a) { return a.new_empty(a.sizes()); });
}
"""

from habana_frameworks.torch.utils.lib_utils import get_include_dir, get_lib_dir
build = pathlib.Path("/tmp/tpc_probe_plugin_build")
build.mkdir(exist_ok=True)
src = build / "probe_plugin.cpp"
src.write_text(CPP)
load(name="fft_probe_plugin", sources=[str(src)],
     build_directory=str(build), verbose=False, is_python_module=False,
     extra_include_paths=[get_include_dir(), "/usr/include/habanalabs"],
     extra_ldflags=[f"-L{get_lib_dir()}", "-lhabana_pytorch2_plugin"])
print("PLUGIN LOADED", flush=True)
x = torch.arange(512, dtype=torch.float32, device="hpu") % 64
out = torch.ops.custom_op.fft_probe(x)
torch.hpu.synchronize()
o = out.cpu()
inp = torch.arange(64, dtype=torch.float32)
for slot, name in [(0, "vec roundtrip"), (64, "stride 1"), (128, "stride 4"), (192, "stride 256")]:
    seg = o[slot:slot + 8]
    print(f"slot {name}: {seg.tolist()} match={bool((seg == inp[:8]).all())}", flush=True)
print("PROBE_DONE")

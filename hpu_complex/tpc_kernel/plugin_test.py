"""Pipeline proof: template add kernel via registerUserCustomOp on Gaudi.
Run in pod with GC_KERNEL_PATH=<libcustom_tpc_perf_lib.so>:/usr/lib/habanalabs/libtpc_kernels.so
"""
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
      "custom_op::tpc_add", "custom_add_f32_gaudi2", output_meta, nullptr);
  return true;
}
static bool registered = register_it();

at::Tensor tpc_add(at::Tensor a, at::Tensor b) {
  std::vector<c10::IValue> inputs{a, b};
  auto desc = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
      "custom_op::tpc_add");
  auto out = desc.execute(inputs);
  return out[0];
}

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
  m.def("tpc_add(Tensor self, Tensor other) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl("tpc_add", tpc_add);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
  m.impl("tpc_add", [](at::Tensor a, at::Tensor b) { return a.new_empty(a.sizes()); });
}
"""

here = pathlib.Path(__file__).parent
build = pathlib.Path("/tmp/tpc_plugin_build")
build.mkdir(exist_ok=True)
src = build / "plugin_test.cpp"
src.write_text(CPP)
from habana_frameworks.torch.utils.lib_utils import get_include_dir, get_lib_dir

load(name="tpc_plugin_test", sources=[str(src)],
     build_directory=str(build), verbose=False, is_python_module=False,
     extra_include_paths=[get_include_dir(), "/usr/include/habanalabs"],
     extra_ldflags=[f"-L{get_lib_dir()}", "-lhabana_pytorch2_plugin"])
print("PLUGIN LOADED", flush=True)
a = torch.randn(4, 256, device="hpu")
b = torch.randn(4, 256, device="hpu")
out = torch.ops.custom_op.tpc_add(a, b)
torch.hpu.synchronize()
print("add err:", (out.cpu() - (a.cpu() + b.cpu())).abs().max().item(), flush=True)
print("PIPELINE_DONE")

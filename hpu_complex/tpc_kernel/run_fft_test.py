"""Fused TPC FFT kernel correctness vs torch.fft (Gaudi pod)."""
import pathlib

import torch
from torch.utils.cpp_extension import load

CPP = r"""
#include <torch/extension.h>
#include "hpu_custom_op_pt2.h"

static habana::PartialOutputMetaDataVector output_meta(const at::Stack& inputs) {
  auto self = inputs[0].toTensor();
  auto rows = self.size(0);
  auto n = self.size(1);
  habana::PartialOutputMetaData meta{self.scalar_type(), {rows, 2, n}};
  return {meta};
}

bool register_it() {
  habana::custom_op::registerUserCustomOp(
      "custom_op::fft_r2c", "fft_r2c_f32_gaudi2", output_meta, nullptr);
  return true;
}
static bool registered = register_it();

at::Tensor fft_r2c(at::Tensor x, at::Tensor twid) {
  std::vector<c10::IValue> inputs{x, twid};
  auto desc = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
      "custom_op::fft_r2c");
  auto out = desc.execute(inputs);
  return out[0];
}

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
  m.def("fft_r2c(Tensor self, Tensor twid) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl("fft_r2c", fft_r2c);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
  m.impl("fft_r2c", [](at::Tensor x, at::Tensor t) {
    return x.new_empty({x.size(0), 2, x.size(1)});
  });
}
"""

from habana_frameworks.torch.utils.lib_utils import get_include_dir, get_lib_dir
build = pathlib.Path("/tmp/tpc_fft_plugin_build")
build.mkdir(exist_ok=True)
src = build / "fft_plugin.cpp"
src.write_text(CPP)
load(name="fft_tpc_plugin", sources=[str(src)],
     build_directory=str(build), verbose=False, is_python_module=False,
     extra_include_paths=[get_include_dir(), "/usr/include/habanalabs"],
     extra_ldflags=[f"-L{get_lib_dir()}", "-lhabana_pytorch2_plugin"])
print("PLUGIN LOADED", flush=True)


def make_lut():
    lut = torch.zeros(64, 256, dtype=torch.uint8)
    for p in range(64):
        lut[p, :] = 0x80 | (p & 31) | ((p >> 5) << 5)
    return lut.to("hpu")


def make_twiddles(n):
    j = torch.arange(n, dtype=torch.float32).view(-1, 1)
    k = torch.arange(n, dtype=torch.float32).view(1, -1)
    ang = -2 * torch.pi * j * k / n
    return torch.stack([torch.cos(ang), torch.sin(ang)]).permute(0, 2, 1).contiguous().to("hpu")


torch.manual_seed(0)
for n in (256, 1024):
    x = torch.randn(4, n)
    twid = make_twiddles(n)
    out = torch.ops.custom_op.fft_r2c(x.to("hpu"), twid)
    torch.hpu.synchronize()
    got = out.cpu()  # [rows, 2, n]
    got_c = torch.complex(got[:, 0], got[:, 1])
    ref = torch.fft.fft(x.double())
    err = (got_c.to(torch.complex128) - ref).abs().max().item()
    print(f"n={n} tpc-fft err={err:.2e}", flush=True)
print("FFT_TPC_DONE")

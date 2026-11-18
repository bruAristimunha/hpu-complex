"""Optional fused TPC FFT fast path (Gaudi2, n % 64 == 0, n <= 2048).

Requires the kernel library built by tpc_kernel/build.sh and
GC_KERNEL_PATH pointing at it (plus the stock libtpc_kernels.so).
enable_tpc_fft() loads the plugin and routes aten::_fft_r2c through
the fused kernel via the dispatch mode, so calls must run inside
hpu_complex.complex_mode() (or enable(enter_mode=True)). Everything
else (and any failure) falls back to the GEMM extension path.
"""
import torch

_TWIDDLES = {}
_ENABLED = False

_CPP = r"""
#include <torch/extension.h>
#include "hpu_custom_op_pt2.h"

static habana::PartialOutputMetaDataVector output_meta(const at::Stack& inputs) {
  auto self = inputs[0].toTensor();
  habana::PartialOutputMetaData meta{self.scalar_type(),
                                     {self.size(0), 2, self.size(1)}};
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


def _twiddles(n, device):
    if n not in _TWIDDLES:
        j = torch.arange(n, dtype=torch.float32).view(-1, 1)
        k = torch.arange(n, dtype=torch.float32).view(1, -1)
        ang = -2 * torch.pi * j * k / n
        _TWIDDLES[n] = torch.stack([torch.cos(ang), torch.sin(ang)]) \
            .permute(0, 2, 1).contiguous().to(device)
    return _TWIDDLES[n]


def _r2c_via_tpc(x, dim, normalization, onesided):
    """aten::_fft_r2c via the fused kernel, or None to fall back."""
    if not (onesided and normalization == 0 and x.dtype == torch.float32):
        return None
    n = x.size(-1)
    if n % 64 != 0 or n > 2048 or (dim not in ([-1], [x.dim() - 1])):
        return None
    rows = x.numel() // n
    out = torch.ops.custom_op.fft_r2c(
        x.reshape(rows, n).contiguous(), _twiddles(n, x.device))
    k = n // 2 + 1
    # stack of contiguous selects (Habana transpose rejects strided views)
    pair = torch.stack(
        [out[:, 0, :k].contiguous(), out[:, 1, :k].contiguous()], dim=-1)
    # Reshape the float pair, not the complex view (Habana as_strided
    # rejects complex tensors).
    pair = pair.reshape(*x.shape[:-1], k, 2)
    result = torch.view_as_complex(pair.contiguous())
    # Absorb the one-shot strided_view compile failure that follows a
    # custom-kernel graph, so the caller's next view op does not hit it.
    try:
        torch.view_as_real(result)
    except RuntimeError:
        torch.view_as_real(result)
    return result


def enable_tpc_fft():
    global _ENABLED
    if _ENABLED:
        return True
    from torch.utils.cpp_extension import load_inline

    from habana_frameworks.torch.utils.lib_utils import \
        get_include_dir, get_lib_dir

    # Built in torch's default extension directory (TORCH_EXTENSIONS_DIR, else ~/.cache/torch_extensions).
    load_inline(name="hpu_complex_tpc_plugin", cpp_sources=[_CPP], verbose=False, is_python_module=False,
                extra_include_paths=[get_include_dir(), "/usr/include/habanalabs"],
                extra_ldflags=[f"-L{get_lib_dir()}", "-lhabana_pytorch2_plugin"])
    from . import mode as _mode

    _mode.FFT_R2C_HOOK = _r2c_via_tpc
    _ENABLED = True
    return True

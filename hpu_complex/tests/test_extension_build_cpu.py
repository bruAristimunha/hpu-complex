"""The C++ extension builds against stock (CPU) torch, and registers the kernels its entry points rely on.

Without a Gaudi the kernels cannot run (they allocate HPU memory), but the dispatcher table shows which keys they
take. The build runs in a subprocess (it overrides BackendSelect factories for the whole process), cached under
TORCH_EXTENSIONS_DIR. Run: python -m pytest hpu_complex/tests/test_extension_build_cpu.py -q
"""
import json
import pathlib
import shutil
import subprocess
import sys

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "cpp_ext" / "fft_ext.cpp"

_SCRIPT = """
import json, sys, torch
from torch.utils.cpp_extension import load
load(name="hpu_complex_cpu_build_check", sources=[sys.argv[1]], is_python_module=False, verbose=False)
ops = ["aten::view.dtype", "aten::view_as_complex", "aten::view_as_real", "aten::mm.out", "aten::copy_",
       "aten::mul.Tensor", "aten::add_.Scalar", "aten::rsub.Tensor", "aten::div.out_mode",
       "aten::cat", "aten::cat.out", "aten::exp", "aten::exp.out", "aten::neg", "aten::neg.out",
       "aten::masked_fill_.Scalar", "aten::masked_fill_.Tensor", "aten::linalg_vector_norm",
       "aten::linalg_vector_norm.out", "aten::mul.Scalar_out", "aten::div.Tensor_mode", "aten::add.out",
       "aten::angle", "aten::angle.out", "aten::eq.Scalar_out",
       "aten::im2col", "aten::im2col.out", "aten::col2im", "aten::col2im.out"]
table = {op: [line.split(":")[0] for line in torch._C._dispatch_dump(op).splitlines() if "fft_ext" in line]
         for op in ops}
# The complex-scalar kernels at AutogradHPU hand every other call to AutogradOther: its entry must be torch's
# autograd kernel (VariableType), the one AutogradHPU had before.
table["autograd_other"] = {op: next(line for line in torch._C._dispatch_dump_table(op).splitlines()
                                    if line.startswith("AutogradOther")) for op in ops[5:9]}
print("TABLE" + json.dumps(table))
"""


@pytest.fixture(scope="module")
def registered():
    if shutil.which("c++") is None and shutil.which("g++") is None:
        pytest.skip("no C++ compiler")
    proc = subprocess.run([sys.executable, "-c", _SCRIPT, str(SRC)], check=False, capture_output=True, text=True, timeout=1200)
    lines = [line for line in proc.stdout.splitlines() if line.startswith("TABLE")]
    assert proc.returncode == 0 and lines, proc.stderr[-3000:]
    return json.loads(lines[-1][len("TABLE"):])


def test_real_to_complex_view_dtype_has_an_hpu_kernel(registered):
    """x_hpu.view(torch.complex64) must reach the extension at the HPU key (else torch's composite keeps the HPU key
    on a complex dtype: "Complex datatype is not supported on HPU device"); z.view(float) at PrivateUse1."""
    assert "HPU" in registered["aten::view.dtype"]
    assert "PrivateUse1" in registered["aten::view.dtype"]


def test_entry_points_are_registered(registered):
    assert "HPU" in registered["aten::view_as_complex"]
    assert {"PrivateUse1"} <= set(registered["aten::view_as_real"]) & set(registered["aten::mm.out"]) & set(
        registered["aten::copy_"])


def test_complex_scalar_kernels_sit_at_autograd_hpu(registered):
    """x_hpu * 2j and friends are promoted at AutogradHPU, before Habana's float kernels (lazy mode aborts there)."""
    for op in ("aten::mul.Tensor", "aten::add_.Scalar", "aten::rsub.Tensor", "aten::div.out_mode"):
        assert "AutogradHPU" in registered[op], op


def test_complex_scalar_kernels_also_sit_at_backend_select(registered):
    """The same calls when they skip the autograd keys (torch.inference_mode, inference tensors) are taken at
    BackendSelect, which every call's key set holds, below autograd and above Habana's HPU kernels."""
    for op in ("aten::mul.Tensor", "aten::add_.Scalar", "aten::rsub.Tensor", "aten::div.out_mode",
               "aten::mul.Scalar_out", "aten::div.Tensor_mode", "aten::add.out"):
        assert "BackendSelect" in registered[op], op


def test_other_calls_reach_torch_autograd_kernel_through_autograd_other(registered):
    for op, line in registered["autograd_other"].items():
        assert "VariableType" in line and "fft_ext" not in line, (op, line)


def test_green_ops_have_card_kernels(registered):
    """Green's complex cat, exp, neg, masked_fill_ and linalg.vector_norm run on the pairs, not the CPU fallback."""
    for op in ("aten::cat", "aten::cat.out", "aten::exp", "aten::exp.out", "aten::neg", "aten::neg.out",
               "aten::masked_fill_.Scalar", "aten::masked_fill_.Tensor", "aten::linalg_vector_norm",
               "aten::linalg_vector_norm.out", "aten::angle", "aten::angle.out", "aten::eq.Scalar_out",
               "aten::im2col", "aten::im2col.out", "aten::col2im", "aten::col2im.out"):
        assert "PrivateUse1" in registered[op], op

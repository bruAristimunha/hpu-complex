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
ops = ["aten::view.dtype", "aten::view_as_complex", "aten::view_as_real", "aten::mm.out", "aten::copy_"]
table = {op: [line.split(":")[0] for line in torch._C._dispatch_dump(op).splitlines() if "fft_ext" in line]
         for op in ops}
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

"""CPU checks of the DS complex128 route (HPU_COMPLEX_DS=1): add/sub/mul/div, integer pow and
abs of host complex128 tensors compute from (hi, lo) complex64 pairs at ~48-bit precision
instead of the host float64 path. Opt-in: without the env the host path stays bit-exact."""
import importlib.util
import os
import pathlib
import shutil
import sys
import types
import warnings

import pytest
import torch

os.environ["HPU_COMPLEX_DS"] = "1"
HERE = pathlib.Path(__file__).parent

if importlib.util.find_spec("habana_frameworks") is not None:
    pytest.skip("Gaudi image: card checks cover DS", allow_module_level=True)
if shutil.which("ninja") is None:
    pytest.skip("ninja not found: cannot build the extension", allow_module_level=True)


@pytest.fixture(scope="module", autouse=True)
def extension():
    from torch.utils.cpp_extension import load

    from hpu_complex.extension import load_extension

    stub = types.ModuleType("torch.hpu")
    stub._lazy_init = lambda: None
    stub.is_available = lambda: False
    stub.is_current_stream_capturing = lambda: False
    saved, saved_attr = sys.modules.get("torch.hpu"), getattr(torch, "hpu", None)
    sys.modules["torch.hpu"] = stub
    torch.hpu = stub
    try:
        load_extension()
        load(name="hpu_guard_stub", sources=[str(HERE / "cpu_stub" / "hpu_guard_stub.cpp")],
             is_python_module=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        if saved is None:
            sys.modules.pop("torch.hpu", None)
        else:
            sys.modules["torch.hpu"] = saved
        if saved_attr is None:
            del torch.hpu
        else:
            torch.hpu = saved_attr


HPU = torch.device("hpu")


def _pair():
    torch.manual_seed(0)
    mk = lambda: (torch.randn(64, 256, dtype=torch.float64)
                  + 1j * torch.randn(64, 256, dtype=torch.float64)).to(torch.complex128)  # noqa: E731
    return mk(), mk()


def _check(fn, tol=1e-12):
    z, w = _pair()
    got = fn(z.to(HPU), w.to(HPU)).cpu()
    ref = fn(z, w)
    rel = ((got - ref).abs() / ref.abs().clamp_min(1e-300)).max().item()
    assert rel < tol, rel


def test_ds_arithmetic():
    _check(lambda a, b: a * b)
    _check(lambda a, b: a / b)
    _check(lambda a, b: a + b)
    _check(lambda a, b: a - b)


def test_ds_pow_and_abs():
    z, _ = _pair()
    got = (z.to(HPU) ** 3).cpu()
    rel = ((got - z**3).abs() / (z**3).abs().clamp_min(1e-300)).max().item()
    assert rel < 1e-12, rel
    mag = z.to(HPU).abs().cpu()
    rel = ((mag - z.abs()).abs() / z.abs().clamp_min(1e-300)).max().item()
    assert rel < 1e-12, rel

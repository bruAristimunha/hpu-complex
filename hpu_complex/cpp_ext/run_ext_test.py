"""Load the HPU extension and run the issue's cases, unmodified torch API."""
import pathlib

import torch
from torch.utils.cpp_extension import load

here = pathlib.Path(__file__).parent
pathlib.Path("/tmp/fft_ext_build").mkdir(exist_ok=True)
load(name="hpu_complex_ext", sources=[str(here / "fft_ext.cpp")],
     build_directory="/tmp/fft_ext_build", verbose=False,
     is_python_module=False)
print("EXT LOADED")
import sys
sys.path.insert(0, str(here.parent.parent))
from hpu_complex.mode import MODE

MODE.__enter__()
print("MODE ACTIVE")

torch.manual_seed(0)
x = torch.randn(4, 200)
xh = x.to("hpu")


def show(name, fn, check=None):
    try:
        y = fn()
        torch.hpu.synchronize()
        extra = check(y) if check else ""
        print(f"{name}: OK {y.dtype} {y.device} {extra}")
        return y
    except Exception as e:
        print(f"{name}: {type(e).__name__}: {str(e)[:110]}")


z = show("torch.fft.rfft(x)", lambda: torch.fft.rfft(xh),
         lambda y: f"err={(torch.view_as_real(y).cpu() - torch.view_as_real(torch.fft.rfft(x))).abs().max():.2e}")
show("roundtrip irfft(rfft)", lambda: torch.fft.irfft(z, n=200),
     lambda y: f"err={(y.cpu() - x).abs().max():.2e}")
show("torch.complex(x,x)", lambda: torch.complex(xh, xh))
show("view_as_complex", lambda: torch.view_as_complex(xh.view(4, 100, 2).contiguous()))
show("torch.stft", lambda: torch.stft(xh, 200, 100, return_complex=True))
show("rfft.abs()", lambda: torch.fft.rfft(xh).abs())
show("zeros complex64 hpu", lambda: torch.zeros(2, dtype=torch.complex64, device="hpu"))


def _backward():
    v = torch.randn(4, 256, device="hpu", requires_grad=True)
    loss = torch.fft.rfft(v).abs().sum()
    loss.backward()
    return v.grad


show("rfft.abs().backward()", _backward,
     lambda g: f"grad_finite={bool(torch.isfinite(g).all())}")
print("EXT_TEST_DONE")

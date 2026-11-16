"""On-device checks. Run in the Gaudi pod: python3 -m pytest tests/test_on_hpu.py
Asserts the float FFT core computes correctly ON HPU (vs CPU torch.fft),
gradients flow, and reports torch.compile behavior (expected: decomp path).
"""
import torch

import hpu_complex as H


def test_core_fft_on_hpu():
    # Gaudi computes float64 as float32 (syn Double->float): test in float32
    # against a float64 CPU reference, relative to peak magnitude.
    torch.manual_seed(0)
    for n in (200, 1024, 1920):
        x = torch.randn(4, n)
        got = H.magnitude(H.rfft(x.to("hpu"))).cpu()
        ref = torch.fft.rfft(x.double()).abs()
        rel = ((got - ref.float()).abs().max() / ref.max()).item()
        print(f"n={n} on-HPU magnitude rel_err={rel:.2e}")
        assert rel < 1e-3, n


def test_stft_and_grad_on_hpu():
    x = torch.randn(2, 4000, device="hpu", requires_grad=True)
    mag = H.stft_magnitude(x, 512, 256)
    assert mag.device.type == "hpu"
    mag.sum().backward()
    assert torch.isfinite(x.grad).all()
    print(f"stft_magnitude {tuple(mag.shape)} grad ok")


def test_compile_attempt():
    H.enable()
    f = torch.compile(lambda t: H.magnitude(H.rfft(t)), backend="hpu_backend")
    x = torch.randn(2, 1024, device="hpu")
    y = f(x)
    torch.hpu.synchronize()
    assert y.device.type == "hpu" and y.shape == (2, 513)
    print(f"compile via core ops: OK {tuple(y.shape)} {y.device}")

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


def test_complex_without_mode_and_clip_grad():
    # Default enable(): no dispatch mode entered, yet complex forward
    # (via the C++ .out kernels) and clip_grad_norm_ (0-d total_norm:
    # regression for the stack({a,b}, -2) crash in float_addsub) work.
    H.enable()
    a = torch.complex(torch.randn(4, 8, device="hpu"),
                      torch.randn(4, 8, device="hpu"))
    b = torch.complex(torch.randn(4, 8, device="hpu"),
                      torch.randn(4, 8, device="hpu"))
    ac, bc = a.cpu(), b.cpu()
    for got, ref in [(a * b, ac * bc), (a / b, ac / bc), (a + b, ac + bc)]:
        err = (got.cpu() - ref).abs().max() / ref.abs().max()
        assert err < 1e-4, err
    assert torch.isfinite(a.abs()).all()
    m = torch.nn.Sequential(
        torch.nn.Conv1d(22, 32, 5), torch.nn.Flatten(),
        torch.nn.Linear(32 * 10, 4)).to("hpu")
    loss = m(torch.randn(8, 22, 14, device="hpu")).sum()
    loss.backward()
    n = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    torch.hpu.synchronize()
    assert torch.isfinite(n)
    with H.complex_mode():  # same clip inside the scoped mode
        loss = m(torch.randn(8, 22, 14, device="hpu")).sum()
        loss.backward()
        n = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        torch.hpu.synchronize()
        assert torch.isfinite(n)
    print("complex without mode + clip_grad_norm_ (both mode states): OK")


def test_compile_attempt():
    H.enable()
    f = torch.compile(lambda t: H.magnitude(H.rfft(t)), backend="hpu_backend")
    x = torch.randn(2, 1024, device="hpu")
    y = f(x)
    torch.hpu.synchronize()
    assert y.device.type == "hpu" and y.shape == (2, 513)
    print(f"compile via core ops: OK {tuple(y.shape)} {y.device}")

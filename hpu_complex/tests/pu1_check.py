"""On one Gaudi: the PrivateUse1 extension against torch on the CPU. One line per check; python3 pu1_check.py"""
import traceback

import habana_frameworks.torch.core  # noqa: F401
import torch

import hpu_complex

hpu = torch.device("hpu")
hpu_complex.enable()
torch.manual_seed(0)
results = []


def check(name, fn, atol):
    try:
        err = fn()
        torch.hpu.synchronize()
        ok = err <= atol
        print(f"{name:44s} max|err| {err:.1e} {'OK' if ok else 'FAIL'}", flush=True)
    except Exception as e:
        ok = False
        print(f"{name:44s} ERROR {type(e).__name__}: {str(e).splitlines()[0][:160]}", flush=True)
        traceback.print_exc(limit=4)
    results.append(ok)


def err(a, b):
    a, b = a.detach().cpu(), b.detach().cpu()
    if a.is_complex():
        a, b = torch.view_as_real(a), torch.view_as_real(b.to(a.dtype))
    return (a.double() - b.double()).abs().max().item()


def grad_err(f, x, atol_shape=None):
    """f(x) on HPU vs CPU: output and input gradient (sum of outputs)."""
    xc = x.clone().requires_grad_()
    xh = x.to(hpu).requires_grad_()
    yc, yh = f(xc), f(xh)
    # Contiguous ones, not .sum(): this image's oneMKL rejects the stride-0 grad of sum() in irfft's backward.
    yc.backward(torch.ones_like(yc))
    yh.backward(torch.ones_like(yh))
    return max(err(yh, yc), err(xh.grad, xc.grad))


x = torch.randn(6, 30, 200)
for norm in ("backward", "forward", "ortho"):
    check(f"rfft norm={norm}", lambda: err(torch.fft.rfft(x.to(hpu), norm=norm), torch.fft.rfft(x, norm=norm)),
          2e-4 if norm == "backward" else 2e-5)
z = torch.fft.rfft(x)
check("irfft(rfft) n=200", lambda: err(torch.fft.irfft(torch.fft.rfft(x.to(hpu)), n=200), x), 1e-5)
check("irfft odd n=201", lambda: err(torch.fft.irfft(z.to(hpu), n=201), torch.fft.irfft(z, n=201)), 1e-5)
zc = torch.randn(4, 64, dtype=torch.complex64)
check("fft / ifft complex", lambda: max(err(torch.fft.fft(zc.to(hpu)), torch.fft.fft(zc)),
                                         err(torch.fft.ifft(zc.to(hpu)), torch.fft.ifft(zc))), 1e-4)
a, b = torch.randn(4, 8), torch.randn(4, 8)
check("torch.complex", lambda: err(torch.complex(a.to(hpu), b.to(hpu)), torch.complex(a, b)), 0)
check("view_as_complex", lambda: err(torch.view_as_complex(x[..., :200].reshape(6, 30, 100, 2).to(hpu)),
                                     torch.view_as_complex(x.reshape(6, 30, 100, 2))), 0)
check("zeros complex64 hpu", lambda: torch.zeros(3, 5, dtype=torch.complex64, device=hpu).abs().sum().item(), 0)
ca, cb = torch.complex(a, b), torch.complex(b, a)
check("complex * / + -", lambda: max(err(ca.to(hpu) * cb.to(hpu), ca * cb), err(ca.to(hpu) / cb.to(hpu), ca / cb),
                                     err(ca.to(hpu) + cb.to(hpu), ca + cb), err(ca.to(hpu) - 2 * cb.to(hpu), ca - 2 * cb)),
      1e-5)
check("complex * real, complex * 2.5", lambda: max(err(ca.to(hpu) * a.to(hpu), ca * a), err(ca.to(hpu) * 2.5, ca * 2.5)),
      1e-5)
check("rfft(norm=forward).abs() fwd+grad", lambda: grad_err(lambda t: torch.fft.rfft(t, norm="forward").abs(), x),
      1e-5)
signal = torch.randn(8, 2000)
check("stft(200, 100).abs() fwd+grad", lambda: grad_err(
    lambda t: torch.stft(t, 200, 100, center=False, onesided=True, return_complex=True,
                         window=torch.ones(200, device=t.device)).abs(), signal), 2e-3)
check(".real/.imag grad", lambda: grad_err(lambda t: torch.fft.rfft(t).real + 2 * torch.fft.rfft(t).imag, x), 2e-3)
kernel = torch.randn(30, 200)
check("irfft(rfft(x) * rfft(k)) fwd+grad (conj)", lambda: grad_err(  # FFT convolution: complex x complex backward
    lambda t: torch.fft.irfft(torch.fft.rfft(t, n=400) * torch.fft.rfft(kernel.to(t.device), n=400), n=400), x),
      2e-3)
check("conj().resolve_conj()", lambda: err(ca.to(hpu).conj().resolve_conj(), ca.conj().resolve_conj()), 0)
check("complex sum(dim) / sum()", lambda: max(err(ca.to(hpu).sum(0), ca.sum(0)), err(ca.to(hpu).sum(), ca.sum())), 1e-5)
shared_kernel = torch.randn(1, 30, 200)  # broadcast over the batch: its gradient is a complex sum (SGConv)
check("broadcast complex product fwd+grad", lambda: grad_err(
    lambda t: torch.fft.irfft(torch.fft.rfft(t, n=400) * torch.fft.rfft(t[:1] + shared_kernel.to(t.device), n=400),
                              n=400), x), 5e-3)


def autocast_case():
    with torch.autocast("hpu", dtype=torch.bfloat16):
        out = torch.fft.rfft(x.to(hpu), norm="forward").abs()
    assert out.dtype == torch.float32, out.dtype
    return err(out, torch.fft.rfft(x, norm="forward").abs())


check("rfft under bf16 autocast stays float32", autocast_case, 2e-5)


def float_ops():  # the ops that crashed or were taxed with the HPU-key overrides
    w = torch.zeros(10, 25, device=hpu)
    w[3].copy_(torch.ones(25, device=hpu))
    mask = torch.ones(4, 90, dtype=torch.bool).to(hpu, non_blocking=True)
    y = (w.abs() * 2 + 1)[2:5].clone()
    return abs(w.sum().item() - 25) + abs(mask.sum().item() - 360) + abs(y.sum().item() - 125)


check("plain float ops (copy_, bool H2D, abs, slice)", float_ops, 0)


def compiled_case():
    f = torch.compile(lambda t: torch.fft.rfft(t, norm="forward").abs(), backend="hpu_backend")
    return grad_err(f, x)


check("compiled rfft.abs fwd+grad", compiled_case, 1e-5)


def compiled_stft_case():
    f = torch.compile(lambda t: torch.stft(t, 200, 100, center=False, onesided=True, return_complex=True,
                                           window=torch.ones(200, device=t.device)).abs(), backend="hpu_backend")
    return grad_err(f, signal)


check("compiled stft(200, 100).abs() fwd+grad", compiled_stft_case, 2e-3)
print(f"PU1_CHECK {sum(results)}/{len(results)} passed", flush=True)

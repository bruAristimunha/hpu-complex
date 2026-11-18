"""On one Gaudi: the PrivateUse1 extension against torch on the CPU. One line per check; exits 1 if any fails.

python3 pu1_check.py
"""
import io
import math
import sys
import traceback

import habana_frameworks.torch.core  # noqa: F401
import torch
from torch._dynamo.utils import counters

import hpu_complex

hpu = torch.device("hpu")
hpu_complex.enable()
torch.manual_seed(0)
results = []


def worst(*errors):
    """The largest error, NaN if any is NaN: Python's max keeps its first argument against a NaN."""
    return math.nan if any(math.isnan(e) for e in errors) else max(errors)


def check(name, fn, atol):
    """fn returns an error, or (error, note); NaN fails."""
    try:
        out = fn()
        torch.hpu.synchronize()
        error, note = out if isinstance(out, tuple) else (out, "")
        ok = error <= atol  # False for NaN
        print(f"{name:58s} max|err| {error:.1e} {'OK' if ok else 'FAIL'}{'  ' + note if note else ''}", flush=True)
    except Exception as e:  # noqa: BLE001
        ok = False
        first_line = (str(e).splitlines() or [""])[0]
        print(f"{name:58s} ERROR {type(e).__name__}: {first_line[:160]}", flush=True)
        traceback.print_exc(limit=4)
    results.append(ok)


def err(a, b):
    a, b = a.detach().cpu(), b.detach().cpu()
    assert a.shape == b.shape, (tuple(a.shape), tuple(b.shape))
    if a.is_complex():
        a, b = torch.view_as_real(a.resolve_conj()), torch.view_as_real(b.to(a.dtype).resolve_conj())
    return (a.double() - b.double()).abs().max().item()


def nan_aware_err(a, b):
    """max |a - b| of two float tensors, equal infinities counting as equal; NaN unless their NaNs coincide."""
    a, b = a.double(), b.double()
    if a.shape != b.shape or not torch.equal(a.isnan(), b.isnan()):
        return math.nan
    return torch.where((a == b) | a.isnan(), 0.0, a - b).abs().max().item()


def tagged(t):
    return torch._C._dispatch_keys(t).has(torch._C.DispatchKey.PrivateUse1)


def grad_err(f, x, relative=False, reference=None):
    """f(x) on HPU vs reference(x) (default f) on the CPU: output and input gradient, absolute or relative to the
    largest reference value."""
    xc = x.clone().requires_grad_()
    xh = x.to(hpu).requires_grad_()
    yc, yh = (reference or f)(xc), f(xh)
    # Contiguous ones, not .sum(): this image's oneMKL rejects the stride-0 grad of sum() in irfft's backward.
    yc.backward(torch.ones_like(yc))
    yh.backward(torch.ones_like(yh))
    pairs = ((yh, yc), (xh.grad, xc.grad))
    return worst(*(err(got, ref) / (ref.detach().abs().max().item() if relative else 1.0) for got, ref in pairs))


x = torch.randn(6, 30, 200)


def inference_warmup_then_real_grad():
    # Must run first: the process's first complex ops under inference_mode (a validation pass before training)
    # once left every later float view of a complex tensor without autograd keys.
    with torch.inference_mode():
        torch.fft.rfft(x.to(hpu)).real.sum().item()
    return grad_err(lambda t: torch.fft.rfft(t).real, x)


def first_derivative(t):
    z = torch.fft.rfft(t)
    (g,) = torch.autograd.grad((z * z).real.sum(), t, create_graph=True)
    return g


check("inference_mode warm-up, then .real grad", inference_warmup_then_real_grad, 2e-3)
# Right after the warm-up: the basis it cached is saved by the recorded backward GEMM.
check("double backward after the warm-up (relative error)", lambda: grad_err(first_derivative, x, relative=True), 1e-5)
for norm in ("backward", "forward", "ortho"):
    check(f"rfft norm={norm}", lambda: err(torch.fft.rfft(x.to(hpu), norm=norm), torch.fft.rfft(x, norm=norm)),
          2e-4 if norm == "backward" else 2e-5)
z = torch.fft.rfft(x)
check("irfft(rfft) n=200", lambda: err(torch.fft.irfft(torch.fft.rfft(x.to(hpu)), n=200), x), 1e-5)
check("irfft odd n=201", lambda: err(torch.fft.irfft(z.to(hpu), n=201), torch.fft.irfft(z, n=201)), 1e-5)
zc = torch.randn(4, 64, dtype=torch.complex64)
check("fft / ifft complex", lambda: worst(err(torch.fft.fft(zc.to(hpu)), torch.fft.fft(zc)),
                                          err(torch.fft.ifft(zc.to(hpu)), torch.fft.ifft(zc))), 1e-4)
a, b = torch.randn(4, 8), torch.randn(4, 8)
check("torch.complex", lambda: err(torch.complex(a.to(hpu), b.to(hpu)), torch.complex(a, b)), 0)
check("view_as_complex", lambda: err(torch.view_as_complex(x[..., :200].reshape(6, 30, 100, 2).to(hpu)),
                                     torch.view_as_complex(x.reshape(6, 30, 100, 2))), 0)
check("zeros complex64 hpu", lambda: torch.zeros(3, 5, dtype=torch.complex64, device=hpu).abs().sum().item(), 0)
ca, cb = torch.complex(a, b), torch.complex(b, a)
check("complex * / + -", lambda: worst(err(ca.to(hpu) * cb.to(hpu), ca * cb), err(ca.to(hpu) / cb.to(hpu), ca / cb),
                                       err(ca.to(hpu) + cb.to(hpu), ca + cb),
                                       err(ca.to(hpu) - 2 * cb.to(hpu), ca - 2 * cb)), 1e-5)
check("complex * real, complex * 2.5", lambda: worst(err(ca.to(hpu) * a.to(hpu), ca * a),
                                                     err(ca.to(hpu) * 2.5, ca * 2.5)), 1e-5)
check("z * 1j, 1j * z, z + 1j, z / (1 + 1j) (complex128 scalars)", lambda: worst(
    err(ca.to(hpu) * 1j, ca * 1j), err(1j * ca.to(hpu), 1j * ca), err(ca.to(hpu) + 1j, ca + 1j),
    err(ca.to(hpu) / (1 + 1j), ca / (1 + 1j))), 1e-6)
check("rfft(norm=forward).abs() fwd+grad", lambda: grad_err(lambda t: torch.fft.rfft(t, norm="forward").abs(), x),
      1e-5)
signal = torch.randn(8, 2000)
check("stft(200, 100).abs() fwd+grad", lambda: grad_err(
    lambda t: torch.stft(t, 200, 100, center=False, onesided=True, return_complex=True,
                         window=torch.ones(200, device=t.device)).abs(), signal), 2e-3)
check(".real/.imag grad", lambda: grad_err(lambda t: torch.fft.rfft(t).real + 2 * torch.fft.rfft(t).imag, x), 2e-3)


def ihfft_parts(t):
    spectrum = torch.fft.ihfft(t)  # a conjugate view
    return spectrum.real + 2 * spectrum.imag + spectrum.abs()


check("ihfft .real/.imag/.abs() fwd+grad (conjugate view)", lambda: grad_err(ihfft_parts, x), 1e-5)
kernel = torch.randn(30, 200)
check("irfft(rfft(x) * rfft(k)) fwd+grad (conj)", lambda: grad_err(  # FFT convolution: complex x complex backward
    lambda t: torch.fft.irfft(torch.fft.rfft(t, n=400) * torch.fft.rfft(kernel.to(t.device), n=400), n=400), x),
      2e-3)
check("irfft(rfft(x) * rfft(k).conj()) fwd+grad, k from x", lambda: grad_err(  # cross-correlation
    lambda t: torch.fft.irfft(torch.fft.rfft(t, n=400) * torch.fft.rfft(t.flip(-1) + kernel.to(t.device), n=400).conj(),
                              n=400), x), 2e-3)
# Bins 1..99 only: DC and Nyquist are real, and their angle (0 or +-pi) flips with the sign of a zero imaginary part.
check("torch.angle(rfft(x)) fwd+grad", lambda: grad_err(lambda t: torch.angle(torch.fft.rfft(t)[..., 1:100]), x), 1e-3)
check("conj().resolve_conj()", lambda: err(ca.to(hpu).conj().resolve_conj(), ca.conj().resolve_conj()), 0)
check("complex sum(dim) / sum()", lambda: worst(err(ca.to(hpu).sum(0), ca.sum(0)), err(ca.to(hpu).sum(), ca.sum())),
      1e-5)
shared_kernel = torch.randn(1, 30, 200)  # broadcast over the batch: its gradient is a complex sum (SGConv)
check("broadcast complex product fwd+grad", lambda: grad_err(
    lambda t: torch.fft.irfft(torch.fft.rfft(t, n=400) * torch.fft.rfft(t[:1] + shared_kernel.to(t.device), n=400),
                              n=400), x), 5e-3)
check("rfft backward from a strided gradient (cat of spectra)", lambda: grad_err(
    lambda t: torch.cat([torch.fft.rfft(t, norm="forward"), torch.fft.rfft(t.flip(-1), norm="forward")], -1).abs(),
    x), 1e-5)


def sgn_special_values():
    re = torch.tensor([math.nan, math.inf, 0.0, 3.0, -2.0, 0.0])
    im = torch.tensor([0.0, 0.0, 0.0, 4.0, math.inf, -0.0])
    values = torch.complex(re, im)
    pair, modulus = torch.view_as_real(values), values.abs().unsqueeze(-1)
    expected = torch.where(modulus == 0, 0.0, pair / modulus)  # torch._refs.sgn: where(|z| == 0, 0, z / |z|)
    return nan_aware_err(torch.view_as_real(torch.sgn(values.to(hpu)).cpu()), expected)


check("sgn of NaN / inf / 0 like torch._refs.sgn", sgn_special_values, 1e-7)


def complex_factories():
    xr = torch.randn(3, 4)
    xh = xr.to(hpu)
    made = {  # name: (HPU tensor, CPU reference or None)
        "x.to(complex64)": (xh.to(torch.complex64), xr.to(torch.complex64)),
        "zeros_like": (torch.zeros_like(xh, dtype=torch.cfloat), torch.zeros(3, 4, dtype=torch.cfloat)),
        "empty_like": (torch.empty_like(xh, dtype=torch.cfloat), None),
        "new_zeros": (xh.new_zeros((2, 3), dtype=torch.cfloat), torch.zeros(2, 3, dtype=torch.cfloat)),
        "full": (torch.full((2, 3), 1 + 2j, dtype=torch.cfloat, device=hpu),
                 torch.full((2, 3), 1 + 2j, dtype=torch.cfloat)),
        "zeros": (torch.zeros(2, 3, dtype=torch.cfloat, device=hpu), torch.zeros(2, 3, dtype=torch.cfloat)),
        "randn": (torch.randn(2, 3, dtype=torch.cfloat, device=hpu), None),
        "scalar_tensor": (torch.scalar_tensor(1 + 2j, dtype=torch.cfloat, device=hpu),
                          torch.scalar_tensor(1 + 2j, dtype=torch.cfloat)),
    }
    errors = []
    for name, (t, ref) in made.items():
        assert t.dtype == torch.complex64 and tagged(t), (name, t.dtype, torch._C._dispatch_keys(t))
        errors.append(err(t, ref) if ref is not None else 0.0)
    assert torch.isfinite(torch.view_as_real(made["randn"][0].cpu())).all()
    return worst(*errors)


check("complex64 factories (to, *_like, new_*, full, randn, ...)", complex_factories, 0)


def autocast_case():
    with torch.autocast("hpu", dtype=torch.bfloat16):
        out = torch.fft.rfft(x.to(hpu), norm="forward").abs()
    assert out.dtype == torch.float32, out.dtype
    return err(out, torch.fft.rfft(x, norm="forward").abs())


check("rfft under bf16 autocast stays float32", autocast_case, 2e-5)


def autocast_arithmetic():
    za, zb = ca.to(hpu), cb.to(hpu)
    with torch.autocast("hpu", dtype=torch.bfloat16):
        product, quotient, total = za * zb, za / zb, za + zb
    assert product.dtype == quotient.dtype == total.dtype == torch.complex64, (product.dtype, quotient.dtype)
    return worst(err(product, ca * cb), err(quotient, ca / cb), err(total, ca + cb))


check("z1 * z2, z1 / z2, z1 + z2 under bf16 autocast", autocast_arithmetic, 1e-5)


def load_to_hpu():
    buffer = io.BytesIO()
    torch.save({"from_cpu": ca, "from_hpu": ca.to(hpu), "strided_view": ca.to(hpu)[1:, ::2]}, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, map_location="hpu")
    for name, t in loaded.items():
        assert t.device.type == "hpu" and tagged(t), (name, t.device, torch._C._dispatch_keys(t))
    return worst(err(loaded["from_cpu"], ca), err(loaded["from_hpu"], ca), err(loaded["strided_view"], ca[1:, ::2]))


check("torch.load(map_location='hpu') of complex tensors", load_to_hpu, 0)


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


def compiled_dynamic_case():
    # torch 2.6's fake tensors reject symbolic shapes for aten._fft_r2c (its meta strides are wrong), so dynamo
    # breaks the graph there: rfft and its backward run eagerly, .abs() stays compiled for every batch shape.
    graphs, breaks = counters["stats"]["unique_graphs"], sum(counters["graph_break"].values())

    def fn(t):
        return torch.fft.rfft(t, norm="forward").abs()

    f = torch.compile(fn, backend="hpu_backend", dynamic=True)
    error = worst(*(grad_err(f, t, reference=fn) for t in (x, x[:4], x[:, :17].contiguous())))
    return error, (f"({counters['stats']['unique_graphs'] - graphs} graph(s) for 3 batch shapes, "
                   f"{sum(counters['graph_break'].values()) - breaks} graph break(s))")


check("compiled(dynamic=True) rfft.abs fwd+grad, 3 batch shapes", compiled_dynamic_case, 1e-5)


def compiled_non_last_dim():
    f = torch.compile(lambda t: torch.fft.rfft(t, dim=0).abs(), backend="hpu_backend")
    try:
        f(x.to(hpu).requires_grad_())
    except Exception as e:  # noqa: BLE001
        return 0.0 if "only last-dim FFT" in str(e) else math.nan
    return math.nan  # accepted: its backward would be a last-dim GEMM


check("compiled rfft(dim=0) raises the eager kernel's error", compiled_non_last_dim, 0)


def compiled_stft_case():
    f = torch.compile(lambda t: torch.stft(t, 200, 100, center=False, onesided=True, return_complex=True,
                                           window=torch.ones(200, device=t.device)).abs(), backend="hpu_backend")
    return grad_err(f, signal)


check("compiled stft(200, 100).abs() fwd+grad", compiled_stft_case, 2e-3)
passed = sum(results)
print(f"PU1_CHECK {passed}/{len(results)} passed", flush=True)
sys.exit(0 if passed == len(results) else 1)

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

# A permuted 5-D complex tensor: its float pair is a non-contiguous rank-6 view, which Habana misreads as a bare
# TensorImpl or an as_strided view (MetaNeuromotorHand's cross-spectral density: 98% off); sliced too, which is not a
# permutation of a contiguous block.
z5 = torch.complex(torch.randn(2, 5, 3, 4, 6), torch.randn(2, 5, 3, 4, 6))
for layout, view in (("permuted", lambda t: t.permute(0, 1, 3, 4, 2)),
                     ("sliced permuted", lambda t: t.permute(0, 2, 1, 3, 4)[:, :, 1:])):
    for label, op in (("* 2", lambda t: t * 2), ("+ itself", lambda t: t + t), ("abs", lambda t: t.abs()),
                      ("sum(-1)", lambda t: t.sum(-1)), ("contiguous", lambda t: t.contiguous()),
                      ("reshape(-1, 6, 3)", lambda t: t.reshape(-1, 6, 3)),
                      ("view_as_real.contiguous", lambda t: torch.view_as_real(t).contiguous())):
        check(f"{layout} 5-D complex: {label}", lambda op=op, view=view: err(op(view(z5.to(hpu))), op(view(z5))), 1e-5)


def cross_spectra(t):  # X X^H per window, then |.|^2, as MetaNeuromotorHand does it after a rearrange
    w = t.reshape(-1, t.shape[-2], t.shape[-1])
    return ((w @ w.conj().transpose(1, 2)) / t.shape[-1]).abs().pow(2)


spectra = torch.complex(torch.randn(2, 7, 3, 16, 16), torch.randn(2, 7, 3, 16, 16))
check("cross-spectra of a permuted 5-D complex (relative)",
      lambda: grad_err(lambda t: cross_spectra(t.permute(0, 2, 1, 3, 4)), spectra, relative=True), 1e-5)


def cplx(*shape):
    return torch.complex(torch.randn(*shape), torch.randn(*shape))


def written(device, write):
    """A complex (2, 5, 3, 4, 6) of zeros after write(it)."""
    base = torch.zeros(2, 5, 3, 4, 6, dtype=torch.cfloat, device=device)
    write(base)
    return base


# Results of rank-6 pairs whose layout comes from a permuted operand of lower rank or a permuted condition; writes
# through permuted and sliced destinations; more layouts: offsets, steps, chunks, rank 7, dtype views, expansions.
c4, c5, r5, x5, src, z6 = cplx(3, 5, 4, 6), cplx(2, 3, 4, 5, 6), torch.randn(2, 3, 4, 5, 6), torch.randn(2, 5, 4, 6, 3), \
    cplx(2, 3, 4, 4, 6), cplx(2, 3, 2, 4, 3, 5)
permuted = lambda t: t.permute(0, 1, 3, 4, 2)  # noqa: E731
for label, case in {
    "permuted 4-D + 5-D complex": lambda d: c4.to(d).permute(0, 2, 1, 3) + c5.to(d),
    "permuted 4-D complex * 5-D real": lambda d: c4.to(d).permute(0, 2, 1, 3) * r5.to(d),
    "permuted 4-D complex / 5-D real": lambda d: c4.to(d).permute(0, 2, 1, 3) / (r5.to(d).abs() + 1),
    "where(permuted 5-D condition, z, 0)": lambda d: torch.where(x5.to(d).permute(0, 1, 4, 2, 3) > 0, z5.to(d), 0),
    "permuted 5-D real .to(complex64)": lambda d: permuted(z5.real.contiguous().to(d)).to(torch.complex64),
    "zeros_like(permuted 5-D complex)": lambda d: torch.zeros_like(permuted(z5.to(d))) + permuted(z5.to(d)),
    "full_like(permuted 5-D complex, 1+2j)": lambda d: torch.full_like(permuted(z5.to(d)), 1 + 2j),
    "permuted 5-D complex .conj() * 2": lambda d: permuted(z5.to(d)).conj() * 2,
    "copy_ into a sliced permuted view": lambda d: written(d, lambda b: b.permute(0, 2, 1, 3, 4)[:, :, 1:]
                                                             .copy_(src.to(d))),
    "mul_(2) of a sliced permuted view": lambda d: written(d, lambda b: b.copy_(z5.to(d)).permute(0, 2, 1, 3, 4)
                                                             [:, :, 1:].mul_(2)),
    "permuted at an offset (z[1:])": lambda d: permuted(z5.to(d)[1:]) * 2,
    "stepped permuted (::2)": lambda d: permuted(z5.to(d))[..., ::2] * 2,
    "chunk(2, 1)[1] permuted": lambda d: permuted(z5.to(d).chunk(2, 1)[1]) * 2,
    "permuted 6-D complex (rank-7 pair)": lambda d: z6.to(d).permute(0, 2, 1, 3, 5, 4) * 2,
    "permuted 6-D complex .view(float32)": lambda d: z6.to(d).permute(0, 2, 1, 3, 4, 5).view(torch.float32).contiguous(),
    "expanded 6-D complex * 2": lambda d: z5.to(d).unsqueeze(1).expand(2, 3, 5, 3, 4, 6) * 2,
    "fill_ on offset permuted 5-D": lambda d: written(d, lambda b: b[1:].permute(0, 2, 1, 3, 4).fill_(3 + 4j)),
}.items():
    check(f"rank-6 pairs: {label}", lambda case=case: err(case(hpu), case("cpu")), 1e-5)

cond_scalar = torch.tensor(True)
for label, case in {
    "z6.to(float32)": lambda d: z6.to(d).to(torch.float32),
    "r6.copy_(z6)": lambda d: torch.empty(z6.shape, dtype=torch.float32, device=d).copy_(z6.to(d)),
    "where((x5 > 0).permute(...), z5, 0)": lambda d: torch.where(
        (x5.to(d) > 0).permute(0, 1, 4, 2, 3), z5.to(d), 0),
    "where(cond_scalar.expand(2, 3, 4, 5, 6), c5, 0)": lambda d: torch.where(
        cond_scalar.to(d).expand(2, 3, 4, 5, 6), c5.to(d), 0),
    "where(cond_scalar.expand(2, 5, 3, 4, 6), z5, 0)": lambda d: torch.where(
        cond_scalar.to(d).expand(2, 5, 3, 4, 6), z5.to(d), 0),
}.items():
    check(f"rank-6+ exact: {label}", lambda case=case: err(case(hpu), case("cpu")), 0)

# <6D strided views (slices, transposes, diagonals, stepped, expanded/stride=0) & in-place storage mutation.
z3, r3, s3 = cplx(3, 5, 5), torch.randn(3, 5, 5), cplx(3, 1, 5)
z2 = cplx(3, 5)
for vname, view_fn in (
    ("slice offset", lambda t: t[1:, 1:, 1:]),
    ("transpose", lambda t: t.transpose(1, 2)),
    ("mH", lambda t: t.mH),
    ("diagonal", lambda t: t.diagonal(dim1=-2, dim2=-1)),
    ("stepped", lambda t: t[::2, 1::2, ::2]),
):
    check(f"<6D {vname}: mul/div/add/sub/abs/sgn/view_as_real", lambda vf=view_fn: worst(
        err(vf(z3.to(hpu)) * vf(z3.to(hpu)), vf(z3) * vf(z3)),
        err(vf(z3.to(hpu)) / (vf(z3.to(hpu)).abs() + 1), vf(z3) / (vf(z3).abs() + 1)),
        err(vf(z3.to(hpu)) + 1.5, vf(z3) + 1.5),
        err(vf(z3.to(hpu)) - vf(z3.to(hpu)), vf(z3) - vf(z3)),
        err(vf(z3.to(hpu)).abs(), vf(z3).abs()),
        err(vf(z3.to(hpu)).sgn(), vf(z3).sgn()),
        err(torch.view_as_real(vf(z3.to(hpu)).resolve_conj()).contiguous(),
            torch.view_as_real(vf(z3).resolve_conj())),
    ), 1e-5)

    def inplace_view(vf=view_fn):
        errors = []
        for mutate in (
            lambda b: vf(b).fill_(1.25 - 0.75j),
            lambda b: vf(b).zero_(),
            lambda b: vf(b).mul_(2.0 + 0.5j),
            lambda b: vf(b).add_(0.5 - 1.0j),
            lambda b: vf(b).copy_(vf(z3.to(b.device)).conj()),
            lambda b: vf(b).copy_(torch.ones_like(vf(b).real)),
        ):
            bc, bh = z3.clone(), z3.to(hpu)
            mutate(bc)
            mutate(bh)
            errors.append(err(bh, bc))
        return worst(*errors)

    check(f"<6D {vname}: inplace fill_/zero_/mul_/add_/copy_", inplace_view, 1e-5)

check("<6D expanded (offset, stride=0): ops & view_as_real", lambda: worst(
    err(s3[1:].to(hpu).expand(2, 5, 5) * 2.0, s3[1:].expand(2, 5, 5) * 2.0),
    err(s3[1:].to(hpu).expand(2, 5, 5) + z3[1:].to(hpu), s3[1:].expand(2, 5, 5) + z3[1:]),
    err(s3[1:].to(hpu).expand(2, 5, 5).abs(), s3[1:].expand(2, 5, 5).abs()),
    err(s3[1:].to(hpu).expand(2, 5, 5).sgn(), s3[1:].expand(2, 5, 5).sgn()),
    err(torch.view_as_real(s3[1:].to(hpu).expand(2, 5, 5)).contiguous(),
        torch.view_as_real(s3[1:].expand(2, 5, 5))),
), 1e-5)

check("<6D real stride-0 / diag / 5D-perm with complex", lambda: worst(
    err(z3.to(hpu) * torch.ones((), device=hpu).expand(3, 5, 5), z3 * torch.ones(()).expand(3, 5, 5)),
    err(z3.to(hpu) / torch.full((), 2.0, device=hpu).expand(3, 5, 5), z3 / torch.full((), 2.0).expand(3, 5, 5)),
    err(torch.full((), 2.0, device=hpu).expand(3, 5, 5) / (z3.to(hpu) + 10),
        torch.full((), 2.0).expand(3, 5, 5) / (z3 + 10)),
    err(z2.to(hpu) * r3.to(hpu).diagonal(dim1=-2, dim2=-1), z2 * r3.diagonal(dim1=-2, dim2=-1)),
    err(c5.to(hpu).permute(0, 2, 1, 3, 4) * r5.to(hpu).permute(0, 2, 1, 3, 4),
        c5.permute(0, 2, 1, 3, 4) * r5.permute(0, 2, 1, 3, 4)),
    err(torch.where(torch.tensor(True, device=hpu).expand(3, 5, 5), z3.to(hpu), r3.to(hpu)),
        torch.where(torch.tensor(True).expand(3, 5, 5), z3, r3)),
    err(z3.to(hpu)[2, 3].sum(), z3[2, 3].sum()),
), 1e-5)

# Issue #4: complex64 linalg forward + backward gradient accuracy vs CPU (with .sum() / sliced / .real.sum() losses).
def linalg_grad_err(loss_fn, make_inp):
    xc = make_inp().requires_grad_()
    xh = xc.detach().clone().to(hpu).requires_grad_()
    yc, yh = loss_fn(xc), loss_fn(xh)
    yc.backward()
    yh.backward()
    return worst(
        err(yh, yc) / max(yc.detach().abs().max().item(), 1.0),
        err(xh.grad, xc.grad) / max(xc.grad.abs().max().item(), 1e-12),
    )


mk_herm = lambda: (lambda a: (a @ a.mH + 8 * torch.eye(8, dtype=torch.complex64)).detach())(cplx(4, 8, 8))  # noqa: E731
mk_sq = lambda: cplx(4, 8, 8)  # noqa: E731
mk_rect = lambda: cplx(4, 8, 6)  # noqa: E731

for label, loss_fn, mk in (
    ("linalg.eigh (w.sum + V.abs.sum)",
     lambda t: (lambda w, V: w.sum() + V.abs().sum())(*torch.linalg.eigh(t)), mk_herm),
    ("linalg.eigh (sliced gauge-inv V @ V.mH)",
     lambda t: (lambda w, V: w[:, :4].sum() + (V[:, :4, :4] @ V[:, :4, :4].mH).real.sum())(*torch.linalg.eigh(t)),
     mk_herm),
    ("linalg.svd (U.abs + S + Vh.abs)",
     lambda t: (lambda U, S, Vh: U.abs().sum() + S.sum() + Vh.abs().sum())(
         *torch.linalg.svd(t, full_matrices=False)), mk_rect),
    ("linalg.svd (sliced gauge-inv U @ U.mH + S)",
     lambda t: (lambda U, S, Vh: (U[:, :4, :4] @ U[:, :4, :4].mH).real.sum() + S[:, :4].sum())(
         *torch.linalg.svd(t, full_matrices=False)), mk_rect),
    ("linalg.qr (Q.abs + R.abs)",
     lambda t: (lambda Q, R: Q.abs().sum() + R.abs().sum())(*torch.linalg.qr(t)), mk_rect),
    ("linalg.qr (Q.real + R[:, :2].abs)",
     lambda t: (lambda Q, R: Q.real.sum() + R[:, :2].abs().sum())(*torch.linalg.qr(t)), mk_rect),
    ("linalg.eig (w.abs + V.abs)",
     lambda t: (lambda w, V: w.abs().sum() + V.abs().sum())(*torch.linalg.eig(t)), mk_sq),
    ("linalg.eig (w.real + gauge-inv V*V.conj)",
     lambda t: (lambda w, V: w[:, :4].real.sum() + (V[:, :4, :4] * V[:, :4, :4].conj()).real.sum())(
         *torch.linalg.eig(t)), mk_sq),
    ("linalg.matrix_exp (abs.sum)", lambda t: torch.linalg.matrix_exp(t).abs().sum(), mk_sq),
    ("linalg.matrix_exp (real.sum & slice)",
     lambda t: torch.linalg.matrix_exp(t).real.sum() + torch.linalg.matrix_exp(t)[:, :2, :2].abs().sum(), mk_sq),
    ("linalg.matrix_exp (mH input)", lambda t: torch.linalg.matrix_exp(t.mH).abs().sum(), mk_sq),
):
    check(f"{label} fwd+bwd (rel)", lambda lf=loss_fn, mk=mk: linalg_grad_err(lf, mk), 1e-4)


def hh_grad_err(slice_real=False):
    # Reflectors from geqrf: with a random tau the product is ill-conditioned, and two float32 runs differ by
    # their summation order alone (CPU complex64 vs complex128: 5e-05 relative already).
    a0, t0 = torch.geqrf(cplx(4, 8, 6))
    ac, tc = a0.requires_grad_(), t0.requires_grad_()
    ah = ac.detach().clone().to(hpu).requires_grad_()
    th = tc.detach().clone().to(hpu).requires_grad_()
    loss_fn = (lambda a, t: torch.linalg.householder_product(a, t)[:, :4, :4].real.sum()) if slice_real else (
        lambda a, t: torch.linalg.householder_product(a, t).abs().sum())
    yc, yh = loss_fn(ac, tc), loss_fn(ah, th)
    yc.backward()
    yh.backward()
    return worst(
        err(yh, yc) / max(yc.detach().abs().max().item(), 1.0),
        err(ah.grad, ac.grad) / max(ac.grad.abs().max().item(), 1e-12),
        err(th.grad, tc.grad) / max(tc.grad.abs().max().item(), 1e-12),
    )


check("linalg.householder_product (abs.sum) fwd+bwd (rel)", lambda: hh_grad_err(False), 5e-4)
check("linalg.householder_product (slice real.sum) fwd+bwd (rel)", lambda: hh_grad_err(True), 5e-4)

# Issue #5: real float32 linalg.eig / linalg.eigvals (+ complex64 eigvals) fwd + bwd vs CPU.
# Use an isolated Generator so existing checks above/below keep their exact RNG sequence.
_eig_gen = torch.Generator().manual_seed(42)
_eig_f32_2d = torch.randn(8, 8, generator=_eig_gen)
_eig_f32_3d = torch.randn(4, 8, 8, generator=_eig_gen)
_eig_f32_32 = torch.randn(32, 22, 22, generator=_eig_gen)
_eig_f32_4d = torch.randn(2, 3, 6, 6, generator=_eig_gen)
_eig_c64_3d = torch.complex(torch.randn(4, 8, 8, generator=_eig_gen),
                            torch.randn(4, 8, 8, generator=_eig_gen))


def real_eig_fwd_err(base, view_fn=lambda t: t, eigvals_only=False):
    xc = view_fn(base)
    xh = view_fn(base.detach().clone().to(hpu))
    with torch.no_grad():
        if eigvals_only:
            wc = torch.linalg.eigvals(xc)
            wh = torch.linalg.eigvals(xh)
            assert wh.dtype == torch.complex64 and tagged(wh), (wh.dtype, torch._C._dispatch_keys(wh))
            return err(wh, wc)
        wc, Vc = torch.linalg.eig(xc)
        wh, Vh = torch.linalg.eig(xh)
    assert wh.dtype == torch.complex64 and Vh.dtype == torch.complex64, (wh.dtype, Vh.dtype)
    assert tagged(wh) and tagged(Vh), (torch._C._dispatch_keys(wh), torch._C._dispatch_keys(Vh))
    return worst(err(wh, wc), err(Vh, Vc))


def real_eig_bwd_err(loss_fn, base, view_fn=lambda t: t):
    bc = base.detach().clone().requires_grad_()
    bh = base.detach().clone().to(hpu).requires_grad_()
    yc, yh = loss_fn(view_fn(bc)), loss_fn(view_fn(bh))
    yc.backward()
    yh.backward()
    assert bh.grad.dtype == torch.float32, bh.grad.dtype
    assert not tagged(bh.grad), torch._C._dispatch_keys(bh.grad)
    return worst(
        err(yh, yc) / max(yc.detach().abs().max().item(), 1.0),
        err(bh.grad, bc.grad) / max(bc.grad.abs().max().item(), 1e-12),
    )


for label, base, vf in (
    ("2D (8, 8)", _eig_f32_2d, lambda t: t),
    ("3D (4, 8, 8)", _eig_f32_3d, lambda t: t),
    ("3D (32, 22, 22)", _eig_f32_32, lambda t: t),
    ("4D (2, 3, 6, 6)", _eig_f32_4d, lambda t: t),
    (".mT", _eig_f32_3d, lambda t: t.mT),
    ("[:, 1:7, 1:7]", _eig_f32_3d, lambda t: t[:, 1:7, 1:7]),
):
    check(f"real f32 linalg.eig fwd {label}", lambda b=base, vf=vf: real_eig_fwd_err(b, vf), 1e-4)

for label, loss_fn, base, vf in (
    ("3D (w.abs + V.abs)", lambda t: (lambda w, V: w.abs().sum() + V.abs().sum())(*torch.linalg.eig(t)),
     _eig_f32_3d, lambda t: t),
    ("3D (gauge-inv)", lambda t: (lambda w, V: w[:, :4].real.sum() + (V[:, :4, :4] * V[:, :4, :4].conj()).real.sum())(
        *torch.linalg.eig(t)), _eig_f32_3d, lambda t: t),
    ("3D (w.abs only)", lambda t: torch.linalg.eig(t)[0].abs().sum(), _eig_f32_3d, lambda t: t),
    ("3D (V.abs only)", lambda t: torch.linalg.eig(t)[1].abs().sum(), _eig_f32_3d, lambda t: t),
    ("2D (8, 8)", lambda t: (lambda w, V: w.abs().sum() + V.abs().sum())(*torch.linalg.eig(t)),
     _eig_f32_2d, lambda t: t),
    ("4D (2, 3, 6, 6)", lambda t: (lambda w, V: w.abs().sum() + V.abs().sum())(*torch.linalg.eig(t)),
     _eig_f32_4d, lambda t: t),
    (".mT", lambda t: (lambda w, V: w.abs().sum() + V.abs().sum())(*torch.linalg.eig(t)),
     _eig_f32_3d, lambda t: t.mT),
    ("[:, 1:7, 1:7]", lambda t: (lambda w, V: w.abs().sum() + V.abs().sum())(*torch.linalg.eig(t)),
     _eig_f32_3d, lambda t: t[:, 1:7, 1:7]),
    ("V.permute(1, 0, 2) grad_V",
     lambda t: (lambda w, V: w.abs().sum() + V.permute(1, 0, 2).abs().sum())(*torch.linalg.eig(t)),
     _eig_f32_3d, lambda t: t),
):
    check(f"real f32 linalg.eig {label} fwd+bwd (rel)",
          lambda lf=loss_fn, b=base, vf=vf: real_eig_bwd_err(lf, b, vf), 1e-4)

for label, base, vf in (
    ("2D (8, 8)", _eig_f32_2d, lambda t: t),
    ("3D (4, 8, 8)", _eig_f32_3d, lambda t: t),
    (".mT", _eig_f32_3d, lambda t: t.mT),
    ("[:, 1:7, 1:7]", _eig_f32_3d, lambda t: t[:, 1:7, 1:7]),
):
    check(f"real f32 linalg.eigvals fwd {label}",
          lambda b=base, vf=vf: real_eig_fwd_err(b, vf, eigvals_only=True), 1e-4)

for label, base, vf in (
    ("2D (8, 8)", _eig_f32_2d, lambda t: t),
    ("3D (4, 8, 8)", _eig_f32_3d, lambda t: t),
    ("4D (2, 3, 6, 6)", _eig_f32_4d, lambda t: t),
    (".mT", _eig_f32_3d, lambda t: t.mT),
    ("[:, 1:7, 1:7]", _eig_f32_3d, lambda t: t[:, 1:7, 1:7]),
):
    check(f"real f32 linalg.eigvals {label} fwd+bwd (rel)",
          lambda b=base, vf=vf: real_eig_bwd_err(lambda t: torch.linalg.eigvals(t).abs().sum(), b, vf), 1e-4)

def c64_eigvals_fwd_err():
    with torch.no_grad():
        wc = torch.linalg.eigvals(_eig_c64_3d)
        wh = torch.linalg.eigvals(_eig_c64_3d.detach().clone().to(hpu))
    assert wh.dtype == torch.complex64 and tagged(wh), (wh.dtype, torch._C._dispatch_keys(wh))
    return err(wh, wc)


check("complex64 linalg.eigvals fwd (4, 8, 8)", c64_eigvals_fwd_err, 1e-4)
check("linalg.eigvals (w.abs.sum) fwd+bwd (rel)",
      lambda: linalg_grad_err(lambda t: torch.linalg.eigvals(t).abs().sum(), lambda: _eig_c64_3d.detach().clone()), 1e-4)

# Complex mm / bmm on the card (one real GEMM): complex x complex, real x complex, complex x real, conjugate and
# strided operands, matmul broadcasting, gradients through @ and .mH.
_mm_gen = torch.Generator().manual_seed(7)
_mc = lambda *s: torch.complex(torch.randn(*s, generator=_mm_gen), torch.randn(*s, generator=_mm_gen))  # noqa: E731
_A, _B, _R = _mc(5, 7, 4), _mc(5, 4, 3), torch.randn(5, 4, 3, generator=_mm_gen)
_A2, _B2 = _mc(7, 4), _mc(4, 3)
def _empty_or_zero_err(got, ref):
    """err() for results that may hold no element: shape, dtype and tag must match; values when there are any."""
    if got.shape != ref.shape or got.dtype != ref.dtype or not tagged(got):
        return math.inf
    return err(got, ref) if ref.numel() else 0.0


def _mixed_dtype_raises():
    try:
        torch.bmm(_A.to(hpu), _R.to(hpu))
    except RuntimeError as e:
        return 0.0 if "same dtype" in str(e) else 1.0, str(e)[:60]
    return 1.0, "no error"


check("complex bmm with a float32 operand raises, as on the CPU", _mixed_dtype_raises, 0)


def _float_out_raises():
    """mm.out / bmm.out into a float32 out raise on the CPU (dtype check); they must not write the real part here."""
    msgs = []
    for name, fn, a, b, shape in (("mm", torch.mm, _A2, _B2, (7, 3)), ("bmm", torch.bmm, _A, _B, (5, 7, 3))):
        try:
            fn(a.to(hpu), b.to(hpu), out=torch.empty(*shape, device=hpu))
        except RuntimeError as e:
            if "Expected out tensor to have dtype" not in str(e):
                return 1.0, f"{name}: {str(e)[:60]}"
            msgs.append(name)
            continue
        return 1.0, f"{name}: no error"
    return 0.0, "raised: " + ", ".join(msgs)


for _route, _threshold in (("card", 0), ("CPU<2^22", 1 << 22)):
    hpu_complex.set_matmul_cpu_threshold(_threshold)
    check(f"complex mm.out / bmm.out into a float32 out raise, as on the CPU {_route}", _float_out_raises, 0)
# Both routes: every product on the card's GEMM (threshold 0), then the default (these small ones on the CPU).
for _route, _threshold in (("card", 0), ("CPU<2^22", 1 << 22)):
    hpu_complex.set_matmul_cpu_threshold(_threshold)
    check(f"complex bmm / mm / matmul (incl. cast, conj, strided, broadcast) {_route}", lambda: worst(
        err(torch.bmm(_A.to(hpu), _B.to(hpu)), torch.bmm(_A, _B)),
        err(torch.mm(_A2.to(hpu), _B2.to(hpu)), torch.mm(_A2, _B2)),
        err(_A.to(hpu) @ _R.to(hpu).to(torch.complex64), _A @ _R.to(torch.complex64)),
        err(_A.to(hpu) @ _A.to(hpu).mH, _A @ _A.mH),
        err(_A.to(hpu).conj() @ _B.to(hpu), _A.conj() @ _B),
        err(_A.to(hpu)[:, 1:6:2] @ _B.to(hpu)[:, :, 1:], _A[:, 1:6:2] @ _B[:, :, 1:]),
        err(_A.to(hpu) @ _B2.to(hpu), _A @ _B2),
        err(_A2.to(hpu) @ _A2.to(hpu)[0], _A2 @ _A2[0]),
        err(torch.bmm(_A.to(hpu), _B.to(hpu), out=torch.empty(5, 7, 3, dtype=torch.complex64, device=hpu)),
            torch.bmm(_A, _B)),
        (lambda u, v: err(u.to(hpu) @ v.to(hpu), u @ v))(_mc(1, 1, 1), _mc(1, 1, 1)),
        (lambda u, v: err(u.to(hpu) @ v.to(hpu), u @ v))(_mc(3, 4, 5), _mc(3, 5, 1)),
        (lambda u, v: err(u.to(hpu) @ v.to(hpu), u @ v))(_mc(3, 1, 5), _mc(3, 5, 4)),
        (lambda u, v: _empty_or_zero_err(u.to(hpu) @ v.to(hpu), u @ v))(_mc(0, 4, 5), _mc(0, 5, 2)),
        (lambda u, v: _empty_or_zero_err(u.to(hpu) @ v.to(hpu), u @ v))(_mc(4, 0), _mc(0, 3)),
    ), 1e-5)
    check(f"complex matmul X Xᴴ fwd+bwd (rel) {_route}", lambda: grad_err(lambda t: (t @ t.mH).abs(), _A, relative=True), 1e-5)
    check(f"complex matmul complex@cast(real) fwd+bwd (rel) {_route}",
          lambda: grad_err(lambda r: (_A.to(r.device) @ r.to(torch.complex64)).abs(), _R, relative=True), 1e-5)

# Adversarial: extreme values through the card GEMM (NaN/inf propagate as on the CPU, overflow to inf), a
# compiled product (complex nodes run eagerly on these kernels), and one large enough for the default card route.
hpu_complex.set_matmul_cpu_threshold(0)
_X = _mc(3, 4, 5)
_X[0, 1, 2] = complex(float("inf"), 0.0)
_X[1, 0, 0] = complex(float("nan"), 1.0)
_X[2] *= 1e30
_Y = _mc(3, 5, 2)
_Y[2, :, 0] *= 1e10  # batch 2, column 0: |products| ~ 1e40 overflow float32 to inf


def _extreme_rel_err():
    """Elementwise |got - ref| / max(|ref|, 1) where finite; NaN unless the same entries are non-finite and the NaN
    input's row is NaN. Overflowing sums are not compared by kind (CPU NaN, card +-inf here): Habana's own float32
    GEMM treats overflowing intermediates differently from the CPU (measured: [3e38, -3e38] @ [10, 10] is 0.0 on the
    card, inf on the CPU), while an explicit inf - inf is NaN on both."""
    got, ref = torch.view_as_real((_X.to(hpu) @ _Y.to(hpu)).cpu()), torch.view_as_real(_X @ _Y)
    same = torch.equal(got.isfinite(), ref.isfinite()) and bool(got[1, 0].isnan().all())
    if not same:
        return math.nan
    fin = ref.isfinite()
    return ((got[fin] - ref[fin]).abs() / ref[fin].abs().clamp_min(1)).max().item()


check("complex bmm card: inf / NaN / overflow as on the CPU (rel)", _extreme_rel_err, 1e-5)
hpu_complex.set_matmul_cpu_threshold(1 << 22)


def _compiled_matmul_err():
    f = torch.compile(lambda t: (t @ t.mH).abs().sum(-1), backend="hpu_backend")
    return grad_err(f, _A, relative=True, reference=lambda t: (t @ t.mH).abs().sum(-1))


check("compiled (t @ t.mH).abs() fwd+bwd (rel)", _compiled_matmul_err, 1e-5)
_Big = _mc(64, 22, 300)
check("complex X Xᴴ above the threshold (card route, 64x22x300x22) (rel)",
      lambda: err(_Big.to(hpu) @ _Big.to(hpu).mH, _Big @ _Big.mH) / (_Big @ _Big.mH).abs().max().item(), 1e-5)

x6_cpu = torch.randn(2, 3, 2, 4, 3, 5)
c6_cpu_perm = (x6_cpu > 0).permute(0, 2, 4, 1, 3, 5)
c6_cpu_off_perm = (x6_cpu > 0)[1:].permute(0, 2, 4, 1, 3, 5)
f6_cpu_perm = x6_cpu.permute(0, 2, 4, 1, 3, 5)
z6_perm_shape = cplx(*f6_cpu_perm.shape)
x7_cpu = torch.randn(2, 2, 2, 3, 2, 3, 2)
c7_cpu_perm = (x7_cpu > 0).permute(1, 0, 2, 3, 4, 5, 6)
z7_perm_shape = cplx(*c7_cpu_perm.shape)
for label, case in {
    "where(6D CPU perm cond, z6_hpu, 0)": lambda d: torch.where(c6_cpu_perm, z6_perm_shape.to(d), 0),
    "where(6D CPU offset perm cond, z6_hpu[1:], 0)": lambda d: torch.where(
        c6_cpu_off_perm, z6_perm_shape.to(d)[1:], 0),
    "where(7D CPU perm cond, z7_hpu, 0)": lambda d: torch.where(c7_cpu_perm, z7_perm_shape.to(d), 0),
}.items():
    check(f"rank-6+ CPU exact: {label}", lambda case=case: err(case(hpu), case("cpu")), 0)

c6_hpu_perm = (x6_cpu.to(hpu) > 0).permute(0, 2, 4, 1, 3, 5)
for label, hpu_fn, cpu_fn in (
    ("z6_hpu * f6_cpu_perm", lambda: z6_perm_shape.to(hpu) * f6_cpu_perm, lambda: z6_perm_shape * f6_cpu_perm),
    ("z6_hpu + f6_cpu_perm", lambda: z6_perm_shape.to(hpu) + f6_cpu_perm, lambda: z6_perm_shape + f6_cpu_perm),
    ("z6_hpu - f6_cpu_perm", lambda: z6_perm_shape.to(hpu) - f6_cpu_perm, lambda: z6_perm_shape - f6_cpu_perm),
    ("z6_hpu / (f6_cpu_perm.abs() + 1)", lambda: z6_perm_shape.to(hpu) / (f6_cpu_perm.abs() + 1),
     lambda: z6_perm_shape / (f6_cpu_perm.abs() + 1)),
    ("where(cond6_hpu, z6_hpu, f6_cpu_perm)", lambda: torch.where(c6_hpu_perm, z6_perm_shape.to(hpu), f6_cpu_perm),
     lambda: torch.where(c6_cpu_perm, z6_perm_shape, f6_cpu_perm)),
    ("z6_hpu * f6_cpu_perm.double()", lambda: z6_perm_shape.to(hpu) * f6_cpu_perm.double(),
     lambda: z6_perm_shape * f6_cpu_perm.double()),
    ("z6_hpu.copy_(z6_cpu_perm.cdouble())", lambda: torch.zeros_like(z6_perm_shape, device=hpu).copy_(
        z6.cdouble().permute(0, 2, 4, 1, 3, 5)),
     lambda: z6.cdouble().permute(0, 2, 4, 1, 3, 5).to(torch.complex64)),
    ("f6_cpu_perm.double().to(hpu, complex64)", lambda: f6_cpu_perm.double().to(device=hpu, dtype=torch.complex64),
     lambda: f6_cpu_perm.double().to(torch.complex64)),
):
    check(f"rank-6+ CPU float/copy: {label}", lambda hf=hpu_fn, cf=cpu_fn: err(hf(), cf()), 1e-5)

# A real HPU operand of rank 6 with permuted strides that is the OUTPUT of a Habana op on a permuted view: Habana's
# own contiguous() misreads it (also without hpu_complex), so the kernels must not call it on such an operand.
_x8 = torch.randn(2, 2, 3, 2, 2, 3, 2, 2)
_sel = lambda t: t[1, 1:].permute(0, 2, 1, 3, 4, 5, 6)[..., 0]  # noqa: E731  rank 6, offset, non-dense
_r6h, _r6c = _sel(_x8.to(hpu)) * 1, _sel(_x8) * 1  # strides (72, 12, 24, 6, 2, 1) on both
_z6p = cplx(*_r6c.shape)
_x7 = cplx(2, 2, 3, 2, 2, 3, 2)
_v7 = lambda t: t[1, 1:].permute(0, 2, 1, 3, 4, 5)  # noqa: E731
for label, hpu_fn, cpu_fn in (
    ("z6 * r6", lambda: _z6p.to(hpu) * _r6h, lambda: _z6p * _r6c),
    ("z6 + r6", lambda: _z6p.to(hpu) + _r6h, lambda: _z6p + _r6c),
    ("r6 / z6", lambda: _r6h / (_z6p.to(hpu) + 3), lambda: _r6c / (_z6p + 3)),
    ("where(r6 > 0, z6, 0)", lambda: torch.where(_r6h > 0, _z6p.to(hpu), 0), lambda: torch.where(_r6c > 0, _z6p, 0)),
    ("where(r6 > 0, z6, r6)", lambda: torch.where(_r6h > 0, _z6p.to(hpu), _r6h),
     lambda: torch.where(_r6c > 0, _z6p, _r6c)),
    ("z6.clone().copy_(r6)", lambda: _z6p.to(hpu).clone().copy_(_r6h), lambda: _z6p.clone().copy_(_r6c)),
    ("torch.complex(r6, r6)", lambda: torch.complex(_r6h, _r6h), lambda: torch.complex(_r6c, _r6c)),
    ("r6.to(complex64)", lambda: _r6h.to(torch.complex64), lambda: _r6c.to(torch.complex64)),
    ("offset permuted v: where(v.real > 0, v, 0)", lambda: torch.where(_v7(_x7.to(hpu)).real > 0, _v7(_x7.to(hpu)), 0),
     lambda: torch.where(_v7(_x7).real > 0, _v7(_x7), 0)),
):
    exact = "/" not in label  # the division rounds differently on the card (1.2e-7 measured)
    check(f"rank-6 permuted Habana result{' (exact)' if exact else ''}: {label}",
          lambda hf=hpu_fn, cf=cpu_fn: err(hf(), cf()), 0 if exact else 1e-6)


def _mixed_device_raises():
    """torch.mm / bmm of an HPU and a CPU complex operand raise on the device check, as torch does."""
    for name, fn, a, b in (("mm", torch.mm, _A2, _B2), ("bmm", torch.bmm, _A, _B)):
        for u, v in ((a.to(hpu), b), (a, b.to(hpu))):
            try:
                fn(u, v)
            except RuntimeError as e:
                if "same device" not in str(e):
                    return 1.0, f"{name}: {str(e)[:60]}"
                continue
            return 1.0, f"{name}: no error"
    return 0.0, "raised for mm and bmm, both orders"


for _route, _threshold in (("card", 0), ("CPU<2^22", 1 << 22)):
    hpu_complex.set_matmul_cpu_threshold(_threshold)
    check(f"complex mm / bmm of HPU and CPU operands raise {_route}", _mixed_device_raises, 0)
hpu_complex.set_matmul_cpu_threshold(1 << 22)

# D2H copy_ into non-dense CPU targets (sentinel-filled) and 5-D complex->real D2H.
def d2h_nondense_copy(view_fn, src_conj=False, dst_conj=False, real_dst=False):
    if real_dst:
        ref = torch.full_like(r3, 99.0)
        got = torch.full_like(r3, 99.0)
        src_c = view_fn(z3).conj() if src_conj else view_fn(z3)
        src_h = view_fn(z3.to(hpu)).conj() if src_conj else view_fn(z3.to(hpu))
        view_fn(ref).copy_(src_c.real)
        view_fn(got).copy_(src_h)
        return err(got, ref)
    ref = torch.full_like(z3, 99.0 + 99.0j)
    got = torch.full_like(z3, 99.0 + 99.0j)
    dst_c = view_fn(ref).conj() if dst_conj else view_fn(ref)
    dst_h = view_fn(got).conj() if dst_conj else view_fn(got)
    src_c = view_fn(z3).conj() if src_conj else view_fn(z3)
    src_h = view_fn(z3.to(hpu)).conj() if src_conj else view_fn(z3.to(hpu))
    dst_c.copy_(src_c)
    dst_h.copy_(src_h)
    return err(got, ref)


for vname, vf in (
    ("diagonal", lambda t: t.diagonal(dim1=-2, dim2=-1)),
    ("narrow", lambda t: t.narrow(-1, 1, 3)),
    ("step", lambda t: t[:, ::2, 1::2]),
):
    check(f"D2H cplx<-cplx {vname}", lambda vf=vf: d2h_nondense_copy(vf), 0)
    check(f"D2H cplx<-cplx {vname} (src conj)", lambda vf=vf: d2h_nondense_copy(vf, src_conj=True), 0)
    check(f"D2H cplx<-cplx {vname} (dst conj)", lambda vf=vf: d2h_nondense_copy(vf, dst_conj=True), 0)
    check(f"D2H real<-cplx {vname}", lambda vf=vf: d2h_nondense_copy(vf, real_dst=True), 0)

check("D2H 5-D real<-cplx (contig & diagonal)", lambda: worst(
    err(torch.full_like(z5.real, 99.0).copy_(z5.to(hpu)), torch.full_like(z5.real, 99.0).copy_(z5.real)),
    err((lambda b: (b.diagonal(-2, -1).copy_(z5.to(hpu).diagonal(-2, -1)), b)[1])(torch.full_like(z5.real, 99.0)),
        (lambda b: (b.diagonal(-2, -1).copy_(z5.diagonal(-2, -1).real), b)[1])(torch.full_like(z5.real, 99.0))),
), 0)

sc_cplx, sc_real = torch.tensor(1.5 - 2.25j), torch.tensor(3.25)
v1_cplx, v1_real = cplx(5), torch.randn(5)
check("D2H broadcast smaller HPU->CPU (cplx & real)", lambda: worst(
    err(torch.full_like(z3, 99.0 + 99.0j).copy_(sc_cplx.to(hpu)), torch.full_like(z3, 99.0 + 99.0j).copy_(sc_cplx)),
    err((lambda b: (b.diagonal(-2, -1).copy_(sc_cplx.to(hpu).conj()), b)[1])(torch.full_like(z3, 99.0 + 99.0j)),
        (lambda b: (b.diagonal(-2, -1).copy_(sc_cplx.conj()), b)[1])(torch.full_like(z3, 99.0 + 99.0j))),
    err(torch.full_like(r3, 99.0).copy_(sc_cplx.to(hpu)), torch.full_like(r3, 99.0).copy_(sc_cplx.real)),
), 0)

# H2D copy_ from broadcast-smaller CPU tensors.
def h2d_broadcast_copy(make_target, src):
    bc = make_target("cpu")
    bh = make_target(hpu)
    (bc[1:] if bc.shape == z3.shape else bc).copy_(src)
    (bh[1:] if bh.shape == z3.shape else bh).copy_(src)
    return err(bh, bc)


check("H2D broadcast-smaller CPU->HPU copies", lambda: worst(
    err(torch.zeros_like(z3, device=hpu).copy_(sc_cplx), torch.zeros_like(z3).copy_(sc_cplx)),
    h2d_broadcast_copy(lambda d: torch.zeros_like(z3, device=d), sc_cplx),
    err((lambda b: (b.diagonal(-2, -1).copy_(sc_cplx.conj()), b)[1])(torch.zeros_like(z3, device=hpu)),
        (lambda b: (b.diagonal(-2, -1).copy_(sc_cplx.conj()), b)[1])(torch.zeros_like(z3))),
    err(torch.zeros_like(z3, device=hpu).copy_(sc_real), torch.zeros_like(z3).copy_(sc_real)),
    h2d_broadcast_copy(lambda d: torch.zeros_like(z3, device=d), sc_real),
    h2d_broadcast_copy(lambda d: torch.zeros_like(z3, device=d), v1_cplx),
    h2d_broadcast_copy(lambda d: torch.zeros_like(z3, device=d), v1_real),
    err((lambda b: (b.diagonal(-2, -1).copy_(sc_cplx), b)[1])(torch.zeros_like(z6, device=hpu)),
        (lambda b: (b.diagonal(-2, -1).copy_(sc_cplx), b)[1])(torch.zeros_like(z6))),
), 0)

passed = sum(results)
print(f"PU1_CHECK {passed}/{len(results)} passed", flush=True)
sys.exit(0 if passed == len(results) else 1)

"""Card checks shared by pu1_check.py (eager) and lazy_check.py (PT_HPU_LAZY_MODE=1): a real HPU tensor and a Python
complex scalar, and complex cat / exp / neg / masked_fill_ / linalg.vector_norm on the float pairs, against torch on
the CPU, gradients included. ``add_checks(check, err, grad_err, worst, sync)`` registers one row per case through the
caller's ``check``; every case also fails if hpu_complex warns that an op ran on the CPU.
"""
import math
import warnings

import torch

hpu = torch.device("hpu")


def _no_cpu_fallback(fn):
    def wrapped():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = fn()
        ran = [str(w.message)[:90] for w in caught if "runs on the CPU" in str(w.message)]
        assert not ran, ran
        return out
    return wrapped


def _raises(fn, match):
    try:
        fn()
    except RuntimeError as e:
        assert match in str(e), str(e)[:200]
        return 0.0
    raise AssertionError("no error")


def _c99_exp(v):
    """C99 Annex G cexp (cmath raises at the non-finite points instead)."""
    x, y = v.real, v.imag
    if y == 0:
        return complex(math.exp(x) if x < 89 else math.inf, y) if not math.isnan(x) else complex(math.nan, y)
    if math.isinf(x) and not math.isfinite(y):
        return 0j if x < 0 else complex(math.inf, math.nan)
    if not math.isfinite(y) or math.isnan(x):
        return complex(math.nan, math.nan)
    e = math.exp(x) if x < 89 else math.inf
    return complex(e * math.cos(y), e * math.sin(y))


def _nan_bits_err(got, ref, signed_zeros=True):
    """0 if NaN sits exactly where torch has it and every other value is equal (signed zeros included unless
    signed_zeros=False), else 1."""
    g, r = torch.view_as_real(got.cpu()).contiguous(), torch.view_as_real(ref).contiguous()
    nan = r.isnan()
    same = (torch.equal(g.view(torch.int32)[~nan], r.view(torch.int32)[~nan]) if signed_zeros
            else torch.equal(g[~nan], r[~nan]))
    return float(not (torch.equal(g.isnan(), nan) and same))


def add_checks(check, err, grad_err, worst, sync=lambda: None):
    torch.manual_seed(0)
    xr = torch.linspace(-1, 1, 24).reshape(4, 6)
    tt = torch.linspace(-0.25, 0.25, 125)
    f = torch.tensor([2.0, 4.5, 8.0])

    def cmp(fn, *args):
        """fn(*hpu args) against fn(*cpu args): values and dtype."""
        got = fn(*[a.to(hpu) if isinstance(a, torch.Tensor) else a for a in args])
        sync()
        ref = fn(*args)
        assert got.dtype == ref.dtype and got.device.type == "hpu", (got.dtype, ref.dtype, got.device)
        return err(got, ref)

    scalar_ops = {
        "x * 2j": lambda x: x * 2j, "2j * x": lambda x: 2j * x, "x + 1j": lambda x: x + 1j,
        "add(x, 1j, alpha=2)": lambda x: torch.add(x, 1j, alpha=2), "x - (1+2j)": lambda x: x - (1 + 2j),
        "1j - x": lambda x: 1j - x, "x / 2j": lambda x: x / 2j, "(2+1j) / x": lambda x: (2 + 1j) / x,
        "mul(x, tensor(1-2j))": lambda x: torch.mul(x, torch.tensor(1 - 2j)),
        "tensor(1-2j) * x": lambda x: torch.tensor(1 - 2j) * x,
        "x[0] * 2j (0-d)": lambda x: x[0, 0] * 2j, "x.mT * 2j (strided)": lambda x: x.mT * 2j,
        "x[:, ::2] + 1j (sliced)": lambda x: x[:, ::2] + 1j,
    }
    def rel(fn, *args):
        ref = fn(*args)
        return cmp(fn, *args) / ref.abs().max().item()

    # relative: Habana's float division is not correctly rounded (x / 3 is 1.2e-7 off on the card, without complex)
    check("real HPU x complex scalar: * + - / rsub, 0-d, strided (relative)", _no_cpu_fallback(lambda: worst(
        *[rel(fn, xr) for fn in scalar_ops.values()])), 2e-7)
    check("real HPU x complex scalar: dtype promotion (bf16, int, bool -> c64; f64 -> c128)", lambda: worst(
        # against the card's own bfloat16 values: in lazy mode Habana's float32 -> bfloat16 cast rounds differently
        # from the CPU's (x.bfloat16().float() is 1.9e-3 off, no complex op involved; Gaudi2, bridge 1.21.4)
        (lambda xb: err(xb * 2j, xb.cpu() * 2j))(xr.to(hpu).bfloat16()), cmp(lambda x: (x * 4).long() * 2j, xr), cmp(lambda x: (x > 0) + 1j, xr),
        cmp(lambda x: x.double() * 2j, xr.double()), cmp(lambda x: x.double() - 1j, xr.double())), 0)
    check("real HPU x complex scalar: in-place / out= on a real tensor raise like torch", lambda: worst(
        *[_raises(lambda fn=fn: fn(xr.to(hpu).clone()), "can't be cast to the desired output type") for fn in (
            lambda x: x.mul_(2j), lambda x: x.add_(1j), lambda x: x.sub_(1j), lambda x: x.div_(2j),
            lambda x: torch.mul(x, 2j, out=torch.empty_like(x)), lambda x: torch.add(x, 1j, out=torch.empty_like(x)))]),
        0)
    check("real HPU x complex scalar: float16 raises (complex32)", lambda: _raises(
        lambda: xr.to(hpu).half() * 2j, "ComplexHalf"), 0)
    check("real HPU x real scalar unchanged (x * 2 + 1, x / 3 - 1)", lambda: worst(cmp(lambda x: x * 2 + 1, xr),
                                                                                  cmp(lambda x: x / 3 - 1, xr)), 2e-7)

    def gabor(fq, t):  # spd_learn's compute_gabor_wavelet
        w = torch.stack([torch.exp(2j * math.pi * a * t) * torch.exp(-4 * math.log(2) * t ** 2 / 0.3 ** 2) for a in fq])
        w = w / torch.linalg.norm(w, dim=-1, keepdim=True)
        return torch.cat((w.real, w.imag))

    check("Gabor wavelets exp(2j pi f t), norm, cat(real, imag): fwd+grad (relative)", _no_cpu_fallback(
        lambda: grad_err(lambda fq: gabor(fq, tt.to(fq.device)), f, relative=True)), 1e-5)

    z = torch.randn(3, 4, 5, dtype=torch.complex64)
    inf, nan = math.inf, math.nan
    edge = torch.tensor([0j, -0.0 + 0j, 1 + 0j, 3j, -1 - 1j, 80 + 0.5j, -100 + 2j, complex(inf, 0),
                         complex(-inf, 0), complex(-inf, 1), complex(nan, 0), complex(-inf, inf), complex(inf, inf),
                         complex(-inf, nan), complex(inf, nan), complex(nan, nan), complex(0, inf), complex(1, nan)],
                        dtype=torch.complex64)
    def exp_edges():
        # Reference: C99 cexp (Python's cmath), elementwise. torch's CPU kernel is not one reference here: torch 2.6
        # gives exp(inf + 0j) = inf + nan j and exp(nan + 0j) = nan + nan j on the Gaudi image's x86 build, and the
        # C99 inf + 0j and nan + 0j on an ARM Mac build (both measured).
        got = torch.view_as_real(torch.exp(edge.to(hpu))).cpu().double()
        ref = torch.view_as_real(torch.tensor([_c99_exp(v) for v in edge.tolist()], dtype=torch.complex128))
        assert torch.equal(got.isnan(), ref.isnan()) and torch.equal(got.isinf(), ref.isinf()), (got, ref)
        assert torch.equal(got[got.isinf()], ref[ref.isinf()]), (got, ref)
        fin = got.isfinite()
        return ((got[fin] - ref[fin]).abs() / ref[fin].abs().clamp_min(1)).max().item()

    check("complex exp: values, conj, strided, 0-d", _no_cpu_fallback(lambda: worst(
        cmp(torch.exp, z), cmp(lambda t: torch.exp(t.conj()), z), cmp(lambda t: torch.exp(t.mT), z),
        cmp(lambda t: torch.exp(t[1, 2, 3]), z))), 1e-5)
    check("complex exp: inf / NaN / large / subnormal edges like C99 cexp (relative)", exp_edges, 1e-6)
    check("complex exp / neg fwd+grad (relative)", _no_cpu_fallback(lambda: grad_err(
        lambda t: torch.view_as_real(-torch.exp(torch.complex(t, 2 * t))), xr, relative=True)), 1e-6)
    check("complex neg: values, conj, strided", _no_cpu_fallback(lambda: worst(
        cmp(torch.neg, z), cmp(lambda t: -(t.conj()), z), cmp(lambda t: -t[:, ::2].mT, z),
        cmp(lambda t: t.clone().neg_(), z))), 0)

    mask = torch.rand(4, 5) > 0.5
    check("complex masked_fill_: broadcast mask, 0 / real / complex value, 0-d value tensor, strided self",
          _no_cpu_fallback(lambda: worst(
              cmp(lambda t, m: t.clone().masked_fill_(m, 0), z, mask),
              cmp(lambda t, m: t.clone().masked_fill_(m, 2.5), z, mask),
              cmp(lambda t, m: t.clone().masked_fill_(m, 1 - 2j), z, mask),
              cmp(lambda t, m: t.clone().masked_fill_(m, torch.tensor(3j)), z, mask),
              cmp(lambda t, m: t.masked_fill(m[0], -1j), z, mask),
              cmp(lambda t, m: (lambda c: (c.mT.masked_fill_(m.mT, 7j), c)[1])(t.clone()), z, mask))), 0)
    check("complex masked_fill_: non-broadcastable / non-bool mask raise", lambda: worst(
        _raises(lambda: z.to(hpu)[0].masked_fill_(mask.to(hpu).expand(3, 4, 5), 1j), "expand"),
        _raises(lambda: z.to(hpu).masked_fill_(mask.to(hpu).float(), 1j), "boolean masks")), 0)
    check("complex masked_fill fwd+grad", _no_cpu_fallback(lambda: grad_err(
        lambda t: torch.view_as_real(torch.complex(t, t * t).masked_fill(t.detach() > 0, 1j)), xr)), 1e-6)

    a, b = torch.randn(2, 3, dtype=torch.complex64), torch.randn(2, 3, dtype=torch.complex64)
    r = torch.randn(2, 3)
    check("complex cat: dims, real operand, conj, legacy empty, strided", _no_cpu_fallback(lambda: worst(
        *[cmp(lambda p, q, d=d: torch.cat([p, q], d), a, b) for d in (0, 1, -1, -2)],
        cmp(lambda p, q, s: torch.cat([p, s, q.conj()], 1), a, b, r),
        cmp(lambda p, e: torch.cat([e, p, e]), a, torch.empty(0, dtype=torch.complex64)),
        cmp(lambda p, q: torch.cat([p.mT, q.mT[1:]]), a, b),
        cmp(lambda p, q: torch.cat([p[:, ::2], q[:, 1:]], 1), a, b))), 0)
    check("complex cat with float64 / CPU operand: complex128 / raises", lambda: worst(
        cmp(lambda p, s: torch.cat([p, s]), a, r.double()),
        _raises(lambda: torch.cat([a.to(hpu), b]), "same device")), 0)
    check("complex cat fwd+grad", _no_cpu_fallback(lambda: grad_err(
        lambda t: torch.view_as_real(torch.cat([torch.complex(t, -t), torch.complex(t * t, t)[1:]], 0)), xr)), 1e-6)

    cases = [(2, None, False), (2, -1, True), (2, (0, 2), False), (1, 1, False), (math.inf, -1, True),
             (-math.inf, 0, False), (0, None, False), (3.5, (1, 2), True)]
    check("complex linalg.vector_norm: ord / dim / keepdim, conj, strided, 0-d", _no_cpu_fallback(lambda: worst(
        *[cmp(lambda t, o=o, d=d, k=k: torch.linalg.vector_norm(t, o, d, k), z) for o, d, k in cases],
        cmp(lambda t: torch.linalg.vector_norm(t.conj().mT, dim=0), z), cmp(lambda t: torch.linalg.norm(t[1, 2, 3]), z),
        cmp(lambda t: torch.linalg.vector_norm(t, dtype=torch.complex128), z))), 1e-5)
    check("complex linalg.vector_norm fwd+grad (ord 2, 1, inf; relative)", _no_cpu_fallback(lambda: worst(
        *[grad_err(lambda t, o=o: torch.linalg.vector_norm(torch.complex(t, t.flip(-1)), o, -1), xr, relative=True)
          for o in (2, 1, math.inf)])), 1e-5)


    # ---- revision round: an independent review's cases ----------------------------------------------------------
    xe = torch.tensor([0.0, -0.0, 1.5, -2.0, inf, -inf, nan, 3e38])
    # Zero signs are not compared: Habana's own float x * 0.0 gives +0 for a negative x (torch -0), without complex.
    # The formula's signed zeros are checked bit for bit on the CPU (test_pair_kernels_formula_cpu.py).
    check("real x complex scalar at 0, -0, inf, nan: NaN where torch, values equal elsewhere", lambda: worst(
        *[_nan_bits_err(fn(xe.to(hpu)), fn(xe), signed_zeros=False) for fn in (
            lambda x: x * 2j, lambda x: (1 - 2j) * x, lambda x: x * (-0.0 - 0j), lambda x: x + 1j,
            lambda x: torch.add(x, -1j, alpha=-2), lambda x: x - (2 + 0j))]), 0)

    def scalar_tensor_grad(first):
        grads = []
        for dev in (hpu, "cpu"):
            w = torch.tensor(1 + 1j, requires_grad=True)
            x = xr.to(dev)
            ((w * x) if first else (x * w)).abs().sum().backward()
            grads.append(w.grad)
        assert grads[0] is not None, "w.grad lost"
        return err(grads[0], grads[1]) / grads[1].abs().item()

    check("gradient of a complex 0-d CPU tensor operand (x * w, w * x; relative)", lambda: worst(
        scalar_tensor_grad(False), scalar_tensor_grad(True)), 1e-6)
    zc = lambda: torch.empty(4, 6, dtype=torch.complex64, device=hpu)
    check("out= forms: tensor(1j) * x out=, mul/add/sub/div Scalar_out, rsub Scalar_out/Tensor_out, Scalar_mode_out",
          lambda: worst(
              err(torch.mul(torch.tensor(1j), xr.to(hpu), out=zc()), torch.tensor(1j) * xr),
              err(torch.ops.aten.mul.Scalar_out(xr.to(hpu), 2j, out=zc()), xr * 2j),
              err(torch.ops.aten.add.Scalar_out(xr.to(hpu), 2j, alpha=3, out=zc()), torch.add(xr, 2j, alpha=3)),
              err(torch.ops.aten.sub.Scalar_out(xr.to(hpu), 2j, alpha=1, out=zc()), xr - 2j),
              err(torch.ops.aten.div.Scalar_out(xr.to(hpu), 2j, out=zc()), xr / 2j),
              err(torch.ops.aten.rsub.Scalar_out(xr.to(hpu), 2j, alpha=1, out=zc()), 2j - xr),
              err(torch.ops.aten.rsub.Tensor_out(xr.to(hpu), torch.tensor(2j), alpha=1, out=zc()), 2j - xr)), 1e-6)
    check("add(x, y, alpha=1j) of two real HPU tensors raises like torch", lambda: _raises(
        lambda: torch.add(xr.to(hpu), xr.to(hpu), alpha=1j), "alpha must not be a complex number"), 0)

    zm = torch.tensor([1e-30, 3e19 + 3e19j, 1e-25j, complex(inf, nan), complex(nan, 1), -0.0 - 0j],
                      dtype=torch.complex64)
    check("complex abs at extremes (hypot: 1e-30, 3e19+3e19j, inf+nan j, nan, -0)", lambda: _nan_bits_err(
        torch.complex(zm.to(hpu).abs(), torch.zeros(6, device=hpu)), torch.complex(zm.abs(), torch.zeros(6))), 0)
    check("complex vector_norm at extremes: ord 0, 1, inf, -inf, 0.5 (relative)", lambda: worst(
        *[cmp(lambda t, o=o: torch.linalg.vector_norm(t, o), zm[:3]) / torch.linalg.vector_norm(zm[:3], o).item()
          for o in (0, 1, inf, -inf, 0.5)]), 1e-6)

    def norm_grad_at_zero():
        errs = []
        for o in (2, 1, 0.5, 3, inf):
            g = []
            for dev in (hpu, "cpu"):
                t = torch.zeros(3, 4, dtype=torch.complex64, device=dev).requires_grad_()
                torch.linalg.vector_norm(t, o, -1).sum().backward()
                g.append(t.grad)
            errs.append(err(g[0], g[1]))
        return worst(*errs)

    check("complex vector_norm gradient at z = 0 (ord 2, 1, 0.5, 3, inf)", norm_grad_at_zero, 0)
    check("complex vector_norm fwd+grad, fractional ord 0.5 / 1.5 / -1 (relative)", lambda: worst(
        *[grad_err(lambda t, o=o: torch.linalg.vector_norm(torch.complex(t, 1 - t), o, -1), xr, relative=True)
          for o in (0.5, 1.5, -1)]), 1e-5)
    check("complex masked_fill_ with -0.0 and -0j: signed zeros kept", lambda: worst(
        *[_nan_bits_err(z.to(hpu).masked_fill(mask.to(hpu), v), z.masked_fill(mask, v))
          for v in (-0.0, complex(-0.0, -0.0), complex(0.0, -0.0))]), 0)

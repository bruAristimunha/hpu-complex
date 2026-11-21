"""Card checks shared by pu1_check.py (eager) and lazy_check.py (PT_HPU_LAZY_MODE=1): a real HPU tensor and a Python
complex scalar, and complex cat / exp / neg / masked_fill_ / linalg.vector_norm on the float pairs, against torch on
the CPU, gradients included. ``add_checks(check, err, grad_err, worst, sync)`` registers one row per case through the
caller's ``check``; every case also fails if hpu_complex warns that an op ran on the CPU.
"""
import cmath
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
    edge = torch.tensor([0j, -0.0 + 0j, 1 + 0j, 3j, -1 - 1j, 80 + 0.5j, -100 + 2j, complex(math.inf, 0),
                         complex(-math.inf, 0), complex(-math.inf, 1), complex(math.nan, 0)], dtype=torch.complex64)
    def exp_edges():
        # Reference: C99 cexp (Python's cmath), elementwise. torch's CPU kernel is not one reference here: torch 2.6
        # gives exp(inf + 0j) = inf + nan j and exp(nan + 0j) = nan + nan j on the Gaudi image's x86 build, and the
        # C99 inf + 0j and nan + 0j on an ARM Mac build (both measured).
        got = torch.view_as_real(torch.exp(edge.to(hpu))).cpu().double()
        ref = torch.view_as_real(torch.tensor([cmath.exp(complex(v)) for v in edge.tolist()],
                                              dtype=torch.complex128))
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

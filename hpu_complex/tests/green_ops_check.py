"""Card checks shared by pu1_check.py (eager) and lazy_check.py (PT_HPU_LAZY_MODE=1): a real HPU tensor and a Python
complex scalar, and complex cat / exp / neg / masked_fill_ / linalg.vector_norm on the float pairs, against torch on
the CPU, gradients included. ``add_checks(check, err, grad_err, worst, sync)`` registers one row per case through the
caller's ``check``; every case also fails if hpu_complex warns that an op ran on the CPU.
"""
import math
import warnings

import torch

hpu = torch.device("hpu")


_STRICT = [False]  # inside _no_cpu_fallback: the card side of cmp / rel128 is watched for CPU computation


def _cpu_allocations(fn, at_least):
    """fn() and the CPU allocations of at least ``at_least`` bytes its ops made (torch.profiler's memory events).

    The dispatcher counter below sees only the extension's CPU fallback: a kernel that computes on the CPU itself
    (vector_norm(z.cpu().cdouble()).float().to(z.device), an explicit .cpu()) allocates a CPU copy of its input, and is
    caught here. Smaller host allocations are the bridge's own (24 bytes around a strided view, Gaudi2, 1.21.4).
    """
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        out = fn()
    return out, [e.cpu_memory_usage for e in prof.events() if e.cpu_memory_usage >= at_least]


def _card(fn, args=()):
    """fn(), the card side of a comparison; inside _no_cpu_fallback it must not copy an input's worth of memory to the
    CPU (the smallest input tensor of args sets the size)."""
    if not _STRICT[0]:
        return fn()
    sizes = [a.numel() * a.element_size() for a in args if isinstance(a, torch.Tensor) and a.numel() > 1]
    out, big = _cpu_allocations(fn, max(min(sizes, default=32), 32))
    assert not big, ("CPU allocations on the card side", big[:8])
    return out


def _host_path(fn):
    """fn() inside _no_cpu_fallback, for a case that computes on the CPU by design (complex128 lives in host memory)."""
    def wrapped():
        strict, _STRICT[0] = _STRICT[0], False
        try:
            return fn()
        finally:
            _STRICT[0] = strict
    return wrapped


def _no_cpu_fallback(fn):
    """fn's ops all ran on the card: no hpu_complex CPU-fallback call (counted per call: the warning is once per op and
    process, so a later fallback of an op that warned before would pass unseen), no such warning, and the card side of
    every cmp / rel128 in fn allocated no CPU memory beyond 0-d scalars (an explicit CPU computation)."""
    def wrapped():
        before = torch.ops.hpu_complex.cpu_fallback_calls()
        _STRICT[0] = True
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                out = fn()
        finally:
            _STRICT[0] = False
        ran = [str(w.message)[:90] for w in caught if "runs on the CPU" in str(w.message)]
        calls = torch.ops.hpu_complex.cpu_fallback_calls() - before
        assert not ran and calls == 0, (ran, calls)
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
        inputs = [a.to(hpu) if isinstance(a, torch.Tensor) else a for a in args]
        got = _card(lambda: (fn(*inputs), sync())[0], args)
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
        _host_path(lambda: cmp(lambda t: torch.linalg.vector_norm(t, dtype=torch.complex128), z))())), 1e-5)
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

    # ---- fix/complex-review: CODEX_REVIEW findings of PRs #9/#10, on the card ------------------------------------
    def rel128(fn, *args):
        """fn on the card against the exact value (fn in complex128 on the CPU) rounded to the result's dtype, relative:
        a part that overflows float32 is inf in both, a part that underflows is 0 in both."""
        inputs = [a.to(hpu) if isinstance(a, torch.Tensor) else a for a in args]
        got = _card(lambda: (fn(*inputs), sync())[0], args)
        ref = fn(*[a.to(torch.complex128 if a.is_complex() else torch.float64) for a in args])
        g, r = got.detach().cpu(), ref.detach().to(got.dtype).to(ref.dtype)
        g, r = (torch.view_as_real(g), torch.view_as_real(r)) if g.is_complex() else (g, r)
        assert torch.equal(g.isnan(), r.isnan()) and torch.equal(g.isinf(), r.isinf()), (got, ref)
        assert torch.equal(g[r.isinf()].double(), r[r.isinf()]), (got, ref)  # an infinity of the wrong sign fails
        fin = r.isfinite()
        return ((g[fin].double() - r[fin]).abs() / r[fin].abs().clamp_min(1e-300)).max().item() if fin.any() else 0.0

    xd = torch.tensor([1.0, -2.0, 0.5])
    # No subnormal result: the card flushes them to zero (1 / (3e37 + 1e37j) has a 1e-38 part, -0 there: Habana float32).
    wd = torch.tensor([1e20j, 1e-30j, 3e36 + 1e36j, 1e-30 + 1e-30j, 3 + 4j, -2j], dtype=torch.complex64)
    zd = torch.tensor([1 + 1j, -3e38 + 1j, 1e-30 - 2e-30j, 2.5, 1j, 7 - 7j], dtype=torch.complex64)
    check("review #10: division (Smith): x / 1e20j, x / 1e-30j, z / w at extremes (relative)", _no_cpu_fallback(
        lambda: worst(rel128(lambda x: x / 1e20j, xd), rel128(lambda x: x / 1e-30j, xd), rel128(torch.div, zd, wd),
                      rel128(torch.div, xd[:1], wd))), 2e-6)
    winf = torch.tensor([complex(inf, 0), complex(-inf, 2), complex(nan, 0), complex(0, inf)], dtype=torch.complex64)
    zf = torch.tensor([1 + 1j, 2.0, -1j, 0.5], dtype=torch.complex64)
    # Two complex card tensors with a real alpha stay pair arithmetic (speed; README Limits): not in this row.
    check("review #10: add / sub with inf: NaN where torch puts it (scalar, real operand, alpha 1j, rsub)", lambda: worst(
        *[_nan_bits_err(fn(*[t.to(hpu) for t in ts]), fn(*ts), signed_zeros=False) for fn, ts in (
            (lambda p: p + complex(inf, 0), (zf,)), (lambda p: p - complex(-inf, 2), (zf,)),
            (lambda p, q: torch.add(p, q, alpha=1j), (zf, winf)), (lambda p: complex(inf, 0) - p, (zf,)),
            (lambda x: x + complex(inf, 0), (xd,)), (lambda p, q: p.real + q, (zf, winf)))]), 0)
    check("review #10/medium: add(x, 1j, alpha=2j) = -1; sub(bool, 1j) / alpha=True raise like torch", lambda: worst(
        err(torch.add(xd.to(hpu), 1j, alpha=2j), torch.add(xd, 1j, alpha=2j)),
        _raises(lambda: torch.sub(xd.to(hpu) > 0, 1j), "bool tensor is not supported"),
        _raises(lambda: torch.add(xd.to(hpu), 1j, alpha=True), "Boolean alpha")), 0)
    c64v, f64v = torch.tensor([1 + 0.5j], dtype=torch.complex64), torch.tensor([1 + 2 ** -25], dtype=torch.float64)
    def promoted_exact():
        # The float64 HPU operand is a Habana float64 tensor (float32 precision on the card): the reference takes it
        # as the card holds it. The point: the result is complex128, computed in complex128 (main: complex64).
        zh, fh = c64v.to(hpu), f64v.to(hpu)
        errs = []
        for fn in (torch.add, torch.sub, torch.mul, torch.div, lambda p, q: q * p):
            got = fn(zh, fh)
            assert got.dtype == torch.complex128, got.dtype
            errs.append(err(got, fn(c64v, fh.cpu())))
        return worst(*errs)

    check("review #8: complex64 (+ - * /) float64 is complex128, exact", promoted_exact, 0)
    zn = {"tiny": [1e-30 + 0j], "huge": [3e19 + 3e19j], "mixed": [1e-30 + 0j, 3e19 + 0j, 1 + 1j],
          "small": [1e-25 + 1e-25j, 2e-25j]}
    check("review #11: vector_norm at extremes, ord 2, 1, 3, -1, inf (relative to the exact norm)", _no_cpu_fallback(
        lambda: worst(*[rel128(lambda t, o=o: torch.linalg.vector_norm(t, o), torch.tensor(v, dtype=torch.complex64))
                        for v in zn.values() for o in (2, 1, 3, -1, inf)])), 2e-6)

    def tiny_norm_grad():
        t = torch.tensor([1e-30 + 0j], dtype=torch.complex64, device=hpu).requires_grad_()
        torch.linalg.vector_norm(t).backward()
        return err(t.grad, torch.tensor([1 + 0j]))

    check("review #11: vector_norm gradient of [1e-30] is 1", tiny_norm_grad, 1e-6)

    def cat_out():
        out = torch.empty(2, dtype=torch.complex128, device=hpu)
        torch.cat([a.to(hpu)[0, :1], torch.tensor([16777217], dtype=torch.int32, device=hpu)], out=out)
        assert out.cpu()[1].real.item() == 16777217, out
        _raises(lambda: torch.cat([a.to(hpu)], out=torch.empty(1, device=hpu)), "can't be cast")
        zz = torch.tensor([1 + 2j], dtype=torch.complex128, device=hpu)
        return _raises(lambda: torch.cat([zz], out=zz), "memory location")

    check("review #12: cat.out in out's dtype (int32 16777217 exact), real out / out=input raise", cat_out, 0)
    check("review medium: exp(88.75 + i pi/4) and exp(100 + i (pi/2 - 1e-7)) fit (relative)", _no_cpu_fallback(
        lambda: rel128(torch.exp, torch.tensor([complex(88.75, math.pi / 4), complex(100.0, math.pi / 2 - 1e-7)],
                                               dtype=torch.complex64))), 2e-6)

    def masked_fill_cases():
        _raises(lambda: torch.zeros(2, dtype=torch.complex64, device=hpu).masked_fill_(
            torch.tensor([True, False], device=hpu), complex(-0.0, 1e40)), "without overflow")
        base, ref = torch.tensor([1 + 1j, 5j], device=hpu), torch.tensor([1 + 1j, 5j])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base[:1].expand(2).masked_fill_(torch.tensor([False, True], device=hpu), 2j)
            ref[:1].expand(2).masked_fill_(torch.tensor([False, True]), 2j)
        return err(base, ref)

    check("review medium: masked_fill_ overflow raises, expanded self writes as torch", masked_fill_cases, 0)

    def storage_cases():
        # #5: partial overlap of an in-place add_ / mul_ raises
        v = torch.tensor([1, 2, 3, 4, 5], dtype=torch.complex64, device=hpu)
        _raises(lambda: v[1:].add_(v[:-1]), "memory location")
        _raises(lambda: v[1:].mul_(v[:-1]), "memory location")
        # #3: a growing out= keeps the values outside its range
        base = torch.tensor([11, 22, 33], dtype=torch.complex64, device=hpu)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.neg(torch.tensor([7, 8, 9, 10], dtype=torch.complex64, device=hpu), out=base[1:])
        sync()
        e = err(base, torch.tensor([11, -7, -8], dtype=torch.complex64))
        # #4: set_() detaches; #1: a failed set_ leaves the tensor unchanged
        y = torch.tensor([1, 2], dtype=torch.complex64, device=hpu)
        alias = y.view(-1)
        y.set_()
        y.resize_(3).fill_(99)
        x = torch.tensor([2 + 0j], device=hpu)
        _raises(lambda: x.set_(torch.zeros(1, dtype=torch.complex64, device=hpu).untyped_storage(), 0, (2,), (1,)),
                "out of bounds")
        sync()
        return worst(e, err(alias, torch.tensor([1, 2], dtype=torch.complex64)), err(x, torch.tensor([2 + 0j])))

    check("review #1/#3/#4/#5: failed set_, growing out= prefix, set_() detach, overlapping add_/mul_", storage_cases, 0)

    # ---- fix/complex-review-2: the second review of PR #11 (CODEX_REVIEW.txt, Part B), on the card ----------------
    def one_by_one(fn, *args):
        """fn per element on the CPU: torch's scalar kernel, whose NaN placement is the same on every CPU (its
        vectorized complex kernels differ between x86 and ARM)."""
        return torch.cat([fn(*[a[i:i + 1] for a in args]) for i in range(args[0].shape[0])])

    small_ord = (([1e20, 1e-30], 0.01), ([1e30, 1e-20], -0.01), ([1e20, 1e-30], 0.5), ([1, 2, 3], 0.7),
                 ([1e-30, 2e-30], 0.25), ([0j, 1 + 1j], -0.5))
    check("review2 B5: vector_norm, |ord| < 1 keeps terms whose ratio under/overflows (relative x |ord|)",
          _no_cpu_fallback(lambda: worst(*[
              abs(o) * rel128(lambda t, o=o: torch.linalg.vector_norm(t, o), torch.tensor(v, dtype=torch.complex64))
              for v, o in small_ord])), 1e-5)

    def norm_nan_placement():
        zz = torch.tensor([[inf, nan], [complex(inf, nan), 1.0], [nan, 1.0], [inf, 1.0], [complex(nan, 1), 1j]],
                          dtype=torch.complex64)
        errs = []
        for o in (2, 1, 3.0, 0.5, -1, inf, -inf, 0):
            got = torch.linalg.vector_norm(zz.to(hpu), o, dim=1).cpu()
            ref = torch.linalg.vector_norm(zz.abs().double(), o, dim=1).float()  # torch's rule on the moduli
            nan_ref = ref.isnan()
            errs.append(float(not (torch.equal(got.isnan(), nan_ref) and torch.equal(got[~nan_ref], ref[~nan_ref]))))
        return worst(*errs)

    check("review2 B5: vector_norm NaN placement: NaN modulus anywhere -> NaN, inf + nan i is inf", norm_nan_placement, 0)

    def cat_out_cast_and_alias():
        out = torch.zeros(2, dtype=torch.complex64, device=hpu)
        torch.cat([torch.tensor([4097], dtype=torch.int32, device=hpu),
                   torch.tensor([1], dtype=torch.float16, device=hpu)], out=out)
        zz = torch.tensor([1 + 2j, 3 + 4j], device=hpu)
        _raises(lambda: torch.cat([zz.view(torch.float32)], out=zz), "memory location")
        return worst(err(out, torch.tensor([4097, 1], dtype=torch.complex64)),
                     err(zz, torch.tensor([1 + 2j, 3 + 4j])))

    check("review2 B6: cat.out casts each input (int32 4097 with float16), a real view of out raises",
          cat_out_cast_and_alias, 0)
    zq = torch.tensor([complex(inf, 1), complex(1, inf), complex(nan, 1), 1 + 1j, complex(inf, 1), 3 + 4j, 3 + 4j,
                       complex(-0.0, 1), 1 + 1j], dtype=torch.complex64)
    xq = torch.tensor([2.0, -3.0, 2.0, 0.0, -2.0, inf, -inf, -2.0, nan])
    wq = torch.tensor([complex(inf, 0), complex(nan, 0), 1 + 2j], dtype=torch.complex64)

    def nan_and_rel(got, ref):
        """1 unless NaN sits where torch has it, infinities match, and finite parts agree to 2.4e-7 relative (Habana's
        float division is not correctly rounded; torch multiplies by 1 / c)."""
        g, r = torch.view_as_real(got.cpu()), torch.view_as_real(ref)
        if not torch.equal(g.isnan(), r.isnan()) or not torch.equal(g[r.isinf()], r[r.isinf()]):
            return 1.0
        fin = r.isfinite()
        return float(((g[fin] - r[fin]).abs() > 2.4e-7 * r[fin].abs()).any())

    check("review2 B9: complex / real tensor and alpha=1+0j: NaN where torch puts it", lambda: worst(
        nan_and_rel(zq.to(hpu) / xq.to(hpu), one_by_one(torch.div, zq, xq)),
        _nan_bits_err(torch.add(zq[:3].to(hpu), wq.to(hpu), alpha=1 + 0j),
                      one_by_one(lambda p, q: torch.add(p, q, alpha=1 + 0j), zq[:3], wq), signed_zeros=False)), 0)

    def sum_out_dtype():
        f = torch.empty((), device=hpu)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.sum(torch.tensor([1 + 3j, 2 - 1j], device=hpu), 0, out=f)
        d = torch.empty((), dtype=torch.complex128, device=hpu)
        torch.sum(torch.tensor([2 ** 24, 1, -2 ** 24], dtype=torch.complex64, device=hpu), 0, out=d)
        _raises(lambda: torch.neg(torch.tensor([1j], device=hpu), out=torch.empty(1, dtype=torch.complex128,
                                                                                 device=hpu)), "Found dtype")
        return worst(err(f, torch.tensor(3.0)), err(d, torch.tensor(1 + 0j, dtype=torch.complex128)))

    check("review2 B10: sum(out=float32) sums real parts, sum(out=complex128) sums in complex128, neg out dtype",
          sum_out_dtype, 0)

    def set_and_inplace_cases():
        x = torch.tensor([2 + 0j], device=hpu)
        x.set_(torch.tensor([7 + 0j], device=hpu).untyped_storage(), 0, (1,), (-1,))  # stock: accepted
        y = torch.tensor([2 + 0j, 3 + 0j], device=hpu)
        _raises(lambda: y.set_(torch.zeros(3, dtype=torch.complex64, device=hpu).untyped_storage(), 0, (2,), (-1,)),
                "Negative strides")
        _raises(lambda: y.set_(torch.zeros(3, dtype=torch.complex64, device=hpu).untyped_storage(), -1, (2,), (1,)),
                "invalid storage offset")
        s = torch.tensor([1j, 2j], dtype=torch.complex128, device=hpu).untyped_storage()
        _raises(lambda: torch.empty(0, dtype=torch.complex128, device=hpu).set_(s[16:32]), "slice of a complex128")
        d = torch.tensor([2.0], dtype=torch.float64, device=hpu)
        _raises(lambda: d.add_(torch.tensor([1 + 1j], device=hpu)), "can't be cast")
        sync()
        return worst(err(x, torch.tensor([7 + 0j])), err(y, torch.tensor([2 + 0j, 3 + 0j])), err(d.float(),
                                                                                                torch.tensor([2.0])))

    check("review2 B1/B2/B4: set_ (singleton negative stride, failed set_, host slice), float64 add_ of complex",
          set_and_inplace_cases, 0)

    # ---- review2 B15: the detectors themselves catch the review's mutations ----------------------------------------
    def detectors_catch_mutations():
        zz = torch.randn(3, 4, 5, dtype=torch.complex64)
        caught = 0
        try:  # an explicit CPU computation: vector_norm on a complex128 CPU copy, moved back
            _no_cpu_fallback(lambda: cmp(lambda t: torch.linalg.vector_norm(t.cpu().cdouble()).float().to(t.device),
                                         zz))()
        except AssertionError as e:
            caught += "CPU allocations" in str(e)
        try:  # an infinity of the wrong sign
            rel128(lambda t: -t if t.device.type == "hpu" else t, torch.tensor([complex(inf, 1.0)],
                                                                             dtype=torch.complex64))
        except AssertionError:
            caught += 1
        return float(caught != 2)

    check("review2 B15: the CPU-computation and inf-sign detectors catch the review's mutations",
          detectors_catch_mutations, 0)

    # ---- M2 G10 (angle + eq.Scalar_out) & G11 (im2col + col2im) ------------------------------------------------
    z_ang = torch.tensor([0j, -0.0 + 0j, 1 + 2j, -3 + 0.5j, 0 - 1j, complex(inf, 1)], dtype=torch.complex64)
    z_ang_nan = torch.tensor([complex(nan, 0), complex(0, nan), 1 - 2j], dtype=torch.complex64)
    check("M2 G10: complex angle values, conj, strided, out=", _no_cpu_fallback(lambda: worst(
        cmp(torch.angle, z_ang),
        float(not torch.equal(torch.angle(z_ang_nan.to(hpu)).cpu().isnan(), torch.angle(z_ang_nan).isnan())),
        cmp(lambda t: torch.angle(t.conj()), z),
        cmp(lambda t: torch.angle(t.mT), z),
        cmp(lambda t: torch.angle(t, out=torch.empty(t.shape, dtype=torch.float32, device=t.device)), z))), 1e-6)
    check("M2 G10: complex angle fwd+grad (0j-safe, eq.Scalar_out on card)", _no_cpu_fallback(
        lambda: grad_err(lambda t: torch.angle(torch.complex(t, t.flip(-1) * 0.5))
                         + torch.angle(torch.complex(torch.zeros_like(t), torch.zeros_like(t))),
                         xr[..., :4].contiguous())), 1e-5)

    x_t1 = torch.complex(torch.randn(2, 3, 7, 1), torch.randn(2, 3, 7, 1))
    x_t2 = torch.complex(torch.randn(2, 3, 5, 6), torch.randn(2, 3, 5, 6))
    c_t2 = torch.complex(torch.randn(2, 12, 9), torch.randn(2, 12, 9))
    check("M2 G11: complex im2col / col2im Tier-1 & Tier-2 values", _no_cpu_fallback(lambda: worst(
        cmp(lambda t: torch.ops.aten.im2col(t, (3, 1), (1, 1), (0, 0), (3, 1)), x_t1),
        cmp(lambda t: torch.ops.aten.im2col(t, (2, 3), (1, 1), (1, 1), (1, 2)), x_t2),
        cmp(lambda t: torch.ops.aten.col2im(t, (5, 5), (2, 2), (2, 2), (0, 0), (1, 1)), c_t2))), 1e-5)
    check("M2 G11: complex im2col / col2im fwd+grad", _no_cpu_fallback(lambda: worst(
        grad_err(lambda t: torch.view_as_real(torch.ops.aten.im2col(
            torch.complex(t, 1 - t).reshape(2, 1, 6, 2), (2, 2), (1, 1), (1, 0), (1, 1))), xr),
        grad_err(lambda t: torch.view_as_real(torch.ops.aten.col2im(
            torch.complex(t, t * 0.5).reshape(2, 4, 3), (4, 2), (2, 2), (1, 1), (0, 0), (1, 1))), xr))), 1e-5)

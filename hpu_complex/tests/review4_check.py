"""On one Gaudi, eager: the review-round-4 cases, eager and under torch.compile(backend="hpu_backend"), vs the CPU.

1. float views of complex storage read inside fused graphs (view_as_real(z.diagonal())[..., 0] * 0.5, ...)
2. eig's V.real.mT.unsqueeze(0) through the view-chain fusion pass
3. x_hpu.view(torch.complex64) of a real HPU tensor
plus the out= / copy_ / view_as_complex edge rules and exact signs of the complex GEMM. Exits 1 if any fails.

python3 review4_check.py
"""
import math
import sys
import traceback
import warnings

import habana_frameworks.torch.core  # noqa: F401
import torch

import hpu_complex

hpu = torch.device("hpu")
hpu_complex.enable()
results = []


def check(name, fn, atol=0.0):
    try:
        error, note = (lambda r: r if isinstance(r, tuple) else (r, ""))(fn())
        torch.hpu.synchronize()
        ok = error <= atol
        print(f"{name:66s} max|err| {error:.1e} {'OK' if ok else 'FAIL'}{'  ' + note if note else ''}", flush=True)
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"{name:66s} ERROR {type(e).__name__}: {(str(e).splitlines() or [''])[0][:150]}", flush=True)
        traceback.print_exc(limit=3)
    results.append(ok)


def err(a, b):
    a, b = a.detach().cpu(), b.detach().cpu()
    assert a.shape == b.shape, (tuple(a.shape), tuple(b.shape))
    if a.numel() == 0:
        return 0.0
    if a.is_complex():
        a, b = torch.view_as_real(a.resolve_conj()), torch.view_as_real(b.to(a.dtype).resolve_conj())
    return (a.double() - b.double()).abs().max().item()


def tagged(t):
    return torch._C._dispatch_keys(t).has(torch._C.DispatchKey.PrivateUse1)


def cplx(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(*shape, generator=g), torch.randn(*shape, generator=g))


def fwd_bwd_err(f, x, compiled):
    """f(x) on HPU (eager or compiled) vs f on the CPU: output and input gradient, relative to the reference."""
    torch._dynamo.reset()
    fh = torch.compile(f, backend="hpu_backend") if compiled else f
    xc, xh = x.clone().requires_grad_(), x.to(hpu).requires_grad_()
    yc, yh = f(xc), fh(xh)
    yc.backward(torch.ones_like(yc))
    yh.backward(torch.ones_like(yh))
    return max(err(g, r) / max(r.detach().abs().max().item(), 1e-30) for g, r in ((yh, yc), (xh.grad, xc.grad)))


def raises_like_cpu(fn_cpu, fn_hpu, needle):
    """Both raise RuntimeError containing needle (torch's message)."""
    msgs = []
    for fn in (fn_cpu, fn_hpu):
        try:
            fn()
        except RuntimeError as e:
            msgs.append(str(e).splitlines()[0])
        else:
            return float("inf"), f"did not raise ({len(msgs) and 'hpu' or 'cpu'})"
    ok = all(needle in m for m in msgs)
    return (0.0 if ok else float("inf")), msgs[1][:90]


# ---- 1. float views of complex storage in fused graphs --------------------------------------------------------
_W = torch.linspace(0.5, 2.0, 25).reshape(5, 5)
_z = cplx(3, 5, 5)
for label, f in (
    ("view_as_real(z.diagonal())[..., 0] * 0.5", lambda t: (torch.view_as_real(t.diagonal(0, -2, -1))[..., 0] * 0.5)
     .exp()),
    ("z.diagonal().real * 2", lambda t: (t.diagonal(0, -2, -1).real * 2).exp()),
    ("z[:, 1:].diagonal().imag * 3 (offset)", lambda t: (t[:, 1:].diagonal(0, -2, -1).imag * 3).exp()),
    ("view_as_real(z)[..., 1].mT.unsqueeze(0) * W", lambda t: (torch.view_as_real(t)[..., 1].mT.unsqueeze(0)
                                                              * _W.to(t.device)).sin()),
):
    for compiled in (False, True):
        check(f"1 {'compiled' if compiled else 'eager   '} {label} fwd+bwd (rel)",
              lambda f=f, c=compiled: fwd_bwd_err(f, _z, c), 1e-5)

# ---- 2. eig's V.real.mT.unsqueeze(0) through the view-chain fusion ---------------------------------------------
_a = torch.randn(3, 5, 5, generator=torch.Generator().manual_seed(8))


def _eig_v_loss(t):  # |V|^2 elementwise: gauge invariant, so eig's backward accepts it; read through V.real.mT
    V = torch.linalg.eig(t)[1]
    return (V.real.mT.unsqueeze(0) ** 2 + V.imag.mT.unsqueeze(0) ** 2) * _W.mT.to(t.device)


def _eig_v_fwd(t):
    return torch.linalg.eig(t)[1].real.mT.unsqueeze(0) * 2


for compiled in (False, True):
    tag = "compiled" if compiled else "eager   "
    check(f"2 {tag} eig(real) V.real.mT.unsqueeze(0) * 2 fwd (rel)", lambda c=compiled: (lambda fh: err(
        fh(_a.to(hpu)), _eig_v_fwd(_a)) / _eig_v_fwd(_a).abs().max().item())(
        torch.compile(_eig_v_fwd, backend="hpu_backend") if c else _eig_v_fwd), 1e-5)
    check(f"2 {tag} eig(real) |V.real.mT|^2 + |V.imag.mT|^2 fwd+bwd (rel)",
          lambda c=compiled: fwd_bwd_err(_eig_v_loss, _a, c), 1e-4)
    check(f"2 {tag} eig(complex) |V.real.mT|^2 + |V.imag.mT|^2 fwd+bwd (rel)",
          lambda c=compiled: fwd_bwd_err(_eig_v_loss, cplx(3, 5, 5, seed=3), c), 1e-4)


# ---- 3. x_hpu.view(torch.complex64) ----------------------------------------------------------------------------
_x = torch.randn(3, 8)


def _view_dtype_eager():
    xh = _x.to(hpu)
    z = xh.view(torch.complex64)
    assert tagged(z), "not tagged"
    e = max(err(z, _x.view(torch.complex64)), err(z * z, _x.view(torch.complex64) ** 2),
            err(z.view(torch.float32), _x), err(xh[:, 2:].view(torch.complex64), _x[:, 2:].view(torch.complex64)))
    z.mul_(2)  # writes reach the float base
    return max(e, err(xh, _x * 2)), "tagged, values, z*z, round trip, offset view, write-through"


check("3 eager    x_hpu.view(complex64)", _view_dtype_eager, 1e-6)
check("3 eager    x_hpu[:, 1:7].view(complex64) raises as on the CPU", lambda: raises_like_cpu(
    lambda: _x[:, 1:7].view(torch.complex64), lambda: _x.to(hpu)[:, 1:7].view(torch.complex64), "storage_offset"))
check("3 eager    x_hpu.view(int32) unchanged", lambda: float(not torch.equal(
    _x.to(hpu).view(torch.int32).cpu(), _x.view(torch.int32))))


def _view_dtype_compiled():
    torch._dynamo.reset()
    f = lambda t: (t.view(torch.complex64) * (1 + 2j)).abs() + t[..., ::2]
    return err(torch.compile(f, backend="hpu_backend")(_x.to(hpu)), f(_x)) / f(_x).abs().max().item()


check("3 compiled (x.view(complex64) * (1+2j)).abs() + x[..., ::2] (rel)", _view_dtype_compiled, 1e-6)

# ---- minors -----------------------------------------------------------------------------------------------------
_buf = torch.randn(10)
check("m view_as_complex odd stride on a size-1 dim raises as on the CPU", lambda: raises_like_cpu(
    lambda: torch.view_as_complex(_buf.as_strided((3, 1, 2), (2, 3, 1))),
    lambda: torch.view_as_complex(_buf.to(hpu).as_strided((3, 1, 2), (2, 3, 1))), "stride divisible by 2"))
check("m view_as_complex empty with last stride 2 raises as on the CPU", lambda: raises_like_cpu(
    lambda: torch.view_as_complex(_buf.as_strided((0, 2), (1, 2))),
    lambda: torch.view_as_complex(_buf.to(hpu).as_strided((0, 2), (1, 2))), "stride 1"))
check("m view_as_complex 0-d raises as on the CPU", lambda: raises_like_cpu(
    lambda: torch.view_as_complex(torch.tensor(1.0)), lambda: torch.view_as_complex(torch.tensor(1.0, device=hpu)),
    "one or more dimensions"))
check("m view_as_complex valid (offset, size-1, empty) as on the CPU", lambda: max(
    err(torch.view_as_complex(_buf.to(hpu)[2:].view(4, 2)), torch.view_as_complex(_buf[2:].view(4, 2))),
    err(torch.view_as_complex(_buf.to(hpu).as_strided((3, 1, 2), (2, 2, 1))),
        torch.view_as_complex(_buf.as_strided((3, 1, 2), (2, 2, 1)))),
    err(torch.view_as_complex(_buf.to(hpu)[:0].view(0, 2)), torch.view_as_complex(_buf[:0].view(0, 2)))))

_A2, _B2 = cplx(7, 5, seed=1), cplx(5, 3, seed=2)
# exact signs of [ar | ai] @ [[br, bi], [-bi, br]] with a conjugate on either side: integers, so any sign change
# of the conj .neg() or the -bi block is a nonzero error
_ia = torch.complex(torch.tensor([[1.0, -2.0], [3.0, 4.0]]), torch.tensor([[2.0, 1.0], [-1.0, 5.0]]))
_ib = torch.complex(torch.tensor([[3.0, 0.0], [-1.0, 2.0]]), torch.tensor([[4.0, -3.0], [2.0, 1.0]]))


def _cpu_out_raises():
    try:
        torch.mm(_A2.to(hpu), _B2.to(hpu), out=torch.empty(7, 3, dtype=torch.complex64))
    except RuntimeError as e:
        return "same device" in str(e) and str(e).splitlines()[0][:60]
    return ""


for route, threshold in (("card", 0), ("CPU<2^22", 1 << 22)):
    hpu_complex.set_matmul_cpu_threshold(threshold)

    def _out_into_storage():
        big_h = torch.zeros(40, dtype=torch.complex64, device=hpu)
        big_c = torch.zeros(40, dtype=torch.complex64)
        torch.mm(_A2.to(hpu), _B2.to(hpu), out=big_h[10:10])
        torch.mm(_A2, _B2, out=big_c[10:10])
        grow_h = torch.empty(0, dtype=torch.complex64, device=hpu)
        alias = grow_h.view(-1)
        torch.mm(_A2.to(hpu), _B2.to(hpu), out=grow_h)
        same = alias.untyped_storage().nbytes() == grow_h.untyped_storage().nbytes() == 21 * 8
        return max(err(big_h, big_c), 0.0 if same else float("inf")), "slice of a buffer; growth keeps the StorageImpl"

    def _out_resize_warns():
        out = torch.empty(2, 2, dtype=torch.complex64, device=hpu)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            torch.mm(_A2.to(hpu), _B2.to(hpu), out=out)
        warned = any("was resized" in str(x.message) for x in w)
        return max(err(out, _A2 @ _B2), 0.0 if warned else float("inf")), "warned" if warned else "no warning"

    check(f"m mm.out writes into the out storage {route}", _out_into_storage, 1e-5)
    check(f"m mm.out resize of a non-empty out warns {route}", _out_resize_warns, 1e-5)
    check(f"m mm.out into an expanded out raises {route}", lambda: raises_like_cpu(
        lambda: torch.mm(_A2, _B2, out=torch.empty(1, 3, dtype=torch.complex64).expand(7, 3)),
        lambda: torch.mm(_A2.to(hpu), _B2.to(hpu), out=torch.empty(1, 3, dtype=torch.complex64, device=hpu)
                         .expand(7, 3)), "more than one element"))
    check(f"m mm.out into a CPU out with HPU inputs raises {route}", lambda: (lambda r: (
        0.0 if r else float("inf"), r))(_cpu_out_raises()))
    check(f"m complex mm exact signs, 4 conj combinations {route}", lambda: max(
        err(fa(_ia.to(hpu)) @ fb(_ib.to(hpu)), fa(_ia) @ fb(_ib))
        for fa in (lambda t: t, torch.conj) for fb in (lambda t: t, torch.conj)))


hpu_complex.set_matmul_cpu_threshold(1 << 22)
# Habana's mul of a real HPU tensor and a Python complex scalar computes into a complex HPU out next to a CPU scalar
# (spd_learn's Gabor wavelets, torch.exp(2j * pi * f * t)): no device check may reject it.
_t = torch.linspace(-0.25, 0.25, 125)
check("m real_hpu * 2j and exp(2j * pi * f * t) as on the CPU", lambda: max(
    err(_t.to(hpu) * 2j, _t * 2j), err(torch.exp(2j * math.pi * _t.to(hpu)[3] * _t.to(hpu)),
                                       torch.exp(2j * math.pi * _t[3] * _t))), 1e-5)
check("m copy_ with a partial self-overlap raises as on the CPU", lambda: raises_like_cpu(
    lambda: (lambda b: b[1:6].copy_(b[0:5]))(cplx(10)),
    lambda: (lambda b: b[1:6].copy_(b[0:5]))(cplx(10).to(hpu)), "single memory location"))
check("m copy_ full self-overlap / disjoint as on the CPU", lambda: max(
    err((lambda b: (b.copy_(b), b)[1])(cplx(10).to(hpu)), cplx(10)),
    err((lambda b: (b[5:].copy_(b[:5]), b)[1])(cplx(10).to(hpu)), (lambda b: (b[5:].copy_(b[:5]), b)[1])(cplx(10)))))

passed = sum(results)
print(f"REVIEW4_CHECK {passed}/{len(results)} passed", flush=True)
sys.exit(0 if passed == len(results) else 1)

"""On one Gaudi: complex128 HPU tensors (host memory, computed on the CPU in float64) against torch on the CPU.

python3 c128_check.py [--hpu-ops] [--time]     (PT_HPU_LAZY_MODE=0 eager or 1 lazy)

Every complex128 result must match torch on the CPU, applied to the same input values, to 1e-12, and come back as a
complex128 HPU tensor. One line per check with the bridge's CPU-fallback count and any "runs on the CPU" warning of
hpu_complex; exits 1 if any check fails. --time: mean of 20 synced calls of eigvals/eig of float64 input against the
CPU op on the same values (float64 inputs held on the card) and against complex64 (float32 input) on the card.
"""
import math
import os
import sys
import time
import traceback
import warnings

import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu  # noqa: F401
import torch
from habana_frameworks.torch.hpu.metrics import metric_global

import hpu_complex

if "--hpu-ops" in sys.argv:
    import hpu_ops

    hpu_ops.enable()
hpu_complex.enable()
hpu = torch.device("hpu")
LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
torch.manual_seed(0)
results = []
ATOL = 1e-12


def _fallback_count():
    """The bridge's CPU-fallback counter (TotalNumber of metric_global("cpu_fallback"))."""
    stats = metric_global("cpu_fallback").stats()
    total = 0
    for name, value in stats:
        if name == "TotalNumber":
            total = value
    return total


def sync():
    if LAZY:
        htcore.mark_step()
    torch.hpu.synchronize()


def to_cpu(t):
    return t.detach().cpu() if isinstance(t, torch.Tensor) else t


def err(got, ref):
    """max |got - ref| (float64), NaN on a shape or dtype mismatch."""
    got, ref = to_cpu(got), to_cpu(ref)
    if got.shape != ref.shape or got.dtype != ref.dtype:
        print(f"    mismatch: got {got.dtype}{tuple(got.shape)}, want {ref.dtype}{tuple(ref.shape)}")
        return math.nan
    if got.is_complex():
        got, ref = torch.view_as_real(got.resolve_conj()), torch.view_as_real(ref.resolve_conj())
    if got.numel() == 0:
        return 0.0
    return (got.double() - ref.double()).abs().max().item()


def check(name, fn, atol=ATOL, c64=False):
    """fn returns (outputs on HPU, references on the CPU): tensors or tuples of them.

    c64: the check also needs complex64 on the card, which lazy mode does not support yet (hpu-complex #3): skipped there.
    """
    if c64 and LAZY:
        print(f"{name:44s} SKIP (complex64 in lazy mode: hpu-complex #3)", flush=True)
        return
    before = _fallback_count()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            got, ref = fn()
            sync()
        got = got if isinstance(got, (tuple, list)) else (got,)
        ref = ref if isinstance(ref, (tuple, list)) else (ref,)
        errors = [err(g, r) for g, r in zip(got, ref)]
        on_hpu = all(g.device.type == "hpu" for g in got if isinstance(g, torch.Tensor))
        error = math.nan if any(math.isnan(e) for e in errors) else max(errors)
        on_cpu = [str(w.message).split(" on a complex")[0] for w in caught if "runs on the CPU" in str(w.message)]
        ok = error <= atol and on_hpu
        print(f"{name:44s} max|err| {error:.1e} hpu={on_hpu} bridge_fallbacks+{_fallback_count() - before}"
              f"{' ext_cpu=' + ','.join(on_cpu) if on_cpu else ''} {'OK' if ok else 'FAIL'}", flush=True)
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"{name:44s} ERROR {type(e).__name__}: {(str(e).splitlines() or [''])[0][:200]}", flush=True)
        traceback.print_exc(limit=3)
    results.append(ok)


def c128(*shape):
    return torch.randn(*shape, dtype=torch.complex128)


A = c128(4, 6, 6)
B = c128(4, 6, 3)
SPD = A @ A.mH + 6 * torch.eye(6, dtype=torch.complex128)
X64 = torch.randn(32, 22, 22, dtype=torch.float64)
Ah, Bh, SPDh = A.to(hpu), B.to(hpu), SPD.to(hpu)
X64h = X64.to(hpu)
X64_held = X64h.cpu()  # float64 as the card holds it (float32 precision): the reference input

# creation, copies, conversions, views
check("to(hpu) / cpu() round trip", lambda: (A.to(hpu), A))
check("1+1e-12j survives the round trip", lambda: (torch.tensor([1 + 1e-12j]).to(torch.complex128).to(hpu),
                                                   torch.tensor([1 + 1e-12j]).to(torch.complex128)), atol=0.0)
check("zeros / ones / full (complex128)", lambda: (
    (torch.zeros(3, 2, dtype=torch.complex128, device=hpu), torch.ones(3, dtype=torch.complex128, device=hpu),
     torch.full((2,), 1 - 2j, dtype=torch.complex128, device=hpu)),
    (torch.zeros(3, 2, dtype=torch.complex128), torch.ones(3, dtype=torch.complex128),
     torch.full((2,), 1 - 2j, dtype=torch.complex128))))
check("randn(complex128, hpu) dtype and device", lambda: (torch.randn(5, dtype=torch.complex128, device=hpu).abs() * 0,
                                                          torch.zeros(5, dtype=torch.float64)))
check("float64 hpu .to(complex128)", lambda: (X64h.to(torch.complex128), X64_held.to(torch.complex128)))
check("complex128 .to(complex64) and back", lambda: (Ah.to(torch.complex64).to(torch.complex128),
                                                     A.to(torch.complex64).to(torch.complex128)), c64=True)
check("complex64 hpu .to(complex128)", lambda: (A.to(torch.complex64).to(hpu).to(torch.complex128),
                                               A.to(torch.complex64).to(torch.complex128)), atol=1e-6, c64=True)
check("torch.complex(float64 hpu, float64 hpu)", lambda: (torch.complex(X64h, 2 * X64h),
                                                         torch.complex(X64_held, 2 * X64_held)), atol=1e-6)
check("view_as_real / view_as_complex", lambda: (
    (torch.view_as_real(Ah), torch.view_as_complex(torch.view_as_real(Ah))), (torch.view_as_real(A), A)))
check(".real / .imag / conj / resolve_conj", lambda: (
    (Ah.real, Ah.imag, Ah.conj(), Ah.conj().resolve_conj(), Ah.conj().imag), (A.real, A.imag, A.conj(), A.conj(),
                                                                              A.conj().imag)))


def write_through():
    z = Ah.clone()
    torch.view_as_real(z)[0, 0, 0, 1] = 7.0
    z[1].mul_(2)
    ref = A.clone()
    torch.view_as_real(ref)[0, 0, 0, 1] = 7.0
    ref[1].mul_(2)
    return z, ref


check("writes through views (view_as_real, slices)", write_through)
check("permute / slice / reshape", lambda: (Ah.permute(2, 0, 1)[1:, ::2].reshape(-1), A.permute(2, 0, 1)[1:, ::2].reshape(-1)))
check("abs / angle / sgn", lambda: ((Ah.abs(), Ah.angle(), Ah.sgn()), (A.abs(), A.angle(), A.sgn())), atol=1e-6)
check("arithmetic: * / + - with float64, 1j", lambda: (
    (Ah * Ah, Ah / Bh[..., :1], Ah + 1j, 2.5 - Ah, Ah * X64h[:4, :6, :6]),
    (A * A, A / B[..., :1], A + 1j, 2.5 - A, A * X64_held[:4, :6, :6])))
check("arithmetic with complex64 operands", lambda: (
    (Ah - Ah.to(torch.complex64), Ah.to(torch.complex64) * Ah), (A - A.to(torch.complex64), A.to(torch.complex64) * A)),
      c64=True)
check("sum / mean / where", lambda: ((Ah.sum(), Ah.sum(-1), Ah.mean(0), torch.where(Ah.real > 0, Ah, 0)),
                                     (A.sum(), A.sum(-1), A.mean(0), torch.where(A.real > 0, A, 0))))

# the ops of the tracking issue (#7), in complex128
OPS = {
    "matmul": lambda a, b, s: a @ b,
    "A @ A.mH": lambda a, b, s: a @ a.mH,
    "einsum": lambda a, b, s: torch.einsum("bij,bjk->bik", a, b),
    "trace": lambda a, b, s: a[0].trace(),
    "tril / triu": lambda a, b, s: (a.tril(), a.triu(1)),
    "inv": lambda a, b, s: torch.linalg.inv(a),
    "inv_ex": lambda a, b, s: torch.linalg.inv_ex(a)[0],
    "solve": lambda a, b, s: torch.linalg.solve(a, b),
    "solve_ex": lambda a, b, s: torch.linalg.solve_ex(a, b)[0],
    "solve_triangular": lambda a, b, s: torch.linalg.solve_triangular(s.tril(), b, upper=False),
    "cholesky": lambda a, b, s: torch.linalg.cholesky(s),
    "cholesky_ex": lambda a, b, s: torch.linalg.cholesky_ex(s)[0],
    "cholesky_solve": lambda a, b, s: torch.cholesky_solve(b, torch.linalg.cholesky(s)),
    "lu_factor + lu_solve": lambda a, b, s: torch.linalg.lu_solve(*torch.linalg.lu_factor(a), b),
    "lstsq": lambda a, b, s: torch.linalg.lstsq(a, b).solution,
    "det / slogdet": lambda a, b, s: (torch.linalg.det(a), torch.linalg.slogdet(a)[0]),
    "pinv": lambda a, b, s: torch.linalg.pinv(a),
    "matrix_exp": lambda a, b, s: torch.linalg.matrix_exp(a * 0.1),
    "householder_product": lambda a, b, s: torch.linalg.householder_product(*torch.geqrf(a)),
    "eigvals": lambda a, b, s: torch.linalg.eigvals(a),
    "qr (Q R)": lambda a, b, s: torch.linalg.qr(a),
    "fft / ifft": lambda a, b, s: (torch.fft.fft(a), torch.fft.ifft(a)),
}
for name, op in OPS.items():
    check(f"complex128 {name}", lambda op=op: (op(Ah, Bh, SPDh), op(A, B, SPD)))
# float64 outputs (eigvalsh, svdvals, norms) stay in host memory; matrix_rank's int64 is a Habana tensor
REAL_OPS = {
    "eigvalsh": lambda s: torch.linalg.eigvalsh(s),
    "svdvals": lambda s: torch.linalg.svdvals(s),
    "matrix_norm 2 / nuc": lambda s: (torch.linalg.matrix_norm(s, 2), torch.linalg.matrix_norm(s, "nuc")),
    "matrix_rank": lambda s: torch.linalg.matrix_rank(s).double(),
}
for name, op in REAL_OPS.items():
    check(f"complex128 {name} (real out)", lambda op=op: (op(SPDh), op(SPD)))


# gauge-invariant checks of eig / eigh / svd: eigenvector phases are not unique
def eig_reconstruct(a):
    w, v = torch.linalg.eig(a)
    return v @ torch.diag_embed(w) @ torch.linalg.inv(v)


check("complex128 eig: V diag(w) V^-1", lambda: (eig_reconstruct(Ah), eig_reconstruct(A)), atol=1e-11)
def eigh_reconstruct(s):
    w, v = torch.linalg.eigh(s)
    return v @ torch.diag_embed(w.to(torch.complex128)) @ v.mH


def svd_reconstruct(a):
    u, sv, vh = torch.linalg.svd(a)
    return u @ torch.diag_embed(sv.to(torch.complex128)) @ vh


# against the input: reconstruction error, float64 rounding of a 6x6 product (the eigenvalues stay in host memory)
check("complex128 eigh: V diag(w) V^H", lambda: (eigh_reconstruct(SPDh), SPD), atol=1e-12)
check("complex128 svd: U diag(S) Vh", lambda: (svd_reconstruct(Ah), A), atol=1e-12)

# issue #6: eig / eigvals of float64 input
check("#6 eigvals(float64 hpu) -> complex128", lambda: (torch.linalg.eigvals(X64h), torch.linalg.eigvals(X64_held)))
check("#6 eig(float64 hpu) -> complex128 (w, V)", lambda: (torch.linalg.eig(X64h), torch.linalg.eig(X64_held)))
check("#6 eigvals(float64) then .abs().max()", lambda: (torch.linalg.eigvals(X64h).abs().amax(-1),
                                                       torch.linalg.eigvals(X64_held).abs().amax(-1)))


def grad_case(device):
    a = (A if device == "cpu" else Ah).detach().clone().requires_grad_()
    loss = torch.view_as_real(torch.linalg.inv(a) @ a.mH).pow(2).sum()
    loss.backward()
    return a.grad


check("complex128 backward: inv(A) @ A^H", lambda: (grad_case("hpu"), grad_case("cpu")), atol=1e-10)

# complex64 is untouched: eigvals of float32 input stays complex64
X32h = X64_held.float().to(hpu)
check("float32 eigvals still complex64", lambda: (torch.linalg.eigvals(X32h), torch.linalg.eigvals(X64_held.float())),
      atol=1e-3, c64=True)


# adversarial edges: odd shapes, strided/offset views, extreme values, repeated calls, serialization, compile
check("n=1, batch 1: eig / inv / matmul", lambda: (
    (torch.linalg.eigvals(Ah[:1, :1, :1]), torch.linalg.inv(Ah[:1, :1, :1]), Ah[:1, :1, :1] @ Ah[:1, :1, :1]),
    (torch.linalg.eigvals(A[:1, :1, :1]), torch.linalg.inv(A[:1, :1, :1]), A[:1, :1, :1] @ A[:1, :1, :1])))
check("empty tensors (0 batch, 0 elements)", lambda: (
    (torch.linalg.eigvals(X64h[:0]), Ah[:, :0] * 2, torch.zeros(0, dtype=torch.complex128, device=hpu).sum()),
    (torch.linalg.eigvals(X64_held[:0]), A[:, :0] * 2, torch.zeros(0, dtype=torch.complex128).sum())))
check("0-d complex128", lambda: ((Ah[0, 0, 0] * Ah[1, 1, 1], Ah.sum().abs()), (A[0, 0, 0] * A[1, 1, 1], A.sum().abs())))
check("transposed, stepped, offset views as operands", lambda: (
    (Ah.mT[1:, ::2] @ Bh[1:], torch.linalg.eigvals(Ah.mT[2:]), torch.linalg.eigvals(X64h.mT[3:5])),
    (A.mT[1:, ::2] @ B[1:], torch.linalg.eigvals(A.mT[2:]), torch.linalg.eigvals(X64_held.mT[3:5]))))
check("broadcast with an expanded operand", lambda: (Ah * Ah[:1].expand(4, 6, 6), A * A[:1].expand(4, 6, 6)))
EXT = torch.tensor([1e300 + 1e-300j, float("inf"), complex(float("nan"), 1), -0.0j, 5e-324], dtype=torch.complex128)


check("extreme values keep their bits (1e300, inf, nan, -0, denormal)", lambda: (
    torch.view_as_real(EXT.to(hpu) * 1).view(torch.int64), torch.view_as_real(EXT * 1).view(torch.int64)), atol=0.0)
check("eigvals(float64 hpu) twice in a row (same values)", lambda: (
    (torch.linalg.eigvals(X64h), torch.linalg.eigvals(X64h)), (torch.linalg.eigvals(X64_held),) * 2))


def save_load():
    import io
    buf = io.BytesIO()
    torch.save(Ah, buf)
    buf.seek(0)
    return torch.load(buf, map_location="hpu"), A


check("torch.save / torch.load(map_location='hpu')", save_load)


def deepcopy_keeps_aliases():
    import copy
    state = copy.deepcopy({"z": Ah.clone(), "conj": Ah.conj()})
    state = copy.deepcopy({"z": state["z"], "real": state["z"].real, "tail": state["z"][1:], "conj": state["conj"]})
    copied_real = state["real"].clone()
    state["z"].zero_()  # the copied .real and slice alias the copied z, as torch's deepcopy keeps them
    return ((copied_real, state["real"], state["tail"], state["conj"], Ah),
            (A.real, torch.zeros_like(A.real), torch.zeros_like(A[1:]), A.conj().resolve_conj(), A))


check("copy.deepcopy (aliases kept, source untouched)", deepcopy_keeps_aliases, atol=0.0)


def restride_inplace():
    outs, refs = [], []
    for op in (lambda t: t.t_(), lambda t: t.unsqueeze_(0), lambda t: t.unsqueeze_(1).squeeze_(1),
               lambda t: t.as_strided_((2, 2), (1, 4), 1)):
        z, ref = A[0, :3, :4].to(hpu), A[0, :3, :4].clone()
        r, rr = z.real, ref.real
        op(z)
        op(ref)
        assert z.stride() == ref.stride() and z.storage_offset() == ref.storage_offset(), (z.stride(), ref.stride())
        z.mul_(2)
        ref.mul_(2)
        outs += [z, r]  # r: the .real view taken before the restride sees the write
        refs += [ref, rr]
    return outs, refs


check("t_ / unsqueeze_ / squeeze_ / as_strided_ in place", restride_inplace, atol=0.0)
check("repr of a complex128 HPU tensor", lambda: (
    Ah[0, 0, :2] if "complex128" in repr(Ah[0, 0, :2]) and "hpu" in repr(Ah[0, 0, :2]) else Ah[0, 0, :1], A[0, 0, :2]))


def eigvals_backward(device):
    a = (A if device == "cpu" else Ah).detach().clone().requires_grad_()
    torch.linalg.eigvals(a).abs().sum().backward()
    return a.grad


check("eigvals backward (complex128)", lambda: (eigvals_backward("hpu"), eigvals_backward("cpu")), atol=1e-10)


def compiled():
    f = torch.compile(lambda a, x: (torch.linalg.eigvals(x), a @ a.mH + 1), backend="hpu_backend")
    return f(Ah, X64h), (torch.linalg.eigvals(X64_held), A @ A.mH + 1)


check("torch.compile(hpu_backend): eigvals(float64) and complex128 ops", compiled)


def timed(fn, n=20):
    outs = [fn() for _ in range(3)]
    sync()
    outs = []
    t = time.perf_counter()
    for _ in range(n):
        outs.append(fn())
    sync()
    return 1e3 * (time.perf_counter() - t) / n, outs


if "--time" in sys.argv:
    Z128 = X64_held.to(torch.complex128) * (1 + 1j)
    Z128h = Z128.to(hpu)
    Z64h = None if LAZY else Z128.to(torch.complex64).to(hpu)
    cases = {
        "eigvals(float64 hpu) -> complex128 hpu": lambda: torch.linalg.eigvals(X64h),
        "eig(float64 hpu) -> complex128 hpu": lambda: torch.linalg.eig(X64h),
        "baseline: eigvals(x.cpu()) same values": lambda: torch.linalg.eigvals(X64h.cpu()),
        **({} if LAZY else {"baseline: eigvals(float32 hpu) -> complex64": lambda: torch.linalg.eigvals(X32h)}),
        "complex128 matmul (32x22x22) on hpu": lambda: Z128h @ Z128h,
        "baseline: complex128 matmul on cpu": lambda: Z128 @ Z128,
        **({} if LAZY else {"baseline: complex64 matmul on the card": lambda: Z64h @ Z64h}),
    }
    for name, fn in cases.items():
        before = _fallback_count()
        ms, _ = timed(fn)
        print(f"TIME {name:44s} {ms:8.3f} ms/call (20 synced) bridge_fallbacks+{_fallback_count() - before}",
              flush=True)

mode = "lazy" if LAZY else "eager"
print(f"C128_CHECK {mode} {sum(results)}/{len(results)} passed", flush=True)
sys.exit(0 if all(results) else 1)

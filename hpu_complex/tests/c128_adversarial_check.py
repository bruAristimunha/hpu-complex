"""Adversarial card check of complex128 on the HPU. One Gaudi, eager or PT_HPU_LAZY_MODE=1.

python3 hpu_complex/tests/c128_adversarial_check.py [--hpu-ops] [--rounds N] [--only substr] [--skip substr] [--models]

Each case builds its inputs on the card, runs the op there and on the CPU (torch, float64/complex128) from the values
the card holds, and prints OK / DIFF / ERR with the bridge cpu_fallback delta and any "runs on the CPU" warning. A
case where torch raises must raise on the HPU too (else DIFF). All cases run --rounds times (default 2) in the same
process, so lazy mode runs twice in a row; a case whose result changes between rounds is RACE. --models adds a
spd_learn Green train step in float64 (complex128 wavelets) and float32 (complex64), HPU vs CPU.
"""
import copy
import math
import os
import sys
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
LAZY = os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"
HPU = torch.device("hpu")
C128, F64 = torch.complex128, torch.float64
ROUNDS = int(sys.argv[sys.argv.index("--rounds") + 1]) if "--rounds" in sys.argv else 2
ONLY = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else ""
SKIP = sys.argv[sys.argv.index("--skip") + 1] if "--skip" in sys.argv else None  # e.g. compile (segfaults)


def sync():
    if LAZY:
        htcore.mark_step()
    torch.hpu.synchronize()


def fallbacks():
    return dict(dict(metric_global("cpu_fallback").stats()).get("FallbackOps", {}) or {})


def fb_delta(a, b):
    return {k: b[k] - a.get(k, 0) for k in b if b[k] - a.get(k, 0)}


def flat(x):
    if isinstance(x, (tuple, list)):
        return [y for e in x for y in flat(e)]
    return [x]


def host(x):
    return x.detach().cpu().resolve_conj().resolve_neg() if isinstance(x, torch.Tensor) else x


def maxerr(g, r):
    if g.dtype != r.dtype or g.shape != r.shape:
        return f"got {g.dtype}{tuple(g.shape)} want {r.dtype}{tuple(r.shape)}"
    if g.numel() == 0:
        return 0.0
    if g.dtype == torch.bool or not (g.is_floating_point() or g.is_complex()):
        return 0.0 if torch.equal(g, r) else "values differ"
    both_nan = torch.isnan(g) & torch.isnan(r) if not g.is_complex() else torch.isnan(g.abs()) & torch.isnan(r.abs())
    d = (g - r).abs()
    d = torch.where(both_nan | (g == r), torch.zeros_like(d), d)
    return d.max().item()


CASES = {}


def case(tol=1e-12):
    def deco(fn):
        CASES[fn.__name__] = (fn, tol)
        return fn
    return deco


def rnd(*shape, dtype=C128, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.randn(shape, dtype=dtype, generator=g)
    return t.float().double() if dtype == F64 else t  # float64 the card holds exactly (it keeps float32)


# ---- silent imaginary drops (torch raises) ----------------------------------------------------------------------

@case()
def f32_add_inplace_c128(d):
    x = torch.ones(3, 4, device=d)
    x.add_(rnd(3, 4).to(d))
    return x


@case()
def f32_mul_inplace_c128(d):
    x = torch.ones(3, 4, device=d)
    x.mul_(rnd(3, 4).to(d))
    return x


@case()
def f64_add_inplace_c128(d):
    x = torch.ones(3, 4, dtype=F64, device=d)
    x.add_(rnd(3, 4).to(d))
    return x


@case()
def mul_out_into_f32(d):
    z = rnd(3, 4).to(d)
    out = torch.empty(3, 4, device=d)
    torch.mul(z, z, out=out)
    return out


@case()
def c64_add_inplace_c128(d):  # torch allows it: cast to complex64
    x = torch.ones(3, 4, dtype=torch.complex64, device=d)
    x.add_(rnd(3, 4).to(d))
    return x


# ---- lazy: Habana copies from host memory, then the host memory changes before mark_step -----------------------

@case()
def copy_f64_from_real_then_overwrite(d):
    z = rnd(4, 5).to(d)
    x = torch.zeros(4, 5, dtype=F64, device=d)
    x.copy_(z.real)
    x2 = x * 1  # a Habana op on it, still in the lazy graph
    z.real.fill_(7.0)
    if d != "cpu":
        sync()
    return x2, z


@case(tol=1e-6)
def copy_c64_from_c128_then_overwrite(d):
    z = rnd(4, 5).to(d)
    w = torch.zeros(4, 5, dtype=torch.complex64, device=d)
    w.copy_(z)
    z.zero_()
    if d != "cpu":
        sync()
    return w


@case()
def to_float_then_overwrite(d):
    z = rnd(4, 5).to(d)
    x = z.real.to(torch.float32) * 2
    z.zero_()
    if d != "cpu":
        sync()
    return x


# ---- float64 / float32 Habana tensors into complex128 -----------------------------------------------------------

@case(tol=1e-6)
def f64_to_c128(d):
    return rnd(3, 4, dtype=F64).to(d).to(C128)


@case(tol=1e-6)
def f32_to_c128(d):
    return rnd(3, 4, dtype=torch.float32).to(d).to(C128)


@case(tol=1e-5)
def f64_times_1j(d):
    return rnd(3, 4, dtype=F64).to(d) * 1j


@case(tol=1e-5)
def exp_2j_pi_f64(d):
    return torch.exp(2j * math.pi * rnd(3, 4, dtype=F64).to(d))


@case(tol=1e-6)
def complex_of_f64_habana(d):
    return torch.complex(rnd(3, dtype=F64).to(d), rnd(3, dtype=F64, seed=1).to(d))


# ---- eig / eigvals of float64 input: odd shapes, views, values ---------------------------------------------------

def eigvals_of(a):
    def f(d):
        x = a.to(d) if d != "cpu" else a
        return torch.linalg.eigvals(x)
    return f


def eig_sorted(d, x):
    L = torch.linalg.eigvals(x)
    return L  # same LAPACK on the same values: no sort needed


@case()
def eigvals_n1(d):
    return torch.linalg.eigvals(rnd(1, 1, dtype=F64).to(d))


@case()
def eigvals_batch1(d):
    return torch.linalg.eigvals(rnd(1, 22, 22, dtype=F64).to(d))


@case()
def eigvals_empty_batch(d):
    return torch.linalg.eigvals(torch.zeros(0, 3, 3, dtype=F64, device=d))


@case()
def eigvals_n0(d):
    return torch.linalg.eigvals(torch.zeros(2, 0, 0, dtype=F64, device=d))


@case()
def eig_n0(d):
    return torch.linalg.eig(torch.zeros(2, 0, 0, dtype=F64, device=d))


@case()
def eigvals_transposed_offset_view(d):
    x = rnd(3, 6, 6, dtype=F64).to(d)[1:, 1:, 1:].mT
    return torch.linalg.eigvals(x)


@case()
def eigvals_expanded_batch(d):
    x = rnd(1, 5, 5, dtype=F64).to(d).expand(4, 5, 5)
    return torch.linalg.eigvals(x)


@case()
def eigvals_singular(d):
    return torch.linalg.eigvals(torch.ones(4, 4, dtype=F64, device=d))


@case()
def eigvals_hilbert12(d):
    i = torch.arange(12, dtype=F64)
    return torch.linalg.eigvals((1.0 / (i[:, None] + i[None, :] + 1)).float().double().to(d))


@case()
def eigvals_large_1e30(d):
    return torch.linalg.eigvals((rnd(5, 5, dtype=F64) * 1e30).float().double().to(d))


@case()
def eigvals_overflow_f32_1e39(d):  # finite float64, inf at float32
    return torch.linalg.eigvals((rnd(5, 5, dtype=F64) * 1e39).to(d))


@case()
def eigvals_subnormal_1e_40(d):  # subnormal at float32: flushed to zero on the card?
    return torch.linalg.eigvals((rnd(5, 5, dtype=F64) * 1e-40).to(d))


@case()
def eigvals_nan_inf(d):
    x = rnd(4, 4, dtype=F64)
    x[0, 0] = float("nan")
    return torch.linalg.eigvals(x.to(d))


@case()
def eig_out_into_c128(d):
    x = rnd(2, 4, 4, dtype=F64).to(d)
    L = torch.empty(0, dtype=C128, device=d)
    V = torch.empty(0, dtype=C128, device=d)
    torch.linalg.eig(x, out=(L, V))
    return L, V


@case()
def eigvals_of_host_f64(d):  # float64 that is a view of a complex128 tensor: exact float64 on the CPU
    return torch.linalg.eigvals(rnd(5, 5).to(d).real)


# ---- autograd --------------------------------------------------------------------------------------------------

@case(tol=1e-9)
def eig_backward_f64_leaf(d):
    x = rnd(3, 5, 5, dtype=F64).to(d).requires_grad_()
    L, V = torch.linalg.eig(x)
    (L.real.sum() + L.imag.pow(2).sum() + V.abs().pow(3).sum()).backward()
    return x.grad, torch.tensor(str(x.grad.dtype) + str(x.grad.device.type) == str(x.dtype) + str(x.device.type))


@case(tol=1e-5)
def eigvals_backward_then_adam(d):
    x = rnd(4, 4, dtype=F64).to(d).requires_grad_()
    opt = torch.optim.Adam([x], lr=0.1)
    for _ in range(3):
        opt.zero_grad()
        torch.linalg.eigvals(x).abs().sum().backward()
        opt.step()
        if d != "cpu":
            sync()
    return x.detach()


@case(tol=1e-10)
def double_backward_c128(d):
    z = rnd(3, 4).to(d).requires_grad_()
    y = (z.abs() ** 3).sum()
    (g,) = torch.autograd.grad(y, z, create_graph=True)
    (gg,) = torch.autograd.grad(g.abs().sum(), z)
    return g.detach(), gg


@case(tol=1e-10)
def grad_through_real_imag_views(d):
    z = rnd(3, 4).to(d).requires_grad_()
    (z.real[1:].pow(2).sum() + z.imag.t().exp().sum() + torch.view_as_real(z)[..., 0].sum()).backward()
    return z.grad


# ---- views and indexing ---------------------------------------------------------------------------------------

@case()
def unfold(d):
    return rnd(3, 8).to(d).unfold(1, 3, 2)


@case()
def stft_c128_signal(d):
    return torch.stft(rnd(64).to(d), 16, return_complex=True, window=torch.hann_window(16, dtype=F64).to(d))


@case()
def bool_mask_and_index(d):
    x = rnd(3, 4)
    z = x.clone().to(d)  # fix round 2: x.to("cpu") is x, so the CPU reference changed x and the masks built from it
    m = (x.real > 0).to(d)
    z[m] = 1j
    z[torch.tensor([0, 2], device=d)] *= 2
    z.masked_fill_((x.imag > 1).to(d), 0)
    z.index_add_(0, torch.tensor([1, 1], device=d), rnd(2, 4, seed=3).to(d))
    return z, z[m], z[[0, 2]]


@case()
def set_source_tensor_aliases(d):
    a = rnd(3, 4).to(d)
    b = rnd(2, 2, seed=5).to(d)
    a.set_(b)
    a.zero_()
    return b


@case()
def reductions(d):
    z = rnd(3, 4).to(d)
    return z.cumsum(1), z.prod(0), z.mean(), z.var(0), torch.linalg.norm(z), z.abs().sort(-1).values


@case()
def cat_c64_c128(d):
    return torch.cat([rnd(2, 3).to(d), rnd(2, 3, dtype=torch.complex64, seed=1).to(d)])


@case()
def inference_mode_tensor(d):
    with torch.inference_mode():
        z = torch.zeros(3, dtype=C128, device=d)
    return torch.tensor(z.is_inference())


@case(tol=1e-6)
def conv1d_c128(d):
    return torch.nn.functional.conv1d(rnd(2, 1, 16).to(d), rnd(3, 1, 5, seed=1).to(d))


@case(tol=1e-5)
def conv1d_c64(d):
    return torch.nn.functional.conv1d(rnd(2, 1, 16, dtype=torch.complex64).to(d),
                                      rnd(3, 1, 5, dtype=torch.complex64, seed=1).to(d))


@case()
def tiny_huge_c128(d):
    z = torch.tensor([1e-310 + 1e-310j, 1e300 + 1e300j, 1e-200 - 3j], dtype=C128).to(d)
    return z * z, z.abs(), z / 1e300


@case()
def deepcopy_module_f64_param_c128_buffer(d):
    m = torch.nn.Linear(3, 3, dtype=F64)
    torch.nn.init.constant_(m.weight, 0.5)
    m.register_buffer("zbuf", rnd(3))
    m = m.to(d)
    m2 = copy.deepcopy(m)
    m2.zbuf.mul_(2)
    return m.zbuf, m2.zbuf


# ---- round 2b (second tester): ops that reach ATen composites, Habana results, device checks ---------------------

@case()
def repeat_c128(d):  # aten::repeat uses unfold internally
    return rnd(3, 4).to(d).repeat(2, 1)


@case()
def tile_c128(d):
    return rnd(3, 4).to(d).tile(2)


@case()
def device_mismatch_raises(d):  # torch: a non-scalar CPU operand with a device tensor raises. (c128, c64, f32) on HPU
    if d == "cpu":
        return torch.tensor([True, True, True])
    raised = []
    for x in (rnd(3, 4).to(d), rnd(3, 4, dtype=torch.complex64).to(d), torch.ones(3, 4, device=d)):
        try:
            y = x + torch.ones(3, 4, dtype=x.dtype)
            sync()
            raised.append(False)
        except RuntimeError:
            raised.append(True)
    return torch.tensor(raised)


@case()
def predicates_and_compare(d):
    a, b = rnd(3, 4).to(d), rnd(3, 4, seed=1).to(d)
    return a.isfinite(), a.isnan(), a == b, a != b, torch.tensor(torch.allclose(a, a * 1)), torch.tensor(torch.equal(a, a.clone()))


@case(tol=1e-10)
def linalg_int_results(d):
    m = rnd(4, 4, seed=2) + 4 * torch.eye(4, dtype=C128)
    h = m @ m.mH
    m, h, b = m.to(d), h.to(d), rnd(4, 3, seed=1).to(d)
    return (torch.linalg.cholesky(h), torch.linalg.inv(m), torch.linalg.det(m), *torch.linalg.slogdet(m),
            torch.linalg.solve(m, b), torch.linalg.lstsq(m, b).solution, torch.linalg.matrix_power(m, -2))


@case()
def mask_select_and_scatter(d):
    a, b = rnd(3, 4).to(d), rnd(3, 4, seed=1).to(d)
    m = a.real > 0
    return a[m], a.masked_scatter(m, b), torch.nonzero(a.real > 0)


@case(tol=1e-9)
def double_backward_eigvals(d):
    x = (rnd(4, 4, seed=2) + 4 * torch.eye(4, dtype=C128)).to(d).requires_grad_()
    (g,) = torch.autograd.grad(torch.linalg.eigvals(x).abs().sum(), x, create_graph=True)
    (h,) = torch.autograd.grad(g.abs().sum(), x)
    return g.detach(), h


@case(tol=1e-12)
def grad_accumulates_twice(d):
    z = rnd(3, 4).to(d).requires_grad_()
    for k in range(2):
        (z.abs() ** (k + 2)).sum().backward()
    return z.grad


@case()
def version_counter_raises(d):
    x = rnd(3, 4).to(d).requires_grad_()
    w = (x * 1).exp()
    w.mul_(2)
    w.sum().abs().backward()
    return x.grad


@case()
def out_view_of_bigger(d):
    base = torch.zeros(6, 4, dtype=C128, device=d)
    torch.mul(rnd(3, 4).to(d), 2, out=base[::2])
    return base


@case(tol=1e-6)
def eig_into_habana_graph_twice(d):  # lazy: complex128 host result feeds float32 Habana ops, then the host changes
    x = rnd(3, 5, 5, dtype=F64).to(d)
    outs = []
    for k in range(2):
        L = torch.linalg.eigvals(x)
        y = (L.abs().float() * (k + 1)).sum(-1)
        L.zero_()
        outs.append(y)
    if d != "cpu":
        sync()
    return outs


@case()
def view_as_complex_of_f64_slice_backward(d):
    x = rnd(3, 4, 2, dtype=F64).to(d).requires_grad_()
    torch.view_as_complex(x).abs().pow(2).sum().backward()
    return x.grad


@case(tol=1e-12)
def eigvals_c128_input_batch_broadcast(d):
    a = rnd(1, 4, 4).to(d).expand(3, 4, 4)
    return torch.linalg.eigvals(a), torch.linalg.eig(a.mT)[0]


@case()
def empty_c128_ops(d):
    z = torch.zeros(0, 4, dtype=C128, device=d)
    return z.exp(), z.sum(0), z.abs(), z @ rnd(4, 2).to(d), torch.linalg.eigvals(torch.zeros(0, 0, dtype=C128, device=d))


@case()
def zero_d_c128(d):
    z = torch.tensor(1 + 2j, dtype=C128).to(d)
    return z * z, z.abs(), z.exp(), torch.tensor(z.item()), z.real + 1


@case()
def to_hpu_noncontig_and_offset(d):
    x = rnd(6, 8)[1::2, ::3].mT
    y = x.to(d) if d != "cpu" else x.clone()  # .to(device) keeps the layout as clone does (preserve_format)
    return y, y.contiguous(), torch.tensor(list(y.stride()))


@case(tol=1e-5)
def float_tensor_requires_grad_times_c128(d):  # float32 Habana weight * complex128 host signal: grad to the weight
    w = torch.ones(4, device=d, requires_grad=True)
    z = rnd(3, 4).to(d)
    (w * z).abs().sum().backward()
    return w.grad


@case()
def save_load_host_f64(d):  # a float64 result of complex128 ops (exact in host memory) through torch.save/load
    import io
    r = (rnd(3, 4).to(d).abs() * (1 + 1e-12))
    buf = io.BytesIO()
    torch.save(r, buf)
    buf.seek(0)
    back_ = torch.load(buf)
    return back_, torch.tensor(str(back_.device.type) == str(r.device.type))


@case()
def randn_c128_on_hpu(d):  # shape, dtype, device and finiteness only (generators differ)
    z = torch.randn(3, 4, dtype=C128, device=d)
    w = torch.randn_like(z)
    z.normal_()
    return torch.tensor(list(z.shape)), torch.tensor(z.isfinite().all().item() and w.isfinite().all().item())


@case()
def overlapping_out_raises(d):  # torch raises (partial overlap of out= and an input): returns whether it raised
    z = rnd(3, 4).to(d)
    try:
        torch.add(z[:, :-1], 1, out=z[:, 1:])
        return torch.tensor(False)
    except RuntimeError:
        return torch.tensor(True)


@case(tol=1e-10)
def istft_c128_roundtrip(d):
    w = torch.hann_window(16, dtype=F64).to(d)
    S = torch.stft(rnd(64).to(d), 16, return_complex=True, window=w)
    return torch.istft(S, 16, window=w, return_complex=True)


@case(tol=1e-6)
def c128_to_c64_then_card_op(d):  # the documented way to compute on the card
    z = rnd(3, 4).to(d)
    w = z.to(torch.complex64)
    return w * w, (w * w).to(C128)


# ---- torch.compile ---------------------------------------------------------------------------------------------

_compiled = {}


def compiled(key, fn):
    if key not in _compiled:
        torch._dynamo.reset()
        _compiled[key] = (torch.compile(fn, backend="hpu_backend"), torch.compile(fn, backend="eager"))
    return _compiled[key]


@case()
def compile_eigvals_f64(d):
    f = lambda x: torch.linalg.eigvals(x).abs().sum(-1) * 2  # noqa: E731
    fh, fc = compiled("eigvals", f)
    x = rnd(4, 5, 5, dtype=F64).to(d)
    return (fh if d != "cpu" else fc)(x)


@case()
def compile_c128_input(d):
    f = lambda z: (z * z.conj()).real.sum(-1) + z.abs().sum(-1)  # noqa: E731
    fh, fc = compiled("c128in", f)
    return (fh if d != "cpu" else fc)(rnd(3, 4).to(d))


@case(tol=1e-9)
def compile_eig_backward(d):
    def f(x):
        return torch.linalg.eigvals(x).abs().sum()
    fh, fc = compiled("eigbw", f)
    x = rnd(4, 4, dtype=F64).to(d).requires_grad_()
    (fh if d != "cpu" else fc)(x).backward()
    return x.grad


# ---- driver ------------------------------------------------------------------------------------------------------

def run_case(name, fn, tol):
    torch.manual_seed(0)
    try:
        ref = fn("cpu")
        ref_err = None
    except Exception as e:  # noqa: BLE001
        ref, ref_err = None, e
    f0 = fallbacks()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            torch.manual_seed(0)
            got = fn(HPU)
            sync()
            got = [host(t) for t in flat(got)]
            got_err = None
        except Exception as e:  # noqa: BLE001
            got, got_err = None, e
    fb = fb_delta(f0, fallbacks())
    cpu_w = sorted({str(w.message)[:80] for w in caught if "runs on the CPU" in str(w.message)})
    tail = f"fallback {fb or '+0'} cpu_warn {cpu_w or 'none'}"
    if ref_err is not None:
        if got_err is not None:
            return "OK", f"both raise (torch: {str(ref_err)[:60]!r}) {tail}", None
        return "DIFF", f"torch raises {type(ref_err).__name__}: {str(ref_err)[:90]!r}; HPU returns {got!r:.120} {tail}", got
    if got_err is not None:
        return "ERR", f"{type(got_err).__name__}: {str(got_err).splitlines()[0][:200]} {tail}", None
    refs = [host(t) for t in flat(ref)]
    errs = [maxerr(g, r) if isinstance(r, torch.Tensor) else (0.0 if g == r else f"{g!r} vs {r!r}")
            for g, r in zip(got, refs)]
    bad = [e for e in errs if isinstance(e, str) or not (e <= tol)]
    worst = max((e for e in errs if not isinstance(e, str)), default=0.0)
    return ("DIFF" if bad else "OK"), f"maxerr {bad[0] if bad else worst:.3g}" if not bad or not isinstance(bad[0], str) \
        else f"{bad[0]}", got


def main():
    print(f"PROBE lazy={LAZY} torch={torch.__version__} hpu_complex={hpu_complex.__file__} rounds={ROUNDS}", flush=True)
    first = {}
    status = {}
    for r in range(ROUNDS):
        for name, (fn, tol) in CASES.items():
            if (ONLY and (ONLY[1:] != name if ONLY.startswith("=") else ONLY not in name)) or (SKIP and any(k in name for k in SKIP.split(","))):
                continue
            try:
                st, msg, got = run_case(name, fn, tol)
            except Exception:  # noqa: BLE001
                st, msg, got = "CRASH", traceback.format_exc().splitlines()[-1], None
            if r == 0:
                first[name] = got
            elif got is not None and first.get(name) is not None:
                same = all(torch.equal(torch.nan_to_num(a), torch.nan_to_num(b)) if isinstance(a, torch.Tensor)
                           else a == b for a, b in zip(got, first[name]))
                if not same:
                    st = "RACE"
            print(f"R{r} {st:5s} {name}: {msg}", flush=True)
            status[name] = status.get(name, set()) | {st}
    if "--models" in sys.argv:
        for dt in (torch.float64, torch.float32):
            green_step(dt)
    bad = [n for n, s in status.items() if s != {"OK"}]
    print(f"SUMMARY {len(status)} cases, {len(bad)} not OK: {bad}", flush=True)


def green_step(dtype, steps=3):
    """spd_learn Green: forward + backward + Adam, HPU vs CPU from the same weights and batch."""
    from spd_learn.models import Green

    torch.manual_seed(0)
    X = torch.randn(8, 22, 250, dtype=dtype)
    y = torch.randint(0, 4, (8,))
    tag = f"green_{str(dtype).split('.')[-1]}"
    try:
        ref_m = Green(4, 22, sfreq=250, dtype=dtype)
        state = copy.deepcopy(ref_m.state_dict())
        out = {}
        for dev in ("cpu", HPU):
            m = Green(4, 22, sfreq=250, dtype=dtype)
            m.load_state_dict(state)
            m = m.to(dev)
            opt = torch.optim.Adam(m.parameters(), 1e-3)
            f0 = fallbacks()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                for _ in range(steps):
                    opt.zero_grad()
                    loss = torch.nn.functional.cross_entropy(m(X.to(dev)), y.to(dev))
                    loss.backward()
                    if dev != "cpu" and LAZY:
                        htcore.mark_step()
                    opt.step()
                    if dev != "cpu":
                        sync()
            out[str(dev)] = (loss.item(), {k: v.detach().cpu() for k, v in m.state_dict().items()},
                             fb_delta(f0, fallbacks()),
                             sorted({str(w.message)[:70] for w in caught if "runs on the CPU" in str(w.message)}))
        (lc, sc, _, _), (lh, sh, fb, cw) = out["cpu"], out["hpu"]
        perr = max(((sh[k].double() - sc[k].double()).abs().max().item() for k in sc if sc[k].is_floating_point()),
                   default=0.0)
        print(f"MODEL {tag}: OK loss cpu {lc:.10g} hpu {lh:.10g} |dloss| {abs(lc - lh):.3g} max param err after "
              f"{steps} Adam steps {perr:.3g} fallback {fb} cpu_warn {cw}", flush=True)
    except Exception as e:  # noqa: BLE001
        where = [f"{f.filename.split('site-packages/')[-1]}:{f.lineno}" for f in traceback.extract_tb(e.__traceback__)]
        print(f"MODEL {tag}: ERR {type(e).__name__}: {str(e).splitlines()[0][:200]} at {where[-3:]}", flush=True)


if __name__ == "__main__":
    main()

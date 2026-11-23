"""The C++ kernels on the CPU stand-in (cpu_stub/hpu_guard_stub.cpp): the review's Medium and Low findings, and the
card pair kernels (exp, neg, masked_fill_, cat, vector_norm, the real x complex-scalar path) run for real against
torch.

python -m pytest hpu_complex/tests/test_review_mediums_cpu.py -q
"""
import math
import warnings

import pytest
import torch
import torch.autograd.forward_ad as fwAD

from hpu_complex.tests.test_complex128_cpu import HPU, extension, tagged  # noqa: F401
from hpu_complex.tests.test_review_fixes_cpu import bits_equal, div_equal, h

C64, C128 = torch.complex64, torch.complex128
INF, NAN = math.inf, math.nan


# ---- alpha and bool validation of the complex-scalar interception ---------------------------------------


def test_complex_alpha_with_a_real_tensor_and_a_complex_scalar():
    x = torch.tensor([1.0])
    bits_equal(torch.add(x.to(HPU), 1j, alpha=2j), torch.add(x, 1j, alpha=2j))  # main raised: [-1+0j] in torch


@pytest.mark.parametrize("call", [
    lambda x, b: torch.sub(b, 1j), lambda x, b: torch.rsub(b, 1j), lambda x, b: torch.add(x, 1j, alpha=True),
    lambda x, b: torch.sub(x, 1j, alpha=True), lambda x, b: x.clone().add_(x, alpha=1j),
    lambda x, b: torch.rsub(x, x, alpha=1j), lambda x, b: x.clone().sub_(x, alpha=1j)])
def test_interception_rejects_what_torch_rejects(call):
    x, b = torch.tensor([1.0]), torch.tensor([True])
    with pytest.raises(RuntimeError) as ref:
        call(x, b)
    with pytest.raises(RuntimeError) as got:
        call(x.to(HPU), b.to(HPU))
    assert str(got.value).splitlines()[0] == str(ref.value).splitlines()[0]


# ---- 0-d complex tensors keep their autograd semantics (no .item()) -------------------------------------


def test_forward_ad_tangent_of_a_0d_complex_tensor():
    x = torch.tensor([2.0])
    with fwAD.dual_level():
        w = fwAD.make_dual(torch.tensor(1 + 1j), torch.tensor(3 + 4j))
        got = fwAD.unpack_dual(x.to(HPU) * w).tangent
        ref = fwAD.unpack_dual(x * w).tangent
    assert got is not None and torch.equal(got.cpu(), ref)  # main dropped it: 6 + 8j in torch


def test_modified_0d_scalar_is_detected_before_backward():
    x = torch.tensor([2.0], requires_grad=True)
    w = torch.tensor(1 + 1j)
    y = (x.to(HPU) * w).real.sum()
    w.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        y.backward()


def test_0d_scalar_mul_supports_double_backward_and_functorch():
    x = torch.tensor([2.0, -3.0], requires_grad=True)
    w = torch.tensor(1 + 2j)
    xh = x.detach().to(HPU).requires_grad_()
    (g_ref,) = torch.autograd.grad((x * w).abs().pow(2).sum(), x, create_graph=True)
    (gg_ref,) = torch.autograd.grad(g_ref.sum(), x)
    (g_got,) = torch.autograd.grad((xh * w).abs().pow(2).sum(), xh, create_graph=True)
    (gg_got,) = torch.autograd.grad(g_got.sum(), xh)
    torch.testing.assert_close(g_got.cpu(), g_ref)
    torch.testing.assert_close(gg_got.cpu(), gg_ref)
    # functorch grad skips MulWith0dComplexScalar cleanly
    fg_got = torch.func.grad(lambda t: (t * w).abs().sum())(xh.detach())
    fg_ref = torch.func.grad(lambda t: (t * w).abs().sum())(x.detach())
    torch.testing.assert_close(fg_got.cpu(), fg_ref)


def test_lazy_inference_mode_storage_and_view_paths():
    """#13 Finding 5: exercise lazy C++ storage/view paths under inference_mode via set_lazy_mode(True)."""
    import gc

    torch.ops.hpu_complex.set_lazy_mode(True)
    try:
        gc.collect()
        alive0 = torch.ops.hpu_complex.lazy_storages_alive()
        with torch.inference_mode():
            e = torch.empty(3, 4, dtype=C64, device=HPU)
            r = torch.randn(3, 4, 2, device=HPU)
            vc = torch.view_as_complex(r)
            vd = r.reshape(3, 8).view(C64)
            cx = torch.complex(r[..., 0], r[..., 1])
            vr = torch.view_as_real(cx)
            rl = cx.real
            im = cx.imag
            assert all(t.is_inference() for t in (e, vc, vd, cx, vr, rl, im))
            assert all(tagged(t) for t in (e, vc, vd, cx))
            assert not any(tagged(t) for t in (vr, rl, im))
            # Rank >= 5 permuted complex64 works in lazy mode, also under inference_mode (M6C lifted the refusal)
            perm5 = torch.empty(2, 2, 2, 2, 2, dtype=C64, device=HPU).permute(4, 3, 2, 1, 0)
            vr5 = torch.view_as_real(perm5)
            assert vr5.is_inference() and vr5.shape == (2, 2, 2, 2, 2, 2)
            del e, r, vc, vd, cx, vr, rl, im, perm5, vr5
        gc.collect()
        assert torch.ops.hpu_complex.lazy_storages_alive() == alive0
    finally:
        torch.ops.hpu_complex.set_lazy_mode(False)


# ---- inference mode: views of inference tensors are inference tensors ----------------------------------


def test_real_view_made_in_inference_mode_is_an_inference_tensor():
    with torch.inference_mode():
        z = h([1 + 2j], C128)
        r = z.real
        v = torch.view_as_real(z)
    assert z.is_inference() and r.is_inference() and v.is_inference()
    with pytest.raises(RuntimeError, match="[Ii]nference"):
        r.fill_(42)  # main: no error (r was a normal tensor); torch raises (after the write, as on the CPU)


# ---- exp: no overflow where the result fits ---------------------------------------------------------------


@pytest.mark.parametrize("v", [complex(88.75, math.pi / 4), complex(89.3, 1.5), complex(100.0, math.pi / 2 - 1e-7),
                               complex(88.0, 0.3), complex(87.9, -2.0), complex(95.0, 0.0), complex(-100.0, 1.0)])
def test_exp_large_real_parts(v):
    z = torch.tensor([v], dtype=C64)
    got = torch.view_as_real(h([v]).exp().cpu())
    ref = torch.view_as_real(z.to(C128).exp().to(C64))  # the exact value, rounded (torch's complex64: inf + inf j)
    torch.testing.assert_close(got, ref, rtol=3e-6, atol=3e-45, equal_nan=True)  # atol: two subnormal ulps


# ---- masked_fill_: checked conversion, expanded self ---------------------------------------------------


@pytest.mark.parametrize("value", [complex(-0.0, 1e40), complex(1.0, 1e40), complex(1e39, 0)])
def test_masked_fill_overflow_raises(value):
    with pytest.raises(RuntimeError, match="without overflow"):
        torch.zeros(2, dtype=C64).masked_fill_(torch.tensor([True, False]), value)
    z = h([0j, 0j])
    with pytest.raises(RuntimeError, match="without overflow"):
        z.masked_fill_(torch.tensor([True, False]).to(HPU), value)  # main wrote -0 + inf j
    assert torch.equal(z.cpu(), torch.zeros(2, dtype=C64))


@pytest.mark.parametrize("mask", [[True, True], [True, False], [False, False]])
def test_masked_fill_on_an_expanded_self_writes_like_torch(mask):
    ref_base = torch.tensor([1 + 1j, 5j])
    got_base = ref_base.to(HPU)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref_base[:1].expand(2).masked_fill_(torch.tensor(mask), 2j)
        got_base[:1].expand(2).masked_fill_(torch.tensor(mask).to(HPU), 2j)  # main raised (internal overlap)
    assert torch.equal(got_base.cpu(), ref_base)


# ---- per-op out= dtype rules ------------------------------------------------------------------------------


def test_exp_of_a_real_tensor_into_a_complex_out():
    x = torch.tensor([1.0, -2.0])
    out, ref = torch.empty(2, dtype=C64, device=HPU), torch.empty(2, dtype=C64)
    torch.exp(x.to(HPU), out=out)  # main: values_pair of a real tensor raised
    torch.exp(x, out=ref)
    assert torch.equal(out.cpu(), ref)


def test_neg_of_a_real_tensor_into_a_complex_out_raises():
    with pytest.raises(RuntimeError, match="Found dtype"):
        torch.neg(torch.ones(1).to(HPU), out=torch.empty(1, dtype=C64, device=HPU))


def test_complex_result_into_a_real_out_raises():
    """Elementwise ops only: a reduction casts its input to out's dtype first (sum(complex, out=float) is the sum of the
    real parts, test_review2_cpu.py)."""
    for fn in (lambda o: torch.exp(h([1j]), out=o), lambda o: torch.mul(h([1j]), 2, out=o),
               lambda o: torch.neg(h([1j]), out=o)):
        with pytest.raises(RuntimeError, match="can't be cast|Found dtype Float but expected ComplexFloat"):  # torch's
            fn(torch.empty(1, device=HPU))


# ---- Lows: memory_format of complex factories ------------------------------------------------------------


@pytest.mark.parametrize("dtype", [C64, C128])
def test_empty_honors_memory_format(dtype):
    got = torch.empty(2, 3, 4, 5, dtype=dtype, device=HPU, memory_format=torch.channels_last)
    assert got.stride() == torch.empty(2, 3, 4, 5, dtype=dtype, memory_format=torch.channels_last).stride()


# ---- the card pair kernels themselves, against torch -------------------------------------------------------

EDGE = torch.tensor([0j, complex(-0.0, 0.0), complex(0.0, -0.0), 1 + 2j, -3.5 + 0.25j, complex(INF, 0),
                     complex(-INF, 1), complex(INF, NAN), complex(NAN, 0), complex(1, INF), complex(-INF, INF),
                     complex(0, NAN), 1e-30 + 1e30j, complex(40.0, 3.0)], dtype=C64)


# C99 Annex G.6.3.1 cexp at its special values, (input, expected, signs specified). The table is the reference, not
# the platform's torch.exp (torch 2.6's x86 kernel gives exp(inf + 0j) = inf + nan j, its arm64 one inf + 0j).
# cexp(conj(z)) = conj(cexp(z)); -inf + iy gives +0 cis(y), +inf + iy (y finite, nonzero) +inf cis(y). Where C99 leaves
# the signs unspecified (the last four infinite cases), the kernel's choice is listed and only magnitudes are checked.
C99_CEXP = [
    (complex(0.0, 0.0), complex(1.0, 0.0), True),
    (complex(-0.0, 0.0), complex(1.0, 0.0), True),
    (complex(0.0, -0.0), complex(1.0, -0.0), True),
    (complex(-0.0, -0.0), complex(1.0, -0.0), True),
    (complex(INF, 0.0), complex(INF, 0.0), True),
    (complex(INF, -0.0), complex(INF, -0.0), True),
    (complex(-INF, 0.0), complex(0.0, 0.0), True),
    (complex(-INF, -0.0), complex(0.0, -0.0), True),
    (complex(-INF, 1.0), complex(0.0, 0.0), True),      # +0 (cos 1 + i sin 1)
    (complex(-INF, -2.0), complex(-0.0, -0.0), True),   # +0 (cos -2 + i sin -2): both negative
    (complex(INF, 1.0), complex(INF, INF), True),
    (complex(INF, 3.0), complex(-INF, INF), True),      # cos 3 < 0, sin 3 > 0
    (complex(NAN, 0.0), complex(NAN, 0.0), True),
    (complex(NAN, -0.0), complex(NAN, -0.0), True),
    (complex(NAN, 1.0), complex(NAN, NAN), True),
    (complex(1.0, INF), complex(NAN, NAN), True),
    (complex(0.0, INF), complex(NAN, NAN), True),
    (complex(1.0, NAN), complex(NAN, NAN), True),
    (complex(0.0, NAN), complex(NAN, NAN), True),
    (complex(NAN, NAN), complex(NAN, NAN), True),
    (complex(-INF, INF), complex(0.0, 0.0), False),
    (complex(-INF, NAN), complex(0.0, 0.0), False),
    (complex(INF, INF), complex(INF, NAN), False),
    (complex(INF, NAN), complex(INF, NAN), False),
]


def check_c99_cexp(exp):
    """exp (complex64 tensor -> complex64 tensor) against C99_CEXP: NaN exactly where C99 has NaN, every other part
    bit-equal (signs included) where C99 specifies the sign, equal in magnitude where it leaves the sign open."""
    z = torch.tensor([v for v, _, _ in C99_CEXP], dtype=torch.complex64)
    got = torch.view_as_real(exp(z)).contiguous()
    want = torch.view_as_real(torch.tensor([w for _, w, _ in C99_CEXP], dtype=torch.complex64))
    signed = torch.tensor([s for _, _, s in C99_CEXP])[:, None].expand_as(want)
    nan = want.isnan()
    assert torch.equal(got.isnan(), nan), (z, got)
    exact = signed & ~nan
    assert torch.equal(got[exact].view(torch.int32), want[exact].view(torch.int32)), (z, got)
    assert torch.equal(got[~signed & ~nan].abs(), want[~signed & ~nan].abs()), (z, got)


def test_exp_kernel_matches_torch_at_edges():
    """The special values against the C99 Annex G table (signs bit-exact where C99 fixes them), as torch's own exp is
    not the same on every platform there; finite inputs within 2 ulp of torch (e^a is computed as two halves, one
    rounding more than torch's e^a cos b), an exactly zero imaginary part kept with its sign."""
    check_c99_cexp(lambda z: h(z.tolist()).exp().cpu())
    fin = EDGE[EDGE.real.isfinite() & EDGE.imag.isfinite()]
    got = torch.view_as_real(h(fin.tolist()).exp().cpu())
    ref = torch.view_as_real(fin.exp())
    assert torch.equal(got.isfinite(), ref.isfinite()), (got, ref)
    zero = ref == 0
    assert torch.equal(got[zero].view(torch.int32), ref[zero].view(torch.int32)), (got, ref)
    torch.testing.assert_close(got, ref, rtol=2.5e-7, atol=0)


@pytest.mark.parametrize("shape", [(), (2, 3, 4)])
def test_exp_kernel_shapes_and_conj(shape):
    z = torch.randn(shape, dtype=C64, generator=torch.Generator().manual_seed(0))
    for inp in (z, z.conj()):
        torch.testing.assert_close(inp.to(HPU).exp().cpu(), inp.exp(), rtol=2e-6, atol=2e-6)

def test_neg_kernel_with_conjugate_views():
    z = torch.randn(3, 4, dtype=C64, generator=torch.Generator().manual_seed(1))
    bits_equal(-(z.to(HPU).conj()), -(z.conj()))
    # The pair kernel negates each float part: a sign flip, torch's float neg on every platform.
    pairs = torch.view_as_real(EDGE)
    flip = torch.view_as_complex(-pairs)
    bits_equal(torch.neg(h(EDGE.tolist())), flip)
    # torch's own complex neg is that sign flip on arm64 (-z per element) but 0 - z in its x86 AVX kernel, which
    # differ only at a +0 part (0 - (+0) = +0, where the flip gives -0): each part of torch.neg(EDGE) is one of the two.
    sub = torch.view_as_complex(torch.zeros_like(pairs) - pairs)
    ref_bits, flip_bits, sub_bits = (torch.view_as_real(t).view(torch.int32) for t in (torch.neg(EDGE), flip, sub))
    known = torch.view_as_real(flip).isnan().logical_not()
    assert torch.equal(torch.view_as_real(torch.neg(EDGE)).isnan(), ~known)
    assert ((ref_bits == flip_bits) | (ref_bits == sub_bits))[known].all(), torch.view_as_real(torch.neg(EDGE))
    assert torch.equal((flip_bits != sub_bits)[known], ((pairs == 0) & ~pairs.signbit())[known])


@pytest.mark.parametrize("value", [0, -0.0, complex(0, -0.0), 2.5, -1j, 1 + 2j, INF, complex(NAN, 1)])
@pytest.mark.parametrize("mask_shape", [(3, 4), (4,), (1, 4), (3, 1), ()])
def test_masked_fill_kernel_like_torch(value, mask_shape):
    z = torch.randn(3, 4, dtype=C64, generator=torch.Generator().manual_seed(2))
    m = torch.rand(mask_shape, generator=torch.Generator().manual_seed(3)) > 0.5
    got = z.to(HPU).masked_fill_(m.to(HPU), value)
    bits_equal(got, z.clone().masked_fill_(m, value))


@pytest.mark.parametrize("dim", [0, 1, -1, -2])
def test_cat_kernel_of_complex_and_real(dim):
    a = torch.randn(2, 3, dtype=C64, generator=torch.Generator().manual_seed(4))
    b = torch.randn(2, 3, generator=torch.Generator().manual_seed(5))
    bits_equal(torch.cat([a.to(HPU), b.to(HPU), a.conj().to(HPU)], dim),
               torch.cat([a, b, a.conj()], dim))
    bits_equal(torch.cat([torch.empty(0).to(HPU), a.to(HPU)]), torch.cat([torch.empty(0), a]))


@pytest.mark.parametrize("c", [2j, 1 + 2j, -0.5 - 0.25j, complex(0, -0.0), 3.0 + 0j])
@pytest.mark.parametrize("op", ["mul", "rmul", "add", "sub", "rsub", "div"])
def test_real_tensor_and_complex_scalar_bitwise(c, op):
    x = torch.tensor([1.5, -2.0, 0.0, -0.0, 1e-40, 1e-30, 3e38, INF, -INF, NAN])
    f = {"mul": lambda t: t * c, "rmul": lambda t: c * t, "add": lambda t: t + c, "sub": lambda t: t - c,
         "rsub": lambda t: c - t, "div": lambda t: t / c}[op]
    # div: torch's CPU complex division differs by platform (see div_equal); the rest stay bitwise.
    (div_equal if op == "div" else bits_equal)(f(x.to(HPU)), f(x))


def test_vector_norm_kernel_gradient_like_torch():
    z = torch.randn(3, 4, dtype=C64, generator=torch.Generator().manual_seed(6))
    for ord in (2, 1, INF, 3.0):
        a = z.clone().requires_grad_()
        b = z.to(HPU).detach().requires_grad_()
        torch.linalg.vector_norm(a, ord, dim=1).sum().backward()
        torch.linalg.vector_norm(b, ord, dim=1).sum().backward()
        torch.testing.assert_close(b.grad.cpu(), a.grad, rtol=2e-6, atol=1e-7)


# ---- the complex-scalar interception when the call skips the autograd keys --------------------------------

SCALAR_OPS = {"mul": lambda t: t * 2j, "rmul": lambda t: (1 - 2j) * t, "add": lambda t: t + 1j,
              "sub": lambda t: t - 1j, "rsub": lambda t: 1j - t, "div": lambda t: t / 2j,
              "exp_mul": lambda t: torch.exp(t * 2j), "0d_tensor": lambda t: t * torch.tensor(2j)}


@pytest.mark.parametrize("op", list(SCALAR_OPS))
def test_real_tensor_and_complex_scalar_under_inference_mode(op):
    """torch.inference_mode excludes the autograd keys: the call is taken at BackendSelect (main: it reached Habana's
    float kernel, which aborts the process in lazy mode; here the stub's CPU kernel gave an untagged complex result)."""
    x = torch.tensor([1.5, -2.0, 0.0, -0.0, 3e38])
    with torch.inference_mode():
        got = SCALAR_OPS[op](x.to(HPU))
        assert got.is_inference()
    assert tagged(got) and got.dtype == C64
    bits_equal(got, SCALAR_OPS[op](x))


def test_inference_tensor_and_complex_scalar_outside_inference_mode():
    """An inference tensor has no autograd keys, so outside inference mode too its call skips AutogradHPU."""
    with torch.inference_mode():
        x = torch.tensor([1.5, -2.0]).to(HPU)
    got = x * 2j
    assert tagged(got) and not got.requires_grad
    bits_equal(got, torch.tensor([1.5, -2.0]) * 2j)


def test_inference_mode_keeps_torch_errors_and_other_calls():
    x = torch.tensor([1.0, 2.0]).to(HPU)
    with torch.inference_mode():
        with pytest.raises(RuntimeError, match="can't be cast to the desired output type"):
            x.clone().mul_(2j)
        with pytest.raises(RuntimeError, match="alpha must not be a complex number"):
            torch.add(x, x, alpha=1j)
        with pytest.raises(RuntimeError, match="Subtraction, the `-` operator, with a bool tensor"):
            torch.sub(torch.tensor([True]).to(HPU), 1j)
        bits_equal(x * 2, torch.tensor([2.0, 4.0]))  # not complex: Habana's kernel, unchanged
        z = h([1 + 1j, 2])
        bits_equal(z * 2j, torch.tensor([1 + 1j, 2]) * 2j)  # a complex tensor: the PrivateUse1 kernels


def test_no_grad_and_grad_paths_unchanged():
    """With autograd keys the AutogradHPU kernel promotes first; BackendSelect then sees a complex operand."""
    x = torch.tensor([1.5, -2.0]).to(HPU).requires_grad_()
    y = x * 2j
    assert tagged(y) and y.requires_grad
    (y * (1 - 1j)).real.sum().backward()
    torch.testing.assert_close(x.grad.cpu(), torch.tensor([2.0, 2.0]))


def test_complex64_made_under_inference_mode_is_an_inference_tensor():
    """As torch: complex64 results and allocations under inference_mode, and their views, are inference tensors;
    a normal tensor used there stays normal (main: every complex64 kept the autograd keys)."""
    x = torch.tensor([1.0, 2.0]).to(HPU)
    z0 = h([1 + 2j, 3])
    with torch.inference_mode():
        z = torch.complex(x, x)
        e = torch.zeros(2, dtype=C64, device=HPU)
        r, v = z.real, torch.view_as_real(z)
        n = z0[:1]
    assert all(t.is_inference() for t in (z, e, r, v)) and not n.is_inference() and not z0.is_inference()
    assert tagged(z) and tagged(e) and not tagged(r)
    bits_equal(z * 2, torch.tensor([2 + 2j, 4 + 4j]))  # an inference tensor in a normal op outside: allowed
    with pytest.raises(RuntimeError, match="[Ii]nference"):
        z.add_(1)
    with pytest.raises(RuntimeError, match="[Ii]nference"):
        torch.view_as_real(z).fill_(0)


@pytest.mark.parametrize("device", ["cpu", HPU])
def test_backend_select_kernels_return_like_torch(device):
    """The BackendSelect kernels (unboxed for the common overloads) return self for in-place forms and out for out=,
    on CPU tensors (handed on at once) and HPU ones, in and out of inference mode."""
    for inference in (False, True):
        with torch.inference_mode(inference):
            x = torch.tensor([1.0, 2.0]).to(device)
            y = torch.tensor([3.0, 4.0]).to(device)
            assert x.add_(y) is x and x.mul_(2) is x and x.sub_(y, alpha=2) is x and x.div_(2.0) is x
            assert torch.equal(x.cpu(), torch.tensor([1.0, 2.0]))
            out = torch.empty(2).to(device)
            assert torch.add(x, y, alpha=3, out=out) is out and torch.mul(x, y, out=out) is out
            assert torch.div(x, y, out=out) is out and torch.equal(out.cpu(), torch.tensor([1 / 3, 0.5]))
            zc = torch.empty(2, dtype=C64).to(device)
            assert torch.mul(x, 2j, out=zc) is zc  # real x, complex scalar, complex out
            bits_equal(zc, torch.tensor([1.0, 2.0]) * 2j)
            bits_equal(torch.div(x, y, rounding_mode="floor"), torch.tensor([0.0, 0.0]))
            bits_equal(torch.rsub(x, 1j), 1j - torch.tensor([1.0, 2.0]))

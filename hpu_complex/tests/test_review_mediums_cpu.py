"""The C++ kernels on the CPU stand-in (cpu_stub/hpu_guard_stub.cpp): the review's Medium and Low findings, and the
card pair kernels (exp, neg, masked_fill_, cat, vector_norm, the real x complex-scalar path) run for real against
torch, where test_pair_kernels_formula_cpu.py only checks Python mirrors of their formulas.

python -m pytest hpu_complex/tests/test_review_mediums_cpu.py -q
"""
import math
import warnings

import pytest
import torch
import torch.autograd.forward_ad as fwAD

from hpu_complex.tests.test_complex128_cpu import HPU, extension, tagged  # noqa: F401
from hpu_complex.tests.test_review_fixes_cpu import bits_equal, h

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


@pytest.mark.xfail(strict=True, reason="kept fast path: a 0-d tensor read with .item() has no version to check "
                   "(README Limits); the promoted op cost Green 90 dispatched ops a step")
def test_modified_0d_scalar_is_detected_before_backward():
    x = torch.tensor([2.0], requires_grad=True)
    w = torch.tensor(1 + 1j)
    y = (x.to(HPU) * w).real.sum()
    w.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        y.backward()


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
    for fn in (lambda o: torch.exp(h([1j]), out=o), lambda o: torch.mul(h([1j]), 2, out=o),
               lambda o: torch.neg(h([1j]), out=o), lambda o: torch.sum(h([1j]), 0, out=o)):
        with pytest.raises(RuntimeError, match="can't be cast"):
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


def test_exp_kernel_matches_torch_at_edges():
    """NaN and inf exactly where torch has them, an exact-zero imaginary part (signed) kept, finite values within 2 ulp
    (e^a is computed as two halves, one rounding more than torch's e^a cos b)."""
    got = torch.view_as_real(h(EDGE.tolist()).exp().cpu())
    ref = torch.view_as_real(EDGE.exp())
    assert torch.equal(got.isnan(), ref.isnan()) and torch.equal(got.isinf(), ref.isinf()), (got, ref)
    assert torch.equal(got[ref.isinf()], ref[ref.isinf()])
    zero = ref == 0
    assert torch.equal(got[zero].view(torch.int32), ref[zero].view(torch.int32)), (got, ref)
    fin = ref.isfinite()
    torch.testing.assert_close(got[fin], ref[fin], rtol=2.5e-7, atol=0)


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
@pytest.mark.parametrize("mask_shape", [(3, 4), (4,), (3, 1), ()])
def test_masked_fill_kernel_like_torch(value, mask_shape):
    z = torch.randn(3, 4, dtype=C64, generator=torch.Generator().manual_seed(2))
    m = torch.rand(mask_shape, generator=torch.Generator().manual_seed(3)) > 0.5
    got = z.to(HPU).masked_fill_(m.to(HPU), value)
    bits_equal(got, z.clone().masked_fill_(m, value))


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_cat_kernel_of_complex_and_real(dim):
    a = torch.randn(2, 3, dtype=C64, generator=torch.Generator().manual_seed(4))
    b = torch.randn(2, 3, generator=torch.Generator().manual_seed(5))
    bits_equal(torch.cat([a.to(HPU), b.to(HPU), a.conj().to(HPU)], dim),
               torch.cat([a, b, a.conj()], dim))
    bits_equal(torch.cat([torch.empty(0).to(HPU), a.to(HPU)]), torch.cat([torch.empty(0), a]))


@pytest.mark.parametrize("c", [2j, 1 + 2j, -0.5 - 0.25j, complex(0, -0.0), 3.0 + 0j])
@pytest.mark.parametrize("op", ["mul", "rmul", "add", "sub", "rsub", "div"])
def test_real_tensor_and_complex_scalar_bitwise(c, op):
    x = torch.tensor([1.5, -2.0, 0.0, -0.0, 1e-30, 3e38])
    f = {"mul": lambda t: t * c, "rmul": lambda t: c * t, "add": lambda t: t + c, "sub": lambda t: t - c,
         "rsub": lambda t: c - t, "div": lambda t: t / c}[op]
    if op == "div" and c == complex(0, -0.0):
        pytest.skip("division by zero: NaN payloads differ")
    bits_equal(f(x.to(HPU)), f(x))


def test_vector_norm_kernel_gradient_like_torch():
    z = torch.randn(3, 4, dtype=C64, generator=torch.Generator().manual_seed(6))
    for ord in (2, 1, INF, 3.0):
        a = z.clone().requires_grad_()
        b = z.to(HPU).detach().requires_grad_()
        torch.linalg.vector_norm(a, ord, dim=1).sum().backward()
        torch.linalg.vector_norm(b, ord, dim=1).sum().backward()
        torch.testing.assert_close(b.grad.cpu(), a.grad, rtol=2e-6, atol=1e-7)

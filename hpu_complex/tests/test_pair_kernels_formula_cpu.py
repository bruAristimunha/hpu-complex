"""CPU checks of the float-pair formulas behind the card kernels for complex neg, exp, masked_fill_, cat and
linalg.vector_norm, and of the promotion rule for a real HPU tensor and a Python complex scalar.

The kernels (cpp_ext/fft_ext.cpp: neg_pu1, exp_pu1, masked_fill_pu1, cat_pu1, vector_norm_pu1,
complex_scalar_autograd_hpu) need a Gaudi to run. Each is mirrored here op for op on [..., 2] float pairs, as
test_cmatmul_formula_cpu.py does for the complex GEMM, and compared with torch's complex kernels, edge values included.
The card runs are pu1_check.py (eager) and lazy_check.py (lazy mode).

python -m pytest hpu_complex/tests/test_pair_kernels_formula_cpu.py -q
"""
import math

import pytest
import torch

INF, NAN = math.inf, math.nan


def pair(z):
    return torch.view_as_real(z.resolve_conj()).contiguous()


def as_complex(p):
    return torch.view_as_complex(p.contiguous())


def as_pair(t):
    return pair(t) if t.is_complex() else torch.stack([t.float(), torch.zeros_like(t, dtype=torch.float32)], -1)


# ---- mirrors of the kernels -------------------------------------------------------------------


def exp_pairs(z):
    p = pair(z)
    a, b = p[..., 0], p[..., 1]
    ea = a.exp()
    re, im = ea * b.cos(), torch.where(b != 0, ea * b.sin(), b)
    special = a.isinf() & ~b.isfinite()
    return as_complex(torch.stack([torch.where(special, ea, re), torch.where(special, ea - ea, im)], -1))


def neg_pairs(z):
    return as_complex(-pair(z))


def masked_fill_pairs(z, mask, value):
    m = mask.expand(z.shape)
    c = complex(value)
    p = pair(z)
    if c == 0 and not math.copysign(1, c.real) < 0 and not math.copysign(1, c.imag) < 0:
        filled = p.masked_fill(m.unsqueeze(-1), 0.0)
    else:
        filled = torch.stack([p[..., 0].masked_fill(m, c.real), p[..., 1].masked_fill(m, c.imag)], -1)
    return as_complex(filled)


def cat_pairs(tensors, dim):
    kept = [t for t in tensors if not (t.dim() == 1 and t.numel() == 0)]
    d = dim % kept[0].dim()
    return as_complex(torch.cat([as_pair(t) for t in kept], d))


def vector_norm_pairs(z, ord=2, dim=None, keepdim=False):
    p = pair(z)
    if ord == 2:
        dims = [d % max(z.dim(), 1) for d in ((dim,) if isinstance(dim, int) else dim)] if dim not in (None, ()) else \
            list(range(z.dim()))
        if z.dim() == 0:
            dims = []
        out = torch.linalg.vector_norm(p, 2, dims + [p.dim() - 1], keepdim=True).squeeze(-1)
        return out if keepdim or not dims else out.squeeze(dims)
    return torch.linalg.vector_norm(modulus_pairs(p), ord, dim, keepdim)


def modulus_pairs(p):
    """modulus(): hypot scaled by max(|re|, |im|), inf if either part is infinite."""
    mag = p.abs()
    m = mag.max(-1, keepdim=True).values
    r = m * torch.linalg.vector_norm(p / torch.where(m > 0, m, torch.ones_like(m)), 2, -1, keepdim=True)
    return torch.where((mag == INF).any(-1, keepdim=True), INF, r).squeeze(-1)


def test_modulus_is_torch_abs_at_extremes():
    """Unscaled sqrt(re^2 + im^2) in float gives inf for 3e19 + 3e19j and 0 for 1e-30 (the card's vector_norm did)."""
    got, ref = modulus_pairs(pair(EDGE)), EDGE.abs()
    assert torch.equal(got.isnan(), ref.isnan())
    same(got, ref, rtol=1e-6)
    t = torch.tensor([1e-30, 3e19 + 3e19j, 1e-25j], dtype=torch.complex64)
    for ord in (0, 1, INF, -INF, 0.5):
        same(vector_norm_pairs(t, ord), torch.linalg.vector_norm(t, ord), rtol=1e-6)


def promoted(x, scalar):
    """The kernel's rule: torch's result_type of (real HPU tensor, complex scalar); the real operand is cast to it."""
    return torch.result_type(x, scalar)


# ---- tests ----------------------------------------------------------------------------------------


def cplx(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, dtype=torch.complex64, generator=g)


EDGE = torch.tensor([0j, -0.0 + 0j, 1 + 0j, -2.5 - 0j, 3j, -1 - 1j, 88 + 0.5j, -104 + 2j, complex(INF, 0),
                     complex(-INF, 0), complex(-INF, 1), complex(INF, 1), complex(NAN, 0), complex(1, NAN),
                     complex(-INF, INF), complex(INF, INF), complex(-INF, NAN), complex(INF, NAN), complex(NAN, NAN),
                     complex(0, INF), complex(3e19, 3e19), complex(1e-30, 0), complex(0, -1e-25)],
                    dtype=torch.complex64)


def same(got, ref, atol=2e-6, rtol=2e-6):
    torch.testing.assert_close(got, ref, atol=atol, rtol=rtol, equal_nan=True)


@pytest.mark.parametrize("shape", [(), (1,), (5,), (3, 4), (2, 3, 4)])
def test_exp_matches_torch(shape):
    z = cplx(*shape)
    same(exp_pairs(z), torch.exp(z))
    same(exp_pairs(z.conj()), torch.exp(z.conj()))


def test_exp_edge_values_like_c99():
    """C99 cexp (cmath): torch 2.6's own CPU kernel gives exp(inf + 0j) = inf + nan j on the Gaudi image's x86 build
    and the C99 inf + 0j on an ARM Mac build (both measured), so cmath is the reference."""
    def c99(v):  # C99 Annex G cexp (cmath raises at these points instead)
        x, y = v.real, v.imag
        if y == 0:
            return complex(math.exp(x) if x < 89 else INF, y) if not math.isnan(x) else complex(NAN, y)
        if math.isinf(x) and not math.isfinite(y):
            return 0j if x < 0 else complex(INF, NAN)
        if not math.isfinite(y) or math.isnan(x):
            return complex(NAN, NAN)
        e = math.exp(x) if x < 89 else INF
        return complex(e * math.cos(y), e * math.sin(y))
    got = exp_pairs(EDGE)
    ref = torch.tensor([c99(v) for v in EDGE.tolist()], dtype=torch.complex64)
    # C99 leaves the signs of 0 and of nan's companions unspecified in these cases; values and NaN placement checked
    gp, rp = torch.view_as_real(got), torch.view_as_real(ref)
    assert torch.equal(gp.isnan(), rp.isnan()), (got, ref)
    same(got.nan_to_num(), ref.nan_to_num(), rtol=1e-5)
    # a real-valued input keeps an exactly zero (signed) imaginary part, as C99 cexp
    real = EDGE[EDGE.imag == 0]
    assert torch.equal(torch.signbit(exp_pairs(real).imag), torch.signbit(torch.exp(real).imag))


def test_exp_of_large_imaginary_parts():
    z = torch.tensor([1e4j, -3e5j, 1 + 1e6j], dtype=torch.complex64)
    same(exp_pairs(z), torch.exp(z), atol=1e-3, rtol=1e-3)


def test_neg_matches_torch_with_conjugate_inputs():
    z = cplx(3, 4)
    same(neg_pairs(z), -z, 0, 0)
    same(neg_pairs(z.conj()), -(z.conj()), 0, 0)
    same(neg_pairs(EDGE), -EDGE, 0, 0)


@pytest.mark.parametrize("value", [0, 0.0, -0.0, complex(0, -0.0), 2.5, -1j, 1 + 2j, INF, complex(NAN, 1)])
@pytest.mark.parametrize("mask_shape", [(3, 4), (4,), (1, 4), (3, 1), ()])
def test_masked_fill_broadcasts_the_mask_like_torch(value, mask_shape):
    z = cplx(3, 4)
    mask = torch.rand(mask_shape, generator=torch.Generator().manual_seed(1)) > 0.5
    got, ref = masked_fill_pairs(z, mask, value), z.clone().masked_fill_(mask, value)
    same(got, ref, 0, 0)
    assert torch.equal(torch.view_as_real(got).signbit(), torch.view_as_real(ref).signbit())  # -0.0 fills -0.0


def test_masked_fill_mask_must_broadcast_to_self():
    z = cplx(4)
    with pytest.raises(RuntimeError):
        torch.zeros(3, 4, dtype=torch.bool).expand(z.shape)  # what the kernel's expand(self.sizes()) raises
    with pytest.raises(RuntimeError):
        z.clone().masked_fill_(torch.zeros(3, 4, dtype=torch.bool), 1)  # torch raises too


@pytest.mark.parametrize("dim", [0, 1, -1, -2])
def test_cat_of_complex_and_real_like_torch(dim):
    a, b = cplx(2, 3), cplx(2, 3, seed=1)
    r = torch.randn(2, 3)
    same(cat_pairs([a, b], dim), torch.cat([a, b], dim), 0, 0)
    same(cat_pairs([a, r, b.conj()], dim), torch.cat([a, r, b.conj()], dim), 0, 0)
    assert torch.cat([a, r], dim).dtype == torch.complex64  # the kernel's result_type branch


def test_cat_skips_legacy_empty_tensors_like_torch():
    a = cplx(2, 3)
    empty = torch.empty(0, dtype=torch.complex64)
    same(cat_pairs([empty, a, empty], 0), torch.cat([empty, a, empty], 0), 0, 0)
    same(cat_pairs([cplx(0, 3), a], 0), torch.cat([cplx(0, 3), a], 0), 0, 0)


def test_cat_result_types_routed_off_the_pair_path():
    """complex128 results (a float64 or complex128 input) take the CPU kernel, as torch.cat promotes."""
    assert torch.result_type(torch.tensor([1j]), torch.tensor([1.0], dtype=torch.float64)) == torch.complex128
    assert torch.cat([cplx(2), torch.randn(2, dtype=torch.float64)]).dtype == torch.complex128


@pytest.mark.parametrize("ord", [2, 2.0, 1, 3.5, INF, -INF, 0, -1])
@pytest.mark.parametrize("dim,keepdim", [(None, False), (None, True), (-1, False), (0, True), ((0, 2), False),
                                         ((0, 2), True), ((), False)])
def test_vector_norm_matches_torch(ord, dim, keepdim):
    z = cplx(3, 4, 5)
    got = vector_norm_pairs(z, ord, dim, keepdim)
    ref = torch.linalg.vector_norm(z, ord, dim, keepdim)
    assert got.shape == ref.shape and got.dtype == ref.dtype == torch.float32
    same(got, ref, atol=1e-5, rtol=1e-5)


def test_vector_norm_0d_and_edge_values():
    for ord in (2, 1, INF):
        same(vector_norm_pairs(EDGE[6], ord), torch.linalg.vector_norm(EDGE[6], ord))
    same(vector_norm_pairs(EDGE[:8], 2, 0), torch.linalg.vector_norm(EDGE[:8], 2, 0), rtol=1e-5)
    same(vector_norm_pairs(EDGE, 2, 0), torch.linalg.vector_norm(EDGE, 2, 0))  # inf and NaN entries


def test_vector_norm_green_wavelet_normalization():
    """spd_learn's Green: wavelets / torch.linalg.norm(wavelets, dim=-1, keepdim=True)."""
    w = cplx(10, 125)
    same(vector_norm_pairs(w, 2, -1, True), torch.linalg.norm(w, dim=-1, keepdim=True), atol=1e-5, rtol=1e-6)


@pytest.mark.parametrize("dtype,expected", [(torch.float32, torch.complex64), (torch.bfloat16, torch.complex64),
                                            (torch.float64, torch.complex128), (torch.int64, torch.complex64),
                                            (torch.bool, torch.complex64), (torch.float16, torch.complex32)])
def test_complex_scalar_promotion_rule(dtype, expected):
    """x * 2j: the real HPU operand becomes torch's result type before the op (complex32 raises on the HPU)."""
    x = torch.ones(3, dtype=dtype)
    assert promoted(x, 2j) == expected == (x * 2j).dtype
    assert promoted(x, torch.tensor(2j)) == expected


@pytest.mark.parametrize("op", [
    lambda x: x * 2j, lambda x: 2j * x, lambda x: x + 1j, lambda x: torch.add(x, 1j, alpha=2), lambda x: x - 1j,
    lambda x: 1j - x, lambda x: x / 2j, lambda x: 2j / x, lambda x: torch.mul(x, torch.tensor(1 - 2j)),
    lambda x: torch.exp(2j * math.pi * x)])
def test_promote_then_compute_equals_torch(op):
    x = torch.linspace(-1, 1, 7)
    same(op(x.to(torch.complex64)), op(x), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("op", [lambda x: x.mul_(2j), lambda x: x.add_(1j), lambda x: x.sub_(1j), lambda x: x.div_(2j),
                                lambda x: torch.mul(x, 2j, out=torch.empty_like(x))])
def test_in_place_and_out_on_a_real_tensor_raise_like_torch(op):
    with pytest.raises(RuntimeError, match="can't be cast to the desired output type"):
        op(torch.ones(3))


def fast_mul(x, c):
    """real_times_complex_scalar: x * c without promoting x (float32 constants, as torch casts the scalar)."""
    cr, ci = torch.tensor(c.real, dtype=torch.float32), torch.tensor(c.imag, dtype=torch.float32)
    re = x * cr
    k_re = 0.0 * ci
    if torch.signbit(k_re):
        re = re - k_re
    return torch.complex(re, x * ci + 0.0 * cr)


def fast_add(x, c, alpha=1.0):
    cr, ci = torch.tensor(c.real, dtype=torch.float32), torch.tensor(c.imag, dtype=torch.float32)
    a = torch.tensor(alpha, dtype=torch.float32)
    k_re, k_im = a * cr - 0.0 * ci, 0.0 + (a * ci + 0.0 * cr)
    re = x + k_re
    return torch.complex(re, torch.full_like(re, k_im.item()))


X_EDGE = torch.tensor([0.0, -0.0, 1.5, -2.0, INF, -INF, NAN, 3e38, 1e-40])
SCALARS = [2j, -2j, 1 + 2j, -1 - 2j, 0j, -0.0 - 0j, complex(0, -0.0), 0.5 - 0j, 2 * math.pi * 1j]


def bits(z):
    return torch.view_as_real(z).contiguous().view(torch.int32)


@pytest.mark.parametrize("c", SCALARS)
def test_fast_mul_is_bitwise_torch(c):
    for ref in (X_EDGE * c, c * X_EDGE):
        got = fast_mul(X_EDGE, c)
        nan = torch.view_as_real(ref).isnan()
        assert torch.equal(torch.view_as_real(got).isnan(), nan), (got, ref)  # NaN exactly where torch has NaN
        assert torch.equal(bits(got)[~nan], bits(ref)[~nan]), (got, ref)


@pytest.mark.parametrize("c", SCALARS)
@pytest.mark.parametrize("alpha", [1, 2.5, -1, 0])
def test_fast_add_sub_are_bitwise_torch(c, alpha):
    for got, ref in ((fast_add(X_EDGE, c, alpha), torch.add(X_EDGE, c, alpha=alpha)),
                     (fast_add(X_EDGE, c, -alpha), torch.sub(X_EDGE, c, alpha=alpha))):
        nan = torch.view_as_real(ref).isnan()
        assert torch.equal(torch.view_as_real(got).isnan(), nan)
        assert torch.equal(bits(got)[~nan], bits(ref)[~nan]), (got, ref)

"""CPU tests on the fake HPU backend for G10 (aten::angle, aten::angle.out, aten::eq.Scalar_out) and
G11/G16 (aten::im2col, aten::im2col.out, aten::col2im, aten::col2im.out).

python -m pytest hpu_complex/tests/test_angle_im2col_cpu.py -q
"""
import math
import warnings

import pytest
import torch

from hpu_complex.tests.test_complex128_cpu import HPU, extension, tagged  # noqa: F401
from hpu_complex.tests.test_review_fixes_cpu import bits_equal

C64, C128 = torch.complex64, torch.complex128
INF, NAN = math.inf, math.nan


# ---- G10: aten::angle, aten::angle.out, aten::eq.Scalar_out ---------------------------------------------------------


@pytest.mark.parametrize("dtype", [C64, C128])
def test_angle_values_and_edges(dtype):
    vals = [1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j, 1 + 1j, -3 - 4j, 0j, complex(-0.0, 0.0),
            complex(0.0, -0.0), complex(-0.0, -0.0), complex(INF, 1), complex(-INF, -1),
            complex(1, INF), complex(NAN, 0), complex(0, NAN)]
    z = torch.tensor(vals, dtype=dtype)
    before = torch.ops.hpu_complex.cpu_fallback_calls()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = torch.angle(z.to(HPU))
    assert torch.ops.hpu_complex.cpu_fallback_calls() == before
    assert not [w for w in caught if "runs on the CPU" in str(w.message)]
    ref = torch.angle(z)
    assert got.dtype == ref.dtype
    bits_equal(got, ref)


def test_angle_conj_strided_0d_empty_and_out():
    z = torch.randn(3, 5, dtype=C64, generator=torch.Generator().manual_seed(0))
    zh = z.to(HPU)
    bits_equal(torch.angle(zh.conj().mT), torch.angle(z.conj().mT))
    bits_equal(torch.angle(zh[1, 2]), torch.angle(z[1, 2]))
    empty = torch.empty(0, 3, dtype=C64, device=HPU)
    assert torch.angle(empty).shape == (0, 3) and torch.angle(empty).dtype == torch.float32

    for out_dtype in (torch.float32, torch.float64, C64):
        out_h = torch.empty(1, dtype=out_dtype, device=HPU)
        out_c = torch.empty(1, dtype=out_dtype)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert torch.angle(zh, out=out_h) is out_h
            torch.angle(z, out=out_c)
        bits_equal(out_h, out_c)

    # Full overlap (out=self) succeeds; partial overlap and non-castable out raise
    zc = zh.clone()
    z_ref = z.clone()
    torch.angle(zc, out=zc)
    bits_equal(zc, torch.angle(z_ref, out=z_ref))
    with pytest.raises(RuntimeError, match="single memory location"):
        torch.angle(zh[1:], out=zh[:-1])
    with pytest.raises(RuntimeError, match="can't be cast"):
        torch.angle(zh, out=torch.empty(3, 5, dtype=torch.int64, device=HPU))


def test_angle_backward_zero_safe_and_inference_mode():
    z = torch.tensor([0j, 1 + 2j, -3 + 0.5j, complex(-0.0, -0.0)], dtype=C64, requires_grad=True)
    zh = z.detach().to(HPU).requires_grad_()
    before = torch.ops.hpu_complex.cpu_fallback_calls()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.angle(zh).sum().backward()
    assert torch.ops.hpu_complex.cpu_fallback_calls() == before
    assert not [w for w in caught if "runs on the CPU" in str(w.message)]
    torch.angle(z).sum().backward()
    torch.testing.assert_close(zh.grad.cpu(), z.grad, atol=1e-6, rtol=1e-6)
    assert zh.grad.cpu()[0] == 0j

    with torch.inference_mode():
        before = torch.ops.hpu_complex.cpu_fallback_calls()
        got = torch.angle(zh.detach())
        assert got.is_inference() and torch.ops.hpu_complex.cpu_fallback_calls() == before
        bits_equal(got, torch.angle(z.detach()))


@pytest.mark.parametrize("scalar", [0, 0.0, 1 + 2j, 1e40, float("inf"), float("nan")])
def test_eq_scalar_out_complex64(scalar):
    z = torch.tensor([0j, 1 + 2j, -0.0 + 0j, float("inf") + 0j, float("nan") + 0j], dtype=C64)
    out_h = torch.empty(5, dtype=torch.bool, device=HPU)
    before = torch.ops.hpu_complex.cpu_fallback_calls()
    torch.eq(z.to(HPU), scalar, out=out_h)
    assert torch.ops.hpu_complex.cpu_fallback_calls() == before
    assert torch.equal(out_h.cpu(), torch.eq(z, scalar))


# ---- G11 & G16: aten::im2col, aten::im2col.out, aten::col2im, aten::col2im.out --------------------------------------


@pytest.mark.parametrize("shape,k,d,p,s", [
    # Tier 1 fast paths: (W==1, kw==1) and (H==1, kh==1), exact & remainder crop
    ((2, 3, 6, 1), (2, 1), (1, 1), (0, 0), (2, 1)),
    ((2, 3, 7, 1), (3, 1), (1, 1), (0, 0), (3, 1)),
    ((2, 3, 1, 8), (1, 4), (1, 1), (0, 0), (1, 4)),
    ((3, 1, 7), (1, 3), (1, 1), (0, 0), (1, 3)),  # 3-D unbatched Tier 1
    # Tier 2 general paths: padding, dilation, overlap, non-square, 3-D unbatched, empty batch
    ((2, 3, 5, 6), (2, 3), (1, 1), (1, 1), (1, 2)),
    ((2, 3, 7, 7), (3, 2), (2, 1), (1, 2), (2, 1)),
    ((3, 5, 4), (2, 2), (1, 1), (0, 1), (1, 1)),
    ((0, 3, 5, 5), (2, 2), (1, 1), (0, 0), (1, 1)),
])
@pytest.mark.parametrize("dtype", [C64, C128])
def test_im2col_fwd_bwd_and_inference(shape, k, d, p, s, dtype):
    g = torch.Generator().manual_seed(42)
    x = torch.complex(torch.randn(*shape, generator=g), torch.randn(*shape, generator=g)).to(dtype)
    xr = x.clone().requires_grad_()
    xh = x.to(HPU).requires_grad_()
    before = torch.ops.hpu_complex.cpu_fallback_calls()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yh = torch.ops.aten.im2col(xh, k, d, p, s)
        yr = torch.ops.aten.im2col(xr, k, d, p, s)
        if x.numel() > 0:
            w = torch.complex(torch.randn(yr.shape, generator=g), torch.randn(yr.shape, generator=g)).to(dtype)
            (yh * w.to(HPU)).abs().sum().backward()
            (yr * w).abs().sum().backward()
    assert torch.ops.hpu_complex.cpu_fallback_calls() == before
    assert not [w for w in caught if "runs on the CPU" in str(w.message)]
    assert tagged(yh) and yh.dtype == dtype and yh.is_contiguous()
    assert not torch._C._is_alias_of(yh, xh)
    torch.testing.assert_close(yh.detach().cpu(), yr.detach(), atol=1e-5, rtol=1e-5)
    if x.numel() > 0:
        torch.testing.assert_close(xh.grad.cpu(), xr.grad, atol=1e-5, rtol=1e-5)

    with torch.inference_mode():
        before = torch.ops.hpu_complex.cpu_fallback_calls()
        yi = torch.ops.aten.im2col(xh.detach(), k, d, p, s)
        assert yi.is_inference() and torch.ops.hpu_complex.cpu_fallback_calls() == before
        torch.testing.assert_close(yi.cpu(), yr.detach(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("out_size,k,d,p,s,batched", [
    ((6, 1), (2, 1), (1, 1), (0, 0), (2, 1), True),
    ((7, 1), (3, 1), (1, 1), (0, 0), (3, 1), True),
    ((1, 9), (1, 4), (1, 1), (0, 0), (1, 4), False),
    ((5, 6), (2, 3), (1, 1), (1, 1), (1, 2), True),
    ((7, 7), (3, 2), (2, 1), (1, 2), (2, 1), False),
])
def test_col2im_fwd_bwd_conj_and_out(out_size, k, d, p, s, batched):
    oh = (out_size[0] + 2 * p[0] - d[0] * (k[0] - 1) - 1) // s[0] + 1
    ow = (out_size[1] + 2 * p[1] - d[1] * (k[1] - 1) - 1) // s[1] + 1
    c = 2
    shape = (2, c * k[0] * k[1], oh * ow) if batched else (c * k[0] * k[1], oh * ow)
    g = torch.Generator().manual_seed(7)
    x = torch.complex(torch.randn(*shape, generator=g), torch.randn(*shape, generator=g)).to(C64)
    xr = x.clone().requires_grad_()
    xh = x.to(HPU).requires_grad_()
    before = torch.ops.hpu_complex.cpu_fallback_calls()
    yh = torch.ops.aten.col2im(xh, out_size, k, d, p, s)
    yr = torch.ops.aten.col2im(xr, out_size, k, d, p, s)
    w = torch.complex(torch.randn(yr.shape, generator=g), torch.randn(yr.shape, generator=g)).to(C64)
    (yh * w.to(HPU)).abs().sum().backward()
    (yr * w).abs().sum().backward()
    assert torch.ops.hpu_complex.cpu_fallback_calls() == before
    torch.testing.assert_close(yh.detach().cpu(), yr.detach(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(xh.grad.cpu(), xr.grad, atol=1e-5, rtol=1e-5)

    # conj + strided input + .out resize / dtype / overlap errors
    xc = x.detach().conj()[..., ::1]
    torch.testing.assert_close(
        torch.ops.aten.col2im(xc.to(HPU), out_size, k, d, p, s).cpu(),
        torch.ops.aten.col2im(xc, out_size, k, d, p, s),
        atol=1e-5, rtol=1e-5,
    )
    out_h = torch.empty(0, dtype=C64, device=HPU)
    assert torch.ops.aten.col2im(xh.detach(), out_size, k, d, p, s, out=out_h) is out_h
    torch.testing.assert_close(out_h.cpu(), yr.detach(), atol=1e-5, rtol=1e-5)
    with pytest.raises(RuntimeError, match="expected scalar type|Expected out tensor to have dtype"):
        torch.ops.aten.col2im(xh.detach(), out_size, k, d, p, s, out=torch.empty(0, dtype=torch.float32, device=HPU))
    with pytest.raises(RuntimeError, match="expected scalar type|Expected out tensor to have dtype"):
        torch.ops.aten.im2col(torch.randn(1, 2, 4, 4, dtype=C64, device=HPU), (2, 2), (1, 1), (0, 0), (1, 1),
                              out=torch.empty(0, dtype=C128, device=HPU))


def test_im2col_col2im_invalid_inputs_preserve_torch_errors():
    z3 = torch.randn(2, 3, 4, dtype=C64)
    for args in [
        (torch.randn(2, 4, dtype=C64), (2, 2), (1, 1), (0, 0), (1, 1)),
        (z3, (5, 2), (1, 1), (0, 0), (1, 1)),
        (torch.randn(2, 0, 4, 4, dtype=C64), (2, 2), (1, 1), (0, 0), (1, 1)),
    ]:
        with pytest.raises((RuntimeError, IndexError)) as ref:
            torch.ops.aten.im2col(*args)
        with pytest.raises((RuntimeError, IndexError)) as got:
            torch.ops.aten.im2col(args[0].to(HPU), *args[1:])
        assert type(got.value) is type(ref.value)
        assert str(got.value).splitlines()[0] == str(ref.value).splitlines()[0]

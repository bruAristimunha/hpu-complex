"""CPU reference for the card's complex mm/bmm (cpp_ext/fft_ext.cpp ``matmul_pu1``), in pure torch.

The C++ kernel cannot be built without habana_frameworks, so this mirrors its layout step for step on float32
[..., 2] pairs: rows [re | im] per operand, the stored values of a conjugate view with the imaginary half negated,
one real GEMM [ar | ai] @ [[br, bi], [-bi, br]] = [cr | ci], and complex_view's rule for size-1 dims and empty
results. A change to the formula or the block layout must be made in both places. The kernel itself, the CPU
threshold route, copy_pu1's staging and the lazy guard are exercised only on Gaudi (pu1_check.py, lazy_check.py). Run: python -m pytest hpu_complex/tests/test_cmatmul_formula_cpu.py -q
"""
import pytest
import torch


def _raw_pair(t):
    """The stored values as float [..., 2], as raw_pair does: a conjugate view's storage, not its logical values."""
    return torch.view_as_real(t.conj() if t.is_conj() else t)


def _rows(t):
    """(..., r, c) complex -> (..., r, 2c) float = [re | im] per row, conjugation folded in."""
    p = _raw_pair(t)
    s = list(p.shape[:-1])
    c = s[-1]
    s[-1] = 2 * c
    r = p.transpose(-1, -2).contiguous().view(s)
    return torch.cat([r.narrow(-1, 0, c), r.narrow(-1, c, c).neg()], -1) if t.is_conj() else r


def _complex_view(pair):
    """complex_view's stride rule: a size-1 dim's stride addresses nothing (contiguous() keeps an odd one there), so
    it is replaced; every other stride must be even; an empty pair takes contiguous strides."""
    if pair.numel() == 0:
        return torch.zeros(pair.shape[:-1], dtype=torch.complex64)
    strides = [2 if size == 1 and st % 2 else st for size, st in zip(pair.shape[:-1], pair.stride()[:-1])]
    assert all(st % 2 == 0 for st in strides) and pair.stride(-1) == 1 and pair.storage_offset() % 2 == 0
    return torch.view_as_complex(pair.as_strided(pair.shape, strides + [1], pair.storage_offset()))


def reference_matmul(a, b):
    n = b.shape[-1]
    top = _rows(b)                                                                 # [br | bi]
    bottom = torch.cat([top.narrow(-1, n, n).neg(), top.narrow(-1, 0, n)], -1)     # [-bi | br]
    out = torch.matmul(_rows(a), torch.cat([top, bottom], -2))                     # (..., m, 2n) = [cr | ci]
    os = list(out.shape)
    os[-1] = n
    os.insert(len(os) - 1, 2)                                                      # (..., m, 2, n)
    return _complex_view(out.view(os).transpose(-1, -2).contiguous())


def _c(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(*shape, generator=g), torch.randn(*shape, generator=g))


SHAPES = [((7, 4), (4, 3)), ((5, 7, 4), (5, 4, 3)), ((1, 1), (1, 1)), ((3, 1, 4), (3, 4, 1)),
          ((2, 5, 1), (2, 1, 6)), ((1, 22, 22), (1, 22, 22)), ((0, 3, 4), (0, 4, 2)), ((3, 0), (0, 2))]


@pytest.mark.parametrize("sa,sb", SHAPES)
@pytest.mark.parametrize("conj_a,conj_b", [(False, False), (True, False), (False, True), (True, True)])
def test_real_gemm_matches_complex_matmul(sa, sb, conj_a, conj_b):
    a, b = _c(*sa, seed=1), _c(*sb, seed=2)
    a = a.conj() if conj_a else a
    b = b.conj() if conj_b else b
    assert a.is_conj() == conj_a and b.is_conj() == conj_b  # the views reach the kernel unresolved
    got, want = reference_matmul(a, b), torch.matmul(a, b)
    assert got.shape == want.shape
    if want.numel():
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_strided_and_hermitian_operands():
    x = _c(4, 6, 9, seed=3)
    xh = x.mH  # a transposed conjugate view: strided and is_conj()
    torch.testing.assert_close(reference_matmul(x, xh), x @ xh, rtol=1e-5, atol=1e-5)
    s = _c(4, 12, 9, seed=4)[:, ::2, 1:]  # strided, offset
    torch.testing.assert_close(reference_matmul(s, xh[:, 1:, :]), s @ xh[:, 1:, :], rtol=1e-5, atol=1e-5)


def test_block_layout_is_the_complex_product():
    """[ar | ai] @ [[br, bi], [-bi, br]]: one entry by hand, so a swapped block or sign is caught explicitly."""
    a = torch.tensor([[1 + 2j]], dtype=torch.complex64)
    b = torch.tensor([[3 - 1j]], dtype=torch.complex64)
    assert reference_matmul(a, b).item() == pytest.approx((1 + 2j) * (3 - 1j))
    assert reference_matmul(a.conj(), b).item() == pytest.approx((1 - 2j) * (3 - 1j))

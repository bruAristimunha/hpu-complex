"""Fake-backend CPU tests of the real C++ complex mm/bmm kernel (cpp_ext/fft_ext.cpp ``matmul_pu1``).

Runs ``torch.matmul`` on ``complex64`` HPU tensors with ``set_matmul_cpu_threshold(0)`` so the card GEMM path
([ar | ai] @ [[br, bi], [-bi, br]] = [cr | ci]) executes directly against stock PyTorch.

python -m pytest hpu_complex/tests/test_cmatmul_cpu.py -q
"""
import pytest
import torch

from hpu_complex.tests.test_complex128_cpu import HPU, extension  # noqa: F401


@pytest.fixture(autouse=True)
def _force_card_matmul(extension):
    torch.ops.hpu_complex.set_matmul_cpu_threshold(0)
    try:
        yield
    finally:
        torch.ops.hpu_complex.set_matmul_cpu_threshold(1 << 22)


def _c(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(*shape, generator=g), torch.randn(*shape, generator=g))


SHAPES = [((7, 4), (4, 3)), ((5, 7, 4), (5, 4, 3)), ((1, 1), (1, 1)), ((3, 1, 4), (3, 4, 1)),
          ((2, 5, 1), (2, 1, 6)), ((1, 22, 22), (1, 22, 22)), ((0, 3, 4), (0, 4, 2)), ((3, 0), (0, 2))]


@pytest.mark.parametrize("sa,sb", SHAPES)
@pytest.mark.parametrize("conj_a,conj_b", [(False, False), (True, False), (False, True), (True, True)])
def test_real_gemm_matches_complex_matmul(sa, sb, conj_a, conj_b):
    a_cpu, b_cpu = _c(*sa, seed=1), _c(*sb, seed=2)
    a_ref = a_cpu.conj() if conj_a else a_cpu
    b_ref = b_cpu.conj() if conj_b else b_cpu
    a_hpu = a_cpu.to(HPU).conj() if conj_a else a_cpu.to(HPU)
    b_hpu = b_cpu.to(HPU).conj() if conj_b else b_cpu.to(HPU)
    assert a_hpu.is_conj() == conj_a and b_hpu.is_conj() == conj_b
    got = torch.matmul(a_hpu, b_hpu).cpu()
    want = torch.matmul(a_ref, b_ref)
    assert got.shape == want.shape and got.dtype == want.dtype
    if want.numel():
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_strided_and_hermitian_operands():
    x = _c(4, 6, 9, seed=3)
    xh = x.mH
    torch.testing.assert_close(torch.matmul(x.to(HPU), x.to(HPU).mH).cpu(), x @ xh, rtol=1e-5, atol=1e-5)
    s = _c(4, 12, 9, seed=4)[:, ::2, 1:]
    s_hpu = _c(4, 12, 9, seed=4).to(HPU)[:, ::2, 1:]
    xh_hpu = x.to(HPU).mH[:, 1:, :]
    torch.testing.assert_close(torch.matmul(s_hpu, xh_hpu).cpu(), s @ xh[:, 1:, :], rtol=1e-5, atol=1e-5)


def test_block_layout_is_the_complex_product():
    """[ar | ai] @ [[br, bi], [-bi, br]]: one 1x1 entry by hand, so a swapped block or sign is caught explicitly."""
    a = torch.tensor([[1 + 2j]], dtype=torch.complex64)
    b = torch.tensor([[3 - 1j]], dtype=torch.complex64)
    assert torch.matmul(a.to(HPU), b.to(HPU)).cpu().item() == pytest.approx((1 + 2j) * (3 - 1j))
    assert torch.matmul(a.to(HPU).conj(), b.to(HPU)).cpu().item() == pytest.approx((1 - 2j) * (3 - 1j))

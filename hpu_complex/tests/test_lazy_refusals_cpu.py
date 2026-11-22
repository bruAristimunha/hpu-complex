"""CPU stand-in tests for lifted lazy refusals: rank >= 5 permuted complex64 and inference_mode FFTs.

python -m pytest hpu_complex/tests/test_lazy_refusals_cpu.py -q
"""
import gc

import pytest
import torch

from hpu_complex.tests.test_complex128_cpu import HPU, extension, tagged  # noqa: F401
from hpu_complex.tests.test_review2_cpu import lazy_flag
from hpu_complex.tests.test_review_fixes_cpu import bits_equal

C64 = torch.complex64


@pytest.mark.parametrize("shape,perm", [
    ((2, 3, 4, 5, 6), (0, 2, 1, 3, 4)),
    ((2, 3, 4, 5, 6), (4, 3, 2, 1, 0)),
    ((2, 3, 4, 5, 6), (0, 4, 3, 1, 2)),  # MetaNeuromotorHand pattern
    ((2, 1, 3, 4, 5), (4, 1, 2, 0, 3)),  # size-1 dim with arbitrary stride
    ((2, 2, 3, 2, 3, 4), (5, 4, 3, 2, 1, 0)),  # rank-6 complex / rank-7 float pair
    ((2, 2, 2, 2, 2, 2, 2), (6, 4, 2, 0, 5, 3, 1)),  # rank-7 complex / rank-8 float pair
])
@pytest.mark.parametrize("offset", [0, 3])
def test_rank5plus_permuted_ops_lazy_and_eager(shape, perm, offset):
    torch.manual_seed(0)
    total = int(torch.tensor(shape).prod().item()) + offset
    base_c = torch.randn(total, dtype=C64)
    for use_lazy in (False, True):
        ctx = lazy_flag() if use_lazy else torch.enable_grad()
        with ctx:
            base_h = base_c.to(HPU)
            zc = base_c[offset:].view(shape).permute(perm)
            zh = base_h[offset:].view(shape).permute(perm)
            assert tagged(zh)
            bits_equal(zh * 2, zc * 2)
            bits_equal(zh.reshape(-1, zc.shape[-1]), zc.reshape(-1, zc.shape[-1]))
            bits_equal(zh.clone(), zc.clone())
            bits_equal(zh.clone(memory_format=torch.preserve_format),
                       zc.clone(memory_format=torch.preserve_format))
            torch.testing.assert_close(zh.abs().cpu(), zc.abs(), rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(
                torch.view_as_real(zh.sum(dim=(0, 2)).cpu()),
                torch.view_as_real(zc.sum(dim=(0, 2))),
                rtol=1e-5, atol=1e-5,
            )
            bits_equal(torch.view_as_real(zh), torch.view_as_real(zc))
            bits_equal(zh.real, zc.real)
            if perm[-1] == len(shape) - 1:
                bits_equal(zh.view(torch.float32), zc.view(torch.float32))
            else:
                with pytest.raises(RuntimeError, match=r"stride\(-1\) must be 1"):
                    zh.view(torch.float32)


def test_rank5plus_permuted_empty_lazy():
    with lazy_flag():
        z = torch.empty(2, 0, 3, 4, 5, dtype=C64, device=HPU).permute(4, 3, 2, 1, 0)
        out = z * 2
        assert out.shape == (5, 4, 3, 0, 2) and out.numel() == 0
        z.fill_(1 + 2j)
        z.zero_()
        z.copy_(torch.empty(5, 4, 3, 0, 2, dtype=C64, device=HPU))


def test_rank5plus_permuted_inplace_copy_fill_zero_lazy():
    torch.manual_seed(1)
    base_c = torch.randn(2, 3, 4, 5, 6, dtype=C64)
    with lazy_flag():
        base_h = base_c.to(HPU)
        pc = base_c.permute(4, 3, 2, 1, 0)
        ph = base_h.permute(4, 3, 2, 1, 0)

        ph.fill_(3 - 4j)
        pc.fill_(3 - 4j)
        bits_equal(base_h, base_c)

        ph.zero_()
        pc.zero_()
        bits_equal(base_h, base_c)

        src = torch.randn(6, 5, 4, 3, 2, dtype=C64)
        ph.copy_(src.conj())
        pc.copy_(src.conj())
        bits_equal(base_h, base_c)

        ph.copy_(src.to(HPU).conj())
        pc.copy_(src.conj())
        bits_equal(base_h, base_c)

        # Broadcast copy into dense rank-5 permuted dst
        bcast = torch.randn(3, 2, dtype=C64)
        ph.copy_(bcast.to(HPU))
        pc.copy_(bcast)
        bits_equal(base_h, base_c)

        # Real -> complex copy into dense rank-5 permuted dst
        re_src = torch.randn(6, 5, 4, 3, 2)
        ph.copy_(re_src.to(HPU))
        pc.copy_(re_src)
        bits_equal(base_h, base_c)

        # Complex -> real copy from rank-5 permuted src
        real_dst_h = torch.empty(6, 5, 4, 3, 2, device=HPU)
        real_dst_c = torch.empty(6, 5, 4, 3, 2)
        real_dst_h.copy_(ph)
        real_dst_c.copy_(pc)
        bits_equal(real_dst_h, real_dst_c)

        # In-place write through .real
        ph.real.mul_(2.5)
        pc.real.mul_(2.5)
        bits_equal(base_h, base_c)


def test_rank5_permuted_matmul_fft_sgn_where_lazy():
    torch.manual_seed(2)
    a_c = torch.randn(2, 2, 2, 4, 5, dtype=C64)
    b_c = torch.randn(2, 2, 2, 5, 3, dtype=C64)
    with lazy_flag():
        torch.ops.hpu_complex.set_matmul_cpu_threshold(0)
        try:
            ah = a_c.to(HPU).permute(2, 1, 0, 3, 4)
            bh = b_c.to(HPU).permute(2, 1, 0, 3, 4)
            ac = a_c.permute(2, 1, 0, 3, 4)
            bc = b_c.permute(2, 1, 0, 3, 4)
            torch.testing.assert_close(
                torch.view_as_real((ah @ bh).cpu()), torch.view_as_real(ac @ bc), rtol=1e-5, atol=1e-5
            )
            torch.testing.assert_close(
                torch.view_as_real((ah @ bh.mH.mT).cpu()),
                torch.view_as_real(ac @ bc.mH.mT),
                rtol=1e-5, atol=1e-5,
            )
            # Empty rank-5 permuted card matmul
            e_a = torch.empty(2, 2, 2, 0, 5, dtype=C64, device=HPU).permute(2, 1, 0, 3, 4)
            e_b = torch.empty(2, 2, 2, 5, 3, dtype=C64, device=HPU).permute(2, 1, 0, 3, 4)
            assert (e_a @ e_b).shape == (2, 2, 2, 0, 3)
        finally:
            torch.ops.hpu_complex.set_matmul_cpu_threshold(1 << 22)

        torch.testing.assert_close(
            torch.view_as_real(torch.fft.fft(ah).cpu()),
            torch.view_as_real(torch.fft.fft(ac)),
            rtol=1e-4, atol=1e-4,
        )
        torch.testing.assert_close(
            torch.view_as_real(ah.sgn().cpu()),
            torch.view_as_real(ac.sgn()),
            rtol=1e-5, atol=1e-5,
        )
        cond = ac.real > 0
        bits_equal(torch.where(cond.to(HPU), ah, 0j), torch.where(cond, ac, 0j))


def test_rank5_permuted_backward_lazy():
    torch.manual_seed(2)
    a_c = torch.randn(2, 2, 2, 4, 5, dtype=C64)
    with lazy_flag():
        xc = a_c.clone().requires_grad_()
        xh = a_c.to(HPU).detach().requires_grad_()
        yc = xc.permute(0, 4, 3, 1, 2).reshape(-1, 2, 2).abs().sum()
        yh = xh.permute(0, 4, 3, 1, 2).reshape(-1, 2, 2).abs().sum()
        yc.backward()
        yh.backward()
        torch.testing.assert_close(
            torch.view_as_real(xh.grad.cpu()), torch.view_as_real(xc.grad), rtol=1e-5, atol=1e-5
        )


def test_inference_mode_rfft_stft_fft_irfft_and_perm5_lazy():
    torch.manual_seed(3)
    sig = torch.randn(4, 64)
    z5 = torch.randn(2, 2, 3, 2, 16, dtype=C64)
    win = torch.hann_window(32)
    with lazy_flag():
        gc.collect()
        alive0 = torch.ops.hpu_complex.lazy_storages_alive()
        with torch.inference_mode():
            sig_h = sig.to(HPU)
            rfft_h = torch.fft.rfft(sig_h)
            irfft_h = torch.fft.irfft(rfft_h, n=64)
            stft_h = torch.stft(sig_h, 32, hop_length=16, window=win.to(HPU), return_complex=True)
            p5_h = z5.to(HPU).permute(0, 4, 3, 1, 2)
            fft_h = torch.fft.fft(p5_h)
            mul_h = p5_h * 2 + (sig_h[:2, :16] * 2j).sum()
            assert all(t.is_inference() for t in (rfft_h, irfft_h, stft_h, fft_h, mul_h))
        torch.testing.assert_close(
            torch.view_as_real(rfft_h.cpu()), torch.view_as_real(torch.fft.rfft(sig)), rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(irfft_h.cpu(), sig, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(
            torch.view_as_real(stft_h.cpu()),
            torch.view_as_real(torch.stft(sig, 32, hop_length=16, window=win, return_complex=True)),
            rtol=1e-4, atol=1e-4,
        )
        p5_c = z5.permute(0, 4, 3, 1, 2)
        torch.testing.assert_close(
            torch.view_as_real(fft_h.cpu()), torch.view_as_real(torch.fft.fft(p5_c)), rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(
            torch.view_as_real(mul_h.cpu()),
            torch.view_as_real(p5_c * 2 + (sig[:2, :16] * 2j).sum()),
            rtol=1e-4, atol=1e-4,
        )
        del sig_h, rfft_h, irfft_h, stft_h, p5_h, fft_h, mul_h
        gc.collect()
        assert torch.ops.hpu_complex.lazy_storages_alive() == alive0

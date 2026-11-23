"""Fake-backend CPU tests of the C++ DFT/FFT kernels (fft_r2c_hpu, fft_c2c_pu1, fft_c2r_pu1, FftR2C backward).

python -m pytest hpu_complex/tests/test_fft_cpu.py -q
"""
import pytest
import torch

from hpu_complex.tests.test_complex128_cpu import HPU, extension, tagged  # noqa: F401


def _r(*shape, seed=0):
    return torch.randn(shape, dtype=torch.float32, generator=torch.Generator().manual_seed(seed))


def _c(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(shape, generator=g), torch.randn(shape, generator=g))


@pytest.mark.parametrize("norm", ["backward", "ortho", "forward"])
@pytest.mark.parametrize("n", [7, 8])
def test_rfft_forward_and_backward_norms(norm, n):
    x_cpu = _r(3, 5, n, seed=1)
    x_ref = x_cpu.clone().requires_grad_()
    x_hpu = x_cpu.to(HPU).requires_grad_()
    y_ref = torch.fft.rfft(x_ref, norm=norm)
    y_hpu = torch.fft.rfft(x_hpu, norm=norm)
    assert tagged(y_hpu) and y_hpu.dtype == torch.complex64 and y_hpu.shape == y_ref.shape
    torch.testing.assert_close(y_hpu.detach().cpu(), y_ref.detach(), rtol=2e-5, atol=2e-5)

    w = _c(*y_ref.shape, seed=2)
    (y_ref * w).abs().pow(2).sum().backward()
    (y_hpu * w.to(HPU)).abs().pow(2).sum().backward()
    torch.testing.assert_close(x_hpu.grad.cpu(), x_ref.grad, rtol=3e-5, atol=3e-5)


def test_rfft_conj_and_strided_grad():
    base = _r(4, 16, seed=3)
    x_ref = base[:, ::2].clone().requires_grad_()
    x_hpu = base.to(HPU)[:, ::2].detach().requires_grad_()
    w = _c(4, 5, seed=4)
    (torch.fft.rfft(x_ref).conj() * w).real.sum().backward()
    (torch.fft.rfft(x_hpu).conj() * w.to(HPU)).real.sum().backward()
    torch.testing.assert_close(x_hpu.grad.cpu(), x_ref.grad, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("norm", ["backward", "ortho", "forward"])
@pytest.mark.parametrize("n", [7, 8])
def test_irfft_matches_cpu(norm, n):
    x = _r(2, 4, n, seed=5)
    spec = torch.fft.rfft(x, norm=norm)
    ref = torch.fft.irfft(spec, n=n, norm=norm)
    got = torch.fft.irfft(spec.to(HPU), n=n, norm=norm).cpu()
    torch.testing.assert_close(got, ref, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("norm", ["backward", "ortho", "forward"])
@pytest.mark.parametrize("n", [7, 8])
def test_fft_and_ifft_match_cpu_with_conj_and_grad(norm, n):
    z = _c(2, 3, n, seed=6)
    for fn in (lambda t: torch.fft.fft(t, norm=norm), lambda t: torch.fft.ifft(t, norm=norm)):
        z_ref = z.clone().requires_grad_()
        z_hpu = z.to(HPU).requires_grad_()
        y_ref = fn(z_ref)
        y_hpu = fn(z_hpu)
        assert tagged(y_hpu)
        torch.testing.assert_close(y_hpu.detach().cpu(), y_ref.detach(), rtol=2e-5, atol=2e-5)
        (y_ref.abs() ** 2).sum().backward()
        (y_hpu.abs() ** 2).sum().backward()
        torch.testing.assert_close(z_hpu.grad.cpu(), z_ref.grad, rtol=3e-5, atol=3e-5)
    torch.testing.assert_close(
        torch.fft.fft(z.to(HPU).conj(), norm=norm).cpu(),
        torch.fft.fft(z.conj(), norm=norm),
        rtol=2e-5,
        atol=2e-5,
    )


def test_stft_forward_and_backward():
    x_cpu = _r(2, 32, seed=7)
    win = torch.hann_window(8)
    x_ref = x_cpu.clone().requires_grad_()
    x_hpu = x_cpu.to(HPU).requires_grad_()
    y_ref = torch.stft(x_ref, n_fft=8, hop_length=4, window=win, return_complex=True)
    y_hpu = torch.stft(x_hpu, n_fft=8, hop_length=4, window=win.to(HPU), return_complex=True)
    assert tagged(y_hpu)
    torch.testing.assert_close(y_hpu.detach().cpu(), y_ref.detach(), rtol=2e-5, atol=2e-5)
    y_ref.abs().sum().backward()
    y_hpu.abs().sum().backward()
    torch.testing.assert_close(x_hpu.grad.cpu(), x_ref.grad, rtol=3e-5, atol=3e-5)

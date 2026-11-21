"""Review round 2 (adversarial) of complex128 HPU tensors: CPU cases that break against torch.

Same CPU stand-in as test_complex128_cpu.py (the extension + a no-op HPU guard). Each break was an xfail(strict=True)
until the fix round removed it with its fix: unfold (registered at PrivateUse1 as ATen's view), conv1d (aten::convolution
and convolution_backward take host_fallback), set_(source) (ATen's set_tensor_), inference_mode (host tensors made
without autograd keys) and the gradient through z.real[1:] (slice_backward and diagonal_backward take host_fallback).
The cases that need Habana tensors (bool masks, float32 HPU operands, lazy mode) are in tests/probe_adversarial_j.py.

python -m pytest hpu_complex/tests/test_adversarial_c128_cpu.py -q
"""
import copy

import pytest
import torch

from hpu_complex.tests.test_complex128_cpu import HPU, assert_hpu_exact, c128, extension  # noqa: F401


def test_unfold_is_a_view():
    x = c128(3, 8)
    z = x.to(HPU)
    u = z.unfold(1, 3, 2)
    assert_hpu_exact(u, x.unfold(1, 3, 2))
    u.mul_(2)  # a view: the write reaches z
    ref = x.clone()
    ref.unfold(1, 3, 2).mul_(2)
    assert_hpu_exact(z, ref)


def test_conv1d_complex128():
    x, w = c128(2, 1, 16), c128(3, 1, 5, seed=1)
    assert_hpu_exact(torch.nn.functional.conv1d(x.to(HPU), w.to(HPU)), torch.nn.functional.conv1d(x, w), atol=1e-12)


def test_set_source_tensor_shares_storage():
    a = c128(3, 4).to(HPU)
    src = c128(2, 2, seed=5)
    b = src.to(HPU)
    a.set_(b)
    a.zero_()
    assert_hpu_exact(b, torch.zeros_like(src))


def test_inference_mode_tensor():
    with torch.inference_mode():
        z = torch.zeros(3, dtype=torch.complex128, device=HPU)
        ref = torch.zeros(3, dtype=torch.complex128)
    assert z.is_inference() == ref.is_inference()


def test_views_that_hold():  # composite views, structured in-place ops on views, overlapping in-place (not breaks)
    x = c128(3, 4)
    z = x.to(HPU)
    for got, ref in zip((*z.unbind(0), *z.split(2, 1), z.narrow(1, 1, 2), z.select(0, 2), z.mT),
                        (*x.unbind(0), *x.split(2, 1), x.narrow(1, 1, 2), x.select(0, 2), x.mT)):
        assert_hpu_exact(got, ref)
    ref = x.clone()
    z[1:, ::2].exp_()
    z.real.clamp_(-0.5, 0.5)
    z.addcmul_(z, z)
    ref[1:, ::2].exp_()
    ref.real.clamp_(-0.5, 0.5)
    ref.addcmul_(ref, ref)
    assert_hpu_exact(z, ref)
    with pytest.raises(RuntimeError, match="more than one element"):
        c128(1, 4).to(HPU).expand(3, 4).exp_()
    from hpu_complex import _deepcopy_complex128_in_host_memory

    _deepcopy_complex128_in_host_memory()  # enable() installs it; it needs torch.hpu, absent here
    s = copy.deepcopy([z.conj(), z, z[1:].real])
    s[1].fill_(1 + 1j)
    assert_hpu_exact(s[2], torch.ones(2, 4, dtype=torch.float64))
    assert_hpu_exact(s[0], ref.conj().resolve_conj())


@pytest.mark.parametrize("shape", [(0, 3, 3), (1, 1), (0, 0), (2, 0, 0)])
def test_eig_odd_shapes(shape):
    x = c128(*shape)
    for got, want in zip(torch.linalg.eig(x.to(HPU)), torch.linalg.eig(x)):
        assert_hpu_exact(got, want)


def test_eig_views_and_singular():
    x = c128(3, 5, 5)
    assert_hpu_exact(torch.linalg.eigvals(x.to(HPU)[1:, 1:, 1:].mT), torch.linalg.eigvals(x[1:, 1:, 1:].mT))
    ones = torch.ones(4, 4, dtype=torch.complex128)
    assert_hpu_exact(torch.linalg.eigvals(ones.to(HPU)), torch.linalg.eigvals(ones))


def test_subnormal_and_huge_values_stay_exact():
    x = torch.tensor([1e-310 + 1e-310j, 1e300 + 1e300j, 1e-200 - 3j], dtype=torch.complex128)
    z = x.to(HPU)
    assert_hpu_exact(z * z, x * x)
    assert_hpu_exact(z.abs(), x.abs())
    assert_hpu_exact(z / 1e300, x / 1e300)


@pytest.mark.parametrize("loss", [
    lambda z: z.real[1:].pow(2).sum(),
    lambda z: (z.real[:, 1] * 3).sum(),
    lambda z: (z.imag[1:] * (1 + 1e-12)).sum(),
])
def test_gradient_through_a_slice_of_a_float64_view(loss):
    x = c128(3, 4)
    ref = x.clone().requires_grad_()
    loss(ref).backward()
    z = x.to(HPU).requires_grad_()
    loss(z).backward()
    assert_hpu_exact(z.grad, ref.grad)


def test_conv1d_complex128_backward_and_float64_signal():
    """spd_learn Green's wavelet layer: conv1d of a complex128 filter bank, gradient to the filters and the signal."""
    x, w = c128(2, 1, 16), c128(3, 1, 5, seed=1)
    grads = []
    for dev in ("cpu", HPU):
        xs, ws = x.clone().to(dev).requires_grad_(), w.clone().to(dev).requires_grad_()
        torch.nn.functional.conv1d(xs, ws, padding=2).abs().pow(2).sum().backward()
        grads.append((xs.grad, ws.grad))
    assert_hpu_exact(grads[1][0], grads[0][0], atol=1e-12)
    assert_hpu_exact(grads[1][1], grads[0][1], atol=1e-12)
    r = x.real.contiguous()  # a float64 signal in host memory (the real part of a complex128 tensor)
    assert_hpu_exact(torch.nn.functional.conv1d(x.to(HPU).real, w.to(HPU).real),
                     torch.nn.functional.conv1d(r, w.real.contiguous()), atol=1e-12)


def test_inference_mode_tensor_computes():
    x = c128(3, 4)
    with torch.inference_mode():
        z = x.to(HPU)
        y = (z * z).exp().abs()
        assert z.is_inference() and y.is_inference()
    assert_hpu_exact(y, (x * x).exp().abs())
    with pytest.raises(RuntimeError, match="[Ii]nference"):
        z.add_(1)  # torch: an inference tensor is not updated in place outside inference mode

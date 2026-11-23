"""FFT on HPU without complex dtype: real tensors in, [..., freq, 2] out.


Why: without hpu_complex.enable(), gaudi-pytorch-bridge rejects complex64
(tensor_utils.cpp is_supported_type) and has no FFT kernel, so torch.fft.*
crashes on HPU. This module never creates a complex tensor. It uses only
float reshape/mul/add/matmul, which HPU already supports, in eager mode and
inside torch.compile graphs. Differentiable.

Convention: a complex spectrum is a real tensor whose last dim is [real, imag].
Power-of-two lengths use an O(n log n) radix-2 butterfly; other lengths use a
dense DFT matmul (still on-device, still beats the CPU round trip).
"""
import math

import torch


def _is_pow2(n):
    return n > 0 and (n & (n - 1)) == 0


def _bit_reverse(x, n):
    bits = n.bit_length() - 1
    idx = torch.arange(n, device=x.device)
    rev = torch.zeros_like(idx)
    for b in range(bits):
        rev = rev | ((idx >> b) & 1) << (bits - 1 - b)
    return x.index_select(-1, rev)


def _cmul(ar, ai, br, bi):
    return ar * br - ai * bi, ar * bi + ai * br


def _fft_pow2(re, im):
    """Iterative radix-2 FFT along last dim. re/im: [..., n], n a power of 2."""
    n = re.shape[-1]
    re, im = _bit_reverse(re, n), _bit_reverse(im, n)
    length = 2
    while length <= n:
        half = length // 2
        k = torch.arange(half, device=re.device, dtype=re.dtype)
        ang = -2.0 * math.pi * k / length
        wr, wi = torch.cos(ang), torch.sin(ang)
        shape = list(re.shape[:-1]) + [n // length, length]
        re = re.view(shape)
        im = im.view(shape)
        x1r, x2r = re[..., :half], re[..., half:]
        x1i, x2i = im[..., :half], im[..., half:]
        tr, ti = _cmul(x2r, x2i, wr, wi)
        re = torch.cat([x1r + tr, x1r - tr], dim=-1)
        im = torch.cat([x1i + ti, x1i - ti], dim=-1)
        re, im = re.reshape(shape[:-2] + [n]), im.reshape(shape[:-2] + [n])
        length *= 2
    return re, im


def _dft_dense(re, im):
    """Dense DFT along last dim for any length, via two real matmuls."""
    n = re.shape[-1]
    k = torch.arange(n, device=re.device, dtype=re.dtype)
    ang = -(2.0 * math.pi / n) * torch.outer(k, k)
    c, s = torch.cos(ang), torch.sin(ang)
    out_re = re @ c.T - im @ s.T
    out_im = re @ s.T + im @ c.T
    return out_re, out_im


def _fft(re, im, inverse=False):
    if inverse:  # ifft(X) = conj(fft(conj(X))) / n
        re, im = _fft(re, -im)
        return re / re.shape[-1], -im / re.shape[-1]
    if _is_pow2(re.shape[-1]):
        return _fft_pow2(re, im)
    return _dft_dense(re, im)


def rfft(x, dim=-1):
    """Real input [..., n] -> spectrum [..., n//2+1, 2] (last dim [re, im])."""
    d = dim % x.dim()
    x = x.movedim(d, -1)
    n = x.shape[-1]
    re, im = _fft(x, torch.zeros_like(x))
    out = torch.stack([re[..., : n // 2 + 1], im[..., : n // 2 + 1]], dim=-1)
    return out.movedim(-2, d)


def irfft(spec, n=None, dim=-1):
    """Spectrum [..., k, 2] -> real signal [..., n]. Inverse of rfft."""
    d = dim % (spec.dim() - 1)
    spec = spec.movedim(d, -2)
    if n is None:
        n = 2 * (spec.shape[-2] - 1)
    re, im = spec[..., 0], spec[..., 1]
    # Rebuild the full Hermitian spectrum, then inverse FFT.
    tail_re = re[..., 1 : n - re.shape[-1] + 1].flip(-1)
    tail_im = -im[..., 1 : n - im.shape[-1] + 1].flip(-1)
    full_re = torch.cat([re, tail_re], dim=-1)
    full_im = torch.cat([im, tail_im], dim=-1)
    out, _ = _fft(full_re, full_im, inverse=True)
    return out.movedim(-1, d)


def fft(spec, dim=-1):
    """Complex input/output as [..., n, 2]. Input signal if spec is [re, im]."""
    d = dim % (spec.dim() - 1)
    spec = spec.movedim(d, -2)
    re, im = _fft(spec[..., 0], spec[..., 1])
    out = torch.stack([re, im], dim=-1)
    return out.movedim(-2, d)


def magnitude(spec):
    """|X| from a [..., 2] spectrum. Adds 1e-12 under the sqrt: finite gradient at 0, magnitude floor ~1e-6."""
    # Adds 1e-12 under the sqrt so abs() gradients stay finite at 0; exact-zero bins have ~1e-6 floor.
    return torch.sqrt(spec[..., 0] ** 2 + spec[..., 1] ** 2 + 1e-12)


def stft(x, n_fft, hop, window=None, center=False):
    """1-D signal [batch, time] -> spectrum [batch, freq, frames, 2].

    center=False keeps framing explicit (no reflect pad, which HPU graphs
    handle inconsistently). window defaults to a Hann window.
    """
    if window is None:
        window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    frames = x.unfold(-1, n_fft, hop) * window  # [batch, frames, n_fft]
    return rfft(frames, dim=-1).permute(0, 2, 1, 3)  # [batch, freq, frames, 2]


def stft_magnitude(x, n_fft, hop, window=None):
    """|STFT|: drop-in for torchaudio Spectrogram(power=None) front ends."""
    return magnitude(stft(x, n_fft, hop, window=window))

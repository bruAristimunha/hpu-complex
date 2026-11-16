"""Numerics + autograd vs torch.fft (CPU reference). Run: python test_hpu_fft.py"""
import torch

import hpu_fft as H


def as_complex(spec):
    return torch.complex(spec[..., 0], spec[..., 1])


def main():
    torch.manual_seed(0)
    ok = True
    for n in (200, 256, 960, 1024, 1920):  # issue's lengths: dense + pow2 paths
        x = torch.randn(4, n, dtype=torch.float64)
        ref = torch.fft.rfft(x)
        got = as_complex(H.rfft(x))
        err = (got - ref).abs().max().item()
        back = H.irfft(H.rfft(x.double()), n=n)
        rerr = (back - x).abs().max().item()
        print(f"n={n:5d} rfft_err={err:.2e} roundtrip_err={rerr:.2e}")
        ok &= err < 1e-4 and rerr < 1e-9

    x = torch.randn(3, 200, dtype=torch.float32, requires_grad=True)
    ref_mag = torch.fft.rfft(x.detach().double()).abs()
    got_mag = H.magnitude(H.rfft(x.double())).float()
    merr = (got_mag - ref_mag.float()).abs().max().item()
    got_mag.sum().backward()
    print(f"stft-front-end magnitude_err={merr:.2e} grad_finite={torch.isfinite(x.grad).all().item()}")
    ok &= merr < 1e-4 and torch.isfinite(x.grad).all().item()

    sig = torch.randn(2, 4000, dtype=torch.float64)
    ref_stft = torch.stft(sig, 200, 100, window=torch.hann_window(200, dtype=torch.float64),
                          return_complex=True, center=False)
    got_stft = as_complex(H.stft(sig, 200, 100))
    serr = (got_stft - ref_stft).abs().max().item()
    print(f"stft_err={serr:.2e} shape ref={tuple(ref_stft.shape)} got={tuple(got_stft.shape)}")
    ok &= serr < 1e-4
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

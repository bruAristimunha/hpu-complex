"""CPU validation of the aten decompositions vs torch.ops.aten (reference).
Run: python -m pytest hpu_complex/tests/test_decompositions_cpu.py -q
"""
import torch

from hpu_complex.decompositions import _fft_c2c, _fft_c2r, _fft_r2c


def test_r2c_all_norms():
    torch.manual_seed(0)
    x = torch.randn(3, 200, dtype=torch.float64)
    for norm in (0, 1, 2):
        ref = torch.ops.aten._fft_r2c(x, [1], norm, True)
        got = _fft_r2c(x, [-1], norm, True)
        assert (got - ref).abs().max() < 1e-9, norm


def test_c2r_roundtrip_and_c2c():
    torch.manual_seed(0)
    x = torch.randn(3, 256, dtype=torch.float64)
    spec = torch.ops.aten._fft_r2c(x, [1], 0, True)
    back = _fft_c2r(spec, [-1], 2, 256)  # by_n inverse of unnormalized fwd
    assert (back - x).abs().max() < 1e-9
    z = torch.randn(3, 128, dtype=torch.complex128)
    ref = torch.ops.aten._fft_c2c(z, [1], 0, True)
    got = _fft_c2c(z, [1], 0, True)
    assert (got - ref).abs().max() < 1e-9
    ref_i = torch.ops.aten._fft_c2c(z, [1], 2, False)
    got_i = _fft_c2c(z, [1], 2, False)
    assert (got_i - ref_i).abs().max() < 1e-9

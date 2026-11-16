"""Decompositions of aten::_fft_r2c/_fft_c2r/_fft_c2c into float-only ops.

These are the only three primitives PyTorch needs from a backend; torch.fft.*
and torch.stft compose on top. Each decomposition converts complex<->pair
at the boundary (view_as_complex/view_as_real) and runs the float core, so
no complex tensor ever enters a Synapse graph.

aten normalization int (fft_norm_mode): 0=none, 1=by_root_n, 2=by_n, already
resolved per-direction by the torch.fft wrappers. Forward core is
unnormalized, inverse core divides by n; _rescale fixes the difference.
"""
import math

import torch

from . import _core as core

aten = torch.ops.aten

# Standalone table (tests, plugin path) + habana table (when available).
DECOMP_TABLE = {}


def _norm_factor(n, normalization, inverse):
    # Target op scaling: none=1, by_root_n=1/sqrt(n), by_n=1/n (both dirs).
    # Core forward is unscaled; core inverse already includes 1/n.
    if normalization == 0:
        return float(n) if inverse else 1.0
    if normalization == 1:
        return math.sqrt(n) if inverse else 1.0 / math.sqrt(n)
    if normalization == 2:
        return 1.0 if inverse else 1.0 / n
    raise ValueError(f"unknown fft normalization {normalization}")


def _last_dim_only(self, dim):
    if [d % self.dim() for d in dim] != [self.dim() - 1]:
        # ponytail: last-dim only; multi-dim FFT needs c2c chaining, add when rfftn is required
        raise NotImplementedError(f"hpu_complex: dim={tuple(dim)} not supported yet")


def _fft_r2c(self, dim, normalization, onesided):
    _last_dim_only(self, dim)
    pair = core.rfft(self, dim=-1)
    if not onesided:  # full FFT of the real signal
        sig = torch.stack([self, torch.zeros_like(self)], dim=-1)
        pair = core.fft(sig, dim=-1)
    n = self.shape[-1]
    pair = pair * _norm_factor(n, normalization, inverse=False)
    return torch.view_as_complex(pair.contiguous())


def _fft_c2r(self, dim, normalization, last_dim_size):
    _last_dim_only(self, dim)
    pair = torch.view_as_real(self.contiguous())
    out = core.irfft(pair, n=last_dim_size, dim=-1)
    return out * _norm_factor(last_dim_size, normalization, inverse=True)


def _fft_c2c(self, dim, normalization, forward):
    _last_dim_only(self, dim)
    pair = torch.view_as_real(self.contiguous())
    n = self.shape[-1]
    if forward:
        out = core.fft(pair, dim=-1)
    else:
        ore, oim = core._fft(pair[..., 0], pair[..., 1], inverse=True)
        out = torch.stack([ore, oim], dim=-1)
    out = out * _norm_factor(n, normalization, inverse=not forward)
    return torch.view_as_complex(out.contiguous())


def register_all():
    for op, fn in [(aten._fft_r2c, _fft_r2c), (aten._fft_c2r, _fft_c2r),
                   (aten._fft_c2c, _fft_c2c)]:
        DECOMP_TABLE[op] = fn
        try:  # habana compile path, present only on Gaudi stacks
            from habana_frameworks.torch.dynamo.compile_backend import \
                decomposition as _hdecomp

            _hdecomp.register_custom_decomposition(
                op, _hdecomp.hpu_backend_decompositions_common)(fn)
        except Exception as e:  # never silent: this hid a bug for a full cycle
            print(f"hpu_complex: habana decomp registration skipped: {e}")
    return DECOMP_TABLE

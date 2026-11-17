# hpu-complex

A plugin that brings `complex64` + `torch.fft` / `torch.stft` to Intel
Gaudi HPU — no Intel changes, no changes to your model code.

I made it because Gaudi's PyTorch stack cannot hold a complex number at
all: every FFT and TorchAudio spectrogram crashes with
`Complex datatype is not supported on HPU device.`
([HabanaAI/gaudi-pytorch-bridge#24](https://github.com/HabanaAI/gaudi-pytorch-bridge/issues/24))

## Install

```bash
pip install git+https://github.com/bruAristimunha/hpu-complex
```

Requires PyTorch from Intel's Habana installer (torch is deliberately
not a pip dependency — a PyPI torch would shadow the fork). The C++
extension compiles on first `enable()`.

## Use

```python
import hpu_complex
hpu_complex.enable()

torch.fft.rfft(x_hpu)          # was: RuntimeError
torch.stft(x_hpu, 512, 256, return_complex=True)
torch.fft.rfft(v).abs().sum().backward()   # gradients work
```

## How it works

- Complex tensors are interleaved-float tensor views — the same
  representation CUDA and Apple MPS use internally — tagged with the
  `PrivateUse1` backend key. Only ops that touch a complex tensor reach
  the extension; float HPU tensors keep the bridge's own kernels, so
  float code pays nothing (no eager tax, no float-op crashes).
- Entry points from real HPU tensors use the `HPU` key: `_fft_r2c`,
  `view_as_complex`, `torch.complex`. Complex factories
  (`torch.zeros(..., dtype=torch.complex64, device="hpu")`) are caught at
  `BackendSelect`. Everything else on a complex tensor is a
  `PrivateUse1` kernel computing on float views: views, arithmetic,
  `abs`/`sgn`/`where`, copies and fills, `_fft_c2c/_fft_c2r`. An
  unregistered op falls back to the CPU and warns once with its name.
- FFTs are one GEMM each with a cached real basis; phase indices are
  reduced mod n in integers, so float32 bases stay accurate at any n.
  The GEMMs run in float32 under autocast.
- `torch.compile(backend="hpu_backend")`: complex nodes run eagerly on
  these kernels, the float nodes around them stay in Synapse graphs.
- Optional: a fused TPC kernel (`hpu_complex.tpc.enable_tpc_fft()`,
  needs the kernel lib from `hpu_complex/tpc_kernel/build.sh` on
  `GC_KERNEL_PATH`). Correct, currently slower than the GEMM path.

## Verified on Gaudi2 (habana 1.21.4 / torch 2.6.0)

`hpu_complex/tests/pu1_check.py` passes 21/21 checks against torch on the CPU. Errors are max abs:

| case | error |
|---|---|
| rfft (3 norms), irfft (even/odd n), fft/ifft, `torch.complex`, `view_as_complex`, complex zeros, complex arithmetic | ≤ 1.5e-05 |
| `rfft().abs()` with gradients | 9e-08 |
| `stft().abs()` with gradients | 2.3e-05 |
| `.real`/`.imag` with gradients | 3.1e-05 |
| FFT convolution with gradients (conjugate views), incl. a broadcast kernel | ≤ 8.4e-05 |
| bf16 autocast (spectra stay float32) | 9e-08 |
| compiled `rfft().abs()` with gradients | 9e-08 |

Plain float ops are unchanged, and a float-only eager loop runs at the same speed with the extension loaded (2.75 vs 2.80 ms).

The onesided `_fft_r2c` backward is one GEMM: `view_as_real(grad) @ Wᵀ`. It is registered at `AutogradHPU` and traceable under compile. Its gradients match CPU float64 to 6e-07, relative, for n = 8, 9, 16 and 200, all three norms, eager and compiled (`tests/r2c_grad_check.py`). Compiled `stft().abs()`, forward and backward, at (3000, 6000): 280 ms here, 219 ms with real-DFT matmuls inside the graph.

braindecode models with `braindecode_kernels.patch_braindecode()` (v0.2.2), one card, compiled, seconds per step. The baseline computes the same spectra with real-DFT matmuls inside the graph:

| model | hpu-complex | real-DFT baseline |
|---|---|---|
| CBraMod | 1.010 (0.972 with v0.2.0 gates) | 0.964 |
| EEGDINO | 1.296 | 1.289 |
| CodeBrain | 5.015 | 4.975 |
| BIOT (`torch.stft`) | 0.412 | 0.378 |

CBraMod and BIOT were each measured against their baseline on the same card; EEGDINO and CodeBrain against earlier baseline runs.

Details: [BENCHMARKS.md](BENCHMARKS.md) ·
why native support is blocked: [ECOSYSTEM.md](ECOSYSTEM.md) ·
how other GPUs do it: [INVESTIGATION.md](INVESTIGATION.md)

## Limits

- The dispatch mode is opt-in and scoped
  (`with hpu_complex.complex_mode():`), not global: `enable()` never
  enters it. It is only needed for the fused TPC FFT hook.
- FFT is a dense GEMM (O(n²)), last dimension only, complex64 only.
- Lazy mode untouched. Conjugate views (`.conj()` bit) are not
  supported; ops outside the registered set run on the CPU (warned).
- Gradients through `.abs()`, `.real` and `.imag` are verified. Braindecode integration
  was removed in 0.2.0 (breaking: `hpu_complex.braindecode` is gone);
  it lives in the separate
  [braindecode-kernels](https://github.com/bruAristimunha/braindecode-kernels)
  package, not here.
- First `enable()` compiles the extension (~1 min, then cached).

## Tests

```bash
python -m pytest hpu_complex/tests/ -q            # CPU
python hpu_complex/tests/pu1_check.py            # Gaudi: eager + compiled vs CPU
python hpu_complex/tests/float_tax.py [--ext]    # Gaudi: float-only loop with/without the extension
python hpu_complex/tpc_kernel/run_fft_test.py     # Gaudi + TPC lib
```

— Bruno Aristimunha ([@bruAristimunha](https://github.com/bruAristimunha)) · Apache-2.0

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
  the extension's kernels; float HPU tensors keep the bridge's own. Float
  code meets the extension only when it allocates: `empty` and
  `empty_strided` check the dtype at `BackendSelect` and redispatch (no
  eager tax, no float-op crashes).
- Entry points from real HPU tensors use the `HPU` key: `_fft_r2c`,
  `view_as_complex`, `torch.complex` (and `_neg_view`, which only complex
  code calls, for the imaginary part of a conjugate view). Every complex64
  factory on the HPU allocates through `empty`/`empty_strided`, so all are
  caught at `BackendSelect`: `zeros`, `ones`, `full`, `randn`, `rand`,
  `scalar_tensor`, `*_like`, `new_*`, `.to(torch.complex64)`,
  `torch.tensor` (Habana registers no HPU kernel for these composite
  factories; checked on Gaudi2). Everything else on a complex tensor is a
  `PrivateUse1` kernel computing on float views: views, arithmetic,
  `abs`/`sgn`/`where`, `sum`, copies and fills, `set_` (`torch.load`),
  `_fft_c2c/_fft_c2r`. An unregistered op falls back to the CPU and warns
  once with its name.
- FFTs are one GEMM each with a cached real basis; phase indices are
  reduced mod n in integers, so float32 bases stay accurate at any n.
  Under autocast, the GEMMs and the complex arithmetic stay float32, as
  complex64 does on other backends.
- `torch.compile(backend="hpu_backend")`: complex nodes run eagerly on
  these kernels, the float nodes around them stay in Synapse graphs.
- Optional: a fused TPC kernel (`hpu_complex.tpc.enable_tpc_fft()`,
  needs the kernel lib from `hpu_complex/tpc_kernel/build.sh` on
  `GC_KERNEL_PATH`). Correct, currently slower than the GEMM path.

## Verified on Gaudi2 (habana 1.21.4 / torch 2.6.0)

`hpu_complex/tests/pu1_check.py` passes 66/66 checks against torch on the CPU. Errors are max abs unless noted:

| case | error |
|---|---|
| rfft (3 norms), irfft (even/odd n), fft/ifft, `torch.complex`, `view_as_complex`, complex zeros, complex arithmetic, incl. Python complex scalars (`z * 1j`) | ≤ 1.5e-05 |
| with gradients: `rfft().abs()`, `stft().abs()`, `.real`/`.imag`, `ihfft` (a conjugate view), `torch.angle` | ≤ 4.9e-05 |
| FFT convolution and cross-correlation (`rfft(k).conj()`) with gradients, incl. a broadcast kernel | ≤ 8.4e-05 |
| rfft backward from a strided gradient (`torch.cat` of spectra) | 1.2e-07 |
| permuted, sliced, stepped and offset 5-D complex tensors (rank-6 float pairs) and permuted 6-D ones: arithmetic incl. broadcasts and `where`, `abs`, `sum`, copies, writes (`copy_`, `mul_`, fills, `conj`), `reshape`, dtype views; cross-spectra `X Xᴴ` of a permuted one with gradients (relative) | ≤ 9.5e-07 / 1.9e-07 |
| after a first complex op under `torch.inference_mode`: `.real` gradients / double backward (relative) | 1.3e-05 / 4.2e-07 |
| bf16 autocast: spectra stay float32, complex `*` `/` `+` stay complex64 | ≤ 1.8e-07 |
| complex64 factories (`.to`, `*_like`, `new_zeros`, `full`, `zeros`, `randn`, `scalar_tensor`) give tagged tensors; `torch.load(map_location="hpu")`; `sgn` of NaN/inf/0 as `torch._refs.sgn` | exact |
| compiled `rfft().abs()` with gradients, static and `dynamic=True` (3 batch shapes, one graph); compiled `stft().abs()` | ≤ 2.2e-05 |
| compiled `rfft(dim=0)` raises the eager kernel's error | raises |

Plain float ops are unchanged. A float-only eager loop (`tests/float_tax.py`) takes 1.9–2.8 ms per iteration with or without the extension loaded (medians 2.70 and 2.58 ms over five runs each): the run-to-run spread is larger than any difference.

The onesided `_fft_r2c` backward is one GEMM: `view_as_real(grad) @ Wᵀ`. It is registered at `AutogradHPU` and traceable under compile. Its gradients match CPU float64 to 6.1e-07, relative, for n = 8, 9, 16 and 200, all three norms, eager and compiled with static and dynamic shapes (`tests/r2c_grad_check.py`). Compiled `stft().abs()`, forward and backward, at (3000, 6000): 280 ms here, 219 ms with real-DFT matmuls inside the graph.

braindecode models with `braindecode_kernels.patch_braindecode()` (v0.2.2), one card, compiled, seconds per step. The baseline computes the same spectra with real-DFT matmuls inside the graph (`--hpu-spectral`); each pair ran back to back on the same card:

| model | hpu-complex | real-DFT baseline | difference |
|---|---|---|---|
| CBraMod | 1.015 | 0.959 | +5.8% |
| EEGDINO | 1.322 | 1.280 | +3.2% |
| CodeBrain | 5.023 | 4.980 | +0.9% |
| BIOT (`torch.stft`) | 0.411 | 0.381 | +7.8% |

No op falls back to the CPU in these runs. The gap is in forward plus backward, where the complex nodes run eagerly outside the Synapse graphs.

Details: [BENCHMARKS.md](BENCHMARKS.md) ·
why native support is blocked: [ECOSYSTEM.md](ECOSYSTEM.md) ·
how other GPUs do it: [INVESTIGATION.md](INVESTIGATION.md)

## Limits

- The dispatch mode is opt-in and scoped
  (`with hpu_complex.complex_mode():`), not global: `enable()` never
  enters it. It is only needed for the fused TPC FFT hook.
- FFT is a dense GEMM (O(n²)), last dimension only, complex64 only.
- Habana misreads some non-contiguous layouts of rank 6 or more, its own
  float tensors too: `as_strided` views, and copies of results that keep
  a permuted input's strides (`(x.permute(0, 2, 1, 3, 4, 5) * 2).contiguous()`).
  Complex tensors of rank 5 or more are covered here (their float pairs
  and results are read as permutes of contiguous blocks, compute operands
  are copied contiguous first); float code with such tensors is not.
- Lazy mode untouched. Ops outside the registered set run on the CPU
  (warned), among them the fills of complex `randn`/`rand`
  (`normal_`/`uniform_`). `torch.func` transforms and forward-mode AD do
  not run through `rfft`/`stft`: their backward is a C++ autograd
  Function. Writes through the imaginary part of a conjugate view
  (`z.conj().imag[...] = v`) do not reach `z`.
- `torch.compile` with dynamic shapes (`dynamic=True`, or automatic
  dynamic after a shape change): torch 2.6's fake tensors reject symbolic
  shapes for `aten._fft_r2c`/`_fft_c2r` (their meta strides are wrong), so
  dynamo breaks the graph at those FFTs. They run eagerly on the
  extension, backward included; the rest stays compiled, one graph for
  all batch sizes.
- Gradients are verified through `.abs()`, `.real`/`.imag`,
  `torch.angle` and complex products (above). Braindecode integration
  was removed in 0.2.0 (breaking: `hpu_complex.braindecode` is gone);
  it lives in the separate
  [braindecode-kernels](https://github.com/bruAristimunha/braindecode-kernels)
  package, not here.
- First `enable()` compiles the extension (~1 min, then cached in
  `TORCH_EXTENSIONS_DIR`, by default `~/.cache/torch_extensions`). It
  warns when `habana_frameworks` is installed but `torch.hpu` is not
  loaded yet.

## Tests

```bash
python -m pytest hpu_complex/tests/ -q            # CPU (the HPU tests skip without habana_frameworks)
python hpu_complex/tests/pu1_check.py            # Gaudi: eager + compiled vs CPU
python hpu_complex/tests/r2c_grad_check.py       # Gaudi: rfft gradients vs CPU float64
python hpu_complex/tests/run_on_hpu.py           # Gaudi: test_on_hpu.py without pytest
python hpu_complex/tests/float_tax.py [--ext]    # Gaudi: float-only loop with/without the extension
python hpu_complex/tpc_kernel/run_fft_test.py     # Gaudi + TPC lib
```

The Gaudi checks exit 1 if any check fails; a NaN error counts as a failure.

— Bruno Aristimunha ([@bruAristimunha](https://github.com/bruAristimunha)) · Apache-2.0

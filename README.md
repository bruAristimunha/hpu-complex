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

- A C++ extension registers the three FFT primitives PyTorch needs
  (`_fft_r2c/_fft_c2r/_fft_c2c`) plus the allocation/copy/view ops the
  bridge asserts on, using the Habana fork's own `HPU` dispatch key.
- Complex tensors are interleaved-float tensor views — the same
  representation CUDA and Apple MPS use internally — so the closed
  graph compiler never sees a dtype it doesn't know.
- Complex mul/add/div run through `.out` kernels (autograd engine);
  forward ops reach them via the fork's composite aliases, no dispatch
  mode needed. `torch.compile(backend="hpu_backend")` graphs run
  on-device.
- Optional: a fused TPC kernel (`hpu_complex.tpc.enable_tpc_fft()`,
  needs the kernel lib from `hpu_complex/tpc_kernel/build.sh` on
  `GC_KERNEL_PATH`). Correct, currently slower than the GEMM path.
- `HPU_COMPLEX_FOURSTEP=1` trades speed for accuracy at n ≥ 4096.

## Verified on Gaudi2 (habana 1.21.4 / torch 2.6.0)

- Every case from issue #24 passes: rfft/irfft/fft, stft,
  `torch.complex`, complex64 allocation, `.abs()`, `.backward()`.
- Gradients match CPU autograd to **5.6e-05**; FFT relative error
  ≤ 9e-05 at n ≤ 1920 (6e-06 at n=6000 with the four-step flag).
- rfft at **0.3–0.8 ms** for the issue's shapes — 60–130× faster than
  the CPU round-trip. Apple MPS is 0.06–0.24 ms (one fused kernel);
  closing that gap needs a VLM-resident radix-2 TPC kernel.

Details: [BENCHMARKS.md](BENCHMARKS.md) ·
why native support is blocked: [ECOSYSTEM.md](ECOSYSTEM.md) ·
how other GPUs do it: [INVESTIGATION.md](INVESTIGATION.md)

## Limits

- The dispatch mode is opt-in and scoped
  (`with hpu_complex.complex_mode():`), not global: `enable()` never
  enters it. It is only needed for the fused TPC FFT hook.
- FFT is a dense GEMM (O(n²)) in the fast path; large-n accuracy is
  fp32-bound without the four-step flag.
- Lazy mode untouched. bf16/half arithmetic runs in fp32 while loaded
  (`HPU_COMPLEX_NO_ARITH=1` opts out).
- Gradients through `.real`/`.imag` do not flow (fork autograd view
  tracking); `.abs()` backward is verified. Braindecode integration
  was removed in 0.2.0 (breaking: `hpu_complex.braindecode` is gone);
  it lives in the separate
  [braindecode-kernels](https://github.com/bruAristimunha/braindecode-kernels)
  package, not here.
- First `enable()` compiles the extension (~1 min, then cached).

## Tests

```bash
python -m pytest hpu_complex/tests/ -q            # CPU
python hpu_complex/cpp_ext/run_ext_test.py        # Gaudi (in the stack)
python hpu_complex/tpc_kernel/run_fft_test.py     # Gaudi + TPC lib
```

— Bruno Aristimunha ([@bruAristimunha](https://github.com/bruAristimunha)) · Apache-2.0

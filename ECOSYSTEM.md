# Why Gaudi cannot have complex — ecosystem deep dive

Companion to [INVESTIGATION.md](INVESTIGATION.md) (how other GPUs do it,
routes). This answers: *why* is real `complex64` impossible today, at every
layer of the Intel Gaudi ecosystem. All file:line citations read from bodies
in the bridge v1.24.2 clone (commit `27eccbc`) and opened vendor docs.

## Short answer

A data type is not one switch. It is an entry that must exist in **every**
layer's list of known types, top to bottom. Gaudi's layers each keep their
own list, and `complex` is on **none** of them — including the one list that
matters most, which lives in a closed Intel binary nobody outside Intel can
edit. The chip itself could do the math tomorrow. The *paperwork* for the
number format was never filed, in 2016, by design.

## The death of one tensor, in order

`torch.zeros(2, dtype=torch.complex64, device="hpu")` hits 8 gates:

| # | Gate | Where | Kind |
|---|------|-------|------|
| 0 | Allocator | `habana_device/HPUAllocator.cpp:173` | **Not a gate** — allocation is byte-blind; memory is fine |
| 1 | Creation assert | `habana_eager/ops/empty.cpp:48,85` → `helpers/tensor_utils.cpp:670-673` | Mechanical. First death: "Complex datatype is not supported" |
| 2 | Complex scalars | `helpers/create_tensor.cpp:1125` `assert(!s.isComplex())` | Mechanical |
| 3 | Type translation | `helpers/create_tensor.cpp:1062-1085` PyTorch→Synapse map has 14 real entries, no complex key | Mechanical but pointless — there is nothing to translate *to* |
| 4 | **The type list itself** | `synDataType` enum (mirror in `HabanaAI/gaudi-pytorch-bridge`: `tests/user_custom_op/synapse_common_types.h:160-189`) ends at `syn_type_ufp16 = 1<<17`. No complex entry at any bit. Real header ships inside closed `libSynapse.so`; bridge only `dlopen`s it (`synapse_helpers/synapse_api_shim.cpp:34`) | **Fundamental. Closed Intel binary. This is the wall.** |
| 5 | Element layout | `synapse_helpers/type_conversions.h:22-46` — no "8 bytes, interleaved real/imag" entry. Pretending complex is `float` halves every stride (corrupt DMA); pretending it's `int64` gives right bytes, wrong meaning | Structural, same closed stack |
| 6 | Kernel naming | `synapse_helpers/graph.cpp:968-1025` builds kernel names like `fft_..._f32` from the dtype. No `c64` suffix exists, and no FFT kernel exists to name | Mechanical, pointing at nothing |
| 7 | Lazy mode + FFT ops | `habana_kernels/kernel_input_checks.cpp:3608` rejects *every* complex input for *every* op; `scripts/hpu_op.yaml` has zero `_fft_r2c/_fft_c2r/_fft_c2c` (the only 3 ops PyTorch needs) | Mechanical + missing kernel |
| 8 | Tail | Copy/DMA, recipe cache versioning, HCCL collectives (`collective_utils.cpp` has no complex wire type; even FP8 cheats via `hcclUint8`) | Mechanical, after Gate 4 |

Patching gates 1–3 and 6–8 in the open bridge changes nothing: gate 4's
closed runtime validates the type when the tensor is created and rejects
it. That is proven from the bridge's own build files — Synapse arrives as
installer headers + binary (`PrepareDependencies.cmake`), never as source.

The skeptic found one crack worth naming: because the allocator is
byte-blind, a hand-written C++ extension could mint a "zombie" tensor that
*claims* complex64 on HPU. It cannot be copied, moved, or computed on —
existence without life. It does not change the verdict. (Unverified on real
hardware; no Gaudi host in this environment.)

## Why it was designed this way

- Habana (founded 2016, Intel 2019) built a **training-only** chip: matrix
  engine (GEMM) + SIMD cores, dtype menu = training precisions
  (FP32/TF32/BF16/FP16, later FP8). Intel's own architecture page lists
  exactly those. Complex/FFT belongs to signal processing, audio, HPC —
  workloads with no GEMM demand and no champion inside the product.
- Official confirmation: the PyTorch Support Matrix has marked
  `Complex32 No / Complex64 No / FFT No` continuously from at least
  v1.15.0 through v1.24.2. This is documented policy, not a regression.
  Issue #24 is therefore an **architecture gap**, not a bug.
- Intel's only public answer ever found (Gaudi forum, 2024, on `stft`):
  complex isn't supported, run it on CPU. No roadmap statement exists.
- Amusing contradiction: the same matrix says "TorchAudio: Yes." That
  means the package imports and its CPU fallbacks run — its core
  `Spectrogram` still crashes on HPU tensors.

## Could Intel just add it? The FP8 precedent

Yes, dtypes *can* be added — FP8 was. But look at the bar: FP8 arrived
**with new silicon** (Gaudi2, May 2022, FP8 in the matrix engine for LLM
quantization), then software followed across ~19–48 bridge files **plus**
the closed compiler, collectives, and TransformerEngine. Intel adds a
dtype when (a) hardware implements it and (b) a headline workload pays
for the full-stack cost. Complex meets neither condition.

Estimated Intel fix, if they chose to:
- **Minimum** (unmodified `torch.fft.rfft(x).abs().backward()`, eager
  only): one closed change (new `syn_type_complex64`), one FFT kernel
  (even a simple matrix-based one), complex `abs`, FFT backward.
  Roughly 2–4 engineer-months, gated on the closed compiler release train.
- **Full** (complex128, elementwise ops, lazy + compile + distributed):
  add quarters, the heaviest cost being numeric validation.

## Ecosystem health: alive for LLMs, frozen for breadth

Quarterly bridge releases continue (v1.24.2, aligned to PyTorch/vLLM).
But: `SynapseAI_Core` repo archived Feb 2025, external PRs refused by
policy (`CONTRIBUTING.md`), successor chip (Falcon Shores) reduced to an
internal test chip, and 4+ years of releases shipped zero complex movement.
The working cadence is evidence *against* an imminent fix, not for it.

Downstream, everyone hand-rolls the workaround you rejected, per model:
TorchAudio spectrograms, FNO spectral convolutions (complex weights can't
even allocate), long-convolution/SSM models (FFT convolution), vLLM-Gaudi
(pins the CPU torchaudio wheel), and — closest to home — Braindecode,
where ContraWR, DIVER-1, and CodeBrain each re-implemented device STFT/DFT
separately. Models that need complex *values* (phase, Hilbert,
complex-valued layers) have no workaround at all.

## Correction + the MPS precedent (verified 2026-11-16, local + source)

Earlier drafts cited MPS as a fellow holdout. **Wrong — outdated.** Verified
live on a Mac (torch 2.6.0, macOS 27): complex64 allocation, `fft.rfft`,
`fft.fft`, `stft`, complex multiply and `abs` all run on MPS and stay
on-device (max err vs CPU ~1e-5). Only complex128/float64 remain blocked.

How Apple did it is the important part. Metal's shading language *still*
has no complex type. Apple added complex at the **graph-compiler layer**:
MPSGraph gained `MPSDataTypeComplexFloat32/16` and an FFT descriptor in
macOS 14 (2023), and PyTorch wired the three primitives to it
(`aten/src/ATen/native/mps/operations/FastFourierTransform.mm`,
`_fft_r2c_mps/_fft_c2r_mps/_fft_c2c_mps`, gated by
`supportsComplex()` = "MacOS 14+"; first commit 2024-02-20, PR #119670).
No silicon change, no language change — the graph layer learned to carry
two floats as one type. **MPSGraph is Apple's SynapseAI.** This removes the
last "everyone has this problem" defense: among first-party PyTorch
backends (CPU, CUDA, ROCm, XPU, MPS), Gaudi's stack is now the only one
where a complex tensor cannot exist.

## What other ecosystems teach

Missing complex at launch is normal (Trainium, Cerebras, Tenstorrent all
launched real-only). Two things are unusual about Gaudi in 2026:
1. Google's XLA put complex in its type enum in **2017, before scale** —
   one line then; Gaudi now pays the multi-layer late tax (new enum bit +
   layout + kernel-language convention + both bridge maps + lazy/eager
   gates + an FFT kernel) that XLA never had to pay.
2. Peers without a native type *sanctioned the pair* (Tenstorrent
   `ComplexTensor`, Cerebras `<complex>`, AWS trnfft). Gaudi does neither:
   it asserts at every layer **and** silently falls back to CPU in
   `torch.complex` — the worst of both worlds.

## Bottom line for issue #24

There is no community path to *real* complex, and no bridge-only Intel
path either. The fix is one product decision inside Intel's closed
compiler team, never publicly made. The most useful thing #24 can extract
is a yes/no: **is `complex` in SynapseAI's type system on any roadmap?**
Until that answer is yes, every Gaudi user who needs an FFT pays the CPU
round trip or rewrites their model — which is exactly the cost your issue
already documents with numbers.

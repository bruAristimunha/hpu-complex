# Systematic investigation: can Gaudi handle complex/FFT like NVIDIA/AMD?

Question: make Intel Gaudi support complex dtype + `torch.fft`/`torch.stft`
the way CUDA/ROCm do. Method: 5 parallel investigators (PyTorch contract,
GPU vendors, Gaudi internals, extension points, decomposition route) + 1
skeptic trying to falsify the hopeful answer. All claims below were read in
source bodies, not snippets. Local bridge clone: v1.24.2 HEAD.

> **Correction (2026-11-16):** §1 originally cited MPS as a complex holdout.
> Outdated: MPS supports complex64 + FFT since macOS 14 / PyTorch Feb 2024,
> added at Apple's graph layer (MPSGraph) without Metal language changes.
> See the MPS section in [ECOSYSTEM.md](ECOSYSTEM.md). Gaudi is now the
> only first-party backend without complex.

## 1. How NVIDIA / AMD / Intel-XPU do it (the recipe)

Every working GPU stack has the same 4 ingredients:

1. **Complex = two floats in a trench coat.** No complex ALU exists anywhere.
   The allocator/type system admits `complex64` as interleaved float pairs
   (`float2`), `view_as_real`/`view_as_complex` are pure stride views (no
   copy), and add/mul/conj/abs are decomposed float kernels.
2. **A vendor FFT library with a plan cache.** cuFFT (NVIDIA),
   hipFFT→rocFFT (AMD), oneMKL DFT (Intel XPU). PyTorch itself contains
   almost no FFT math — CUDA's `SpectralOps.cu` just wraps cuFFT plans
   (`CuFFTPlanCache.h`, LRU cache).
3. **Exactly 3 PyTorch primitives registered per backend:**
   `aten::_fft_r2c`, `aten::_fft_c2r`, `aten::_fft_c2c` (+`.out`).
   Everything else is free: `torch.fft.*` are composite wrappers, and
   `stft`/`istft` compose framing + window + rfft in PyTorch itself.
   *No vendor has an STFT call.* Stock PyTorch will NOT decompose FFT into
   real ops — prims terminate at `aten::_fft_*`, Inductor marks them
   fallback. A backend must supply the 3 primitives.
4. **Loud fallback for gaps** (XPU→CPU). Counterexample: Apple MPS, where
   Metal has no complex/double — tensors can't even be allocated. Lesson:
   ingredient 1 (the type system admitting complex) is the hard gate.

## 2. Gaudi, layer by layer

| Layer | State | Owner |
|---|---|---|
| Chip (TPC/MME) | Fine. FFT is real mul/add/trig; TPC does 64 float SIMD ops/cycle. No hardware change needed. | — |
| TPC-C language | Real scalars only (float/bf16/int). No complex, no double. An FFT kernel must take float re/im tensors. | Intel (docs/SDK public) |
| SynapseAI graph compiler | **The fundamental wall.** `synDataType` enum has no complex value, so no kernel name can carry a complex precision suffix. Closed source. | **Intel only** |
| Bridge dtype gates | `tensor_utils.cpp:673` asserts complex unsupported; `create_tensor.cpp:1062` has no PyTorch→Synapse complex mapping; `kernel_input_checks.cpp:3608` lazy mode rejects every complex input. | Intel repo (visible) |
| Bridge ops | Zero `fft` strings in the whole repo; `hpu_op.yaml` has no `_fft_*`/`stft`/`view_as_*`. Only `complex_hpu` exists — silently computes on CPU. | Intel repo (visible) |
| `torch.compile` | `decomposition.py` has zero FFT entries, so compiled graphs hit the same wall. | Intel repo (visible) |
| Upstream policy | `CONTRIBUTING.md`: "please fork it. Any external pull requests won't be reviewed and merged." | **Intel only** |

Key finding: **the FFT computation was never the problem.** Only the
complex *data type* is blocked, and only in Intel's closed compiler.

## 3. Five routes, ranked

| Route | Who | Effort | Gets you | Ceiling |
|---|---|---|---|---|
| R1. Float shim package (`hpu_fft.py`, done) | Us, today | Days | Correct on-device FFT/STFT now, eager + compile, autograd | Returns float `[...,2]`, not `complex64` — call sites change. Unfused, no plan cache. |
| R2. Bridge decomposition (fork) | Us (fork only) | Weeks | Unmodified `torch.compile` graphs produce FFT on HPU; optional slice makes real `complex64` tensors exist via `view_as_real` | Never merges upstream (CONTRIBUTING.md). Same unfused perf. |
| R3. Custom TPC FFT kernel | Us | Weeks–months | Fused fast FFT via official `registerUserCustomOp` + `GC_KERNEL_PATH` plugin (eager/lazy/compile proven pattern in `HabanaAI/gaudi-pytorch-bridge` `tests/user_custom_op/`) | Still float-pair API, own namespace `custom_op::` — not `torch.fft`. |
| R4. Native support | **Intel only** | Roadmap | True parity with cuFFT: `complex64`, unmodified `torch.fft`, vendor FFT guid | The only competitive-performance route. |
| R5. CPU fallback (today) | — | 0 | Correct results | Host copies dominate; cards idle. |

Skeptic's verdict: the strong claim "community can make Gaudi handle this
like other GPUs *without Intel*" is **false** — true `complex64`,
unmodified `torch.fft`, and any upstream merge all require Intel. The weak
claim is **true**: we can ship correct, fast-enough on-device FFT today
(R1/R3) under the float-pair contract.

## 4. Exact file list for the R2 bridge change (all bridge-side, no Synapse)

1. `scripts/hpu_op.yaml` — add `_fft_r2c/_fft_c2r/_fft_c2c/stft` entries
   with `override_fn`.
2. NEW `habana_eager/ops/fft.cpp` + `fft.h` — radix-2/DFT composition
   (port of `hpu_fft.py`), registered in `habana_eager/ops/override_fns.h`.
   Precedent: `masked_select.cpp`, `unique.cpp`, `bincount.cpp` already
   compose aten ops on-device with no TPC kernel.
3. `python_packages/habana_frameworks/torch/dynamo/compile_backend/decomposition.py`
   — `@register_custom_decomposition` for the 3 `_fft_*` ops.
4. Only for true-complex returns: allocation-only exemption in
   `is_supported_type`, `view_as_real/view_as_complex` view overrides
   (pattern: `as_strided.cpp` builds views with no Synapse tensor),
   on-device rewrite of `complex_hpu` (interleave instead of `.to("cpu")`).
   Rationale: the allocator is byte-based — a complex tensor can *exist*
   as interleaved float storage and be `view_as_real`'d before any
   Synapse compute. No `synDataType` change needed for this slice.
5. Tests (in `HabanaAI/gaudi-pytorch-bridge`): `tests/pytest_working/{eager,lazy,compile}/test_fft*.py` vs CPU
   `torch.fft`; update `tests/pytest_working/lazy/test_complex.py` which
   currently asserts the CPU-device result.

## 5. The ask to put on issue #24

1. Accept R2 as a reviewed contribution (bridge-only files above, no
   Synapse/TPC changes), or state which alternative Intel prefers.
2. Publish whether complex support in SynapseAI (`synDataType`) is on the
   roadmap — that single answer decides if `torch.fft` parity is ever
   possible.
3. Meanwhile the community ships R1 so users get on-device FFT/STFT today.

## 6. Unverified (needs Gaudi hardware)

- Radix-2 loop vs dense matmul vs CPU-fallback performance crossover.
- Whether a `load_library` plugin can override `aten::_fft_*` HPU dispatch
  without forking the bridge (boxed `cpu_fallback` registration order).
- Lazy-mode graph capture, dynamic shapes, `torch.compile` fullgraph of
  the shim; `hann_window`/`unfold` eager HPU dispatch (inferred, not run).
- Closed SynapseAI internals: an unpublished Intel FFT guid can't be
  disproven; public evidence is zero `fft` strings repo-wide.

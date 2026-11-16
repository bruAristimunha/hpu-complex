# Benchmarks — Gaudi2, torch 2.6.0 + habana 1.21.4, eager, fp32

Measured in the `hpu-complex-dev` pod (1 card), `hpu_complex/cpp_ext/bench.py`,
median of 10 after warmup, device-synchronized.

## rfft (ms/call, lower is better)

| Shape | ext (first) | ext (plan cache) | ext (now) | CPU round-trip | Apple MPS* |
|---|---|---|---|---|---|---|
| 1024×512 | 230 | 4.07 | **0.48** | 39.7 | 0.06 |
| 1024×1024 | 250 | 4.42 | **0.29** | 50.0 | 0.11 |
| 1024×1920 | 249 | 3.88 | **0.49** | 60.3 | — |
| 256×6000 | 288 | 3.84 | **0.55** | 50.1 | 0.24 |
| 8192×512 | 281 | 6.55 | **0.58** | 90.3 | — |
| STFT 64×16000, n_fft=512 | 263 | 5.36 | **1.28** | — | 0.25 |

\* MPS measured locally (same shapes, torch 2.6.0, Apple Silicon) for scale
reference only — different hardware, vendor-fused FFT.

**Issue gate check:** "faster than the CPU round trip for n ≥ 512" — **met,
60–130×**, at every measured size. Accuracy (rel. err vs CPU float64):
≤9e-05 at n ≤ 1920; 5.4e-04 at n=6000 (dense fp32 accumulation).
Gradients vs CPU autograd: 5.6e-05 max err (unchanged by all speed work).

Round 2 (250→4 ms was host copies; 4→0.3 ms was copies again): profiling
showed 73% of time in `copy_` — every `.out` kernel computed a result and
then copied it into `out`, and dense c2c did 4 GEMMs + sub/add ops. Fixes:
write-into-`out` bmm, one concatenated `[re,im] @ [C;-S]` GEMM per output
part (4 GEMMs → 2, zero sub/add), cached transposed plan matrices, skip
the ×1.0 norm multiply. Two Habana quirks found on the way: `batch_gemm`
rejects strided slice targets and very large elementwise batches (chunked
bmm+copy instead), and Habana `stack` rejects strided slice views
(materialize onesided slices before stacking).

## What the 60× came from (in order of discovery)

1. The 250 ms was never math. Raw matmul/add on this stack: 0.16 ms.
2. Per-call host→device fills: my add/sub fallback built a `ones` row per
   call, and `ones` fills via a CPU copy that stalls (~50 ms each, ×4).
   → Cached the row tensors (plan-cache style).
3. r2c computed 4 GEMMs + a zero-fill; the zero terms vanish. → 2 GEMMs.
4. DFT matrices rebuilt per call looked innocent but weren't the issue;
   the plan cache (cuFFT's trick) is in place regardless.

## How the vendor implementations differ (and what to steal next)

| | Apple MPSGraph | cuFFT / rocFFT | hpu_complex (now) |
|---|---|---|---|
| Complex type | Native in graph compiler (macOS 14+) | Native (`float2`) | TensorImpl view over float storage |
| FFT kernel | One fused radix kernel per graph | Fused mixed-radix kernels, batched | Dense DFT as 2 GEMMs (MME) |
| Planning | Descriptor + graph cache | Plan cache (LRU) | Matrix/row caches ✓ |
| Launches per FFT | 1 | 1 | ~15 aten ops (~0.25 ms each) |
| Large-n accuracy | Radix (log-depth sums) | Radix | Dense fp32: 5e-04 @ 6000 |

**Ranked next steps, by measured leverage:**

1. **Fused TPC radix kernel** (one launch, radix-2/4, twiddle ROM, via
   `registerUserCustomOp` + `GC_KERNEL_PATH`). This is the only route to
   MPS-class 0.1–1 ms; est. 10–40× over current. Weeks of work.
2. **Four-step (sqrt-decomposition) GEMM FFT** — implemented, opt-in via
   `HPU_COMPLEX_FOURSTEP=1`: rel. err at n=6000 drops 5e-04 → 6e-06
   (issue tolerance met), but it is launch-bound slow at large batch
   (241 ms @ 256×6000 vs 0.5 ms dense), so dense stays the default.
   Making four-step fast needs the twiddle folded into the stage-2 GEMM
   matrices (one batched GEMM, no elementwise pass).
3. **Radix-2 via aten ops: not recommended.** On this stack we are
   launch-bound, not flop-bound; a butterfly is ~log₂(n)×6 launches and
   loses to one dense GEMM until n is very large. (The Python core's
   radix path also dies at batch scale: `strided_view` synStatus.)
4. Keep dense GEMM for n ≤ ~2048: MME is faster than any launch-bound
   butterfly there, and the plan cache makes it ~4 ms flat.

## Correctness note found by this round

TensorIterator hands `.out` kernels a *permuted* (transposed-layout)
`out` tensor for strided inputs. Writing via `out.reshape(-1)` silently
wrote into a detached copy — complex mul on transposed tensors (the
four-step twiddle, some backward paths) was wrong. Fixed by writing
through `copy_` for non-contiguous `out` (verified: mul on transposed
tensors now exact, grads unchanged at 5.6e-05).

## Fused TPC kernel (built 2026-11-16)

Toolchain obtained in-pod (habanalabs-tools 1.21.4 from vault.habana.ai).
Minimal standalone kernel lib (`hpu_complex/tpc_kernel/`, tpc-clang) +
`registerUserCustomOp` plugin; wired as opt-in `hpu_complex.tpc.enable_tpc_fft()`
for n % 64 == 0, n <= 2048 (GEMM path otherwise).

| Shape | TPC kernel | GEMM ext |
|---|---|---|
| 1024x512 | 18.3 ms | 0.70 ms |
| 1024x1024 | 21.0 ms | 0.82 ms |
| 8192x512 | 21.5 ms | 0.73 ms |
| 256x4096 | 30.5 ms | 0.87 ms |

Correct at every row count (rel. err ~1e-4, fp32 DFT accumulation) and
through unmodified `torch.fft.rfft` (3.7e-05 @ 512). It is ~25x SLOWER
than the GEMM path: v1 is a DFT (O(n^2)) with per-bin partial stores and
twiddle reload per row; the GEMM extension's MME blocking wins. The
remaining design is a VLM-resident radix-2 (row loaded once into local
memory, shuffle butterflies for intra-vector stages) - the shuffle
encoding and index-space subset semantics needed for it were both
reverse-engineered and verified during this work:
- index space: kernel loops [offset, offset+size) per the TPC docs;
  rows must sit on index dim 0 with preferredSplitDim=1 and
  allRequired on the split mappings, or GC never parallelizes.
- shuffle splat only reaches 16-lane groups (src = B0+2*B1+4*B2+8*B5),
  so radix stages need the mov_dual_group + shuffle combination.

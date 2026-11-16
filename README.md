# hpu-complex: a plugin that brings complex64 + torch.fft to Intel Gaudi

I made this plugin because Intel Gaudi's PyTorch stack cannot hold a
complex number at all — every `torch.fft`, `torch.stft`, and TorchAudio
spectrogram crashes on HPU. `hpu-complex` fixes that from the outside,
with no Intel changes and no changes to your model code:

```python
import hpu_complex
hpu_complex.enable()

torch.fft.rfft(x_hpu)          # was: RuntimeError
torch.stft(x_hpu, 512, 256, return_complex=True)
torch.fft.rfft(v).abs().sum().backward()   # gradients work
```

It works by registering C++ kernels on the Habana fork's own dispatch
key, carrying complex as interleaved-float tensor views (the same trick
CUDA and Apple MPS use internally), so the closed graph compiler never
sees a dtype it doesn't know. Measured on Gaudi2 (exact environment of
the issue): gradients match CPU autograd to 5.6e-05, FFT runs
60–130× faster than the CPU round-trip the issue is forced into today,
and a fused TPC kernel (source included) proves the native route.

— Bruno Aristimunha ([@bruAristimunha](https://github.com/bruAristimunha))

Issue: [HabanaAI/gaudi-pytorch-bridge#24](https://github.com/HabanaAI/gaudi-pytorch-bridge/issues/24)

---

# Unlocking FFT / complex on Gaudi — superplain version

## What is broken, in plain words

Your code wants a Fourier transform. That math produces *complex* numbers
(a real part + an imaginary part, like `3 + 2i`).

The Gaudi card's PyTorch software refuses to even *hold* a complex number.
The moment any complex tensor appears, it stops with:

```
Complex datatype is not supported on HPU device.
```

So `torch.fft.*`, `torch.stft`, and every TorchAudio spectrogram crash.
Your only escapes today are: send the data to the CPU, transform it there,
send it back (slow — the card sits idle), or fake the transform with a big
matrix multiply on the card (works, but wasteful).

## The three locks (all still present in the newest bridge, v1.24.2)

Think of three doors in a row. All three are shut.

| # | Lock | Where | Plain meaning |
|---|------|-------|---------------|
| 1 | Complex tensors are banned | `backend/helpers/tensor_utils.cpp`, `is_supported_type()` | The software rejects the complex *number format* itself, before any math happens. |
| 2 | There is no FFT program for the card | `scripts/hpu_op.yaml` (zero `fft` entries in the whole repo) | Even the transform itself was never written for the chip. Each op in that file points to a kernel in Intel's closed compiler library — no kernel, no op. |
| 3 | `torch.complex` secretly uses the CPU | `habana_eager/ops/complex.cpp`, `complex_hpu()` | The one door that looks open is a trapdoor: it silently moves your data to the CPU and hands back a CPU tensor. |

## Who can open each lock

- **Locks 1 + 2: only Intel.** The FFT kernel and complex-number support live
  in their closed compiler (SynapseAI), not in the open bridge repo. Nobody
  outside Intel can merge a native fix. That is what issue #24 asks them for.
- **Lock 3 + a workaround for 1 + 2: us, today.** See below.

## What we built instead (works today, no Intel needed)

Trick: never let a complex number exist. Keep the real and imaginary parts
as two ordinary floats side by side — a tensor ending in a dimension of 2,
like `[..., frequency, 2]`. The card is perfectly happy with floats.

[`hpu_fft.py`](hpu_fft.py) does the whole transform with float-only math:

- `rfft(x)` → spectrum `[..., freq, 2]` (use `magnitude()` for `|X|`)
- `irfft(spec, n)` → back to a real signal
- `fft(spec)` → full complex-to-complex, still as `[..., n, 2]`
- `stft(x, n_fft, hop)` / `stft_magnitude(x, n_fft, hop)` → drop-in for the
  TorchAudio spectrogram front end that crashes today

Lengths that are powers of two use a real O(n log n) butterfly; other lengths
(200, 960, 1920…) use an on-device DFT matrix multiply — the workaround you
already benchmarked at 1.1–2.4× faster than the CPU round trip, now packaged.
Everything is reshape/multiply/add/matmul, so it runs in eager mode and
inside `torch.compile`, and gradients flow through it.

Verified on CPU against `torch.fft` (no Gaudi here): max error ~1e-11 in
float64 across n = 200/256/960/1024/1920, round-trip and STFT included,
gradients finite. In float32 the power-of-two path is at ~3e-7 relative
error; the dense-matrix path is ~9e-5 at n=1920, so prefer power-of-two
`n_fft` (e.g. 2048 instead of 1920) when accuracy matters.

## The plugin (in progress)

`hpu_complex/` is the planned product: unmodified `torch.fft` on Gaudi.
Internal pair representation (like CUDA/MPS), user code unchanged.

```python
import hpu_complex
hpu_complex.enable()   # then use torch.fft / torch.stft / torch.complex
```

Decision history (measured): Python `torch.library` override is shadowed
(G1), complex tensors cannot exist stock (G2), compile decompositions are
accepted (G3) — and a C++ extension on the fork's `HPU` dispatch key wins,
which is what `enable()` loads. CPU-side tests:
`python -m pytest hpu_complex/tests/ -q` (all passing).

**Measured on Gaudi2 (Voyager dev pod, 2026-11-16):** G1 FAIL for Python
`torch.library` (shadowed), G2 FAIL (the complex assert, live), G3 PASS
(decompositions accepted). Then the breakthrough: Habana's fork has its
own `c10::DispatchKey::HPU`, and a C++ extension registering there
(`TORCH_LIBRARY_IMPL(aten, HPU)`) **wins** — no bridge rebuild needed.

**Status: working.** `import hpu_complex; hpu_complex.enable()` makes
unmodified `torch.fft.rfft/irfft/fft`, `torch.stft`, `torch.complex`,
`view_as_complex`, `complex64` allocation, `.abs()`, and `.backward()`
run on Gaudi2 (grad err 5.6e-05 vs CPU autograd; FFT rel. err ≤9e-05 at
n≤1920, 5e-04 at n=6000 — dense-DFT v1, radix/TPC is the perf phase).
`torch.compile(backend="hpu_backend")` graphs run on-device.
**Performance:** 0.3–0.8 ms/call at issue sizes — 60–130× faster than
the CPU round trip (the issue's speed gate), STFT front-end 3.0 ms.
Apple MPS (fused kernel) is 0.06–0.24 ms. A fused TPC kernel now exists
(`hpu_complex/tpc_kernel/`, opt-in via `hpu_complex.tpc.enable_tpc_fft()`,
correct through `torch.fft.rfft`) but at ~21 ms it loses to the GEMM
path — it is a DFT, not yet the VLM radix-2 design that would close
the gap. Details: [BENCHMARKS.md](BENCHMARKS.md).
Full numbers and the vendor comparison: [BENCHMARKS.md](BENCHMARKS.md).
The C++
extension (`hpu_complex/cpp_ext/fft_ext.cpp`) carries complex as
interleaved-float TensorImpl views — Synapse never sees the dtype —
plus the allocation/copy/view ops the bridge asserts on; complex
mul/add/div live in a TorchDispatchMode (forward) and `.out` kernels
(autograd engine). Persistent test pod: `hpu-complex-dev` (1 card,
ceph mapped) — iterate with
`kubectl exec -n bpinto-sdp190 hpu-complex-dev -- ...`, delete with
`kubectl delete pod -n bpinto-sdp190 hpu-complex-dev`.

Background: [INVESTIGATION.md](INVESTIGATION.md) (how other GPUs do it,
routes) and [ECOSYSTEM.md](ECOSYSTEM.md) (why complex is missing, the
MPS precedent). Intel engagement: issue #24 comments
[6922765074](https://github.com/HabanaAI/gaudi-pytorch-bridge/issues/24#issuecomment-6922765074)
(roadmap yes/no) and
[6922811768](https://github.com/HabanaAI/gaudi-pytorch-bridge/issues/24#issuecomment-6922811768)
(MPS precedent).

## Try it on the cluster

```python
import torch, hpu_fft as H
x = torch.randn(4, 2048, device="hpu")
mag = H.magnitude(H.rfft(x))          # was: RuntimeError on HPU
spec = H.stft_magnitude(x, 512, 256)  # was: RuntimeError on HPU
```

Copy `hpu_fft.py` next to your training script. No complex tensor is ever
created, so lock 1 never triggers, and no FFT kernel is needed, so lock 2
never matters.

## What to still push Intel for

The native fix in issue #24. This shim is the interim primitive the issue
itself asks for in request #3 — if Intel won't ship it in
`habana_frameworks`, this file is the same idea as a drop-in. Native kernels
would still win on speed for large non-power-of-two transforms.

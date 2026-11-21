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
  `view_as_complex`, `torch.complex`, `x.view(torch.complex64)` (with torch's own layout
  checks and messages) (and `_neg_view`, which only complex
  code calls, for the imaginary part of a conjugate view). Every complex64
  factory on the HPU allocates through `empty`/`empty_strided`, so all are
  caught at `BackendSelect`: `zeros`, `ones`, `full`, `randn`, `rand`,
  `scalar_tensor`, `*_like`, `new_*`, `.to(torch.complex64)`,
  `torch.tensor` (Habana registers no HPU kernel for these composite
  factories; checked on Gaudi2). Everything else on a complex tensor is a
  `PrivateUse1` kernel computing on float views: views, arithmetic,
  `abs`/`sgn`/`where`, `sum`, copies and fills, `set_` (`torch.load`),
  `_fft_c2c/_fft_c2r`, `exp`, `neg`, `cat`, `masked_fill_` and `linalg.vector_norm`
  (spd_learn's `Green` ran these five on the CPU every step). An unregistered op falls back to the CPU and warns
  once with its name.
- A real HPU tensor and a Python complex scalar (`x * 2j`, `2j * x`, `x + 1j`, `x - 1j`, `1j - x`, `x / 2j`, and
  the same with a 0-d complex CPU tensor) is taken at `AutogradHPU`, before Habana's float kernels (lazy mode
  aborted the process there): `x * c` and `x ± alpha c` are `torch.complex` of one or two float ops on `x`
  (torch's complex64 arithmetic on `x + 0i`, signed zeros included), the others promote `x` to torch's result type
  (complex64; complex128 for float64) and run on the complex kernels. Gradients are torch's own formulas.
  In-place and `out=` forms with a real `self`/`out` raise torch's "can't be cast" error; every other call goes
  on to torch's autograd kernel unchanged.
- Complex `mm`/`bmm` (so `@`, `matmul`, `einsum` products) are one real GEMM on the card:
  `[ar | ai] @ [[br, bi], [-bi, br]] = [cr | ci]`, conjugate operands included (they reach `mm`/`bmm`
  unresolved). Products below 2²² multiply-adds (batch·m·k·n) go to the CPU instead, one copy each way, where
  they are faster (warned once; `hpu_complex.set_matmul_cpu_threshold(0)` keeps all of them on the card). The
  threshold comes from `tests/cmatmul_bench.py` (per-product timings), not from a model. In spd_learn's `Green`
  no product falls below it; a train step took 296 ms median (291-351) with the card GEMM against 411 ms
  (369-422) with every product on the CPU, 5 rotated rounds of 20 steps in one process. A second 2-round run
  measured only 1.05x and 1.21x: the step time drifts by up to 30% within a process, so read it as 1.0-1.4x.
- FFTs are one GEMM each with a cached real basis; phase indices are
  reduced mod n in integers, so float32 bases stay accurate at any n.
  Under autocast, the GEMMs and the complex arithmetic stay float32, as
  complex64 does on other backends.
- `torch.compile(backend="hpu_backend")`: complex nodes, and nodes that
  read a float view of complex storage (`view_as_real`, `.real`, `.imag`,
  `view(dtype)`, and any view of those, e.g. `z.diagonal().real` or
  `view_as_real(z)[..., 0].mT`; a fused graph misreads a strided one), run eagerly on these
  kernels; the other float nodes, fresh results such as `abs()` included,
  stay in Synapse graphs. Habana's view-chain fusion starts no chain at, and runs no chain into, a view
  touching complex storage (complex views, their float views such as `V.real.mT`, float views feeding
  `view(complex64)`): it bakes the strides traced by the fake tensors into one
  `as_strided`, and torch's meta of `linalg_eig` gives contiguous
  eigenvectors where the kernel returns column-major ones. Without these two
  patches the compiled `eig`/`eigvals`/`qr` gradients were wrong with no error.
- `complex128` (e.g. `torch.linalg.eig`/`eigvals` of a float64 matrix):
  Gaudi2 holds float64 at float32 precision (`1 + 1e-12` copied to the
  card and back is `1`), so a complex128 in Habana storage could not keep
  its values. A complex128 HPU tensor is instead kept in host memory that
  reports the HPU device, and every op on it computes on the CPU in
  float64: exact, with torch's semantics (views, `view_as_real`,
  `.real`/`.imag` and conjugate views alias it; float64 results such as
  `abs`, eigenvalues of `eigh` or `svd`'s `S` stay there too; bool and
  int results are ordinary HPU tensors). One warning at the first
  complex128 allocation. `complex64` keeps computing on the card
  (complex128 `mm`/`bmm` use the CPU's float64 GEMM, not the card's).
- Optional: a fused TPC kernel (`hpu_complex.tpc.enable_tpc_fft()`,
  needs the kernel lib from `hpu_complex/tpc_kernel/build.sh` on
  `GC_KERNEL_PATH`). Correct, currently slower than the GEMM path.

## Verified on Gaudi2 (habana 1.21.4 / torch 2.6.0)

`hpu_complex/tests/pu1_check.py` passes 197/197 checks against torch on the CPU. Errors are max abs unless noted:

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
| complex `mm`/`bmm`/`@` on both routes (card GEMM, CPU below the threshold): conjugate, strided, broadcast, n = 1, empty operands; `X Xᴴ` with gradients, compiled, and one above the threshold (64×22×300×22, relative); a float32 operand, or a float32 `out=` for `mm`/`bmm`, raises as on the CPU | ≤ 1.0e-06 |
| inf / NaN through the card GEMM: the same entries non-finite as on the CPU (overflowing sums: ±inf on the card where the CPU gives NaN; subnormal products flush to zero on the card, e.g. `1e-20 * 1e-20` is 0, where the CPU keeps 1e-40: Habana's float32 GEMM, not fixable here) | finite ≤ 8.3e-07 rel |
| `mm`/`bmm` of an HPU and a CPU complex operand raise "Expected all tensors to be on the same device", both routes | raises |
| rank-6 real HPU operands with permuted strides that are a Habana op's result (`x.permute(...) * 1`, `> 0`) in complex `*`, `+`, `/`, `where`, `copy_`, `torch.complex`, `.to(complex64)` (Habana's own `contiguous()` misreads them, also without hpu_complex) | exact (`/` 1.2e-07) |
| compiled (`hpu_backend`) `eigvals`/`eig` of real and complex input and complex `qr`, forward + backward (relative) | ≤ 1e-04 |
| complex64 `linalg.eigh`, `svd`, `qr`, `eig`, `matrix_exp`, `householder_product` forward + backward (relative), products on either route | ≤ 1.0e-06 |
| float32 `linalg.eig`/`eigvals` (complex output) forward + backward, incl. `.mT` and sliced inputs (relative) | ≤ 2.8e-06 |
| spd_learn `Green` train step (22 channels, batch 32): gradients vs a CPU float64 run, next to the CPU's own float32 error (the card is about 7x float32's error, worst `spd_layers.0.shrinkage_logit`) | 1.1e-03 / 1.5e-04 |
| spd_learn `Green` train step in lazy mode (`PT_HPU_LAZY_MODE=1`), same comparison; 53-80 ms/step, no hpu_complex CPU op | 1.2e-03 / 1.5e-04 |
| real HPU tensor and a Python complex scalar (`*`, `+`, `-`, `/`, rsub, alpha, 0-d, strided; bf16/int/bool → complex64, float64 → complex128), eager and lazy (relative); in-place / `out=` into a real tensor raise as torch | ≤ 7.4e-08 |
| complex `exp` (C99 edges: `exp(inf + 0j) = inf + 0j`), `neg`, `masked_fill_`, `cat` (real operands, conj, legacy empties), `linalg.vector_norm` (ord, dim, keepdim), with gradients, eager and lazy | ≤ 4.8e-07 |

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

`hpu_complex/tests/c128_check.py` (complex128) passes 65/65 checks in eager mode and 61/61 in lazy mode (the 4 left
out need complex64, which lazy mode lacks), with and without `hpu_ops.enable()`. Every complex128 result equals torch
on the CPU bit for bit (lstsq to 1.8e-15; eigh and svd, checked by reconstruction, to 1.1e-14), against the same input values as the card holds them: creation, copies,
`.to`, `view_as_real`/`view_as_complex`, writes through views, conjugate views, arithmetic with float64 and Python
scalars, the linear algebra of #7 (matmul, einsum, inv, solve, cholesky, lu, lstsq, det, pinv, matrix_exp, qr, eig,
eigh, svd, householder_product), FFTs, a backward pass, odd shapes (n = 1, empty, 0-d, offset and transposed views),
`torch.save`/`torch.load`, `copy.deepcopy`, in-place restrides (`t_`, `unsqueeze_`, `squeeze_`, `as_strided_`,
which keep the storage and its views) and `torch.compile(backend="hpu_backend")`. `eigvals` of a float64 (32, 22, 22) batch:

| (32, 22, 22), mean of 20 synced calls | eager | lazy |
|---|---|---|
| `eigvals(x)`, x float64 HPU → complex128 HPU | 2.90 ms | 2.06 ms |
| `eig(x)` → complex128 HPU | 5.03 ms | 3.55 ms |
| baseline: `eigvals(x.cpu())`, same values | 2.80 ms | 2.00 ms |
| baseline: `eigvals(x.float())` → complex64 HPU | 3.54 ms | — |

`eig` has no Habana kernel: the bridge's own CPU fallback runs once per call for float64 and float32 input alike
(`cpu_fallback` +1 per call); complex128 ops after it add no bridge fallback. Before this, both failed with
`Complex datatype is not supported on HPU device.`

Details: [BENCHMARKS.md](BENCHMARKS.md) ·
why native support is blocked: [ECOSYSTEM.md](ECOSYSTEM.md) ·
how other GPUs do it: [INVESTIGATION.md](INVESTIGATION.md)

## Limits

- complex128 computes on the CPU (above): it costs what the CPU op costs (its values are never copied to the card), and runs nothing on the
  card. Five edges: `torch.fft.rfft` of a float64 HPU tensor still returns complex64 (computed on the card in float32,
  as before); `view_as_complex` of a float64 HPU tensor returns a complex128 copy, not a view (warned); a Python float
  written into a float64 view of a complex128 tensor (`z.real[i] = 0.1`) passes through a Habana float64 scalar, at
  float32 precision; the raw storage of a complex128 HPU tensor is host memory, so `untyped_storage()`-level
  copies (`z.untyped_storage().cpu()`) are not supported (`torch.save` and `copy.deepcopy` are); and `torch.save`
  writes each complex128 HPU tensor as its own copy of its values, so tensors saved together that were views of one
  another (`{"z": z, "re": z.real}`) load as independent tensors (`copy.deepcopy` keeps them linked, as torch does);
  float64 results of complex128 ops load back into host memory, exact. An `out=` that partly overlaps an input in a
  layout torch's alias check cannot settle (`torch.add(z[:, :-1], 1, out=z[:, 1:])`, no error on the CPU either) is
  computed from the inputs before the write, so its values can differ from the CPU's in-place order. Under
  `torch.compile(backend="hpu_backend")`, a graph that holds a complex128 value (or, once the process has allocated
  one, a float64 value) runs eagerly as a whole: its complex128 ops compute on the CPU either way, Synapse graphs
  cannot read host memory, and Habana's graph passes gave a wrong gradient through `eig` even with every node eager.

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
- **Lazy mode (`PT_HPU_LAZY_MODE=1`)**: complex64 runs. Habana's lazy kernels take only tensors its own ops
  made (a plain view of the same storage raises `GetHbLazyTensor for a non lazy tensor`, and `empty(0).set_()` over
  lazy storage gives wrong values: `repro/lazy_complex64.py`), so a complex64 tensor made in lazy mode gets a storage
  that holds the Habana float tensor it lives in, and the kernels compute on Habana's own `as_strided` views of that
  tensor; it is released with the last complex tensor over it. spd_learn's `Green` trains in lazy mode (below).
  The guard now refuses, with a clear `RuntimeError` instead of a failure deep in Habana: complex64 under
  `torch.inference_mode` (Habana's views would be inference tensors, which autograd's view tracking of
  `view_as_real`/`.real` cannot take; `torch.no_grad()` works), and permuted complex64 tensors of rank 5 or more at
  storage offset 0 (Habana's lazy graphs read their permute block wrong: errors up to 12; the same layouts at an
  offset read right). `torch.compile(backend="hpu_backend")` fails in lazy mode for float-only functions too
  (`BackendCompilerFailed`; use eager mode for it). In lazy mode a float32 tensor cast to bfloat16 on the card
  rounds differently from the CPU (`x.bfloat16().float()` is 1.9e-3 off, without any complex op). Without
  hpu_complex, a real HPU tensor times a Python complex scalar still aborts the process in lazy mode, inside
  Habana's float `mul`/`add` (`repro/lazy_complex_scalar.py`, the `no-ext` rows).
- The complex linear-algebra kernels themselves run on the CPU (warned): `eigh`, `svd`, `qr`, `eig`,
  `matrix_exp`, `householder_product`, `solve`, `solve_triangular`, `triu`/`tril`; Habana's own CPU fallback
  computes `linalg.eig`/`eigvals` of a float32 input (one op per call in the bridge metric). Their gradients
  match the CPU (above), the rest of the backward stays on the card.
- Ops outside the registered set run on the CPU
  (warned), among them the fills of complex `randn`/`rand`
  (`normal_`/`uniform_`). `torch.func` transforms and forward-mode AD do
  not run through `rfft`/`stft`: their backward is a C++ autograd
  Function. Writes through the imaginary part of a conjugate view
  (`z.conj().imag[...] = v`) do not reach `z`.
- `out=` of the registered ops follows torch: a wrong-shape `out` is resized in its own storage (in place,
  warned when it had elements; growth reallocates the same storage, without keeping the old values), an
  expanded `out` raises, and so does an `mm`/`bmm` `out` on another device; `copy_` raises on a partial overlap
  of source and target.
- Copies between a plain CPU tensor and a plain HPU tensor never reach
  these kernels (neither carries the complex key), so Habana's `copy_`
  decides: `complex_cpu.copy_(real_hpu)` raises "Unsupported pytorch type
  ComplexFloat", and `real_hpu.copy_(complex_cpu)` raises "Could not run
  'aten::_copy_from' ... CPU backend" where torch keeps the real part. Use
  `real_hpu.to(torch.complex64).cpu()` and `real_hpu.copy_(complex_cpu.real)`.
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
python -m pytest hpu_complex/tests/ -q            # CPU (the HPU tests skip without habana_frameworks;
                                                 #  the complex128 ones build the extension: need ninja)
# The CPU suite checks the complex GEMM's formula and block layout in pure torch (test_cmatmul_formula_cpu.py),
# and the hpu_backend pass patches on stub passes plus the rank-6 permute-block rule (test_compile_patches_cpu.py).
# test_extension_build_cpu.py builds the C++ extension against CPU torch and checks its dispatcher registrations.
# test_pair_kernels_formula_cpu.py mirrors the pair formulas of exp, neg, masked_fill_, cat, vector_norm and of the
# real x complex-scalar path against torch (the scalar path bit for bit, signed zeros included).
# The C++ kernels themselves (mm/bmm and its CPU threshold, copy_pu1 staging, complex_view, the lazy guard)
# need a Gaudi to run: they are covered only by pu1_check.py, review4_check.py, run_on_hpu.py and lazy_check.py.
python hpu_complex/tests/pu1_check.py            # Gaudi: eager + compiled vs CPU
python hpu_complex/tests/review4_check.py        # Gaudi: compiled float views of complex storage, x.view(complex64), out=/copy_ rules
python hpu_complex/tests/r2c_grad_check.py       # Gaudi: rfft gradients vs CPU float64
python hpu_complex/tests/run_on_hpu.py           # Gaudi: test_on_hpu.py without pytest
python hpu_complex/tests/lazy_check.py           # Gaudi (PT_HPU_LAZY_MODE=1): complex64 in lazy mode, its guards, green_ops_check.py
                                                 #  (pu1_check.py runs green_ops_check.py in eager mode)
python repro/lazy_complex64.py --mode all        # Gaudi: Issue #3 reproducer (habana-bug, guarded, workaround)
python hpu_complex/tests/float_tax.py [--ext]    # Gaudi: float-only loop with/without the extension
python hpu_complex/tests/c128_check.py [--hpu-ops] [--time]  # Gaudi: complex128 vs CPU (eager and PT_HPU_LAZY_MODE=1)
python hpu_complex/tests/green_step.py [--ab]    # Gaudi: spd_learn Green train steps, gradients vs CPU, timing
python hpu_complex/tests/linalg_bwd_bench.py     # Gaudi: complex linalg fwd+bwd, one process per op
python hpu_complex/tests/cmatmul_bench.py [--sweep]  # Gaudi: complex bmm, card GEMM vs CPU round trip
python hpu_complex/tpc_kernel/run_fft_test.py     # Gaudi + TPC lib
```

The Gaudi checks exit 1 if any check fails; a NaN error counts as a failure.

— Bruno Aristimunha ([@bruAristimunha](https://github.com/bruAristimunha)) · Apache-2.0

// hpu_complex extension: complex64 + torch.fft on Gaudi, PrivateUse1 edition.
//
// A complex tensor is a TensorImpl view (dtype complex64) over interleaved
// float HPU storage, tagged with the PrivateUse1 backend key. Only ops that
// touch such a tensor reach the PrivateUse1 kernels below; plain float HPU
// tensors keep the bridge's own kernels, because nothing here is registered
// for them. The earlier HPU-key overrides (copy_, mul.out, abs, empty, ...)
// ran every float op of a model through this file: a 14x eager tax, and
// "Schema not found for node" in the bridge's eager graph builder.
//
// Entry points:
//   HPU key (real HPU inputs): _fft_r2c, view_as_complex, complex, and
//     _neg_view (reached only from complex code: the imaginary part of a
//     conjugate view).
//   BackendSelect (every factory passes it before a backend is chosen):
//     empty.memory_format / empty_strided with dtype complex64 on hpu; all
//     other calls are redispatched as the generated kernel does. Habana
//     registers no HPU kernel for the composite factories (zeros, ones, full,
//     randn, scalar_tensor, *_like, new_*, _to_copy), so they all allocate
//     through these two (verified on Gaudi2, SynapseAI 1.21.4).
//   AutogradHPU: _fft_r2c, whose onesided backward is one GEMM.
//   PrivateUse1: every op on a complex tensor. Each computes on float views
//     of the same storage, with the plain HPU key set, so Synapse never sees
//     a complex dtype. Unregistered ops fall back to the CPU (warned once).
//
// FFT: dense real DFT as matmuls (O(n^2)), bases cached per (n, norm, device), at most 32 at a time.
// Norm int: 0=none, 1=by_root_n, 2=by_n (aten per-op scaling).
#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/EmptyTensor.h>
#include <ATen/Operators.h>
#include <ATen/ops/_reshape_alias_native.h>
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/view_native.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/native/Resize.h>
#include <c10/core/InferenceMode.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/util/FunctionRef.h>

#include <cmath>
#include <limits>
#include <mutex>
#include <algorithm>
#include <numeric>
#include <set>
#include <string>
#include <tuple>
#include <map>

namespace {

using c10::DispatchKey;
using c10::DispatchKeySet;

DispatchKeySet complex_keys() {
  return DispatchKeySet({DispatchKey::PrivateUse1, DispatchKey::AutogradPrivateUse1, DispatchKey::ADInplaceOrView});
}

// The key set of a plain float HPU tensor, autograd keys included: the first call may run under
// torch.inference_mode (a validation pass before training), whose tensors lack them, and every float view of a
// complex tensor would then be an inference tensor for the rest of the process.
DispatchKeySet hpu_float_keys() {
  static const DispatchKeySet keys = [] {
    auto ks = at::empty({0}, at::TensorOptions().dtype(at::kFloat).device(at::kHPU)).key_set();
    return ks | c10::getAutogradRelatedKeySetFromBackend(ks.highestBackendKey());
  }();
  return keys;
}

bool tagged(const at::Tensor& t) {
  return t.defined() && t.key_set().has(DispatchKey::PrivateUse1);
}

at::Tensor view_of(const at::Tensor& base, DispatchKeySet keys, caffe2::TypeMeta dtype, c10::IntArrayRef sizes,
                   c10::IntArrayRef strides, int64_t storage_offset) {
  auto result = at::detail::make_tensor<at::TensorImpl>(c10::TensorImpl::VIEW, c10::Storage(base.storage()), keys,
                                                        dtype);
  auto* impl = result.unsafeGetTensorImpl();
  impl->set_sizes_and_strides(sizes, strides);
  impl->set_storage_offset(storage_offset);
  return result;
}

// A real view of base's storage for Habana's kernels. Habana misreads a bare TensorImpl (view_of) whenever its
// storage_offset is nonzero or its layout is non-contiguous/expanded (Synapse treats sizes as the physical buffer
// bounds and rejects out-of-bounds offsets or strides with synStatus 1 in strided_view / strided_insert). It also
// misreads as_strided views of rank 6 or more, so dense rank-6+ layouts are built via narrow + view + permute on a
// zero-offset full-storage view_of, while lower-rank / non-dense / expanded layouts go through Habana's as_strided
// over the full storage.
at::Tensor habana_view(const at::Tensor& base, caffe2::TypeMeta dtype, c10::IntArrayRef sizes, c10::IntArrayRef strides,
                       int64_t storage_offset) {
  auto view = view_of(base, hpu_float_keys(), dtype, sizes, strides, storage_offset);
  bool expanded = false;
  for (size_t i = 0; i < sizes.size(); ++i) {
    expanded |= sizes[i] > 1 && strides[i] == 0;
  }
  if (storage_offset == 0 && view.is_contiguous() && !expanded) {
    return view;
  }
  const auto base_numel = static_cast<int64_t>(base.storage().nbytes() / dtype.itemsize());
  if (view.dim() >= 6 && !expanded && view.is_non_overlapping_and_dense()) {
    // The dims from the largest stride to the smallest, as a contiguous block, permuted back.
    const auto n = static_cast<int64_t>(sizes.size());
    std::vector<int64_t> order(n), block(n), block_strides(n, 1), dims(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) { return strides[a] > strides[b]; });
    for (int64_t k = 0; k < n; ++k) {
      block[k] = sizes[order[k]];
      dims[order[k]] = k;
    }
    for (int64_t k = n - 2; k >= 0; --k) {
      block_strides[k] = block_strides[k + 1] * block[k + 1];
    }
    if (storage_offset == 0) {
      return view_of(base, hpu_float_keys(), dtype, block, block_strides, 0).permute(dims);
    }
    const int64_t view_numel = view.numel();
    const int64_t off = view_numel == 0 ? std::min(storage_offset, base_numel) : storage_offset;
    return view_of(base, hpu_float_keys(), dtype, {base_numel}, {1}, 0)
        .narrow(0, off, view_numel)
        .view(block)
        .permute(dims);
  }
  const int64_t inner = base_numel % 2 == 0 ? 2 : 1;
  return at::as_strided(view_of(base, hpu_float_keys(), dtype, {base_numel / inner, inner}, {inner, 1}, 0), sizes,
                        strides, storage_offset);
}

// float HPU [..., k, 2] (last stride 1) -> tagged complex64 [..., k], same storage.
at::Tensor complex_view(const at::Tensor& pair) {
  TORCH_CHECK(pair.scalar_type() == at::kFloat && pair.size(-1) == 2 && pair.stride(-1) == 1,
              "hpu_complex: a complex view needs float32 [..., 2] with last stride 1");
  auto sizes = pair.sizes().vec();
  auto strides = pair.strides().vec();
  sizes.pop_back();
  strides.pop_back();
  for (auto& s : strides) {
    TORCH_CHECK(s % 2 == 0, "hpu_complex: odd float stride cannot be a complex stride");
    s /= 2;
  }
  TORCH_CHECK(pair.storage_offset() % 2 == 0, "hpu_complex: odd storage offset");
  return view_of(pair, complex_keys(), caffe2::TypeMeta::Make<c10::complex<float>>(), sizes, strides,
                 pair.storage_offset() / 2);
}

// complex [...] -> float [..., 2] over the same storage, ignoring a conjugate bit: plain HPU keys for a tagged
// (complex64) tensor, the stock view for any other one, e.g. the complex128 CPU scalar of z * 1j.
at::Tensor raw_pair(const at::Tensor& c) {
  if (!tagged(c)) {
    TORCH_CHECK(c.is_complex(), "hpu_complex: expected a complex tensor, got ", c.scalar_type());
    return at::view_as_real(c.is_conj() ? c.conj() : c);  // conj() of a conjugate view: the stored values
  }
  TORCH_CHECK(c.scalar_type() == at::kComplexFloat, "hpu_complex: complex64 only, got ", c.scalar_type());
  auto sizes = c.sizes().vec();
  auto strides = c.strides().vec();
  for (auto& s : strides) {
    s *= 2;
  }
  sizes.push_back(2);
  strides.push_back(1);
  return habana_view(c, caffe2::TypeMeta::Make<float>(), sizes, strides, c.storage_offset() * 2);
}

// The pair of c's values. Conjugate views only reach copy_ (the Conjugate key
// passes copy_/clone/views through and resolves every other op's inputs first).
at::Tensor pair_view(const at::Tensor& c) {
  TORCH_CHECK(!c.is_conj(), "hpu_complex: unexpected conjugate view (resolve_conj() it first)");
  return raw_pair(c);
}

// The pair of c's values as an operand of Habana's compute kernels. Non-contiguous / expanded pair views are
// materialized contiguous so Habana compute ops (and their output stride propagation) never receive strided or
// stride-0 views.
at::Tensor operand_pair(const at::Tensor& c) {
  auto p = pair_view(c);
  return p.is_contiguous() ? p : p.contiguous();
}

at::Tensor make_complex_empty(c10::IntArrayRef size, c10::IntArrayRef stride, at::Device device) {
  // Storage for the strided layout, in floats (two per complex element); ATen's sizing checks rank and overflow.
  const auto floats = at::detail::computeStorageNbytes(size, stride, sizeof(c10::complex<float>)) / sizeof(float);
  auto flat = at::empty({static_cast<int64_t>(floats)}, at::TensorOptions().dtype(at::kFloat).device(device));
  return view_of(flat, complex_keys(), caffe2::TypeMeta::Make<c10::complex<float>>(), size, stride, 0);
}

// ---- BackendSelect factories -------------------------------------------------

// Exactly the generated BackendSelect kernel of a factory: the backend key of
// the options alone. Python/Fake modes sit above BackendSelect and have already
// intercepted (FakeTensor redispatches here with device=meta).
DispatchKeySet factory_keys(std::optional<at::ScalarType> dtype, std::optional<at::Layout> layout,
                            std::optional<at::Device> device) {
  return DispatchKeySet(c10::computeDispatchKey(dtype, layout, device));
}

bool complex_on_hpu(std::optional<at::ScalarType> dtype, DispatchKeySet ks) {
  return dtype.has_value() && *dtype == at::kComplexFloat && ks.highestPriorityTypeId() == DispatchKey::HPU;
}

at::Tensor empty_backend_select(c10::SymIntArrayRef size, std::optional<at::ScalarType> dtype,
                                std::optional<at::Layout> layout, std::optional<at::Device> device,
                                std::optional<bool> pin_memory, std::optional<at::MemoryFormat> memory_format) {
  auto ks = factory_keys(dtype, layout, device);
  if (complex_on_hpu(dtype, ks)) {
    auto sizes = C10_AS_INTARRAYREF_SLOW(size);
    return make_complex_empty(sizes, c10::contiguous_strides(sizes), *device);
  }
  return at::_ops::empty_memory_format::redispatch(ks, size, dtype, layout, device, pin_memory, memory_format);
}

at::Tensor empty_strided_backend_select(c10::SymIntArrayRef size, c10::SymIntArrayRef stride,
                                        std::optional<at::ScalarType> dtype, std::optional<at::Layout> layout,
                                        std::optional<at::Device> device, std::optional<bool> pin_memory) {
  auto ks = factory_keys(dtype, layout, device);
  if (complex_on_hpu(dtype, ks)) {
    return make_complex_empty(C10_AS_INTARRAYREF_SLOW(size), C10_AS_INTARRAYREF_SLOW(stride), *device);
  }
  return at::_ops::empty_strided::redispatch(ks, size, stride, dtype, layout, device, pin_memory);
}

// ---- DFT bases ------------------------------------------------------------------

// Phase index j*f reduced mod n in integers, so the angle stays in [0, 2pi)
// and float32 bases are accurate at any n (no 2*pi*j*f/n rounding growth).
at::Tensor angles(int64_t n, int64_t rows, int64_t cols) {
  auto opts = at::TensorOptions().dtype(at::kLong);
  auto idx = at::outer(at::arange(rows, opts), at::arange(cols, opts)).remainder(n);
  return idx.to(at::kDouble) * (2.0 * M_PI / static_cast<double>(n));
}

double norm_scale(int64_t normalization, int64_t n) {
  if (normalization == 1) {
    return 1.0 / std::sqrt(static_cast<double>(n));
  }
  if (normalization == 2) {
    return 1.0 / static_cast<double>(n);
  }
  return 1.0;
}

enum class Basis : int64_t { R2C, C2C, C2R };

// One cache for every basis (float32, on the device), keyed by (kind, n, m, normalization, device index); m is k,
// or the direction for c2c. Bases are O(n^2) floats, so the cache is bounded like main's plan cache (cleared when
// full, rebuilt lazily): many distinct FFT lengths cannot grow device memory without limit.
at::Tensor cached_basis(Basis kind, int64_t n, int64_t m, int64_t normalization, const at::Device& device,
                        c10::function_ref<at::Tensor()> build) {
  static std::map<std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t>, at::Tensor> cache;
  static std::mutex mutex;
  constexpr size_t kMaxBases = 32;
  std::lock_guard<std::mutex> lock(mutex);
  auto key = std::make_tuple(static_cast<int64_t>(kind), n, m, normalization, static_cast<int64_t>(device.index()));
  auto it = cache.find(key);
  if (it != cache.end()) {
    return it->second;
  }
  if (cache.size() >= kMaxBases) {
    cache.clear();
  }
  // A basis cached under inference mode would be an inference tensor, which the rfft backward's matmul cannot
  // save when a double backward (create_graph=True) records it.
  std::optional<c10::InferenceMode> normal_mode;
  if (c10::InferenceMode::is_enabled()) {
    normal_mode.emplace(/*enabled=*/false);
  }
  return cache.emplace(key, build().to(at::kFloat).to(device)).first->second;
}

// The r2c basis from angles [n, k]: x [..., n] @ W [n, 2k] = interleaved [re0, im0, re1, ...].
at::Tensor r2c_weights(const at::Tensor& ang, int64_t n, int64_t k, int64_t normalization) {
  const auto scale = norm_scale(normalization, n);
  return at::stack({at::cos(ang) * scale, at::sin(ang) * -scale}, -1).reshape({n, 2 * k});
}

// r2c, one GEMM.
at::Tensor r2c_basis(int64_t n, int64_t k, int64_t normalization, const at::Device& device) {
  return cached_basis(Basis::R2C, n, k, normalization, device,
                      [&] { return r2c_weights(angles(n, n, k), n, k, normalization); });
}

// c2c, one GEMM: [re, im] interleaved [..., 2n] @ W [2n, 2n] = interleaved output.
at::Tensor c2c_basis(int64_t n, bool forward, int64_t normalization, const at::Device& device) {
  return cached_basis(Basis::C2C, n, forward, normalization, device, [&] {
    auto ang = angles(n, n, n);  // [j, f]
    auto scale = norm_scale(normalization, n);
    auto c = at::cos(ang) * scale;
    auto s = at::sin(ang) * (forward ? -scale : scale);  // e^{-+i}: X = sum x (c + i s)
    // out_re = re c - im s ; out_im = re s + im c, rows interleaved (re_j, im_j).
    auto from_re = at::stack({c, s}, -1);       // [j, f, 2]
    auto from_im = at::stack({-s, c}, -1);      // [j, f, 2]
    auto w = at::stack({from_re, from_im}, 1);  // [j, 2, f, 2]
    return w.reshape({2 * n, 2 * n});
  });
}

// c2r, one GEMM from the half spectrum (Hermitian): interleaved [..., 2k] @ W [2k, n].
at::Tensor c2r_basis(int64_t n, int64_t k, int64_t normalization, const at::Device& device) {
  return cached_basis(Basis::C2R, n, k, normalization, device, [&] {
    // x_j = sum_f w_f (re_f cos - im_f sin), w_f = 1 for DC/Nyquist, 2 for the bins counted twice.
    auto ang = angles(n, k, n);  // [f, j]
    auto weight = at::full({k, 1}, 2.0, at::TensorOptions().dtype(at::kDouble));
    weight[0] = 1.0;
    if (n % 2 == 0 && k == n / 2 + 1) {
      weight[k - 1] = 1.0;
    }
    auto scale = norm_scale(normalization, n);
    auto w = at::stack({at::cos(ang) * weight * scale, at::sin(ang) * weight * -scale}, 1);  // [f, 2, j]
    return w.reshape({2 * k, n});
  });
}

void check_last_dim(const at::Tensor& self, c10::IntArrayRef dim) {
  TORCH_CHECK(dim.size() == 1 && (dim[0] == -1 || dim[0] == self.dim() - 1), "hpu_complex: only last-dim FFT");
}

// Autocast would run the DFT GEMMs and the pair arithmetic in bfloat16 (Habana's default lower-precision list
// has mul and matmul): complex math stays float32, as complex64 does under stock autocast.
#define HPU_COMPLEX_NO_AUTOCAST c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::autocast_dispatch_keyset)

// ---- FFT ------------------------------------------------------------------------------

at::Tensor fft_r2c_hpu(const at::Tensor& self, c10::IntArrayRef dim, int64_t normalization, bool onesided) {
  HPU_COMPLEX_NO_AUTOCAST;
  check_last_dim(self, dim);
  const int64_t n = self.size(-1);
  const int64_t k = onesided ? n / 2 + 1 : n;
  auto x = self.to(at::kFloat).contiguous();  // Habana GEMMs mis-handle strided batched inputs (stft frames)
  auto out = at::matmul(x, r2c_basis(n, k, normalization, self.device()));
  auto sizes = self.sizes().vec();
  sizes.back() = k;
  sizes.push_back(2);
  return complex_view(out.view(sizes));
}

at::Tensor fft_c2c_pu1(const at::Tensor& self, c10::IntArrayRef dim, int64_t normalization, bool forward) {
  HPU_COMPLEX_NO_AUTOCAST;
  check_last_dim(self, dim);
  const int64_t n = self.size(-1);
  auto pair = pair_view(self).contiguous();
  auto sizes = pair.sizes().vec();
  auto out = at::matmul(pair.view({-1, 2 * n}), c2c_basis(n, forward, normalization, self.device()));
  return complex_view(out.view(sizes));
}

at::Tensor fft_c2r_pu1(const at::Tensor& self, c10::IntArrayRef dim, int64_t normalization, int64_t last_dim_size) {
  HPU_COMPLEX_NO_AUTOCAST;
  check_last_dim(self, dim);
  const int64_t k = self.size(-1);
  auto pair = pair_view(self).contiguous();
  auto sizes = self.sizes().vec();
  sizes.back() = last_dim_size;
  auto out = at::matmul(pair.view({-1, 2 * k}), c2r_basis(last_dim_size, k, normalization, self.device()));
  return out.view(sizes);
}

at::Tensor view_as_complex_hpu(const at::Tensor& self) {
  return complex_view(self);
}

at::Tensor view_as_real_pu1(const at::Tensor& self) {
  return pair_view(self);
}

// z.view(dtype): the stock kernel copies the PrivateUse1 key onto the new dtype, and nothing here computes on a
// tagged real tensor (its float ops would come straight back): a real view gets the plain HPU keys, as
// view_as_real does.
at::Tensor view_dtype_pu1(const at::Tensor& self, at::ScalarType dtype) {
  auto v = at::native::view_dtype(self, dtype);
  if (v.is_complex()) {
    return v;
  }
  return habana_view(v, v.dtype(), v.sizes(), v.strides(), v.storage_offset());
}

// _neg_view (the imaginary part of a conjugate view, some complex derivatives): the Habana bridge's clone and copy
// ignore the Negative bit, so a lazily negated view would be read with the wrong sign. A physical negation instead:
// values and gradients are right; writes through such a view no longer reach its base.
at::Tensor neg_view_hpu(const at::Tensor& self) {
  return self.neg();
}

at::Tensor neg_view_pu1(const at::Tensor& self) {
  return complex_view(operand_pair(self.resolve_conj()).neg().contiguous());
}

// [..., 2] pair from real and imaginary parts, broadcast. The parts are made contiguous first: Habana's stack
// rejects strided views (BENCHMARKS.md), and a select or an expand is one.
at::Tensor stack_pair(const at::Tensor& re, const at::Tensor& im) {
  auto parts = at::broadcast_tensors({re, im});
  return at::stack({parts[0].contiguous(), parts[1].contiguous()}, -1);
}

at::Tensor complex_hpu(const at::Tensor& real, const at::Tensor& imag) {
  return complex_view(stack_pair(real.to(at::kFloat), imag.to(at::kFloat)));
}

// ---- arithmetic (PrivateUse1) ------------------------------------------------------------

// A real operand as float32 on the HPU (CPU 0-dim tensors are Python numbers).
at::Tensor hpu_float(const at::Tensor& t, const at::Device& device) {
  auto src = !t.is_hpu() && t.dim() >= 5 && !t.is_contiguous() ? t.contiguous() : t;
  auto r = src.to(device, at::kFloat);
  return r.dim() >= 5 && !r.is_contiguous() ? r.contiguous() : r;
}

// [..., 2] float32 pair of an operand: complex -> its pair (on the HPU), real -> nullopt.
std::optional<at::Tensor> pair_of(const at::Tensor& t, const at::Device& device) {
  if (!t.is_complex()) {
    return std::nullopt;
  }
  auto p = operand_pair(t);
  return p.device() == device && p.scalar_type() == at::kFloat ? p : p.to(device, at::kFloat);
}

at::Device hpu_device_of(const at::Tensor& a, const at::Tensor& b) {
  return tagged(a) ? a.device() : b.device();
}

// A float result as complex. One of rank 6 or more can keep a permuted operand's strides (a broadcast, where's
// condition), and Habana copies it as if it were contiguous; its permute view (habana_view) copies right.
at::Tensor complex_result(const at::Tensor& pair) {
  auto p = pair.dim() >= 6 && !pair.is_contiguous()
               ? habana_view(pair, pair.dtype(), pair.sizes(), pair.strides(), pair.storage_offset())
               : pair;
  return complex_view(p.contiguous());
}

// Neither operand complex: only the out= variants get here (out is the complex tensor), the plain op computes the
// value and copy_into writes it into out. A tagged real operand would dispatch straight back here.
bool both_real(const std::optional<at::Tensor>& pa, const std::optional<at::Tensor>& pb, const at::Tensor& a,
               const at::Tensor& b) {
  if (pa || pb) {
    return false;
  }
  TORCH_CHECK(!tagged(a) && !tagged(b), "hpu_complex: a real tensor carries the complex dispatch key");
  return true;
}

at::Tensor mul_pu1(const at::Tensor& a, const at::Tensor& b) {
  HPU_COMPLEX_NO_AUTOCAST;
  const auto device = hpu_device_of(a, b);
  auto pa = pair_of(a, device);
  auto pb = pair_of(b, device);
  if (both_real(pa, pb, a, b)) {
    return at::mul(a, b);
  }
  if (pa && pb) {
    auto ar = pa->select(-1, 0), ai = pa->select(-1, 1), br = pb->select(-1, 0), bi = pb->select(-1, 1);
    return complex_result(at::stack({ar * br - ai * bi, ar * bi + ai * br}, -1));
  }
  // complex x real: scale both parts.
  return complex_result(pa ? *pa * hpu_float(b, device).unsqueeze(-1) : *pb * hpu_float(a, device).unsqueeze(-1));
}

at::Tensor div_pu1(const at::Tensor& a, const at::Tensor& b) {
  HPU_COMPLEX_NO_AUTOCAST;
  const auto device = hpu_device_of(a, b);
  auto pa = pair_of(a, device);
  auto pb = pair_of(b, device);
  if (both_real(pa, pb, a, b)) {
    return at::div(a, b);
  }
  if (!pb) {
    return complex_result(*pa / hpu_float(b, device).unsqueeze(-1));
  }
  auto br = pb->select(-1, 0), bi = pb->select(-1, 1);
  auto inv = (br * br + bi * bi).reciprocal();
  if (!pa) {  // real / complex = a conj(b) / |b|^2
    auto ar = hpu_float(a, device);
    return complex_result(at::stack({ar * br * inv, -ar * bi * inv}, -1));
  }
  auto ar = pa->select(-1, 0), ai = pa->select(-1, 1);
  return complex_result(at::stack({(ar * br + ai * bi) * inv, (ai * br - ar * bi) * inv}, -1));
}

at::Tensor add_pu1(const at::Tensor& a, const at::Tensor& b, const at::Scalar& alpha) {
  HPU_COMPLEX_NO_AUTOCAST;
  TORCH_CHECK(!alpha.isComplex(), "hpu_complex: complex alpha is not supported");
  const auto device = hpu_device_of(a, b);
  const double s = alpha.toDouble();
  auto pa = pair_of(a, device);
  auto pb = pair_of(b, device);
  if (both_real(pa, pb, a, b)) {
    return at::add(a, b, alpha);
  }
  if (pa && pb) {
    return complex_result(s == 1.0 ? *pa + *pb : *pa + *pb * s);
  }
  // complex + real: the real part moves, the imaginary part is the complex operand's (scaled when it is b).
  if (pa) {
    return complex_result(stack_pair(pa->select(-1, 0) + hpu_float(b, device) * s, pa->select(-1, 1)));
  }
  return complex_result(stack_pair(hpu_float(a, device) + pb->select(-1, 0) * s, pb->select(-1, 1) * s));
}

at::Tensor sub_pu1(const at::Tensor& a, const at::Tensor& b, const at::Scalar& alpha) {
  TORCH_CHECK(!alpha.isComplex(), "hpu_complex: complex alpha is not supported");
  return add_pu1(a, b, -alpha.toDouble());
}

// |z| as one kernel: complex nodes run eagerly, so every op is a launch over the whole spectrum.
at::Tensor modulus(const at::Tensor& pair, bool keepdim) {
  return at::linalg_vector_norm(pair, 2, at::IntArrayRef{-1}, keepdim);
}

at::Tensor abs_pu1(const at::Tensor& self) {
  HPU_COMPLEX_NO_AUTOCAST;
  return modulus(operand_pair(self), false);
}

at::Tensor sgn_pu1(const at::Tensor& self) {
  HPU_COMPLEX_NO_AUTOCAST;
  auto pair = pair_view(self).contiguous();  // read twice below: one gather of a strided spectrum, not two
  // where(|z| == 0, 0, z / |z|), as torch._refs.sgn and the CUDA kernel: only exact zeros map to 0, so NaN and inf
  // inputs give NaN as on the other backends.
  const auto mod = modulus(pair, true);
  return complex_result((pair / mod).masked_fill(mod == 0, 0.0));
}

// A [..., 2] pair of any operand (a real one gets imaginary part 0), on the HPU.
at::Tensor as_pair(const at::Tensor& t, const at::Device& device) {
  if (auto p = pair_of(t, device)) {
    return *p;
  }
  auto re = hpu_float(t, device);
  return stack_pair(re, at::zeros_like(re));
}

// where(cond, a, b) with a complex a or b (compiled sgn backward: where(|z| == 0, 0, z / |z|)).
at::Tensor where_pu1(const at::Tensor& cond, const at::Tensor& a, const at::Tensor& b) {
  HPU_COMPLEX_NO_AUTOCAST;
  const auto device = hpu_device_of(a, b);
  auto cond_src = !cond.is_hpu() && cond.dim() >= 5 && !cond.is_contiguous() ? cond.contiguous() : cond;
  auto cond_dev = cond_src.to(device);
  if (cond_dev.dim() >= 5 && !cond_dev.is_contiguous()) {
    cond_dev = cond_dev.contiguous();
  }
  return complex_result(at::where(cond_dev.unsqueeze(-1), as_pair(a, device), as_pair(b, device)));
}

// out variants: compute, then resize and copy into out (complex or real).
at::Tensor& copy_into(at::Tensor& out, const at::Tensor& value);

at::Tensor& mul_out_pu1(const at::Tensor& a, const at::Tensor& b, at::Tensor& out) {
  return copy_into(out, mul_pu1(a, b));
}
at::Tensor& div_out_pu1(const at::Tensor& a, const at::Tensor& b, at::Tensor& out) {
  return copy_into(out, div_pu1(a, b));
}
at::Tensor& add_out_pu1(const at::Tensor& a, const at::Tensor& b, const at::Scalar& alpha, at::Tensor& out) {
  return copy_into(out, add_pu1(a, b, alpha));
}
at::Tensor& sub_out_pu1(const at::Tensor& a, const at::Tensor& b, const at::Scalar& alpha, at::Tensor& out) {
  return copy_into(out, sub_pu1(a, b, alpha));
}
at::Tensor& abs_out_pu1(const at::Tensor& self, at::Tensor& out) {
  return copy_into(out, abs_pu1(self));
}
at::Tensor& sgn_out_pu1(const at::Tensor& self, at::Tensor& out) {
  return copy_into(out, sgn_pu1(self));
}
// In-place variants: the result must keep self's shape and fit its dtype, as TensorIterator enforces; copy_into
// would otherwise resize self (detaching a view from its base) or drop an imaginary part.
at::Tensor& assign_inplace(at::Tensor& self, const at::Tensor& value) {
  TORCH_CHECK(self.sizes().equals(value.sizes()), "output with shape ", self.sizes(),
              " doesn't match the broadcast shape ", value.sizes());
  TORCH_CHECK(self.is_complex() || !value.is_complex(),
              "result type ComplexFloat can't be cast to the desired output type ", self.scalar_type());
  return copy_into(self, value);
}
at::Tensor& mul__pu1(at::Tensor& self, const at::Tensor& other) {
  return assign_inplace(self, mul_pu1(self, other));
}
at::Tensor& add__pu1(at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  return assign_inplace(self, add_pu1(self, other, alpha));
}

// ---- copies and fills (PrivateUse1) ------------------------------------------------------

at::Tensor& copy_pu1(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
  if (self.is_complex()) {
    auto dst = raw_pair(self);
    if (src.is_complex()) {
      auto src_pair = raw_pair(src);
      if (!src_pair.is_contiguous()) {
        src_pair = src_pair.contiguous();
      }
      dst.copy_(src_pair, non_blocking);
      if (self.is_conj() != src.is_conj()) {  // the stored values differ by one conjugation
        dst.select(-1, 1).neg_();
      }
    } else {  // real -> complex: imaginary part 0
      auto src_f = src;
      if ((!src_f.is_hpu() && src_f.dim() >= 5 && !src_f.is_contiguous()) ||
          (src_f.is_hpu() && src_f.dim() >= 6 && !src_f.is_contiguous())) {
        src_f = src_f.contiguous();
      }
      dst.select(-1, 0).copy_(src_f, non_blocking);
      dst.select(-1, 1).zero_();
    }
    return self;
  }
  // complex -> real keeps the real part, as torch does (it warns there).
  TORCH_WARN_ONCE("hpu_complex: casting complex values to real discards the imaginary part");
  auto src_pair = raw_pair(src);
  if (!src_pair.is_contiguous()) {
    src_pair = src_pair.contiguous();
  }
  auto real_part = src_pair.select(-1, 0);
  if (real_part.is_hpu() && real_part.dim() >= 6 && !real_part.is_contiguous()) {
    real_part = real_part.contiguous();
  }
  self.copy_(real_part, non_blocking);
  return self;
}

// Resize a tagged complex tensor in place: fresh contiguous storage, as resize_ does when it grows.
void resize_complex(at::Tensor& t, c10::IntArrayRef sizes) {
  auto strides = c10::contiguous_strides(sizes);
  auto fresh = make_complex_empty(sizes, strides, t.device());
  auto* impl = t.unsafeGetTensorImpl();
  impl->set_storage_keep_dtype(fresh.storage());
  impl->set_sizes_and_strides(sizes, strides);
  impl->set_storage_offset(0);
}

at::Tensor& copy_into(at::Tensor& out, const at::Tensor& value) {
  if (!out.sizes().equals(value.sizes())) {
    if (tagged(out)) {
      resize_complex(out, value.sizes());
    } else {
      out.resize_(value.sizes());
    }
  }
  if (tagged(out)) {
    return copy_pu1(out, value, false);
  }
  out.copy_(value);  // a tagged value dispatches to copy_pu1: complex -> complex, or its real part into a real out
  return out;
}

// The CPU fallback's way back into an out= tensor.
at::Tensor copy_from_and_resize_pu1(const at::Tensor& self, const at::Tensor& dst) {
  auto out = dst;  // same TensorImpl: resizing and copying out changes dst
  copy_into(out, self);
  return dst;
}

// sum over complex dims = sum over the same dims of the pair (its extra last dim stays).
at::Tensor sum_pu1(const at::Tensor& self, at::OptionalIntArrayRef dim, bool keepdim,
                   std::optional<at::ScalarType> dtype) {
  TORCH_CHECK(!dtype.has_value() || *dtype == at::kComplexFloat, "hpu_complex: complex sum to ", *dtype);
  std::vector<int64_t> dims;
  if (dim.has_value() && !dim->empty()) {
    for (auto d : *dim) {
      dims.push_back(c10::maybe_wrap_dim(d, self.dim()));
    }
  } else {
    for (int64_t d = 0; d < self.dim(); ++d) {
      dims.push_back(d);
    }
  }
  if (self.dim() == 0) {  // nothing to reduce: an empty dim list, or dim 0 of the pair, would reduce re/im
    return complex_result(pair_view(self).clone());
  }
  return complex_result(operand_pair(self).sum(dims, keepdim));
}

at::Tensor& sum_out_pu1(const at::Tensor& self, at::OptionalIntArrayRef dim, bool keepdim,
                        std::optional<at::ScalarType> dtype, at::Tensor& out) {
  return copy_into(out, sum_pu1(self, dim, keepdim, dtype));
}

at::Tensor& fill_pu1(at::Tensor& self, const at::Scalar& value) {
  auto pair = pair_view(self);
  auto c = value.toComplexDouble();
  pair.select(-1, 0).fill_(c.real());
  pair.select(-1, 1).fill_(c.imag());
  return self;
}

at::Tensor& zero_pu1(at::Tensor& self) {
  pair_view(self).zero_();
  return self;
}

// torch.load rebuilds every tensor as empty(0).set_(storage, offset, size, stride): attach the device storage in
// place, as the backends' own set_ kernels do (the CPU fallback would hand a CPU tensor the HPU storage).
at::Tensor& set_storage_pu1(at::Tensor& self, at::Storage source, int64_t storage_offset, c10::IntArrayRef size,
                            c10::IntArrayRef stride) {
  const auto strides = stride.empty() ? c10::contiguous_strides(size) : at::DimVector(stride.begin(), stride.end());
  at::native::checkSetStorage(self, std::move(source), storage_offset, size, c10::IntArrayRef(strides));
  at::native::setStrided(self, size, c10::IntArrayRef(strides), storage_offset);
  return self;
}

// ---- autograd of _fft_r2c on HPU: the onesided backward as one GEMM ----------------------

// The r2c basis from int32 device ops, for traced graphs: no cache or host state at trace time.
at::Tensor r2c_basis_traced(int64_t n, int64_t k, int64_t normalization, const at::Device& device) {
  // The phase index j*f must fit int32, or compiled gradients would be silently wrong.
  TORCH_CHECK((n - 1) * (k - 1) <= std::numeric_limits<int32_t>::max(),
              "hpu_complex: the compiled rfft backward supports n <= 65536, got ", n);
  auto iopts = at::TensorOptions().dtype(at::kInt).device(device);
  auto idx = at::outer(at::arange(n, iopts), at::arange(k, iopts)).remainder(n);
  return r2c_weights(idx.to(at::kFloat) * (2.0 * M_PI / static_cast<double>(n)), n, k, normalization);
}

// X = x @ W with W [n, 2k] = interleaved (C, -S): the real pairs of X are linear in x, and torch's
// complex gradient is dL/dRe + i dL/dIm, so grad_x = view_as_real(grad) @ W^T. This replaces
// fft_r2c_backward's zero-padded full spectrum and its [2n, 2n] c2c.
// Under torch.compile this Function runs on fake tensors, whose sizes are symbolic with dynamic=True: shapes are
// read as SymInts here, never with sizes().
class FftR2C : public torch::autograd::Function<FftR2C> {
 public:
  static at::Tensor forward(torch::autograd::AutogradContext* ctx, const at::Tensor& self, std::vector<int64_t> dim,
                            int64_t normalization, bool onesided) {
    // The eager kernel's check, here too: traced graphs reach this Function but not that kernel, and the backward
    // below is a last-dim transform (onesided: k = n / 2 + 1 bins, as the kernel computes them).
    check_last_dim(self, dim);
    ctx->saved_data["n"] = self.sym_size(-1);
    ctx->saved_data["dim"] = dim;
    ctx->saved_data["normalization"] = normalization;
    ctx->saved_data["onesided"] = onesided;
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::_fft_r2c::call(self, dim, normalization, onesided);
  }

  // grads[0] is always defined (custom Functions materialize zeros), and the engine casts the returned
  // gradient to the input's dtype.
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                 torch::autograd::variable_list grads) {
    HPU_COMPLEX_NO_AUTOCAST;
    const auto& grad = grads[0];
    const auto normalization = ctx->saved_data["normalization"].toInt();
    if (!ctx->saved_data["onesided"].toBool()) {  // the full spectrum: fft_r2c_backward's own formula
      const auto dim = ctx->saved_data["dim"].toIntVector();
      return {at::real(at::_fft_c2c(grad, dim, normalization, /*forward=*/false)), at::Tensor(), at::Tensor(),
              at::Tensor()};
    }
    // The basis is built for one length: a symbolic n is specialized (one graph per FFT length, the guard of
    // dynamic=True), the batch dims stay symbolic.
    const int64_t n = ctx->saved_data["n"].toSymInt().guard_int(__FILE__, __LINE__);
    const int64_t k = n / 2 + 1;
    // Eager gradients come from the kernels here (tagged): the cached basis. Traced ones are fake.
    auto basis = tagged(grad) ? r2c_basis(n, k, normalization, grad.device())
                              : r2c_basis_traced(n, k, normalization, grad.device());
    // resolve_conj: the gradient of rfft(x).conj() (cross-spectra, correlation, ihfft) arrives as a conjugate
    // view, which view_as_real refuses. contiguous: a gradient can be a strided view (a slice of cat's gradient,
    // a transpose), and Habana GEMMs mis-handle strided inputs, as in the forward.
    auto pair = at::view_as_real(grad.resolve_conj()).contiguous().view({-1, 2 * k});
    auto sizes = grad.sym_sizes().vec();
    sizes.back() = n;
    return {at::matmul(pair, basis.t()).view_symint(sizes), at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

at::Tensor fft_r2c_autograd(const at::Tensor& self, c10::IntArrayRef dim, int64_t normalization, bool onesided) {
  return FftR2C::apply(self, dim.vec(), normalization, onesided);
}

// ---- everything else: through the CPU (correct, slow; warned once per op) ----------------

void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  static std::set<std::string> warned;
  static std::mutex warned_mutex;
  {
    std::lock_guard<std::mutex> lock(warned_mutex);
    if (warned.insert(op.schema().operator_name().name + "." + op.schema().operator_name().overload_name).second) {
      TORCH_WARN("hpu_complex: ", op.schema().operator_name(), " on a complex HPU tensor runs on the CPU");
    }
  }
  at::native::cpu_fallback(op, stack);
}

}  // namespace

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&cpu_fallback>());
}

TORCH_LIBRARY_IMPL(aten, AutogradHPU, m) {
  m.impl("_fft_r2c", &fft_r2c_autograd);
}

TORCH_LIBRARY_IMPL(aten, BackendSelect, m) {
  m.impl("empty.memory_format", &empty_backend_select);
  m.impl("empty_strided", &empty_strided_backend_select);
}

TORCH_LIBRARY_IMPL(aten, HPU, m) {
  // Real HPU inputs only: these produce the tagged complex tensors.
  m.impl("_fft_r2c", &fft_r2c_hpu);
  m.impl("view_as_complex", &view_as_complex_hpu);
  m.impl("complex", &complex_hpu);
  m.impl("_neg_view", &neg_view_hpu);  // reached only from complex code (imag of a conjugate view)
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("_fft_c2c", &fft_c2c_pu1);
  m.impl("_fft_c2r", &fft_c2r_pu1);
  m.impl("view_as_real", &view_as_real_pu1);
  m.impl("view.dtype", &view_dtype_pu1);
  m.impl("_neg_view", &neg_view_pu1);
  // ATen's own TensorImpl views (same key set and dtype, bounds-checked): no PrivateUse1 kernels upstream.
  m.impl("as_strided", TORCH_FN(at::native::as_strided_tensorimpl));
  m.impl("view", TORCH_FN(at::native::view));
  m.impl("_reshape_alias", TORCH_FN(at::native::_reshape_alias));
  m.impl("abs", &abs_pu1);
  m.impl("abs.out", &abs_out_pu1);
  m.impl("sgn", &sgn_pu1);
  m.impl("sgn.out", &sgn_out_pu1);
  m.impl("mul.Tensor", &mul_pu1);
  m.impl("mul.out", &mul_out_pu1);
  m.impl("mul_.Tensor", &mul__pu1);
  m.impl("div.Tensor", &div_pu1);
  m.impl("div.out", &div_out_pu1);
  m.impl("add.Tensor", &add_pu1);
  m.impl("add.out", &add_out_pu1);
  m.impl("add_.Tensor", &add__pu1);
  m.impl("sub.Tensor", &sub_pu1);
  m.impl("sub.out", &sub_out_pu1);
  m.impl("where.self", &where_pu1);
  m.impl("sum.dim_IntList", &sum_pu1);
  m.impl("sum.IntList_out", &sum_out_pu1);
  m.impl("_copy_from_and_resize", &copy_from_and_resize_pu1);
  m.impl("copy_", &copy_pu1);
  m.impl("fill_.Scalar", &fill_pu1);
  m.impl("zero_", &zero_pu1);
  m.impl("set_.source_Storage_storage_offset", &set_storage_pu1);
}

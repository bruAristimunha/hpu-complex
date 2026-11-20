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
//     complex128 tensors live in host memory and compute on the CPU (see
//     "complex128" below).
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
#include <ATen/ops/_to_copy_native.h>
#include <ATen/ops/clone_native.h>
#include <ATen/ops/empty_like_native.h>
#include <ATen/ops/from_blob.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/native/Resize.h>
#include <c10/core/InferenceMode.h>
#include <c10/core/impl/alloc_cpu.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/util/FunctionRef.h>

#include <cmath>
#include <cstring>
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

// A real view of base's storage for Habana's kernels. Habana misreads a non-contiguous view of rank 6 or more built
// as a bare TensorImpl (view_of) or with as_strided, its own tensors' too, e.g. the float pair of a permuted 5-D complex
// tensor; it reads the same layout right as a permute of a contiguous view. So a dense one (a permuted contiguous
// block) is built that way. Other layouts of rank 6 or more go through Habana's as_strided on a contiguous 2-D base
// over the whole storage, which it reads right for some of them. Lower ranks and expanded views stay bare: Habana
// reads them right, and rejects some expanded views as strided views (synStatus 1 in strided_view / strided_insert).
at::Tensor habana_view(const at::Tensor& base, caffe2::TypeMeta dtype, c10::IntArrayRef sizes, c10::IntArrayRef strides,
                       int64_t storage_offset) {
  auto view = view_of(base, hpu_float_keys(), dtype, sizes, strides, storage_offset);
  bool expanded = false;
  for (size_t i = 0; i < sizes.size(); ++i) {
    expanded |= sizes[i] > 1 && strides[i] == 0;
  }
  if (view.dim() < 6 || view.is_contiguous() || expanded) {
    return view;
  }
  if (view.is_non_overlapping_and_dense()) {
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
    return view_of(base, hpu_float_keys(), dtype, block, block_strides, storage_offset).permute(dims);
  }
  const auto numel = static_cast<int64_t>(base.storage().nbytes() / dtype.itemsize());
  const int64_t inner = numel % 2 == 0 ? 2 : 1;
  return at::as_strided(view_of(base, hpu_float_keys(), dtype, {numel / inner, inner}, {inner, 1}, 0), sizes, strides,
                        storage_offset);
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

// The pair of c's values as an operand of Habana's compute kernels. Their result keeps a non-contiguous operand's
// strides, and Habana copies a tensor of rank 6 or more with such strides as if it were contiguous, its own too
// ((x.permute(0, 2, 1, 3, 4, 5) * 2).contiguous()); they also misread some of habana_view's as_strided views. The
// operand's own copy is right, so such a pair is made contiguous first (complex_result covers results whose layout
// comes from elsewhere).
at::Tensor operand_pair(const at::Tensor& c) {
  auto p = pair_view(c);
  return p.dim() >= 6 && !p.is_contiguous() ? p.contiguous() : p;
}

at::Tensor make_complex_empty(c10::IntArrayRef size, c10::IntArrayRef stride, at::Device device) {
  // Storage for the strided layout, in floats (two per complex element); ATen's sizing checks rank and overflow.
  const auto floats = at::detail::computeStorageNbytes(size, stride, sizeof(c10::complex<float>)) / sizeof(float);
  auto flat = at::empty({static_cast<int64_t>(floats)}, at::TensorOptions().dtype(at::kFloat).device(device));
  return view_of(flat, complex_keys(), caffe2::TypeMeta::Make<c10::complex<float>>(), size, stride, 0);
}

// ---- complex128: HPU tensors held in host memory, computed on the CPU in float64 ---------
//
// Gaudi2 holds float64 at float32 precision (hpu-ops #16: 1 + 1e-12 copied to the card and back is 1), so a
// complex128 kept in Habana storage could not keep its values. A complex128 HPU tensor is instead a tagged tensor
// (PrivateUse1) whose storage is host memory that reports the HPU device: its views, dtype views (view_as_real,
// .real, .imag) and copies keep their exact float64 values, every op on it computes on the CPU from a CPU tensor
// over the same memory (exact torch semantics), and Habana never sees it (it has no HPU key). Float64 results of
// such ops (abs, svd's S, eigvalsh, a gradient through view_as_real) stay in host memory too; results of other dtypes
// (bool masks, int pivots, float32 casts) become ordinary Habana tensors. One warning, at the first complex128
// allocation; complex64 keeps its on-card kernels.

void host_free(void* p) {
  c10::free_cpu(p);
}

// A tagged tensor over host memory: complex128, or a real view of one (view_as_real, .real, .imag, view(dtype)).
bool host_backed(const at::Tensor& t) {
  return tagged(t) && t.has_storage() && t.storage().data_ptr().get_deleter() == &host_free;
}

bool any_host(const at::Tensor& a, const at::Tensor& b) {
  return host_backed(a) || host_backed(b);
}

at::Device hpu_index(const at::Device& device) {
  return device.has_index() ? device : at::Device(device.type(), 0);
}

at::Tensor make_host_empty(c10::IntArrayRef size, c10::IntArrayRef stride, caffe2::TypeMeta dtype,
                           const at::Device& device) {
  TORCH_WARN_ONCE("hpu_complex: complex128 HPU tensors are kept in host memory and computed on the CPU in float64 "
                  "(exact; Gaudi2 holds float64 at float32 precision). Use complex64 to compute on the card.");
  const auto nbytes = at::detail::computeStorageNbytes(size, stride, dtype.itemsize());
  void* data = c10::alloc_cpu(std::max<size_t>(nbytes, 1));
  c10::Storage storage(c10::Storage::use_byte_size_t(), nbytes, c10::DataPtr(data, data, &host_free, hpu_index(device)),
                       /*allocator=*/nullptr, /*resizable=*/false);
  auto t = at::detail::make_tensor<at::TensorImpl>(c10::TensorImpl::VIEW, std::move(storage), complex_keys(), dtype);
  t.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);
  return t;
}

// A CPU tensor over a host-backed tensor's memory: same sizes, strides, offset, conjugate and negative bits.
// Writes through it are writes to t; it keeps t's storage alive.
at::Tensor host_cpu(const at::Tensor& t) {
  auto storage = t.storage();
  // Over the whole storage, then as_strided: kernels that restride their input (set_, as_strided) stay in bounds.
  const auto numel = static_cast<int64_t>(storage.nbytes() / t.dtype().itemsize());
  auto v = at::from_blob(storage.mutable_data(), {numel}, [storage](void*) {},
                         at::TensorOptions().dtype(t.dtype()).device(at::kCPU))
               .as_strided(t.sizes(), t.strides(), t.storage_offset());
  v._set_conj(t.is_conj());
  v._set_neg(t.is_neg());
  return v;
}

// An operand on the CPU: host-backed tensors as a view of their memory, other device tensors copied.
at::Tensor cpu_operand(const at::Tensor& t) {
  if (!t.defined() || t.is_cpu()) {
    return t;
  }
  return host_backed(t) ? host_cpu(t) : t.cpu();
}

// Where a CPU result of operands a, b goes back to: the first device operand.
at::Device device_of(const at::Tensor& a, const at::Tensor& b) {
  return a.defined() && !a.is_cpu() ? a.device() : b.device();
}

at::Tensor& host_inplace(at::Tensor& self, c10::function_ref<void(at::Tensor&)> op) {
  auto cpu = host_cpu(self);
  op(cpu);
  return self;
}

// A CPU result of an op on host-backed tensors, on the HPU device: complex128 and float64 in host memory (Habana
// would hold float64 at float32 precision, so eig, svd, pinv or a gradient through a real intermediate would lose
// it), every other dtype as an ordinary Habana tensor.
at::Tensor from_cpu(const at::Tensor& t, const at::Device& device) {
  if (!t.defined()) {
    return t;
  }
  if (t.scalar_type() != at::kComplexDouble && t.scalar_type() != at::kDouble) {
    return t.to(device);
  }
  const auto contiguous = c10::contiguous_strides(t.sizes());
  auto out = make_host_empty(t.sizes(), t.is_non_overlapping_and_dense() ? t.strides() : c10::IntArrayRef(contiguous),
                             t.dtype(), device);
  host_cpu(out).copy_(t);
  return out;
}

// empty_like of a tagged tensor (zeros_like, clone, the Negative key's resolution, ...): float64 of a host-backed one
// stays in host memory, as from_cpu keeps it; everything else as ATen's own kernel (complex128 reaches make_host_empty
// through empty_strided).
at::Tensor empty_like_pu1(const at::Tensor& self, std::optional<at::ScalarType> dtype, std::optional<at::Layout> layout,
                          std::optional<at::Device> device, std::optional<bool> pin_memory,
                          std::optional<at::MemoryFormat> memory_format) {
  if (host_backed(self) && dtype.value_or(self.scalar_type()) == at::kDouble &&
      (!device || device->type() == self.device().type())) {
    auto like = at::empty_like(host_cpu(self), dtype, layout, at::kCPU, pin_memory, memory_format);
    return make_host_empty(like.sizes(), like.strides(), like.dtype(), self.device());
  }
  return at::native::empty_like(self, dtype, layout, device, pin_memory, memory_format);
}

// clone and _to_copy (.to(dtype), .double(), ...) of a host-backed tensor: the CPU kernel on a view of its memory, the
// result through from_cpu. ATen's composite kernels would allocate with empty_strided: float64 in Habana storage.
at::Tensor clone_pu1(const at::Tensor& self, std::optional<at::MemoryFormat> memory_format) {
  if (host_backed(self)) {
    return from_cpu(at::clone(host_cpu(self), memory_format), self.device());
  }
  return at::native::clone(self, memory_format);
}

at::Tensor to_copy_pu1(const at::Tensor& self, std::optional<at::ScalarType> dtype, std::optional<at::Layout> layout,
                       std::optional<at::Device> device, std::optional<bool> pin_memory, bool non_blocking,
                       std::optional<at::MemoryFormat> memory_format) {
  if (host_backed(self) && (!device || device->type() == self.device().type())) {
    return from_cpu(at::_to_copy(host_cpu(self), dtype, layout, std::nullopt, pin_memory, non_blocking, memory_format),
                    self.device());
  }
  return at::native::_to_copy(self, dtype, layout, device, pin_memory, non_blocking, memory_format);
}

// copy.deepcopy of a host-backed tensor (hpu_complex.__init__ routes Tensor.__deepcopy__ here): torch's own path
// clones the storage through Habana's storage copy, which reads the host pointer as device memory (segfault in eager).
// Without storage_of: a fresh host copy of self's whole storage. With it (a copy made earlier in the same deepcopy, of a
// tensor sharing self's storage): that copy's storage. Either way self's dtype, sizes, strides and offset, so tensors
// that alias one another (z, z.real, z[1:]) still alias after the copy, as torch's deepcopy keeps them. Conjugate and
// negative bits are not carried (torch's deepcopy resolves them: the caller does).
at::Tensor host_deepcopy(const at::Tensor& self, const std::optional<at::Tensor>& storage_of) {
  TORCH_CHECK(host_backed(self), "hpu_complex::host_deepcopy: not a complex128 HPU tensor (or a float64 view of one)");
  c10::Storage storage;
  if (storage_of.has_value() && storage_of->defined()) {
    TORCH_CHECK(host_backed(*storage_of) && storage_of->storage().nbytes() == self.storage().nbytes(),
                "hpu_complex::host_deepcopy: storage_of is not a copy of self's storage");
    storage = storage_of->storage();
  } else {
    const auto nbytes = static_cast<int64_t>(self.storage().nbytes());
    auto fresh = make_host_empty({nbytes}, {1}, caffe2::TypeMeta::Make<uint8_t>(), self.device());
    if (nbytes > 0) {
      std::memcpy(fresh.storage().mutable_data(), self.storage().data(), static_cast<size_t>(nbytes));
    }
    storage = fresh.storage();
  }
  auto out = at::detail::make_tensor<at::TensorImpl>(c10::TensorImpl::VIEW, std::move(storage), complex_keys(),
                                                     self.dtype());
  auto* impl = out.unsafeGetTensorImpl();
  impl->set_sizes_and_strides(self.sizes(), self.strides());
  impl->set_storage_offset(self.storage_offset());
  return out;
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

bool complex128_on_hpu(std::optional<at::ScalarType> dtype, DispatchKeySet ks) {
  return dtype.has_value() && *dtype == at::kComplexDouble && ks.highestPriorityTypeId() == DispatchKey::HPU;
}

at::Tensor empty_backend_select(c10::SymIntArrayRef size, std::optional<at::ScalarType> dtype,
                                std::optional<at::Layout> layout, std::optional<at::Device> device,
                                std::optional<bool> pin_memory, std::optional<at::MemoryFormat> memory_format) {
  auto ks = factory_keys(dtype, layout, device);
  if (complex_on_hpu(dtype, ks)) {
    auto sizes = C10_AS_INTARRAYREF_SLOW(size);
    return make_complex_empty(sizes, c10::contiguous_strides(sizes), *device);
  }
  if (complex128_on_hpu(dtype, ks)) {
    auto sizes = C10_AS_INTARRAYREF_SLOW(size);
    return make_host_empty(sizes, c10::contiguous_strides(sizes), caffe2::TypeMeta::Make<c10::complex<double>>(),
                           *device);
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
  if (complex128_on_hpu(dtype, ks)) {
    return make_host_empty(C10_AS_INTARRAYREF_SLOW(size), C10_AS_INTARRAYREF_SLOW(stride),
                           caffe2::TypeMeta::Make<c10::complex<double>>(), *device);
  }
  return at::_ops::empty_strided::redispatch(ks, size, stride, dtype, layout, device, pin_memory);
}

// scalar_tensor (wrapped Python scalars: torch.where(c, z, 0), z.real[i] = v): Habana registers an HPU kernel for it in
// lazy mode, so it would not allocate through empty. Complex128 on the HPU is built on the CPU and moved to host memory.
at::Tensor scalar_tensor_backend_select(const at::Scalar& s, std::optional<at::ScalarType> dtype,
                                        std::optional<at::Layout> layout, std::optional<at::Device> device,
                                        std::optional<bool> pin_memory) {
  auto ks = factory_keys(dtype, layout, device);
  if (complex128_on_hpu(dtype, ks)) {
    return at::scalar_tensor(s, at::TensorOptions().dtype(at::kComplexDouble)).to(*device);
  }
  return at::_ops::scalar_tensor::redispatch(ks, s, dtype, layout, device, pin_memory);
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
  if (host_backed(self)) {
    return from_cpu(at::_fft_c2c(host_cpu(self), dim, normalization, forward), self.device());
  }
  HPU_COMPLEX_NO_AUTOCAST;
  check_last_dim(self, dim);
  const int64_t n = self.size(-1);
  auto pair = pair_view(self).contiguous();
  auto sizes = pair.sizes().vec();
  auto out = at::matmul(pair.view({-1, 2 * n}), c2c_basis(n, forward, normalization, self.device()));
  return complex_view(out.view(sizes));
}

at::Tensor fft_c2r_pu1(const at::Tensor& self, c10::IntArrayRef dim, int64_t normalization, int64_t last_dim_size) {
  if (host_backed(self)) {
    return from_cpu(at::_fft_c2r(host_cpu(self), dim, normalization, last_dim_size), self.device());
  }
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
  if (self.scalar_type() == at::kDouble) {
    // Habana's float64 storage holds float32 values in its own layout: no complex128 view of it exists.
    TORCH_WARN_ONCE("hpu_complex: view_as_complex of a float64 HPU tensor returns a complex128 copy, not a view: "
                    "writes to it do not reach the float64 tensor");
    return from_cpu(at::view_as_complex(self.cpu()), self.device());
  }
  return complex_view(self);
}

// The complex128 view of a host-backed float64 [..., 2] tensor (view_as_real of a complex128 one, or a slice of it).
at::Tensor view_as_complex_pu1(const at::Tensor& self) {
  TORCH_CHECK(host_backed(self) && self.scalar_type() == at::kDouble,
              "hpu_complex: view_as_complex of a tagged ", self.scalar_type(), " tensor");
  auto c = at::view_as_complex(host_cpu(self));  // torch's checks (last dim 2, stride 1, even strides and offset)
  return view_of(self, complex_keys(), c.dtype(), c.sizes(), c.strides(), c.storage_offset());
}

at::Tensor view_as_real_pu1(const at::Tensor& self) {
  if (host_backed(self)) {
    auto r = at::view_as_real(host_cpu(self));  // torch's checks (no unresolved conjugate)
    return view_of(self, complex_keys(), r.dtype(), r.sizes(), r.strides(), r.storage_offset());
  }
  return pair_view(self);
}

// z.view(dtype): the stock kernel copies the PrivateUse1 key onto the new dtype, and nothing here computes on a
// tagged real tensor (its float ops would come straight back): a real view gets the plain HPU keys, as
// view_as_real does.
at::Tensor view_dtype_pu1(const at::Tensor& self, at::ScalarType dtype) {
  auto v = at::native::view_dtype(self, dtype);
  if (v.is_complex() || host_backed(v)) {  // host memory stays tagged: every op on it computes on the CPU
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
  if (host_backed(self)) {  // host memory is read right with the Negative bit: a true view
    auto v = self.alias();
    v._set_neg(!self.is_neg());
    return v;
  }
  return complex_view(operand_pair(self.resolve_conj()).neg().contiguous());
}

// [..., 2] pair from real and imaginary parts, broadcast. The parts are made contiguous first: Habana's stack
// rejects strided views (BENCHMARKS.md), and a select or an expand is one.
at::Tensor stack_pair(const at::Tensor& re, const at::Tensor& im) {
  auto parts = at::broadcast_tensors({re, im});
  return at::stack({parts[0].contiguous(), parts[1].contiguous()}, -1);
}

at::Tensor complex_hpu(const at::Tensor& real, const at::Tensor& imag) {
  if (real.scalar_type() == at::kDouble || imag.scalar_type() == at::kDouble) {  // complex128: torch's checks
    return from_cpu(at::complex(cpu_operand(real), cpu_operand(imag)), device_of(real, imag));
  }
  return complex_view(stack_pair(real.to(at::kFloat), imag.to(at::kFloat)));
}

// ---- arithmetic (PrivateUse1) ------------------------------------------------------------

// A real operand as float32 on the HPU (CPU 0-dim tensors are Python numbers).
at::Tensor hpu_float(const at::Tensor& t, const at::Device& device) {
  return t.to(device, at::kFloat);
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
  if (any_host(a, b)) {
    return from_cpu(at::mul(cpu_operand(a), cpu_operand(b)), device_of(a, b));
  }
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
  if (any_host(a, b)) {
    return from_cpu(at::div(cpu_operand(a), cpu_operand(b)), device_of(a, b));
  }
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
  if (any_host(a, b)) {
    return from_cpu(at::add(cpu_operand(a), cpu_operand(b), alpha), device_of(a, b));
  }
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
  if (any_host(a, b)) {
    return from_cpu(at::sub(cpu_operand(a), cpu_operand(b), alpha), device_of(a, b));
  }
  TORCH_CHECK(!alpha.isComplex(), "hpu_complex: complex alpha is not supported");
  return add_pu1(a, b, -alpha.toDouble());
}

// |z| as one kernel: complex nodes run eagerly, so every op is a launch over the whole spectrum.
at::Tensor modulus(const at::Tensor& pair, bool keepdim) {
  return at::linalg_vector_norm(pair, 2, at::IntArrayRef{-1}, keepdim);
}

at::Tensor abs_pu1(const at::Tensor& self) {
  if (host_backed(self)) {
    return from_cpu(at::abs(host_cpu(self)), self.device());
  }
  HPU_COMPLEX_NO_AUTOCAST;
  return modulus(operand_pair(self), false);
}

at::Tensor sgn_pu1(const at::Tensor& self) {
  if (host_backed(self)) {
    return from_cpu(at::sgn(host_cpu(self)), self.device());
  }
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
  if (any_host(a, b)) {
    return from_cpu(at::where(cpu_operand(cond), cpu_operand(a), cpu_operand(b)), device_of(a, b));
  }
  HPU_COMPLEX_NO_AUTOCAST;
  const auto device = hpu_device_of(a, b);
  return complex_result(at::where(cond.to(device).unsqueeze(-1), as_pair(a, device), as_pair(b, device)));
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
  if (host_backed(self)) {
    return host_inplace(self, [&](at::Tensor& s) { s.mul_(cpu_operand(other)); });
  }
  if (host_backed(other)) {  // complex64 self: computed in complex128, cast back as torch does
    return assign_inplace(self, at::mul(cpu_operand(self), cpu_operand(other)).to(self.scalar_type()));
  }
  return assign_inplace(self, mul_pu1(self, other));
}
at::Tensor& add__pu1(at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  if (host_backed(self)) {
    return host_inplace(self, [&](at::Tensor& s) { s.add_(cpu_operand(other), alpha); });
  }
  if (host_backed(other)) {
    return assign_inplace(self, at::add(cpu_operand(self), cpu_operand(other), alpha).to(self.scalar_type()));
  }
  return assign_inplace(self, add_pu1(self, other, alpha));
}

// ---- copies and fills (PrivateUse1) ------------------------------------------------------

at::Tensor& copy_pu1(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
  if (host_backed(self)) {  // CPU copy semantics: casts, broadcasts, conjugate and negative bits of both sides
    return host_inplace(self, [&](at::Tensor& s) { s.copy_(cpu_operand(src)); });
  }
  if (host_backed(src)) {
    if (!self.is_complex() && src.is_complex()) {
      TORCH_WARN_ONCE("hpu_complex: casting complex values to real discards the imaginary part");
      self.copy_(at::real(host_cpu(src).resolve_conj()), non_blocking);
    } else {
      self.copy_(host_cpu(src), non_blocking);  // CPU, a Habana tensor, or complex64 (copy_pu1 again, from the CPU)
    }
    return self;
  }
  if (self.is_complex()) {
    auto dst = raw_pair(self);
    if (src.is_complex()) {
      dst.copy_(raw_pair(src), non_blocking);
      if (self.is_conj() != src.is_conj()) {  // the stored values differ by one conjugation
        dst.select(-1, 1).neg_();
      }
    } else {  // real -> complex: imaginary part 0
      dst.select(-1, 0).copy_(src, non_blocking);
      dst.select(-1, 1).zero_();
    }
    return self;
  }
  // complex -> real keeps the real part, as torch does (it warns there).
  TORCH_WARN_ONCE("hpu_complex: casting complex values to real discards the imaginary part");
  self.copy_(raw_pair(src).select(-1, 0), non_blocking);
  return self;
}

// Resize a tagged complex tensor in place: fresh contiguous storage, as resize_ does when it grows.
void resize_complex(at::Tensor& t, c10::IntArrayRef sizes) {
  auto strides = c10::contiguous_strides(sizes);
  auto fresh = host_backed(t) ? make_host_empty(sizes, strides, t.dtype(), t.device())
                              : make_complex_empty(sizes, strides, t.device());
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
  if (host_backed(self)) {
    return from_cpu(at::sum(host_cpu(self), dim, keepdim, dtype), self.device());
  }
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
  if (host_backed(self)) {
    return host_inplace(self, [&](at::Tensor& s) { s.fill_(value); });
  }
  auto pair = pair_view(self);
  auto c = value.toComplexDouble();
  pair.select(-1, 0).fill_(c.real());
  pair.select(-1, 1).fill_(c.imag());
  return self;
}

at::Tensor& zero_pu1(at::Tensor& self) {
  if (host_backed(self)) {
    return host_inplace(self, [](at::Tensor& s) { s.zero_(); });
  }
  pair_view(self).zero_();
  return self;
}

// torch.load rebuilds every tensor as empty(0).set_(storage, offset, size, stride): attach the device storage in
// place, as the backends' own set_ kernels do (the CPU fallback would hand a CPU tensor the HPU storage).
at::Tensor& set_storage_pu1(at::Tensor& self, at::Storage source, int64_t storage_offset, c10::IntArrayRef size,
                            c10::IntArrayRef stride) {
  if (self.scalar_type() != at::kComplexFloat && source.device().type() == at::kHPU &&
      source.data_ptr().get_deleter() != &host_free) {
    // complex128 from Habana storage (torch.load(map_location="hpu") restores storages there): its bytes are copied
    // into host memory, so the tensor holds its own copy rather than an alias of source.
    const auto nbytes = static_cast<int64_t>(source.nbytes());
    auto bytes = at::empty({0}, at::TensorOptions().dtype(at::kByte).device(source.device()));
    bytes.set_(source, 0, {nbytes}, {1});
    auto host = make_host_empty({nbytes}, {1}, caffe2::TypeMeta::Make<uint8_t>(), source.device());
    host_cpu(host).copy_(bytes.cpu());
    source = host.storage();
  }
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

at::Tensor& copy_into(at::Tensor& out, const at::Tensor& value);

bool has_host_arg(const c10::OperatorHandle& op, const torch::jit::Stack* stack) {
  for (const auto& arg : torch::jit::last(*stack, op.schema().arguments().size())) {
    if (arg.isTensor() && host_backed(arg.toTensor())) {
      return true;
    }
    if (arg.isList()) {
      for (const auto& e : arg.toListRef()) {
        if (e.isTensor() && host_backed(e.toTensor())) {
          return true;
        }
      }
    }
  }
  return false;
}

// The CPU fallback of an op with a host-backed (complex128 or its float64 view) argument. Unlike
// at::native::cpu_fallback, host-backed inputs are read in place (no copy), and float64 results stay in host memory
// (from_cpu): the complex128 computation stays exact end to end. Written arguments are computed on a CPU copy and
// written back (copy_into resizes an out= tensor). No warning: make_host_empty warned once already.
void host_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  const auto num_args = schema.arguments().size();
  const auto start = stack->size() - num_args;
  const std::vector<c10::IValue> originals(stack->begin() + start, stack->end());
  std::optional<at::Device> device;
  std::vector<std::pair<at::Tensor, at::Tensor>> written;  // (argument, its CPU copy)
  auto to_cpu = [&](const at::Tensor& t, bool write) {
    if (!t.defined() || t.is_cpu()) {
      return t;
    }
    if (!device || host_backed(t)) {
      device = t.device();
    }
    if (host_backed(t) && !write) {
      return host_cpu(t);
    }
    auto copy = cpu_operand(t).clone();
    if (write) {
      written.emplace_back(t, copy);
    }
    return copy;
  };
  for (size_t i = 0; i < num_args; ++i) {
    auto& iv = (*stack)[start + i];
    const auto* alias = schema.arguments()[i].alias_info();
    const bool write = alias != nullptr && alias->isWrite();
    if (iv.isTensor()) {
      iv = to_cpu(iv.toTensor(), write);
    } else if (iv.isTensorList()) {
      auto list = iv.toTensorList().copy();
      for (size_t j = 0; j < list.size(); ++j) {
        list.set(j, to_cpu(list.get(j), write));
      }
      iv = list;
    } else if (iv.isList()) {  // Tensor?[] (index, index_put_)
      auto list = iv.toList().copy();
      for (size_t j = 0; j < list.size(); ++j) {
        if (list.get(j).isTensor()) {
          list.set(j, to_cpu(list.get(j).toTensor(), write));
        }
      }
      iv = list;
    }
  }
  op.redispatchBoxed(c10::DispatchKeySet(DispatchKey::CPU), stack);
  for (auto& [arg, copy] : written) {
    auto out = arg;
    copy_into(out, copy);
  }
  const auto& returns = schema.returns();
  const auto rstart = stack->size() - returns.size();
  for (size_t i = 0; i < returns.size(); ++i) {
    auto& iv = (*stack)[rstart + i];
    const auto* alias = returns[i].alias_info();
    if (alias != nullptr) {  // in-place and out= ops return their written argument: the original
      TORCH_CHECK(alias->isWrite(), "hpu_complex: the view op ", schema.operator_name(),
                  " has no complex128 kernel on the HPU");
      for (size_t j = 0; j < num_args; ++j) {
        const auto* a = schema.arguments()[j].alias_info();
        if (a != nullptr && *a == *alias) {
          iv = originals[j];
        }
      }
    } else if (iv.isTensor()) {
      iv = from_cpu(iv.toTensor(), *device);
    } else if (iv.isTensorList()) {
      auto list = iv.toTensorList();
      std::vector<at::Tensor> moved;
      for (const at::Tensor& t : list) {
        moved.push_back(from_cpu(t, *device));
      }
      iv = c10::List<at::Tensor>(at::ArrayRef<at::Tensor>(moved));
    }
  }
}

void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  if (has_host_arg(op, stack)) {
    host_fallback(op, stack);
    return;
  }
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

// Structured ops (pow, _linalg_svd, _linalg_eigh, linalg_inv_ex, ...) have a CompositeExplicitAutogradNonFunctional
// kernel, which takes PrivateUse1's slot ahead of the fallback: it allocates the output from the input's options
// (float64 on the HPU: Habana storage at float32 precision) and calls the out= op. For host-backed arguments this
// kernel computes on the CPU instead (host_fallback, float64 kept in host memory); for any other tagged tensor it
// runs that composite kernel, read from PrivateUse3's slot, where nothing else is registered.
void structured_pu1(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  if (has_host_arg(op, stack)) {
    host_fallback(op, stack);
    return;
  }
  op.redispatchBoxed(DispatchKeySet(DispatchKey::PrivateUse3), stack);
}

// Registers structured_pu1 for every aten op whose PrivateUse1 slot holds the NonFunctional composite kernel.
struct RegisterStructured {
  RegisterStructured() {
    static auto* lib = new torch::Library(torch::Library::IMPL, "aten", DispatchKey::PrivateUse1, __FILE__, __LINE__);
    auto& dispatcher = c10::Dispatcher::singleton();
    for (const auto& name : dispatcher.getAllOpNames()) {
      if (name.name.rfind("aten::", 0) != 0) {
        continue;
      }
      auto op = dispatcher.findOp(name);
      if (!op || !op->hasKernelForDispatchKey(DispatchKey::CompositeExplicitAutogradNonFunctional) ||
          op->hasKernelForDispatchKey(DispatchKey::PrivateUse1) ||
          op->hasKernelForDispatchKey(DispatchKey::PrivateUse3)) {
        continue;
      }
      auto unqualified = name.name.substr(6) + (name.overload_name.empty() ? "" : "." + name.overload_name);
      lib->impl(unqualified.c_str(), torch::CppFunction::makeFromBoxedFunction<&structured_pu1>());
    }
  }
};

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
  m.impl("scalar_tensor", &scalar_tensor_backend_select);
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
  m.impl("view_as_complex", &view_as_complex_pu1);
  m.impl("empty_like", &empty_like_pu1);
  m.impl("clone", &clone_pu1);
  m.impl("_to_copy", &to_copy_pu1);
  m.impl("view.dtype", &view_dtype_pu1);
  m.impl("_neg_view", &neg_view_pu1);
  // ATen's own TensorImpl views (same key set and dtype, bounds-checked): no PrivateUse1 kernels upstream.
  m.impl("as_strided", TORCH_FN(at::native::as_strided_tensorimpl));
  m.impl("view", TORCH_FN(at::native::view));
  m.impl("_reshape_alias", TORCH_FN(at::native::_reshape_alias));
  // In-place restride (t_, transpose_, unsqueeze_, squeeze_ call it): sizes and strides change on the same storage,
  // so views of the tensor keep aliasing it. Its NonFunctional composite would otherwise reach structured_pu1, whose
  // host path computes on a copy and writes back into fresh storage.
  m.impl("as_strided_", TORCH_FN(at::native::as_strided__symint));
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

TORCH_LIBRARY_FRAGMENT(hpu_complex, m) {  // a fragment: Team C's branch may define the namespace too
  m.def("host_deepcopy(Tensor self, Tensor? storage_of=None) -> Tensor", &host_deepcopy);
}

// After the registrations above (same translation unit: initialized in order).
const RegisterStructured register_structured;

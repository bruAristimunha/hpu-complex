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
#include <ATen/MemoryOverlap.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/Operators.h>
#include <ATen/ops/set_native.h>
#include <ATen/ops/unfold_native.h>
#include <ATen/ops/_reshape_alias_native.h>
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/view_native.h>
#include <ATen/ops/_to_copy_native.h>
#include <ATen/ops/clone_native.h>
#include <ATen/ops/empty_like_native.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/view_as_complex_native.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TypeProperties.h>
#include <c10/core/InferenceMode.h>
#include <c10/core/impl/alloc_cpu.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/util/FunctionRef.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>

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

// keys for a tensor over base's storage, as torch keys a view or a result: an inference tensor (no autograd keys) when
// base is one (a view of it, or a fresh allocation under torch.inference_mode, which is one), else keys as given.
DispatchKeySet keys_over(DispatchKeySet keys, const at::Tensor& base) {
  return base.is_inference() ? keys - c10::autograd_dispatch_keyset_with_ADInplaceOrView : keys;
}

bool tagged(const at::Tensor& t) {
  return t.defined() && t.key_set().has(DispatchKey::PrivateUse1);
}

std::atomic<int>& lazy_mode_flag() {
  static std::atomic<int> flag{-1};
  return flag;
}

void set_lazy_mode(bool enabled) {
  lazy_mode_flag().store(enabled ? 1 : 0, std::memory_order_relaxed);
}

bool is_lazy_mode() {
  const int flag = lazy_mode_flag().load(std::memory_order_relaxed);
  if (flag >= 0) {
    return flag != 0;
  }
  const char* env = std::getenv("PT_HPU_LAZY_MODE");
  return env != nullptr && std::string_view(env) != "0";
}

// ---- lazy mode: the Habana tensor behind a complex storage -------------------------------
//
// In lazy mode (PT_HPU_LAZY_MODE=1) Habana's kernels take only tensors its own ops made: a bare TensorImpl over the
// same storage (view_of, the eager path of habana_view) raises "GetHbLazyTensor for a non lazy tensor". So a complex64
// tensor made in lazy mode gets a storage of its own whose DataPtr context holds the Habana float tensor it lives in
// (make_complex_empty's allocation, the result pair that complex_view tags, the real input of view_as_complex /
// view(complex64)), and the float views the kernels compute on are Habana's own as_strided views of that tensor: the
// lazy graph sees every read and write of a complex tensor as a view of one of its own tensors. That storage has the
// Habana storage's size; its data pointer is the context's address (overlap checks compare addresses; nothing reads
// through it), and it releases the Habana tensor when the last complex tensor over it goes.
struct LazyOwner {
  at::Tensor tensor;
};

std::atomic<int64_t> lazy_owners_alive{0};

// One lazy storage per Habana storage: complex tensors made separately over the same Habana storage (two
// view_as_complex of one r) share it, as torch's views share a StorageImpl, so a growth through one of them is seen by
// all (each kept its own capacity before) and torch's overlap checks see one storage.
struct LazyStorages {
  std::mutex mutex;
  std::map<const c10::StorageImpl*, std::pair<const LazyOwner*, c10::weak_intrusive_ptr<c10::StorageImpl>>> of;
};

LazyStorages& lazy_storages() {
  static auto* storages = new LazyStorages();  // never destroyed: storages may be freed during interpreter exit
  return *storages;
}

void lazy_owner_free(void* ctx) {
  auto* owner = static_cast<LazyOwner*>(ctx);
  {
    auto& storages = lazy_storages();
    std::lock_guard<std::mutex> lock(storages.mutex);
    auto it = storages.of.find(owner->tensor.storage().unsafeGetStorageImpl());
    if (it != storages.of.end() && it->second.first == owner) {
      storages.of.erase(it);
    }
  }
  delete owner;
  lazy_owners_alive.fetch_sub(1);
}

// The owner is a detached alias of the Habana tensor (its own TensorImpl, same Habana storage): r.set_(), r.data = ...
// or r.resize_() on the caller's tensor cannot retarget the complex tensor, and r's autograd graph is not kept alive.
LazyOwner* lazy_owner(const at::Tensor& owner) {
  lazy_owners_alive.fetch_add(1);
  return new LazyOwner{owner.detach()};
}

c10::Storage lazy_storage(const at::Tensor& owner) {
  auto& storages = lazy_storages();
  const auto* key = owner.storage().unsafeGetStorageImpl();
  std::lock_guard<std::mutex> lock(storages.mutex);
  auto it = storages.of.find(key);
  if (it != storages.of.end()) {
    if (auto alive = it->second.second.lock()) {
      // The Habana storage may have grown since (r.resize_() on the real tensor): the lazy storage reports its size.
      if (alive->nbytes() < owner.storage().nbytes()) {
        alive->set_nbytes(owner.storage().nbytes());
      }
      return c10::Storage(std::move(alive));
    }
  }
  auto* ctx = lazy_owner(owner);
  c10::Storage storage(c10::Storage::use_byte_size_t(), owner.storage().nbytes(),
                       c10::DataPtr(ctx, ctx, &lazy_owner_free, owner.device()), /*allocator=*/nullptr,
                       /*resizable=*/false);
  storages.of.insert_or_assign(key, std::make_pair(static_cast<const LazyOwner*>(ctx), storage.getWeakStorageImpl()));
  return storage;
}

at::Tensor lazy_base(const at::Tensor& t) {
  const auto& dp = t.storage().data_ptr();
  return dp.get_deleter() == &lazy_owner_free ? static_cast<LazyOwner*>(dp.get_context())->tensor : at::Tensor();
}

// The lazy-mode habana_view: a view of base's storage made by Habana's own ops on the storage's Habana tensor.
at::Tensor lazy_view(const at::Tensor& base, caffe2::TypeMeta dtype, c10::IntArrayRef sizes, c10::IntArrayRef strides,
                     int64_t storage_offset) {
  // A plain Habana tensor (dense_copy, complex_result of a float result) is its own owner.
  auto owner = tagged(base) ? lazy_base(base) : base;
  TORCH_CHECK(owner.defined(), "hpu_complex: in Habana lazy mode (PT_HPU_LAZY_MODE=1) this complex64 HPU tensor has no "
              "Habana tensor behind its storage (made in eager mode, or by set_ on a Habana storage): run in eager mode "
              "(PT_HPU_LAZY_MODE=0)");
  if (owner.dtype() != dtype) {
    TORCH_CHECK(owner.dtype().itemsize() == dtype.itemsize(), "hpu_complex: lazy mode: no ", dtype.name(),
                " view of a ", owner.dtype().name(), " storage");
    owner = owner.view(dtype.toScalarType());
  }
  // Always a new view (never owner itself): view ops (view_as_real, .real) return it, and autograd makes its result
  // a differentiable view, which a tensor with autograd metadata already (a user's input) cannot become.
  return owner.as_strided(sizes, strides, storage_offset);
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

// A complex64 view (tagged) of base's storage: base a tagged tensor (same storage), or a Habana float tensor (in lazy
// mode over a lazy_storage that holds it).
at::Tensor complex_of(const at::Tensor& base, c10::IntArrayRef sizes, c10::IntArrayRef strides, int64_t storage_offset) {
  const auto dtype = caffe2::TypeMeta::Make<c10::complex<float>>();
  if (!is_lazy_mode() || !base.is_hpu() || tagged(base)) {
    return view_of(base, keys_over(complex_keys(), base), dtype, sizes, strides, storage_offset);
  }
  auto result = at::detail::make_tensor<at::TensorImpl>(c10::TensorImpl::VIEW, lazy_storage(base),
                                                        keys_over(complex_keys(), base), dtype);
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
  // Lazy mode: the same layouts, built from the storage's Habana tensor (lazy_view); view_of only describes them.
  const bool lazy = is_lazy_mode() && base.is_hpu();
  auto view = view_of(base, keys_over(hpu_float_keys(), base), dtype, sizes, strides, storage_offset);
  bool expanded = false;
  for (size_t i = 0; i < sizes.size(); ++i) {
    expanded |= sizes[i] > 1 && strides[i] == 0;
  }
  if (!lazy && storage_offset == 0 && view.is_contiguous() && !expanded) {
    return view;
  }
  const auto base_numel = static_cast<int64_t>(base.storage().nbytes() / dtype.itemsize());
  if (view.dim() >= 6 && !expanded && view.is_non_overlapping_and_dense()) {
    // Lazy mode: Habana's lazy graphs read this permute block wrong when it starts at offset 0 (permuted 5-D complex:
    // errors up to 12, Gaudi2, bridge 1.21.4; at an offset, behind a narrow, they read it right): raise instead.
    TORCH_CHECK(!lazy || storage_offset != 0 || view.is_contiguous(), "hpu_complex: a permuted complex64 HPU tensor of rank 5 or more (a float "
                "pair of rank ", view.dim(), ") is not supported in Habana lazy mode (PT_HPU_LAZY_MODE=1): its lazy "
                "graph reads it wrong. Make it contiguous at rank 4 or less first, or run in eager mode "
                "(PT_HPU_LAZY_MODE=0).");
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
    if (storage_offset == 0 && !lazy) {
      return view_of(base, keys_over(hpu_float_keys(), base), dtype, block, block_strides, 0).permute(dims);
    }
    const int64_t view_numel = view.numel();
    const int64_t off = view_numel == 0 ? std::min(storage_offset, base_numel) : storage_offset;
    return (lazy ? lazy_view(base, dtype, {base_numel}, {1}, 0)
                 : view_of(base, keys_over(hpu_float_keys(), base), dtype, {base_numel}, {1}, 0))
        .narrow(0, off, view_numel)
        .view(block)
        .permute(dims);
  }
  if (lazy) {
    return lazy_view(base, dtype, sizes, strides, storage_offset);
  }
  const int64_t inner = base_numel % 2 == 0 ? 2 : 1;
  return at::as_strided(
      view_of(base, keys_over(hpu_float_keys(), base), dtype, {base_numel / inner, inner}, {inner, 1}, 0), sizes,
      strides, storage_offset);
}

// t with contiguous strides. Habana's own contiguous() misreads a dense, permuted HPU tensor of rank 6 or more
// that is the output of one of its ops on a permuted view (e.g. x.permute(...) * 1 or > 0; reproduced without
// hpu_complex, Gaudi2, bridge 1.21.4: repro/rank6_permuted_contiguous.py), while its elementwise ops, its D2H copy and a
// permute view over the same storage (habana_view) read it right: such a tensor is copied through that view.
at::Tensor dense_copy(const at::Tensor& t) {
  if (t.is_contiguous()) {
    return t;
  }
  if (t.is_hpu() && t.dim() >= 6 && t.is_non_overlapping_and_dense()) {
    return habana_view(t, t.dtype(), t.sizes(), t.strides(), t.storage_offset()).contiguous();
  }
  return t.contiguous();
}

// float HPU [..., k, 2] (last stride 1) -> tagged complex64 [..., k], same storage.
at::Tensor complex_view(const at::Tensor& pair) {
  TORCH_CHECK(pair.scalar_type() == at::kFloat && pair.size(-1) == 2 && (pair.stride(-1) == 1 || pair.numel() == 0),
              "hpu_complex: a complex view needs float32 [..., 2] with last stride 1");
  auto sizes = pair.sizes().vec();
  sizes.pop_back();
  if (pair.numel() == 0) {  // no element to address: the strides are free, kept contiguous
    return complex_of(pair, sizes, c10::contiguous_strides(sizes), 0);
  }
  auto strides = pair.strides().vec();
  strides.pop_back();
  for (size_t i = 0; i < strides.size(); ++i) {
    // A size-1 dim's stride addresses nothing, and contiguous() keeps an odd one (a transposed [..., 1, 2] is
    // already contiguous): any complex stride will do there.
    if (sizes[i] == 1 && strides[i] % 2 != 0) {
      strides[i] = 1;
      continue;
    }
    TORCH_CHECK(strides[i] % 2 == 0, "hpu_complex: odd float stride cannot be a complex stride");
    strides[i] /= 2;
  }
  TORCH_CHECK(pair.storage_offset() % 2 == 0, "hpu_complex: odd storage offset");
  return complex_of(pair, sizes, strides, pair.storage_offset() / 2);
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
  return complex_of(flat, size, stride, 0);
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

// Set at the first host allocation: from then on float64 tensors of the process may be host-backed, so
// torch.compile(hpu_backend) must not fuse float64 nodes into Synapse graphs (hpu_complex.__init__ asks it).
std::atomic<bool> host_used{false};

bool host_in_use() {
  return host_used.load();
}

// The key set of a host-backed tensor: an inference tensor under torch.inference_mode, as TensorImpl's own
// constructor makes one (the VIEW constructor used here keeps the key set as given).
DispatchKeySet host_keys() {
  return c10::InferenceMode::is_enabled() ? complex_keys() - c10::autograd_dispatch_keyset_with_ADInplaceOrView
                                          : complex_keys();
}

at::Tensor make_host_empty(c10::IntArrayRef size, c10::IntArrayRef stride, caffe2::TypeMeta dtype,
                           const at::Device& device) {
  TORCH_WARN_ONCE("hpu_complex: complex128 HPU tensors are kept in host memory and computed on the CPU in float64 "
                  "(exact; Gaudi2 holds float64 at float32 precision). Use complex64 to compute on the card.");
  host_used.store(true);
  const auto nbytes = at::detail::computeStorageNbytes(size, stride, dtype.itemsize());
  void* data = c10::alloc_cpu(std::max<size_t>(nbytes, 1));
  c10::Storage storage(c10::Storage::use_byte_size_t(), nbytes, c10::DataPtr(data, data, &host_free, hpu_index(device)),
                       /*allocator=*/nullptr, /*resizable=*/false);
  auto t = at::detail::make_tensor<at::TensorImpl>(c10::TensorImpl::VIEW, std::move(storage), host_keys(), dtype);
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

// torch's device rule (TensorIterator): every operand on one device, index included (hpu:0 and hpu:1 raise), except
// CPU 0-d tensors (wrapped Python scalars, CPU scalar tensors).
void check_host_devices(std::initializer_list<const at::Tensor*> operands) {
  std::optional<at::Device> device;
  for (const auto* t : operands) {
    if (t->defined() && !t->is_cpu()) {
      TORCH_CHECK(!device || *device == t->device(), "Expected all tensors to be on the same device, but found at "
                  "least two devices, ", *device, " and ", t->device(), "!");
      device = t->device();
    }
  }
  for (const auto* t : operands) {
    TORCH_CHECK(!t->defined() || !t->is_cpu() || t->dim() == 0, "Expected all tensors to be on the same device, but "
                "found at least two devices, ", device.value_or(at::Device(at::kHPU, 0)), " and cpu!");
  }
}

// torch's result type of a binary op is complex128 while neither operand is host-backed (complex64 with a float64 HPU
// tensor): computed on the CPU in complex128 and kept in host memory, as torch promotes (the card pair kernels would
// round the float64 operand to float32 and return complex64). Not for a 0-d CPU operand: autograd hands saved Python
// scalars back to the backward as such tensors (complex128, no longer wrapped numbers), so every z * 2j in a model
// would send its gradient through host memory and a device sync (lazy Green: 64 -> 76 ms a step, measured); its
// engine casts that gradient back to complex64 anyway. Those stay complex64 on the card, as before (README Limits).
// A ComplexHalf (complex32) result type (a 0-d complex64 with a float16 tensor) raises: there is no complex32 on the
// HPU, and the pair kernels would return complex64.
void check_not_complex32(const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(at::native::result_type(a, b) != at::kComplexHalf, "hpu_complex: the result type of ", a.scalar_type(),
              " and ", b.scalar_type(), " is ComplexHalf (complex32), which is not supported on HPU: cast to complex64 "
              "first");
}

// Called first by every binary kernel (before its host path too: a 0-d complex128 with a float16 vector is complex32).
bool promotes_to_complex128(const at::Tensor& a, const at::Tensor& b) {
  check_not_complex32(a, b);
  const auto cpu_scalar = [](const at::Tensor& t) { return t.defined() && t.is_cpu() && t.dim() == 0; };
  return !cpu_scalar(a) && !cpu_scalar(b) && at::native::result_type(a, b) == at::kComplexDouble;  // no dispatch
}

// TensorIterator's check on a written argument: the result dtype must cast to it (a complex result into a real
// tensor raises; it does not drop the imaginary part).
void check_can_cast(at::ScalarType result, const at::Tensor& out) {
  TORCH_CHECK(at::canCast(result, out.scalar_type()), "result type ", result,
              " can't be cast to the desired output type ", out.scalar_type());
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
  if (host_backed(self) && dtype.value_or(self.scalar_type()) == at::kDouble && (!device || device->is_hpu())) {
    auto like = at::empty_like(host_cpu(self), dtype, layout, at::kCPU, pin_memory, memory_format);
    return make_host_empty(like.sizes(), like.strides(), like.dtype(), device ? hpu_index(*device) : self.device());
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
  if (host_backed(self) && (!device || device->is_hpu())) {  // hpu:N kept: a host copy reporting that device
    return from_cpu(at::_to_copy(host_cpu(self), dtype, layout, std::nullopt, pin_memory, non_blocking, memory_format),
                    device ? hpu_index(*device) : self.device());
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
  auto out = at::detail::make_tensor<at::TensorImpl>(c10::TensorImpl::VIEW, std::move(storage), host_keys(),
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
  if (complex_on_hpu(dtype, ks) || complex128_on_hpu(dtype, ks)) {
    auto sizes = C10_AS_INTARRAYREF_SLOW(size);
    // memory_format's strides (channels_last, ...), as torch's empty gives them
    std::vector<int64_t> strides;
    if (memory_format == at::MemoryFormat::ChannelsLast || memory_format == at::MemoryFormat::ChannelsLast3d) {
      strides = at::empty(sizes, at::TensorOptions().dtype(*dtype).device(at::kMeta), memory_format).strides().vec();
    } else {
      const auto contiguous = c10::contiguous_strides(sizes);
      strides.assign(contiguous.begin(), contiguous.end());
    }
    const auto dev = device ? *device : at::Device(at::kHPU);
    if (*dtype == at::kComplexDouble) {
      return make_host_empty(sizes, strides, caffe2::TypeMeta::Make<c10::complex<double>>(), dev);
    }
    return make_complex_empty(sizes, strides, dev);
  }
  return at::_ops::empty_memory_format::redispatch(ks, size, dtype, layout, device, pin_memory, memory_format);
}

at::Tensor empty_strided_backend_select(c10::SymIntArrayRef size, c10::SymIntArrayRef stride,
                                        std::optional<at::ScalarType> dtype, std::optional<at::Layout> layout,
                                        std::optional<at::Device> device, std::optional<bool> pin_memory) {
  auto ks = factory_keys(dtype, layout, device);
  if (complex_on_hpu(dtype, ks)) {
    const auto dev = device ? *device : at::Device(at::kHPU);
    return make_complex_empty(C10_AS_INTARRAYREF_SLOW(size), C10_AS_INTARRAYREF_SLOW(stride), dev);
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
  if (complex_on_hpu(dtype, ks)) {  // lazy mode: Habana's own scalar_tensor kernel would get the complex dtype
    auto t = make_complex_empty({}, {}, *device);
    auto pair = pair_view(t);
    const auto c = s.toComplexDouble();
    pair.select(-1, 0).fill_(c.real());
    pair.select(-1, 1).fill_(c.imag());
    return t;
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

// The public entry rejects what torch rejects (rank 0, last dim not 2 or not stride 1, an odd stride or offset,
// with torch's messages): torch's own kernel computes and checks the layout, which is then tagged. complex_view,
// lenient about size-1 and empty dims, stays for the extension's own results.
at::Tensor view_as_complex_hpu(const at::Tensor& self) {
  if (self.scalar_type() == at::kDouble) {
    // Habana's float64 storage holds float32 values in its own layout: no complex128 view of it exists. The copy is
    // host memory.
    TORCH_WARN_ONCE("hpu_complex: view_as_complex of a float64 HPU tensor returns a complex128 copy, not a view: "
                    "writes to it do not reach the float64 tensor");
    return from_cpu(at::view_as_complex(self.cpu()), self.device());
  }
  auto v = at::native::view_as_complex(self);
  TORCH_CHECK(self.scalar_type() == at::kFloat, "hpu_complex: view_as_complex on HPU supports float32 (complex64) "
              "only, got ", self.scalar_type());
  return complex_of(self, v.sizes(), v.strides(), v.storage_offset());
}

// x.view(torch.complex64) of a real HPU tensor: torch's view_dtype (CompositeExplicitAutograd) would keep the HPU
// keys on a complex dtype, a tensor no kernel here sees ("Complex datatype is not supported on HPU device" at its
// first op). Its layout, checked by torch, is tagged instead; real-to-real views are torch's.
at::Tensor view_dtype_hpu(const at::Tensor& self, at::ScalarType dtype) {
  if (!c10::isComplexType(dtype) || self.is_complex()) {
    return at::native::view_dtype(self, dtype);
  }
  TORCH_CHECK(dtype == at::kComplexFloat, "hpu_complex: complex64 only on HPU, got ", dtype);
  auto v = at::native::view_dtype(self, dtype);
  return complex_of(self, v.sizes(), v.strides(), v.storage_offset());
}

// The key set of a view of a host-backed tensor: an inference tensor's views are inference tensors (no autograd keys),
// as torch's views keep their base's keys.
DispatchKeySet keys_like(const at::Tensor& self) {
  return self.key_set().has(DispatchKey::AutogradPrivateUse1)
             ? complex_keys()
             : complex_keys() - c10::autograd_dispatch_keyset_with_ADInplaceOrView;
}

// The complex view of a host-backed real [..., 2] tensor (view_as_real of a complex128 one, or a slice of it; float32
// for a complex64 view of complex128 memory). Host memory stays host memory, whatever the dtype.
at::Tensor view_as_complex_pu1(const at::Tensor& self) {
  TORCH_CHECK(host_backed(self) && (self.scalar_type() == at::kDouble || self.scalar_type() == at::kFloat),
              "hpu_complex: view_as_complex of a tagged ", self.scalar_type(), " tensor");
  auto c = at::view_as_complex(host_cpu(self));  // torch's checks (last dim 2, stride 1, even strides and offset)
  return view_of(self, keys_like(self), c.dtype(), c.sizes(), c.strides(), c.storage_offset());
}

at::Tensor view_as_real_pu1(const at::Tensor& self) {
  if (host_backed(self)) {
    auto r = at::view_as_real(host_cpu(self));  // torch's checks (no unresolved conjugate)
    return view_of(self, keys_like(self), r.dtype(), r.sizes(), r.strides(), r.storage_offset());
  }
  return pair_view(self);
}

// z.view(dtype): the stock kernel copies the PrivateUse1 key onto the new dtype, and nothing here computes on a
// tagged real tensor (its float ops would come straight back): a real view gets the plain HPU keys, as
// view_as_real does. Host memory stays tagged whatever the dtype (z128.view(complex64) is complex64 in host memory,
// computed on the CPU); card storage cannot hold a complex128 or float64 view (complex128 lives in host memory, and
// Habana reads float64 at float32 precision), so those raise instead of computing on a misclassified tensor.
at::Tensor view_dtype_pu1(const at::Tensor& self, at::ScalarType dtype) {
  TORCH_CHECK(host_backed(self) || (dtype != at::kComplexDouble && dtype != at::kDouble),
              "hpu_complex: a ", dtype, " view of complex64 HPU storage is not supported: ", dtype, " HPU tensors "
              "live in host memory (complex128) or at float32 precision on the card (float64). Use .to(", dtype,
              ") for a copy");
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
  return at::stack({dense_copy(parts[0]), dense_copy(parts[1])}, -1);
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
  auto src = t.dim() >= 5 ? dense_copy(t) : t;
  auto r = src.to(device, at::kFloat);
  return r.dim() >= 5 ? dense_copy(r) : r;
}

// [..., 2] float32 pair of an operand: complex -> its pair (on the HPU), real -> nullopt.
std::optional<at::Tensor> pair_of(const at::Tensor& t, const at::Device& device) {
  if (!t.is_complex()) {
    return std::nullopt;
  }
  auto p = operand_pair(t);
  return p.device() == device && p.scalar_type() == at::kFloat ? p : p.to(device, at::kFloat);
}

// The device a pair kernel computes on: the complex operand's, else the device operand's. Never a CPU 0-d scalar's
// (x_hpu * tensor(2j) through Habana's mul.out reaches mul_out_pu1 with a real HPU a and a CPU complex b: that value was
// computed on the CPU and copied into out).
at::Device hpu_device_of(const at::Tensor& a, const at::Tensor& b) {
  if (tagged(a)) {
    return a.device();
  }
  if (tagged(b)) {
    return b.device();
  }
  return a.is_cpu() ? b.device() : a.device();
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
  check_host_devices({&a, &b});
  if (promotes_to_complex128(a, b) || any_host(a, b)) {
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

// a / b as torch's c10::complex division (Smith's algorithm, numpy's): the larger part of b is divided out first, so
// |b|^2 is never formed (1 / 1e20j was 0 and 1 / 1e-30j nan - inf i with the |b|^2 form). A real operand is torch's
// x + 0i, so NaN and signed-zero placement follow torch too; b = 0 gives a / 0 per part (complex inf or NaN).
at::Tensor smith_div(const at::Tensor& ar, const at::Tensor& ai, const at::Tensor& br, const at::Tensor& bi) {
  const auto abs_r = br.abs(), abs_i = bi.abs();
  const auto real_major = abs_r >= abs_i;
  const auto rat = at::where(real_major, bi / br, br / bi);
  const auto scl = at::where(real_major, br + bi * rat, bi + br * rat).reciprocal();
  auto re = at::where(real_major, ar + ai * rat, ar * rat + ai) * scl;
  auto im = at::where(real_major, ai - ar * rat, ai * rat - ar) * scl;
  const auto zero = (abs_r == 0).logical_and(abs_i == 0);
  re = at::where(zero, ar / abs_r, re);
  im = at::where(zero, ai / abs_i, im);
  return at::stack({re, im}, -1);
}

// (ar + i ai) / (c + 0i): ar / c + i ai / c, plus torch's cross terms ai * (0 / c) and -ar * (0 / c), which are zeros
// for a finite part and NaN for an infinite one: (inf + 1i) / 2 is inf + nan i in torch (c10::complex's Smith
// division), where the per-part quotient gave inf + 0.5i. Two ops more than the per-part quotient: the exact form
// (Smith with a zero imaginary part, 13 ops) cost lazy Green 18 ms a step (Gaudi2, bridge 1.21.4; Green divides its
// wavelets by their norm, forward and backward). Left as before: zero signs where a quotient part is exactly 0, a
// zero c next to an infinite other part (NaN here, torch inf), and the last ulp (torch multiplies by 1 / c).
// The cross terms as (ai, ar) * 0: no constant tensor (its host-to-device copy synced every eager call: +30 ms an
// eager Green step), at the cost of im + ar * 0 for torch's im - ar * 0, a zero sign (left as before, above).
at::Tensor real_div(const at::Tensor& pair, const at::Tensor& c) {
  return pair / c.unsqueeze(-1) + pair.flip(-1) * 0.0;
}

at::Tensor div_pu1(const at::Tensor& a, const at::Tensor& b) {
  check_host_devices({&a, &b});
  if (promotes_to_complex128(a, b) || any_host(a, b)) {
    return from_cpu(at::div(cpu_operand(a), cpu_operand(b)), device_of(a, b));
  }
  HPU_COMPLEX_NO_AUTOCAST;
  const auto device = hpu_device_of(a, b);
  auto pa = pair_of(a, device);
  auto pb = pair_of(b, device);
  if (both_real(pa, pb, a, b)) {
    return at::div(a, b);
  }
  if (!pb) {  // complex / real: torch's (ar + i ai) / (c + 0i), Smith's algorithm with a zero imaginary part
    return complex_result(real_div(*pa, hpu_float(b, device)));
  }
  auto br = pb->select(-1, 0), bi = pb->select(-1, 1);
  if (!pa) {
    auto ar = hpu_float(a, device);
    return complex_result(smith_div(ar, at::zeros_like(ar), br, bi));
  }
  return complex_result(smith_div(pa->select(-1, 0), pa->select(-1, 1), br, bi));
}

// a + alpha b as torch computes it: alpha b is a complex product even for alpha = 1 (so inf + 0i in b gives
// inf + nan i, as on the CPU), a real operand is x + 0i, and alpha may be complex.
// The add/sub kernel's own checks of alpha (alpha_check, then its cast to the complex64 compute type): a bool alpha
// for a non-bool result raises, and so does one that overflows complex64 (1e39), before anything is written.
c10::complex<float> complex_alpha(const at::Scalar& alpha) {
  TORCH_CHECK(!alpha.isBoolean(), "Boolean alpha only supported for Boolean results.");
  return alpha.to<c10::complex<float>>();
}

// a + alpha b, torch's checks first; sub_pu1 passes its alpha negated (already checked).
at::Tensor add_pairs(const at::Tensor& a, const at::Tensor& b, const at::Scalar& alpha);

at::Tensor add_pu1(const at::Tensor& a, const at::Tensor& b, const at::Scalar& alpha) {
  check_host_devices({&a, &b});
  if (promotes_to_complex128(a, b) || any_host(a, b)) {
    return from_cpu(at::add(cpu_operand(a), cpu_operand(b), alpha), device_of(a, b));
  }
  return add_pairs(a, b, alpha);
}

at::Tensor add_pairs(const at::Tensor& a, const at::Tensor& b, const at::Scalar& alpha) {
  HPU_COMPLEX_NO_AUTOCAST;
  const auto device = hpu_device_of(a, b);
  auto pa = pair_of(a, device);
  auto pb = pair_of(b, device);
  if (both_real(pa, pb, a, b)) {
    return at::add(a, b, alpha);
  }
  const auto s = complex_alpha(alpha);
  const auto sr = static_cast<double>(s.real());
  const auto si = static_cast<double>(s.imag());
  // alpha b = (sr br - si bi) + i (sr bi + si br), with bi = 0 for a real b (then sr bi and si bi are the signed zeros
  // 0 sr and 0 si: x - (+0) is x, x - (-0) is x + 0, -0 + x is x, +0 + x is x + 0).
  at::Tensor kr, ki;
  if (pa && pb && !alpha.isComplex() && !a.is_cpu() && !b.is_cpu()) {
    // Two complex tensors on the card, a real alpha (gradient accumulation, sub): pair arithmetic, one or two ops.
    // torch's alpha * b differs from s b only where b has an inf or NaN part (inf + 0i in b gives inf + nan i there)
    // and in the sign of an exact-zero sum; the exact form below cost 25 ms of a 75 ms lazy Green step (Gaudi2,
    // bridge 1.21.4). A Python scalar, a real operand and a complex alpha (1 + 0j included) keep torch's exact form.
    return complex_result(sr == 1.0 ? *pa + *pb : *pa + *pb * sr);
  }
  const auto br = pb ? pb->select(-1, 0) : hpu_float(b, device);
  if (pb) {
    const auto bi = pb->select(-1, 1);
    kr = br * sr - bi * si;
    ki = bi * sr + br * si;
  } else if (std::isfinite(sr) && std::isfinite(si)) {
    kr = br * sr;
    ki = br * si;
    if (std::signbit(0.0f * static_cast<float>(si))) {
      kr = kr + 0.0;
    }
    if (!std::signbit(0.0f * static_cast<float>(sr))) {
      ki = ki + 0.0;
    }
  } else {
    const auto bi = at::zeros_like(br);
    kr = br * sr - bi * si;
    ki = bi * sr + br * si;
  }
  if (pa) {
    return complex_result(stack_pair(pa->select(-1, 0) + kr, pa->select(-1, 1) + ki));
  }
  return complex_result(stack_pair(hpu_float(a, device) + kr, ki + 0.0));  // a real a is a + 0i
}

at::Tensor sub_pu1(const at::Tensor& a, const at::Tensor& b, const at::Scalar& alpha) {
  check_host_devices({&a, &b});
  if (promotes_to_complex128(a, b) || any_host(a, b)) {
    return from_cpu(at::sub(cpu_operand(a), cpu_operand(b), alpha), device_of(a, b));
  }
  if (a.is_complex() || b.is_complex()) {
    complex_alpha(alpha);  // the checks of torch's alpha, not of its negation (a bool alpha would become -1)
  } else if (both_real(std::nullopt, std::nullopt, a, b)) {
    return at::sub(a, b, alpha);  // out= of two real tensors: torch's own op and checks
  }
  return add_pairs(a, b, alpha.isComplex() ? at::Scalar(-alpha.toComplexDouble()) : at::Scalar(-alpha.toDouble()));
}

// |z| = hypot(re, im), scaled by m = max(|re|, |im|) as C's hypot: Habana's own vector_norm and hypot square in float
// (|3e19 + 3e19j| = inf, |1e-30| = 0 on Gaudi2, bridge 1.21.4, where torch gives 4.2e19 and 1e-30). m = 0 gives 0;
// an infinite part gives inf even next to a NaN (hypot(inf, nan) = inf, as torch's abs).
at::Tensor modulus(const at::Tensor& pair, bool keepdim) {
  auto mag = pair.abs();
  auto m = std::get<0>(mag.max(-1, /*keepdim=*/true));
  auto scale = at::where(m > 0, m, at::ones_like(m));
  auto r = m * at::linalg_vector_norm(pair / scale, 2, at::IntArrayRef{-1}, /*keepdim=*/true);
  const auto inf = std::numeric_limits<float>::infinity();
  r = at::where((mag == inf).any(-1, /*keepdim=*/true), at::Scalar(inf), r);
  return keepdim ? r : r.squeeze(-1);
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
  // Device tensors on one device, index included (a mask or operand on hpu:1 raises). CPU masks and real CPU operands
  // of any shape stay accepted on the card path, as before (pu1_check's rank-6 CPU rows; copied to the card, where
  // torch would raise for a non-0-d one: README Limits); the host path keeps torch's rule.
  std::optional<at::Device> dev;
  for (const auto* t : {&a, &b, &cond}) {
    if (t->defined() && !t->is_cpu()) {
      TORCH_CHECK(!dev || *dev == t->device(), "Expected all tensors to be on the same device, but found at least two "
                  "devices, ", *dev, " and ", t->device(), "!");
      dev = t->device();
    }
  }
  if (promotes_to_complex128(a, b) || any_host(a, b)) {
    check_host_devices({&a, &b});
    return from_cpu(at::where(cpu_operand(cond), cpu_operand(a), cpu_operand(b)), device_of(a, b));
  }
  HPU_COMPLEX_NO_AUTOCAST;
  const auto device = hpu_device_of(a, b);
  auto cond_dev = (cond.dim() >= 5 ? dense_copy(cond) : cond).to(device);
  if (cond_dev.dim() >= 5) {
    cond_dev = dense_copy(cond_dev);
  }
  return complex_result(at::where(cond_dev.unsqueeze(-1), as_pair(a, device), as_pair(b, device)));
}

// ---- matrix products (PrivateUse1) ----------------------------------------------------------

// Below this many complex multiply-adds a product runs faster on the CPU, one copy each way, than as the card's
// GEMM and its layout copies (launch-bound, ~0.3-1 ms; Gaudi2, bridge 1.21.4: tests/cmatmul_bench.py --sweep).
std::atomic<int64_t>& matmul_cpu_below() {
  static std::atomic<int64_t> macs{int64_t{1} << 22};
  return macs;
}

void set_matmul_cpu_threshold(int64_t macs) {
  TORCH_CHECK(macs >= 0, "hpu_complex: the threshold is a count of multiply-adds, got ", macs);
  matmul_cpu_below().store(macs, std::memory_order_relaxed);
}

// a @ b (mm / bmm shapes) of two complex64 tensors, as one real GEMM on the card:
// [ar | ai] @ [[br, bi], [-bi, br]] = [ar br - ai bi | ar bi + ai br] = [cr | ci]. The operand layouts are copied
// contiguous first (Habana GEMMs mis-handle strided inputs, see the rfft backward). Mixed dtypes raise, as in torch.
at::Tensor matmul_pu1(const at::Tensor& a, const at::Tensor& b) {
  HPU_COMPLEX_NO_AUTOCAST;
  TORCH_CHECK(a.scalar_type() == b.scalar_type(), "expected m1 and m2 to have the same dtype, but got: ",
              a.scalar_type(), " != ", b.scalar_type());
  TORCH_CHECK(a.device() == b.device(), "Expected all tensors to be on the same device, but found at least two "
              "devices, ", a.device(), " and ", b.device(), "!");
  if (any_host(a, b)) {  // complex128 (or host float64): the CPU's own GEMM in float64, never the card's float32 one
    return from_cpu(at::matmul(cpu_operand(a), cpu_operand(b)), device_of(a, b));
  }
  const auto device = hpu_device_of(a, b);
  const int64_t macs = a.numel() * b.size(-1);  // batch * m * k * n
  if (macs < matmul_cpu_below().load(std::memory_order_relaxed)) {
    TORCH_WARN_ONCE("hpu_complex: complex mm/bmm below the CPU threshold (default 2^22 multiply-adds; first one: ",
                    macs, ") runs on the CPU, where it is faster than the card's GEMM "
                    "(hpu_complex.set_matmul_cpu_threshold(0) keeps every product on the card)");
    // Copy through raw_pair (not a.to(kCPU) directly on a Conjugate view): in Habana lazy mode, copying a transposed
    // conjugate view via PrivateUse1 copy_ while another lazy result is live corrupts or triggers strided_insert.
    auto to_cpu_resolved = [](const at::Tensor& t) {
      auto c = at::view_as_complex(dense_copy(raw_pair(t)).to(at::kCPU));
      if (t.is_conj()) {
        c = c.conj().resolve_conj();
      }
      if (t.is_neg()) {
        c = c.neg();
      }
      return c;
    };
    return at::matmul(to_cpu_resolved(a), to_cpu_resolved(b)).to(device);
  }
  // (..., r, c) complex -> (..., r, 2c) float = [re | im] per row. mm/bmm fall through the Conjugate key (BLAS folds
  // the conjugation in), so a conjugate view arrives unresolved: raw_pair gives its stored values, and the
  // imaginary half is negated here.
  auto rows = [&](const at::Tensor& t) {
    auto p = raw_pair(t);
    p = p.device() == device && p.scalar_type() == at::kFloat ? p : p.to(device, at::kFloat);
    auto s = p.sizes().vec();
    s.pop_back();
    const int64_t c = s.back();
    s.back() = 2 * c;
    auto r = p.transpose(-1, -2).contiguous().view(s);
    return t.is_conj() ? at::cat({r.narrow(-1, 0, c), r.narrow(-1, c, c).neg()}, -1) : r;
  };
  const int64_t n = b.size(-1);
  auto top = rows(b);                                                                   // [br | bi]
  auto bottom = at::cat({top.narrow(-1, n, n).neg(), top.narrow(-1, 0, n)}, -1);        // [-bi | br]
  auto out = at::matmul(rows(a), at::cat({top, bottom}, -2));                           // (..., m, 2n) = [cr | ci]
  auto os = out.sizes().vec();
  os.back() = n;
  os.insert(os.end() - 1, 2);  // (..., m, 2, n)
  return complex_result(out.view(os).transpose(-1, -2).contiguous());
}

at::Tensor mm_pu1(const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(0), "hpu_complex: mm shapes ", a.sizes(), " @ ",
              b.sizes());
  return matmul_pu1(a, b);
}

at::Tensor bmm_pu1(const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(a.dim() == 3 && b.dim() == 3 && a.size(0) == b.size(0) && a.size(2) == b.size(1),
              "hpu_complex: bmm shapes ", a.sizes(), " @ ", b.sizes());
  return matmul_pu1(a, b);
}

// out variants: compute, then resize and copy into out (complex or real).
at::Tensor& copy_into(at::Tensor& out, const at::Tensor& value);

// An out= tensor as torch checks it: no two of its elements at one address (an expanded out), and on the device of
// the value, index included (the inputs' device: copy_into would copy a result across devices). mm.out/bmm.out
// (strict, matrix_mm=true) also need out of the operands' dtype: copy_into would write the real part into a float out.
void check_out(const at::Tensor& out, const at::Tensor& value, bool matrix_mm) {
  at::assert_no_internal_overlap(out);
  TORCH_CHECK(out.device() == value.device(), "Expected all tensors to be on the same device, but found at least two "
              "devices, ", value.device(), " and ", out.device(), "!");
  if (!matrix_mm) {
    return;
  }
  TORCH_CHECK(out.scalar_type() == value.scalar_type(), "Expected out tensor to have dtype ", value.dtype(),
              ", but got ", out.dtype(), " instead");
}

void check_lazy_overlap(const at::Tensor& self, const at::Tensor& src, bool any_overlap);

// TensorIterator's alias check of a written tensor against one input: a partial overlap raises (the kernels compute
// the value first and copy it, so the CPU's in-place order cannot be reproduced); with any_overlap (cat's rule) a full
// one too. Every input is checked, tagged or not: a real view of out's storage (out.view(float32), out.real,
// view_as_real) is an untagged Habana view of the same storage. Host tensors' CPU views have separate storages and never
// reach a kernel as inputs.
void check_written_against(const at::Tensor& out, const at::Tensor& t, bool any_overlap) {
  if (!tagged(out) || !t.defined()) {
    return;
  }
  if (any_overlap) {
    at::assert_no_overlap(out, t);
  } else {
    at::assert_no_partial_overlap(out, t);
  }
  check_lazy_overlap(out, t, any_overlap);
}

void check_written_overlap(const at::Tensor& out, std::initializer_list<const at::Tensor*> inputs) {
  for (const auto* t : inputs) {
    check_written_against(out, *t, /*any_overlap=*/false);
  }
}

// An out= write, torch's checks first: an expanded out, a result dtype that cannot be cast to out's (a complex result
// into a real out), out partly overlapping an input; then torch's resize warning, the resize and the copy.
at::Tensor& write_out(at::Tensor& out, const at::Tensor& value, std::initializer_list<const at::Tensor*> inputs,
                      bool matrix_mm = false) {
  check_out(out, value, matrix_mm);
  check_can_cast(value.scalar_type(), out);
  check_written_overlap(out, inputs);
  at::native::resize_output_check(out, value.sizes());  // torch's warning when a non-empty out changes shape
  return copy_into(out, value);
}

at::Tensor& host_out(at::Tensor& out, const at::Tensor& value, std::initializer_list<const at::Tensor*> inputs,
                     bool matrix_mm = false) {
  return write_out(out, value, inputs, matrix_mm);
}

at::Tensor& mm_out_pu1(const at::Tensor& a, const at::Tensor& b, at::Tensor& out) {
  return host_out(out, mm_pu1(a, b), {&a, &b}, /*matrix_mm=*/true);
}
at::Tensor& bmm_out_pu1(const at::Tensor& a, const at::Tensor& b, at::Tensor& out) {
  return host_out(out, bmm_pu1(a, b), {&a, &b}, /*matrix_mm=*/true);
}

at::Tensor& mul_out_pu1(const at::Tensor& a, const at::Tensor& b, at::Tensor& out) {
  return host_out(out, mul_pu1(a, b), {&a, &b});
}
at::Tensor& div_out_pu1(const at::Tensor& a, const at::Tensor& b, at::Tensor& out) {
  return host_out(out, div_pu1(a, b), {&a, &b});
}
at::Tensor& add_out_pu1(const at::Tensor& a, const at::Tensor& b, const at::Scalar& alpha, at::Tensor& out) {
  return host_out(out, add_pu1(a, b, alpha), {&a, &b});
}
at::Tensor& sub_out_pu1(const at::Tensor& a, const at::Tensor& b, const at::Scalar& alpha, at::Tensor& out) {
  return host_out(out, sub_pu1(a, b, alpha), {&a, &b});
}
// abs/sgn/vector_norm of a complex tensor: torch's out dtype is the value's (a float out for abs, exactly).
at::Tensor& abs_out_pu1(const at::Tensor& self, at::Tensor& out) {
  return host_out(out, abs_pu1(self), {&self});
}
at::Tensor& sgn_out_pu1(const at::Tensor& self, at::Tensor& out) {
  return host_out(out, sgn_pu1(self), {&self});
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
  check_written_overlap(self, {&other});  // z[1:].mul_(z[:-1]) raises, as TensorIterator does
  if (host_backed(self)) {
    check_host_devices({&self, &other});
    return host_inplace(self, [&](at::Tensor& s) { s.mul_(cpu_operand(other)); });
  }
  if (host_backed(other)) {  // complex64 self: computed in complex128, cast back as torch does; a real self raises
    check_host_devices({&self});
    auto value = at::mul(cpu_operand(self), cpu_operand(other));
    check_can_cast(value.scalar_type(), self);
    return assign_inplace(self, value.to(self.scalar_type()));
  }
  if (promotes_to_complex128(self, other)) {  // complex64 self, float64 other: computed in complex128, cast back
    check_host_devices({&self, &other});
    auto value = at::mul(cpu_operand(self), cpu_operand(other));
    check_can_cast(value.scalar_type(), self);  // a float64 self of a complex other raises, as torch
    return assign_inplace(self, value.to(self.scalar_type()));
  }
  return assign_inplace(self, mul_pu1(self, other));
}
at::Tensor& add__pu1(at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  check_written_overlap(self, {&other});
  if (host_backed(self)) {
    check_host_devices({&self, &other});
    return host_inplace(self, [&](at::Tensor& s) { s.add_(cpu_operand(other), alpha); });
  }
  if (host_backed(other)) {
    check_host_devices({&self});
    auto value = at::add(cpu_operand(self), cpu_operand(other), alpha);
    check_can_cast(value.scalar_type(), self);
    return assign_inplace(self, value.to(self.scalar_type()));
  }
  if (promotes_to_complex128(self, other)) {
    check_host_devices({&self, &other});
    auto value = at::add(cpu_operand(self), cpu_operand(other), alpha);
    check_can_cast(value.scalar_type(), self);
    return assign_inplace(self, value.to(self.scalar_type()));
  }
  return assign_inplace(self, add_pu1(self, other, alpha));
}

// ---- a real HPU tensor and a complex scalar (AutogradHPU) ---------------------------------
//
// x * 2j, 2j * x, x + 1j, x - 1j, 1j - x, x / 2j (spd_learn's Gabor wavelets: exp(2j * pi * f * t)) reach Habana's
// float mul/add/sub/div with a complex CPU scalar. In lazy mode its kernels abort the process ("value cannot be
// converted to type double without overflow", then a device critical error; also without hpu_complex:
// repro/lazy_complex_scalar.py), in eager mode they allocate a complex result and call the out= op (mul_out_pu1).
// These kernels sit at AutogradHPU, above Habana's: the real operand is promoted to the result type (torch's
// result_type: complex64 for float32/bfloat16/int/bool, complex128 for float64), as a differentiable .to(), and the op
// runs on the complex kernels with torch's own autograd formulas. In-place and out= forms with a real self/out raise
// TensorIterator's message. Every other call goes on to torch's autograd kernel unchanged: through AutogradOther,
// whose table entry is the same VariableType kernel (an AutogradHPU redispatch would come back here). A call that skips
// the autograd keys (torch.inference_mode, an inference tensor, kernels below autograd) is taken at BackendSelect.

bool complex_cpu_scalar(const c10::IValue& v) {
  if (v.isTensor()) {
    const auto& t = v.toTensor();
    return t.defined() && t.is_cpu() && t.dim() == 0 && t.is_complex();
  }
  return v.isScalar() && v.toScalar().isComplex();
}

bool real_hpu(const c10::IValue& v) {
  if (!v.isTensor()) {
    return false;
  }
  const auto& t = v.toTensor();
  // Not a traced tensor: torch.compile's fake tensors (Python key) and functional wrappers keep torch's own path,
  // which traced right before (compiled exp(x * 2j) failed in fake-tensor tracing through this kernel).
  const auto ks = t.defined() ? t.key_set() : DispatchKeySet();
  return t.defined() && t.is_hpu() && !t.is_complex() && !tagged(t) && !ks.has(DispatchKey::Python) &&
         !ks.has(DispatchKey::Functionalize);
}

at::ScalarType promoted_type(const at::Tensor& self, const c10::IValue& other) {
  return other.isTensor() ? at::result_type(self, other.toTensor()) : at::result_type(self, other.toScalar());
}

// x * c, c * x, x + alpha c, x - alpha c (x real, c a finite complex scalar, a complex64 result) without promoting x:
// torch.complex of two real parts, each one or two float ops on x, differentiable through torch.complex's own
// gradient. A promoted x costs a copy into a complex tensor and the complex product's ten-odd launches, forward and
// backward (Green's Gabor wavelets: 2j * pi * f * t for every frequency). The parts are torch's complex64 arithmetic
// with x + 0i, constants in float as torch casts the scalar: x c = (x cr - 0 ci) + i (x ci + 0 cr), and
// x + alpha c = (x + (alpha cr - 0 ci)) + i (0 + (alpha ci + 0 cr)), so signed zeros match too. Anything else
// (division, rsub, a complex first operand of add/sub, a non-finite c, a float64 x) is promoted.
std::optional<at::Tensor> real_times_complex_scalar_1d(bool mul, bool sub, c10::ArrayRef<c10::IValue> args,
                                                       c10::complex<float> c, const at::Tensor& x);

// Lightweight autograd::Function for `real * w` (0-d complex CPU scalar tensor w, real.requires_grad(),
// !w.requires_grad(), !w._fw_grad(0).defined()): runs real_times_complex_scalar_1d under
// AutoDispatchBelowADInplaceOrView and saves w via save_for_backward({w}) so PyTorch checks w's version counter
// on backward without paying the ~90-op promoted path.
class MulWith0dComplexScalar : public torch::autograd::Function<MulWith0dComplexScalar> {
 public:
  static at::Tensor forward(torch::autograd::AutogradContext* ctx, const at::Tensor& real, const at::Tensor& w) {
    ctx->save_for_backward({w});
    ctx->saved_data["real_dtype"] = static_cast<int64_t>(real.scalar_type());
    const auto c64 = w.item().toComplexDouble();
    const c10::complex<float> c(static_cast<float>(c64.real()), static_cast<float>(c64.imag()));
    ctx->saved_data["cr"] = static_cast<double>(c.real());
    ctx->saved_data["ci"] = static_cast<double>(c.imag());
    at::AutoDispatchBelowADInplaceOrView guard;
    HPU_COMPLEX_NO_AUTOCAST;
    auto x = real.scalar_type() == at::kFloat ? real : real.to(at::kFloat);
    if (x.dim() == 0) {
      x = x.unsqueeze(0);
      auto value = real_times_complex_scalar_1d(true, false, {}, c, x);
      return value->squeeze(0);
    }
    return *real_times_complex_scalar_1d(true, false, {}, c, x);
  }

  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                 torch::autograd::variable_list grads) {
    const auto saved = ctx->get_saved_variables();  // enforces version-counter check on w
    (void)saved;
    const auto& grad = grads[0];
    if (!grad.defined()) {
      return {at::Tensor(), at::Tensor()};
    }
    HPU_COMPLEX_NO_AUTOCAST;
    const double cr = ctx->saved_data["cr"].toDouble();
    const double ci = ctx->saved_data["ci"].toDouble();
    const auto real_dtype = static_cast<at::ScalarType>(ctx->saved_data["real_dtype"].toInt());
    // Mirror forward's 0-d -> 1-d workaround: Habana's lazy compiler mis-shapes 0-d view arithmetic in unbind/stack backward.
    const bool scalar = grad.dim() == 0;
    auto g = scalar ? grad.unsqueeze(0) : grad;
    auto pair = at::view_as_real(g.resolve_conj());
    auto g_real = pair.select(-1, 0) * cr + pair.select(-1, 1) * ci;
    if (g_real.scalar_type() != real_dtype) {
      g_real = g_real.to(real_dtype);
    }
    return {scalar ? g_real.squeeze(0) : g_real, at::Tensor()};
  }
};

bool in_functorch(const at::Tensor& t) {
  const auto functorch_keys = DispatchKeySet({
      DispatchKey::FuncTorchDynamicLayerFrontMode,
      DispatchKey::FuncTorchDynamicLayerBackMode,
      DispatchKey::FuncTorchBatched,
      DispatchKey::FuncTorchGradWrapper,
      DispatchKey::FuncTorchVmapMode,
  });
  return t.key_set().has_any(functorch_keys) ||
         c10::impl::tls_local_dispatch_key_set().included_.has_any(functorch_keys);
}

std::optional<at::Tensor> real_times_complex_scalar(const std::string& name, c10::ArrayRef<c10::IValue> args,
                                                    int64_t real_at, const at::Tensor& real) {
  const bool mul = name == "aten::mul";
  const bool add = name == "aten::add";
  const bool sub = name == "aten::sub";
  if (!(mul || ((add || sub) && real_at == 0))) {
    return std::nullopt;
  }
  const auto& other = args[1 - real_at];
  if (other.isTensor()) {
    const auto& w = other.toTensor();
    if (w.requires_grad() || w._fw_grad(/*level=*/0).defined()) {
      return std::nullopt;
    }
    const auto c64 = w.item().toComplexDouble();
    const c10::complex<float> c(static_cast<float>(c64.real()), static_cast<float>(c64.imag()));
    if (!std::isfinite(c.real()) || !std::isfinite(c.imag())) {
      return std::nullopt;
    }
    if (mul && torch::autograd::GradMode::is_enabled() && real.requires_grad() && !in_functorch(real)) {
      return MulWith0dComplexScalar::apply(real, w);
    }
    auto x = real.scalar_type() == at::kFloat ? real : real.to(at::kFloat);
    if (x.dim() == 0) {
      x = x.unsqueeze(0);
      auto value = real_times_complex_scalar_1d(mul, sub, args, c, x);
      return value ? std::optional<at::Tensor>(value->squeeze(0)) : std::nullopt;
    }
    return real_times_complex_scalar_1d(mul, sub, args, c, x);
  }
  const auto c64 = other.toScalar().toComplexDouble();
  const c10::complex<float> c(static_cast<float>(c64.real()), static_cast<float>(c64.imag()));
  if (!std::isfinite(c.real()) || !std::isfinite(c.imag())) {
    return std::nullopt;
  }
  auto x = real.scalar_type() == at::kFloat ? real : real.to(at::kFloat);
  if (x.dim() == 0) {  // lazy mode: Habana's graph compile fails on this 0-d stack (synStatus 26, Gaudi2, 1.21.4)
    x = x.unsqueeze(0);
    auto value = real_times_complex_scalar_1d(mul, sub, args, c, x);
    return value ? std::optional<at::Tensor>(value->squeeze(0)) : std::nullopt;
  }
  return real_times_complex_scalar_1d(mul, sub, args, c, x);
}

std::optional<at::Tensor> real_times_complex_scalar_1d(bool mul, bool sub, c10::ArrayRef<c10::IValue> args,
                                                       c10::complex<float> c, const at::Tensor& x) {
  if (mul) {
    auto re = x * static_cast<double>(c.real());
    const float k_re = 0.0f * c.imag();  // re = x cr - 0 ci: subtracting +0 changes nothing
    if (std::signbit(k_re)) {
      re = re - static_cast<double>(k_re);
    }
    auto im = x * static_cast<double>(c.imag()) + static_cast<double>(0.0f * c.real());
    return at::complex(re, im);
  }
  auto alpha = args[2].toScalar();
  if (alpha.isComplex()) {
    return std::nullopt;
  }
  const float a = static_cast<float>(sub ? -alpha.toDouble() : alpha.toDouble());
  const float k_re = a * c.real() - 0.0f * c.imag();
  const float k_im = 0.0f + (a * c.imag() + 0.0f * c.real());
  auto re = x + static_cast<double>(k_re);
  return at::complex(re, at::full_like(re, static_cast<double>(k_im)));
}

// torch's argument checks of add/sub/rsub that the promotion would skip (they live in the CPU/CUDA kernels, not in the
// meta functions): bool subtraction (sub_check) and a bool alpha for a non-bool result (alpha_check).
void check_like_torch(const c10::FunctionSchema& schema, c10::ArrayRef<c10::IValue> args) {
  const auto& name = schema.name();
  const bool sub = name == "aten::sub" || name == "aten::sub_" || name == "aten::rsub";
  const auto is_bool = [](const c10::IValue& v) {
    return v.isTensor() ? v.toTensor().scalar_type() == at::kBool : v.isScalar() && v.toScalar().isBoolean();
  };
  TORCH_CHECK(!sub || !(is_bool(args[0]) || is_bool(args[1])), "Subtraction, the `-` operator, with a bool tensor is "
              "not supported. If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.");
  const auto alpha = schema.argumentIndexWithName("alpha");
  if (alpha && args[*alpha].isScalar()) {
    TORCH_CHECK(!args[*alpha].toScalar().isBoolean(), "Boolean alpha only supported for Boolean results.");
  }
}

// below: the keys to hand every other call to (torch's autograd kernel at AutogradHPU, the backend at BackendSelect).
void complex_scalar_promote(const c10::OperatorHandle& op, DispatchKeySet below, torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  const auto n = schema.arguments().size();
  auto args = torch::jit::last(*stack, n);
  // (real HPU self, complex scalar other), or (complex CPU 0-d self, real HPU other) for the Tensor overloads.
  // In-place forms only promote other, never a written self.
  bool writes = false;
  for (size_t i = 0; i < n; ++i) {
    const auto* alias = schema.arguments()[i].alias_info();
    writes |= alias != nullptr && alias->isWrite();
  }
  const auto* self_alias = schema.arguments()[0].alias_info();
  const bool inplace = self_alias != nullptr && self_alias->isWrite();
  // torch's check, before Habana's float add/sub sees a complex alpha (two real HPU tensors; add_, sub_, rsub too)
  const auto& name = schema.name();
  if ((name == "aten::add" || name == "aten::sub" || name == "aten::add_" || name == "aten::sub_" ||
       name == "aten::rsub") &&
      n >= 3 && args[2].isScalar() && args[2].toScalar().isComplex() && real_hpu(args[0]) && args[1].isTensor() &&
      !args[1].toTensor().is_complex()) {
    TORCH_CHECK(false, "For non-complex input tensors, argument alpha must not be a complex number.");
  }
  int64_t real_at = -1;
  if (real_hpu(args[0]) && complex_cpu_scalar(args[1])) {
    real_at = 0;
  } else if (!inplace && args[0].isTensor() && complex_cpu_scalar(args[0]) && real_hpu(args[1])) {
    real_at = 1;
  }
  const auto& real = real_at >= 0 ? args[real_at].toTensor() : at::Tensor();
  const auto result = real_at < 0                 ? at::ScalarType::Undefined
                      : real_at == 0              ? promoted_type(real, args[1])
                                                  : at::result_type(args[0].toTensor(), real);
  if (real_at < 0 || !c10::isComplexType(result)) {
    op.redispatchBoxed(below, stack);
    return;
  }
  // In-place (self written) and out= forms: a complex result cannot be cast to a real self or out.
  for (size_t i = 0; i < n; ++i) {
    const auto* alias = schema.arguments()[i].alias_info();
    if (alias != nullptr && alias->isWrite() && args[i].isTensor()) {
      const auto& written = args[i].toTensor();
      TORCH_CHECK(at::canCast(result, written.scalar_type()), "result type ", result,
                  " can't be cast to the desired output type ", written.scalar_type());
    }
  }
  // the promotion below must not accept a call torch rejects (sub(bool_x, 1j), add(x, 1j, alpha=True)).
  check_like_torch(schema, args);
  TORCH_CHECK(result == at::kComplexFloat || result == at::kComplexDouble, "hpu_complex: ", result,
              " is not supported on HPU (complex64 and complex128 are)");
  if (!writes && result == at::kComplexFloat) {
    if (auto value = real_times_complex_scalar(schema.name(), args, real_at, real)) {
      torch::jit::drop(*stack, n);
      stack->emplace_back(std::move(*value));
      return;
    }
  }
  (*stack)[stack->size() - n + real_at] = real.to(result);
  op.callBoxed(stack);
}

void complex_scalar_autograd_hpu(const c10::OperatorHandle& op, DispatchKeySet ks, torch::jit::Stack* stack) {
  complex_scalar_promote(op,
                         (ks & DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::AutogradFunctionality)) |
                             DispatchKeySet(DispatchKey::AutogradOther),
                         stack);
}

// The same calls when they skip the autograd keys: under torch.inference_mode (TLS excludes them), with an inference
// tensor (it has none), or from a kernel below autograd. BackendSelect is in every call's key set (TLS includes it by
// default, inference mode keeps it), below autograd and above the backend (Habana's HPU kernels); with no kernel of
// its own an op falls through it. The promoted call dispatches again with the caller's TLS: no autograd under
// inference mode (inference-tensor results, as torch makes them), none below autograd either. A call that the
// AutogradHPU kernel above already promoted arrives here with a complex operand and passes through.
DispatchKeySet after_backend_select(DispatchKeySet ks) {
  return ks & DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
}

void complex_scalar_backend_select(const c10::OperatorHandle& op, DispatchKeySet ks, torch::jit::Stack* stack) {
  complex_scalar_promote(op, after_backend_select(ks), stack);
}

// BackendSelect sees every call of these ops below autograd, on any device (CPU code too). Measured on the CPU: the
// boxed kernel alone cost 0.6-1.2 us a call (x * y: 1.2 -> 1.8 us). The unboxed kernel below hands on at once a call
// without both an HPU tensor and a complex scalar (a Python complex, a complex 0-d CPU tensor: what
// complex_scalar_promote looks for), and boxes the rest (complex_scalar_backend_select).
template <class T>
bool on_hpu(const T& v) {
  if constexpr (std::is_same_v<std::decay_t<T>, at::Tensor>) {
    return v.defined() && v.is_hpu() && !v.is_complex() && !tagged(v);
  } else {
    return false;
  }
}

template <class T>
bool complex_scalar_arg(const T& v) {
  if constexpr (std::is_same_v<std::decay_t<T>, at::Tensor>) {
    return v.defined() && v.is_cpu() && v.dim() == 0 && v.is_complex();
  } else if constexpr (std::is_same_v<std::decay_t<T>, at::Scalar>) {
    return v.isComplex();
  } else {
    return false;
  }
}

// Written: the index of the argument an in-place (0, self) or out= (the last) form returns; -1 for a fresh result.
template <class Op, int Written, class Schema = typename Op::schema>
struct BackendSelectFast;

template <class Op, int Written, class Ret, class... Args>
struct BackendSelectFast<Op, Written, Ret(Args...)> {
  static Ret call(DispatchKeySet ks, Args... args) {
    if (!(on_hpu(args) || ...) || !(complex_scalar_arg(args) || ...)) {
      return Op::redispatch(after_backend_select(ks), args...);
    }
    static const c10::OperatorHandle op =
        c10::Dispatcher::singleton().findSchemaOrThrow(Op::name, Op::overload_name);
    torch::jit::Stack stack;
    stack.reserve(sizeof...(Args));
    (torch::jit::push(stack, args), ...);
    complex_scalar_promote(op, after_backend_select(ks), &stack);
    if constexpr (Written >= 0) {
      return std::get<Written>(std::forward_as_tuple(args...));
    } else {
      return std::move(stack.back()).template to<Ret>();
    }
  }
};

// ---- copies and fills (PrivateUse1) ------------------------------------------------------

// Lazy mode: a real view of a complex tensor (view_as_real, .real, view(float32): Habana's own as_strided of the
// storage's Habana tensor) lives in the Habana storage, not in the complex tensor's lazy storage, so torch's overlap
// check sees two storages. Offsets and strides are both in the Habana storage's coordinates: self is described over
// src's storage, and checked against src. (Two complex tensors over one Habana storage share their lazy storage.)
void check_lazy_overlap(const at::Tensor& self, const at::Tensor& src, bool any_overlap) {
  if (!tagged(self) || !src.has_storage() || self.storage().is_alias_of(src.storage())) {
    return;
  }
  const auto a = lazy_base(self);
  const auto b = tagged(src) ? lazy_base(src) : src;
  if (!a.defined() || !b.defined() || !b.has_storage() ||
      a.storage().unsafeGetStorageImpl() != b.storage().unsafeGetStorageImpl()) {
    return;
  }
  const auto described = view_of(b, b.key_set(), self.dtype(), self.sizes(), self.strides(), self.storage_offset());
  const auto read = tagged(src) ? view_of(b, b.key_set(), src.dtype(), src.sizes(), src.strides(), src.storage_offset())
                                : src;
  if (any_overlap) {
    at::assert_no_overlap(described, read);
  } else {
    at::assert_no_partial_overlap(described, read);
  }
}

at::Tensor& copy_pu1(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
  // torch's copy_ checks (TensorIterator): self has no internal overlap, and src covers self's memory fully or not
  // at all; a partial overlap would read values already overwritten.
  at::assert_no_internal_overlap(self);
  at::assert_no_partial_overlap(self, src);
  check_lazy_overlap(self, src, /*any_overlap=*/false);
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
      auto src_pair = raw_pair(src);
      if (!src_pair.is_contiguous()) {
        src_pair = src_pair.contiguous();
      }
      if (!self.is_hpu() && (!dst.is_contiguous() || !src_pair.sizes().equals(dst.sizes()))) {
        // Habana D2H copy_ fails on non-dense CPU targets and does not broadcast smaller HPU sources; stage via CPU.
        auto cpu_src = src_pair.to(at::kCPU);
        if (self.is_conj() != src.is_conj()) {
          cpu_src.select(-1, 1).neg_();
        }
        dst.copy_(cpu_src);
        return self;
      }
      if (self.is_hpu() && !src.is_hpu() && !src_pair.sizes().equals(dst.sizes())) {
        // Habana H2D copy_ does not broadcast smaller CPU tensors across dst; stage to HPU first.
        src_pair = src_pair.to(dst.scalar_type()).to(dst.device(), non_blocking);
      }
      dst.copy_(src_pair, non_blocking && (self.is_hpu() || self.is_conj() == src.is_conj()));
      if (self.is_conj() != src.is_conj()) {  // the stored values differ by one conjugation
        dst.select(-1, 1).neg_();
      }
    } else {  // real -> complex: imaginary part 0
      auto src_f = src.dim() >= 5 ? dense_copy(src) : src;
      auto dst_re = dst.select(-1, 0);
      if (!self.is_hpu()) {
        if (src_f.is_hpu()) {
          src_f = src_f.to(at::kCPU);
        }
        dst_re.copy_(src_f);
        dst.select(-1, 1).zero_();
        return self;
      }
      if (!src_f.is_hpu() && !src_f.sizes().equals(dst_re.sizes())) {
        src_f = src_f.to(dst_re.scalar_type()).to(dst_re.device(), non_blocking);
      }
      dst_re.copy_(src_f, non_blocking);
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
  if (real_part.is_hpu() && (real_part.dim() >= 5 || !self.is_hpu()) && !real_part.is_contiguous()) {
    real_part = real_part.contiguous();
  }
  if (!self.is_hpu() && (!self.is_contiguous() || !real_part.sizes().equals(self.sizes()))) {
    self.copy_(real_part.to(at::kCPU));
    return self;
  }
  self.copy_(real_part, non_blocking);
  return self;
}

// Resize a tagged complex tensor in place, as resize_ does: contiguous strides at the same offset, in the same
// storage. One too small grows in place, as torch's resize_bytes does: a new allocation in the same StorageImpl with
// the old bytes copied over, so every view of it sees the result and values outside the resized range are kept
// (torch.neg(v, out=base[1:]) keeps base[0]).
void resize_complex(at::Tensor& t, c10::IntArrayRef sizes) {
  auto strides = c10::contiguous_strides(sizes);
  const auto needed = static_cast<size_t>(
      at::detail::computeStorageNbytes(sizes, strides, t.dtype().itemsize(), t.storage_offset()));
  auto storage = t.storage();
  const auto old_nbytes = storage.nbytes();
  if (old_nbytes < needed) {
    if (host_backed(t)) {  // host memory has no allocator: a new host block in the same StorageImpl
      void* data = c10::alloc_cpu(needed);
      if (old_nbytes > 0) {
        std::memcpy(data, storage.data(), old_nbytes);
      }
      storage.set_data_ptr_noswap(c10::DataPtr(data, data, &host_free, hpu_index(t.device())));
    } else if (auto old = lazy_base(t); old.defined()) {
      // Lazy mode: Habana's own resize_ of the storage's Habana tensor (its detached alias): Habana grows the storage
      // in its lazy graph, keeps the old values, and its views taken before (view_as_real, .real of this tensor, the
      // real tensor it was made from) follow it (measured, Gaudi2, bridge 1.21.4: a float view taken before resize_
      // reads later writes). The Habana tensor's sizes do not matter here: its views are as_strided in storage terms.
      // Its elements are counted in its own dtype; one that has no float view (a float16 r of r.view(complex64))
      // cannot be computed on in lazy mode, and raises here, before the storage or self changes.
      const auto item = static_cast<size_t>(old.dtype().itemsize());
      TORCH_CHECK(item == sizeof(float), "hpu_complex: lazy mode: no float view of a ", old.dtype().name(),
                  " storage");
      const auto elements = static_cast<int64_t>((needed + item - 1) / item);
      old.resize_({std::max<int64_t>(elements - old.storage_offset(), 0)});
    } else {
      // torch's own storage growth for the device (resize_bytes_nocuda: Habana's resize_ of a byte tensor over the
      // storage, which copies the old bytes in its stream order). Copying them here around a swapped DataPtr lost them
      // on the card: Habana's eager ops resolve the storage's address when they run, after the swap (measured).
      TORCH_CHECK(storage.resizable(), "Trying to resize storage that is not resizable");
      at::native::resize_bytes_nocuda(storage, static_cast<int64_t>(needed));
    }
    storage.set_nbytes(needed);
  }
  t.unsafeGetTensorImpl()->set_sizes_and_strides(sizes, strides);
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

// sum(..., out=) as torch's out= reductions (make_reduction_from_out_ty): the result dtype is out's (a dtype= must be
// it), and the input is cast to it before the reduction: a complex input into a real out sums the real parts (torch
// warns), and into a complex128 out sums in complex128 (2^24 + 1 - 2^24 is 1, not 0 from a complex64 sum).
at::Tensor& sum_out_pu1(const at::Tensor& self, at::OptionalIntArrayRef dim, bool keepdim,
                        std::optional<at::ScalarType> dtype, at::Tensor& out) {
  TORCH_CHECK(!dtype.has_value() || *dtype == out.scalar_type(), "Expected out tensor to have dtype ",
              c10::scalarTypeToTypeMeta(*dtype), ", but got ", out.dtype(), " instead");
  const auto value = self.scalar_type() == out.scalar_type()
                         ? sum_pu1(self, dim, keepdim, std::nullopt)
                         : at::sum(self.to(out.scalar_type()), dim, keepdim, std::nullopt);
  return write_out(out, value, {&self});
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

// The deleter of a Python storage slice (s[a:b], torch/csrc/Storage.cpp): its context is the sliced StorageImpl, which
// the slice keeps alive. hpu_complex.extension notes it once, from a CPU slice, after loading the extension.
std::atomic<c10::DeleterFnPtr> storage_slice_deleter{nullptr};

void note_storage_slice(const at::Tensor& t) {
  const auto deleter = t.storage().data_ptr().get_deleter();
  TORCH_CHECK(t.storage().data_ptr().get_context() != nullptr && deleter != nullptr &&
                  deleter != c10::GetCPUAllocator()->raw_deleter() && deleter != &host_free &&
                  deleter != &lazy_owner_free,
              "hpu_complex::note_storage_slice: not a tensor over a storage slice");
  storage_slice_deleter.store(deleter);
}

// A slice of a storage made here hides its kind (lazy owner, host memory) behind the slice's own deleter: through it,
// set_ would take an owner context for Habana bytes, or host memory for Habana storage. Slices are followed to the
// storage they were taken of; slices of those raise (a whole storage with an offset is the supported spelling).
void check_not_a_slice_of_ours(const at::Storage& source) {
  const auto slice = storage_slice_deleter.load();
  const c10::StorageImpl* s = source.unsafeGetStorageImpl();
  while (slice != nullptr && s != nullptr && s->data_ptr().get_deleter() == slice &&
         s->data_ptr().get_context() != nullptr) {
    s = static_cast<const c10::StorageImpl*>(s->data_ptr().get_context());
  }
  if (s == nullptr || s == source.unsafeGetStorageImpl()) {
    return;
  }
  TORCH_CHECK(s->data_ptr().get_deleter() != &lazy_owner_free, "hpu_complex: set_ onto a slice of a complex64 HPU "
              "storage made in Habana lazy mode is not supported (that storage holds a Habana tensor, not bytes): "
              "set_ onto the whole storage with a storage offset, or take a view of the tensor instead");
  TORCH_CHECK(s->data_ptr().get_deleter() != &host_free, "hpu_complex: set_ onto a slice of a complex128 HPU storage "
              "(host memory) is not supported: set_ onto the whole storage with a storage offset, or take a view of "
              "the tensor instead");
}

// torch.load rebuilds every tensor as empty(0).set_(storage, offset, size, stride): attach the device storage in
// place, as the backends' own set_ kernels do (the CPU fallback would hand a CPU tensor the HPU storage). Every check
// runs before anything is installed (ATen's checkSetStorage swaps the storage in before its offset check, setStrided
// before its stride check): a set_ that raises leaves self, and a tensor autograd saved, unchanged. The stride of a
// size-1 (or empty) dim addresses nothing: a negative one is accepted, as the CPU kernel does, and made contiguous.
at::Tensor& set_storage_pu1(at::Tensor& self, at::Storage source, int64_t storage_offset, c10::IntArrayRef size,
                            c10::IntArrayRef stride) {
  check_not_a_slice_of_ours(source);
  TORCH_CHECK(stride.empty() || stride.size() == size.size(), "unequal size length (", size.size(),
              ") and stride length (", stride.size(), ")");
  TORCH_CHECK(storage_offset >= 0, "Tensor: invalid storage offset ", storage_offset);
  const auto contiguous = c10::contiguous_strides(size);
  auto strides = stride.empty() ? contiguous : at::DimVector(stride.begin(), stride.end());
  for (size_t i = 0; i < strides.size(); ++i) {
    if (strides[i] < 0 && size[i] <= 1) {
      strides[i] = contiguous[i];
    }
    TORCH_CHECK(strides[i] >= 0, "as_strided: Negative strides are not supported at the moment, got strides: ",
                stride);
  }
  const bool lazy_source = source.data_ptr().get_deleter() == &lazy_owner_free;
  if (self.scalar_type() != at::kComplexFloat && source.device().type() == at::kHPU &&
      source.data_ptr().get_deleter() != &host_free) {
    // A complex64 storage made in lazy mode holds a Habana tensor, not bytes: there is nothing to copy from it.
    TORCH_CHECK(!lazy_source, "hpu_complex: set_ of a ", self.scalar_type(), " HPU tensor onto the storage of a "
                "complex64 tensor made in Habana lazy mode is not supported (", self.scalar_type(), " HPU tensors "
                "live in host memory): use .to(", self.scalar_type(), ") instead");
    // complex128 from Habana storage (torch.load(map_location="hpu") restores storages there): its bytes are copied
    // into host memory, so the tensor holds its own copy rather than an alias of source.
    const auto nbytes = static_cast<int64_t>(source.nbytes());
    at::native::checkInBoundsForStorage(size, c10::IntArrayRef(strides), storage_offset, self.dtype(), source);
    auto bytes = at::empty({0}, at::TensorOptions().dtype(at::kByte).device(source.device()));
    bytes.set_(source, 0, {nbytes}, {1});
    auto host = make_host_empty({nbytes}, {1}, caffe2::TypeMeta::Make<uint8_t>(), source.device());
    host_cpu(host).copy_(bytes.cpu());
    source = host.storage();
  }
  at::native::checkInBoundsForStorage(size, c10::IntArrayRef(strides), storage_offset, self.dtype(), source);
  at::native::checkSetStorage(self, std::move(source), storage_offset, size, c10::IntArrayRef(strides));
  at::native::setStrided(self, size, c10::IntArrayRef(strides), storage_offset);
  return self;
}

// a.set_(storage): the whole storage as a 1-D tensor (ATen's composite would build a CPU tensor first: a device error).
at::Tensor& set_source_storage_pu1(at::Tensor& self, at::Storage source) {
  const auto n = static_cast<int64_t>(source.nbytes() / self.dtype().itemsize());
  return set_storage_pu1(self, std::move(source), 0, {n}, {});
}

// a.set_(): an empty tensor over a fresh empty storage of its own, detached from the old one (its views keep it).
at::Tensor& set_empty_pu1(at::Tensor& self) {
  auto fresh = host_backed(self) || self.scalar_type() != at::kComplexFloat
                   ? make_host_empty({0}, {1}, self.dtype(), self.device())
                   : make_complex_empty({0}, {1}, self.device());
  return set_storage_pu1(self, fresh.storage(), 0, {0}, {});
}

// a.set_(b): a takes b's storage, offset, sizes and strides, so the two alias (ATen's kernel: set_ on the storage,
// which set_storage_pu1 attaches in place). The fallback would compute it on a copy.
at::Tensor& set_tensor_pu1(at::Tensor& self, const at::Tensor& source) {
  return at::native::set_tensor_(self, source);
}

// ---- elementwise and reductions on the pairs (PrivateUse1): neg, exp, masked_fill_, cat, vector_norm -------------
//
// spd_learn's Green ran these on the CPU every step (exp of its Gabor wavelets, cat of their real and imaginary parts,
// linalg.norm, and in the backward neg and masked_fill_). Each is one or a few float ops on the [..., 2] pairs. Their
// gradients are torch's own formulas (autograd sits above this key), which call only ops registered here.

// The pair of c's values for reading. Most ops reach a kernel with a conjugate input resolved by the Conjugate
// fallback; neg is not one of them (conj and neg commute), so resolve here.
at::Tensor values_pair(const at::Tensor& c) {
  return operand_pair(c.is_conj() ? c.resolve_conj() : c);
}

at::Tensor neg_pu1(const at::Tensor& self) {
  if (host_backed(self)) {
    return from_cpu(at::neg(host_cpu(self)), self.device());
  }
  HPU_COMPLEX_NO_AUTOCAST;
  return complex_result(values_pair(self).neg());
}

// exp(a + ib) = e^a (cos b + i sin b). An imaginary part that is exactly 0 stays exactly 0 (signed), as C99 cexp and
// torch's kernels give for a real-valued complex input (exp(inf + 0i) = inf + 0i, not inf * sin(0) = NaN). e^a is
// split in two halves, (e^(a/2) cos b) e^(a/2): a result that fits in float32 (exp(88.75 + i pi/4) = 2.47e38 (1 + i))
// is not lost to an e^a that overflows first (torch's own complex64 kernel gives inf + inf i there). One rounding more
// than e^a cos b (within 2 ulp of torch); three launches more, where a where(a > 88, ...) form cost seven.
at::Tensor exp_pu1(const at::Tensor& self) {
  if (host_backed(self)) {
    return from_cpu(at::exp(host_cpu(self)), self.device());
  }
  HPU_COMPLEX_NO_AUTOCAST;
  auto pair = values_pair(self);
  auto a = pair.select(-1, 0), b = pair.select(-1, 1);
  auto ea = (a * 0.5).exp();  // e^(a/2): (e^(a/2) cos b) e^(a/2) overflows only where the result does
  auto re = ea * b.cos() * ea;
  // where(b != 0, ...), not where(b == 0, b, ...): the same values, and 6x faster on Gaudi2 eager (bridge 1.21.4:
  // 0.26 against 1.5 ms for this exp of a (10, 125) tensor, measured; the == form was slower than the CPU fallback).
  auto im = at::where(b != 0, ea * b.sin() * ea, b);
  // C99 cexp at an infinite real part and a non-finite imaginary part: exp(-inf + i inf/nan) = 0 + 0i,
  // exp(+inf + i inf/nan) = inf + nan i (e^a * cos(b) would be NaN for both). ea is 0 or inf there, and ea - ea gives
  // the imaginary part (0 - 0 = 0, inf - inf = nan).
  auto special = a.isinf().logical_and(b.isfinite().logical_not());
  re = at::where(special, ea, re);
  im = at::where(special, ea - ea, im);
  return complex_result(at::stack({re, im}, -1));
}

// neg/exp with out=: a real self (dispatched here through a complex out) is torch's real op, then cast into out as
// TensorIterator does: exp(float, out=complex64) is accepted (a safe cast), neg's out must have self's dtype exactly
// (check_all_same_dtype: neg(complex64, out=complex128) raises too).
at::Tensor& neg_out_pu1(const at::Tensor& self, at::Tensor& out) {
  TORCH_CHECK(out.scalar_type() == self.scalar_type(), "Found dtype ", out.scalar_type(), " but expected ",
              self.scalar_type());
  return write_out(out, self.is_complex() || host_backed(self) ? neg_pu1(self) : at::neg(self), {&self});
}

at::Tensor& exp_out_pu1(const at::Tensor& self, at::Tensor& out) {
  return write_out(out, self.is_complex() || host_backed(self) ? exp_pu1(self) : at::exp(self), {&self});
}

// masked_fill_ on an expanded self (stride 0, size > 1): torch warns and writes through every alias, so a memory
// location gets the value when any mask element over it is set. The mask is reduced over the expanded dims, and the
// fill runs on one slice of them.
bool fill_expanded(at::Tensor& self, const at::Tensor& m, const at::Scalar& value) {
  std::vector<int64_t> dims;
  for (int64_t d = 0; d < self.dim(); ++d) {
    if (self.stride(d) == 0 && self.size(d) > 1) {
      dims.push_back(d);
    }
  }
  if (dims.empty()) {
    return false;
  }
  TORCH_WARN("Use of masked_fill_ on expanded tensors is deprecated. Please clone() the tensor before performing this "
             "operation. This also applies to advanced indexing e.g. tensor[mask] = scalar");
  auto slice = self;
  for (auto d : dims) {
    slice = slice.narrow(d, 0, 1);
  }
  auto any = m.any(dims, /*keepdim=*/true);  // where some alias of an element is masked
  slice.masked_fill_(any, value);
  return true;
}

at::Tensor& masked_fill_pu1(at::Tensor& self, const at::Tensor& mask, const at::Scalar& value) {
  TORCH_CHECK(mask.scalar_type() == at::kBool, "masked_fill_ only supports boolean masks, but got mask with dtype ",
              mask.scalar_type());
  if (host_backed(self)) {
    return host_inplace(self, [&](at::Tensor& s) { s.masked_fill_(cpu_operand(mask), value); });
  }
  HPU_COMPLEX_NO_AUTOCAST;
  // torch's checked conversion of the value (an overflowing part raises: complex(-0.0, 1e40) is not -0 + inf i).
  const auto c = static_cast<c10::complex<double>>(value.to<c10::complex<float>>());
  // torch's in-place rule: the mask broadcasts to self, never self to the mask.
  auto m = (mask.dim() >= 5 ? dense_copy(mask) : mask).to(self.device()).expand(self.sizes());
  if (fill_expanded(self, m, value)) {
    return self;
  }
  at::assert_no_internal_overlap(self);
  auto pair = values_pair(self);
  const bool plus_zero = c.real() == 0.0 && c.imag() == 0.0 && !std::signbit(c.real()) && !std::signbit(c.imag());
  const bool signed_zero = (c.real() == 0.0 && std::signbit(c.real())) || (c.imag() == 0.0 && std::signbit(c.imag()));
  at::Tensor filled;
  if (plus_zero) {
    filled = pair.masked_fill(m.unsqueeze(-1), 0.0);
  } else if (signed_zero) {
    // A -0.0 part goes in as a tensor, not a scalar: Habana's lazy graph merges the constants -0.0 and 0.0 (the two
    // parts' fills came out with each other's zero sign; Gaudi2, bridge 1.21.4).
    auto value = at::tensor({static_cast<float>(c.real()), static_cast<float>(c.imag())}).to(self.device());
    filled = at::where(m.unsqueeze(-1), value, pair);
  } else {
    filled = at::stack({pair.select(-1, 0).masked_fill(m, c.real()), pair.select(-1, 1).masked_fill(m, c.imag())}, -1);
  }
  copy_pu1(self, complex_result(filled), false);
  return self;
}

at::Tensor& masked_fill_tensor_pu1(at::Tensor& self, const at::Tensor& mask, const at::Tensor& value) {
  TORCH_CHECK(value.dim() == 0, "masked_fill_ only supports a 0-dimensional value tensor, but got tensor with ",
              value.dim(), " dimension(s).");
  return masked_fill_pu1(self, mask, value.item());
}

// cat: torch's result type over the inputs; complex64 is the cat of their pairs along the same dim (a real input gets
// imaginary part 0). 1-D empty inputs are skipped as torch's legacy rule does; every input must be on the HPU.
at::Tensor cat_pairs(const std::vector<at::Tensor>& all, int64_t dim, const at::Device& device);

at::Tensor cat_pu1(const at::ITensorListRef& tensors, int64_t dim) {
  std::vector<at::Tensor> all(tensors.begin(), tensors.end());
  TORCH_CHECK(!all.empty(), "torch.cat(): expected a non-empty list of Tensors");
  const auto result = at::native::result_type(at::ITensorListRef(all));
  at::Device device = all[0].device();
  bool host = false;
  for (const auto& t : all) {
    host |= host_backed(t);
    if (tagged(t)) {
      device = t.device();
    }
  }
  for (const auto& t : all) {
    TORCH_CHECK(t.device() == device, "Expected all tensors to be on the same device, but found at least two devices, ",
                device, " and ", t.device(), "!");
  }
  if (host || result != at::kComplexFloat) {  // complex128 (or a real result into a complex out=): the CPU kernel
    std::vector<at::Tensor> cpu;
    for (const auto& t : all) {
      cpu.push_back(cpu_operand(t));
    }
    auto value = at::cat(cpu, dim);
    return host ? from_cpu(value, device) : value.to(device);
  }
  return cat_pairs(all, dim, device);
}

// The complex64 cat on the card: the cat of the inputs' pairs along the same dim, each input cast to float32 on its own
// (a real one gets imaginary part 0), so nothing is rounded to a narrower common type first.
at::Tensor cat_pairs(const std::vector<at::Tensor>& all, int64_t dim, const at::Device& device) {
  HPU_COMPLEX_NO_AUTOCAST;
  std::vector<at::Tensor> pairs;
  int64_t ndim = -1;
  for (const auto& t : all) {
    if (t.dim() == 1 && t.numel() == 0) {
      continue;
    }
    ndim = t.dim();
    pairs.push_back(as_pair(t.is_conj() ? t.resolve_conj() : t, device));
  }
  if (pairs.empty()) {  // only legacy empties: torch's own result (an empty tensor of the first one's shape)
    return make_complex_empty(all[0].sizes(), c10::contiguous_strides(all[0].sizes()), device);
  }
  TORCH_CHECK(ndim > 0, "zero-dimensional tensor (at position 0) cannot be concatenated");
  const auto d = c10::maybe_wrap_dim(dim, ndim);
  // torch's shape check, here: Habana's lazy cat concatenates mismatched shapes without an error.
  for (size_t i = 1; i < pairs.size(); ++i) {
    TORCH_CHECK(pairs[i].dim() == pairs[0].dim(), "Tensors must have same number of dimensions: got ",
                pairs[0].dim() - 1, " and ", pairs[i].dim() - 1);
    for (int64_t k = 0; k < ndim; ++k) {
      TORCH_CHECK(k == d || pairs[i].size(k) == pairs[0].size(k), "Sizes of tensors must match except in dimension ",
                  d, ". Expected size ", pairs[0].size(k), " but got size ", pairs[i].size(k), " for tensor number ",
                  i, " in the list.");
    }
  }
  return complex_result(at::cat(pairs, d));
}

// cat.out as torch: the inputs' result type must cast to out's dtype (no complex into a real out), out may not overlap
// an input, and the values are concatenated in out's dtype (an int32 16777217 into a complex128 out stays exact): the
// card's pair path for a complex64 out, the CPU otherwise.
at::Tensor& cat_out_pu1(const at::ITensorListRef& tensors, int64_t dim, at::Tensor& out) {
  std::vector<at::Tensor> all(tensors.begin(), tensors.end());
  TORCH_CHECK(!all.empty(), "torch.cat(): expected a non-empty list of Tensors");
  const auto result = at::native::result_type(at::ITensorListRef(all));
  TORCH_CHECK(at::canCast(result, out.scalar_type()), "torch.cat(): input types can't be cast to the desired output "
              "type ", out.scalar_type());
  for (const auto& t : all) {
    TORCH_CHECK(t.device() == out.device(), "Expected all tensors to be on the same device, but found at least two "
                "devices, ", out.device(), " and ", t.device(), "!");
  }
  at::assert_no_internal_overlap(out);
  // Every input against out, tagged or not: a real view of out's storage (out.view(float32), out.real) is untagged.
  for (const auto& t : all) {
    check_written_against(out, t, /*any_overlap=*/true);
  }
  bool host = host_backed(out);
  for (const auto& t : all) {
    host |= host_backed(t);
  }
  at::Tensor value;
  if (!host && out.scalar_type() == at::kComplexFloat && result != at::kComplexDouble) {
    value = cat_pairs(all, dim, out.device());
  } else {
    std::vector<at::Tensor> cpu;
    for (const auto& t : all) {
      cpu.push_back(cpu_operand(t).to(out.scalar_type()));
    }
    value = at::cat(cpu, dim);
  }
  at::native::resize_output_check(out, value.sizes());
  return copy_into(out, value);
}

// ||z||_ord over dim = the real vector norm of |z| (the scaled modulus), for every ord. Habana's float norm squares
// unscaled (|1e-30| gave 0, |3e19 + 3e19j| inf): a positive finite ord is computed as M ||z / M|| with M the largest
// modulus over dim, a negative one with the smallest, so nothing over- or underflows that torch keeps; 0 and +-inf
// need no scaling. Duplicate dims raise as torch (also for a 0-d input).
at::Tensor vector_norm_pu1(const at::Tensor& self, const at::Scalar& ord, at::OptionalIntArrayRef dim, bool keepdim,
                           std::optional<at::ScalarType> dtype) {
  if (host_backed(self)) {
    return from_cpu(at::linalg_vector_norm(host_cpu(self), ord, dim, keepdim, dtype), self.device());
  }
  if (dtype.has_value() && *dtype != at::kComplexFloat) {  // torch's dtype checks, then the computation in that dtype
    TORCH_CHECK(c10::isComplexType(*dtype), "linalg.vector_norm: dtype should be complex for complex input, but got ",
                *dtype);
    return at::linalg_vector_norm(self.to(*dtype), ord, dim, keepdim, std::nullopt);
  }
  if (self.numel() == 0) {  // torch's empty-input rules and errors (Habana's lazy graph fails or skips them)
    return at::linalg_vector_norm(self.cpu(), ord, dim, keepdim, std::nullopt).to(self.device());
  }
  HPU_COMPLEX_NO_AUTOCAST;
  const auto p = ord.toDouble();
  std::vector<int64_t> dims;
  if (dim.has_value()) {
    dims.assign(dim->begin(), dim->end());
  }
  (void)at::dim_list_to_bitset(dims, std::max<int64_t>(self.dim(), 1));  // torch's duplicate and range checks
  if (p == 2.0) {
    // ord 2 (Green's norm): one scaled reduction of the pairs over dim and the pair dim, M sqrt(sum (x / M)^2) with M
    // the largest |part|, half the launches of the general path; an infinite part gives inf even next to a NaN.
    auto pair = values_pair(self);
    std::vector<int64_t> all;
    if (dims.empty()) {
      for (int64_t d = 0; d < self.dim(); ++d) {
        all.push_back(d);
      }
    } else {
      for (auto d : dims) {
        all.push_back(c10::maybe_wrap_dim(d, std::max<int64_t>(self.dim(), 1)));
      }
    }
    if (self.dim() == 0) {
      all.clear();
    }
    all.push_back(pair.dim() - 1);
    const auto inf = std::numeric_limits<float>::infinity();
    // An element's modulus is inf where a part is infinite (hypot(inf, nan) = inf: an infinite part beats a NaN one
    // within an element) and NaN where a part is NaN otherwise; the norm is NaN as soon as one element's modulus is
    // NaN (torch's sum of squares is NaN there, even next to an infinite element), else inf or finite. NaN is flagged
    // explicitly: Habana's amax skips NaN (measured, Gaudi2, bridge 1.21.4), the CPU's propagates it.
    const auto mag = pair.abs();
    const auto elem_inf = (mag == inf).any(-1, /*keepdim=*/true);
    const auto elem_nan = mag.isnan().any(-1, /*keepdim=*/true).logical_and(elem_inf.logical_not());
    const auto bound = at::amax(at::nan_to_num(mag, 0.0, inf, -inf), all, /*keepdim=*/true);
    const auto scale = at::where(bound.isfinite().logical_and(bound > 0), bound, at::ones_like(bound));
    auto norm = at::linalg_vector_norm(pair / scale, 2, all, /*keepdim=*/true) * scale;
    norm = at::where(bound == inf, bound, norm);
    norm = norm.masked_fill(at::amax(elem_nan.to(at::kFloat), all, /*keepdim=*/true) > 0,
                            std::numeric_limits<float>::quiet_NaN());
    norm = norm.squeeze(-1);
    all.pop_back();
    return keepdim || all.empty() ? norm : norm.squeeze(all);
  }
  const auto mod = modulus(values_pair(self), false);
  // A NaN modulus makes the norm NaN for every ord but 0 (which counts it as nonzero), as torch's CPU kernels: flagged
  // explicitly, as Habana's amax / amin (ord +-inf, and the scale below) skip NaN.
  const auto with_nan = [&](const at::Tensor& norm) {
    return p == 0.0 ? norm
                    : norm.masked_fill(at::amax(mod.isnan().to(at::kFloat), dims, keepdim).view(norm.sizes()) > 0,
                                       std::numeric_limits<float>::quiet_NaN());
  };
  if (p == 0.0 || std::isinf(p) || std::isnan(p)) {
    return with_nan(at::linalg_vector_norm(mod, ord, dim, keepdim, std::nullopt));
  }
  const auto bound = p > 0 ? at::amax(mod, dims, /*keepdim=*/true) : at::amin(mod, dims, /*keepdim=*/true);
  const auto scale = at::where(bound.isfinite().logical_and(bound > 0), bound, at::ones_like(bound));
  const auto ratio = mod / scale;
  if (std::abs(p) >= 1.0) {
    // A ratio lost to under- (ord > 0) or overflow (ord < 0) has a term ratio^ord below 1e-38 next to the bound's 1.
    auto norm = at::linalg_vector_norm(ratio, ord, dim, keepdim, std::nullopt);
    return with_nan(norm * scale.view(norm.sizes()));
  }
  // |ord| < 1: ratio^ord can matter where the ratio itself under- or overflows (||[1e20, 1e-30]||_0.01 = 8.6e31, and
  // 1e20 from the ratios). There the term is exp(ord (log |z| - log M)), from the logs, which stay in range.
  auto terms = ratio.pow(p);
  const auto lost = p > 0 ? (ratio < std::numeric_limits<float>::min()).logical_and(mod > 0)
                          : ratio.isinf().logical_and(mod.isfinite());
  terms = at::where(lost, ((mod.log() - scale.log()) * p).exp(), terms);
  auto sum = at::linalg_vector_norm(terms, 1, dim, keepdim, std::nullopt);  // the terms are >= 0 (or NaN)
  // sum^(1 / ord) can over- or underflow float32 before the rescale (3 x 1e-30 at ord 0.01: 3^100 = 5e47): there,
  // and only there, exp(log(sum) / ord + log(scale)) (the log form costs ~|log| ulps, 2.7e-6 at ord 0.5 measured, so
  // the direct power stays wherever it is a normal float). Each branch reads a safe input (1 where it is not taken),
  // so neither puts log(0) or 0 * inf into the gradient (torch's zero gradient at z = 0).
  const auto inv = 1.0 / p;
  const auto s = scale.view(sum.sizes());
  const auto probe = sum.pow(inv);
  const auto use_log = (sum > 0).logical_and(
      probe.isfinite().logical_and(probe >= std::numeric_limits<float>::min()).logical_not());
  const auto one = at::ones_like(sum);
  const auto by_log = (at::where(use_log, sum, one).log() * inv + s.log()).exp();
  const auto by_pow = at::where(use_log, one, sum).pow(inv) * s;
  return with_nan(at::where(use_log, by_log, by_pow));
}

at::Tensor& vector_norm_out_pu1(const at::Tensor& self, const at::Scalar& ord, at::OptionalIntArrayRef dim,
                                bool keepdim, std::optional<at::ScalarType> dtype, at::Tensor& out) {
  auto value = vector_norm_pu1(self, ord, dim, keepdim, dtype);
  // torch's rule: out has the norm's (real) dtype exactly.
  TORCH_CHECK(out.scalar_type() == value.scalar_type(), "Expected out tensor to have dtype ",
              c10::toRealValueType(dtype.value_or(self.scalar_type())), ", but got ", out.scalar_type(), " instead");
  return write_out(out, value, {&self});
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

// TensorIterator's alias check, which the CPU kernel cannot make here (written arguments are computed on a copy): a
// written host-backed argument that partly overlaps another tensor argument raises, as torch does.
void check_no_partial_overlap(const c10::FunctionSchema& schema, const std::vector<c10::IValue>& args) {
  std::vector<at::Tensor> written, read;
  for (size_t i = 0; i < args.size(); ++i) {
    if (!args[i].isTensor() || !host_backed(args[i].toTensor())) {
      continue;
    }
    const auto* alias = schema.arguments()[i].alias_info();
    (alias != nullptr && alias->isWrite() ? written : read).push_back(args[i].toTensor());
  }
  for (const auto& w : written) {
    for (const auto& r : read) {
      at::assert_no_partial_overlap(w, r);
    }
  }
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
  check_no_partial_overlap(schema, originals);
  // torch's device rule for the tensor arguments (index lists, Tensor?[], may stay on the CPU as torch allows): one
  // device, index included; a CPU tensor only as a 0-d scalar.
  std::optional<at::Device> common;
  for (const auto& iv : originals) {
    if (!iv.isTensor() || !iv.toTensor().defined()) {
      continue;
    }
    const auto& t = iv.toTensor();
    if (t.is_cpu()) {
      TORCH_CHECK(t.dim() == 0, "Expected all tensors to be on the same device, but found at least two devices, hpu:0 "
                  "and cpu! (in ", schema.operator_name(), ")");
    } else {
      TORCH_CHECK(!common || *common == t.device(), "Expected all tensors to be on the same device, but found at least "
                  "two devices, ", *common, " and ", t.device(), "! (in ", schema.operator_name(), ")");
      common = t.device();
    }
  }
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

std::atomic<int64_t> cpu_fallback_calls{0};

void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  if (has_host_arg(op, stack)) {
    host_fallback(op, stack);
    return;
  }
  cpu_fallback_calls.fetch_add(1);
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

// CompositeExplicitAutograd ops whose kernel allocates its result from a host-backed float64 argument's options
// (Habana storage, float32 precision: the gradient through z.real[1:] was lost) or reaches a backend op with no
// complex128 kernel (convolution_overrideable). Their host path is the CPU's own kernel, through host_fallback.
bool allocates_like(const c10::OperatorName& name) {
  static const std::set<std::string> ops = {"aten::slice_backward", "aten::diagonal_backward",
                                            "aten::masked_scatter_backward", "aten::convolution",
                                            "aten::convolution_backward"};
  return name.overload_name.empty() && ops.count(name.name) > 0;
}

// Registers structured_pu1 for every aten op whose PrivateUse1/PrivateUse2 slot holds the NonFunctional composite
// kernel, and for the composite ops above.
struct RegisterStructured {
  RegisterStructured() {
    static auto* lib = new torch::Library(torch::Library::IMPL, "aten", DispatchKey::PrivateUse1, __FILE__, __LINE__);
    static auto* lib2 = new torch::Library(torch::Library::IMPL, "aten", DispatchKey::PrivateUse2, __FILE__, __LINE__);
    auto& dispatcher = c10::Dispatcher::singleton();
    for (const auto& name : dispatcher.getAllOpNames()) {
      if (name.name.rfind("aten::", 0) != 0) {
        continue;
      }
      auto op = dispatcher.findOp(name);
      if (!op || !(op->hasKernelForDispatchKey(DispatchKey::CompositeExplicitAutogradNonFunctional) ||
                   (op->hasKernelForDispatchKey(DispatchKey::CompositeExplicitAutograd) && allocates_like(name))) ||
          op->hasKernelForDispatchKey(DispatchKey::PrivateUse3)) {
        continue;
      }
      auto unqualified = name.name.substr(6) + (name.overload_name.empty() ? "" : "." + name.overload_name);
      if (!op->hasKernelForDispatchKey(DispatchKey::PrivateUse1)) {
        lib->impl(unqualified.c_str(), torch::CppFunction::makeFromBoxedFunction<&structured_pu1>());
      }
      if (!op->hasKernelForDispatchKey(DispatchKey::PrivateUse2)) {
        lib2->impl(unqualified.c_str(), torch::CppFunction::makeFromBoxedFunction<&structured_pu1>());
      }
    }
  }
};

// ---- angle, eq.Scalar_out, im2col, col2im (PrivateUse1) ---------------------------------------------------

at::Tensor angle_pu1(const at::Tensor& self) {
  if (host_backed(self)) {
    return from_cpu(at::angle(host_cpu(self)), self.device());
  }
  HPU_COMPLEX_NO_AUTOCAST;
  if (!self.is_complex()) {
    return at::angle(self);
  }
  auto pair = values_pair(self);
  return at::atan2(pair.select(-1, 1), pair.select(-1, 0));
}

at::Tensor& angle_out_pu1(const at::Tensor& self, at::Tensor& out) {
  return write_out(out, angle_pu1(self), {&self});
}

at::Tensor& eq_scalar_out_pu1(const at::Tensor& self, const at::Scalar& other, at::Tensor& out) {
  if (host_backed(self)) {
    return write_out(out, from_cpu(at::eq(host_cpu(self), other), self.device()), {&self});
  }
  HPU_COMPLEX_NO_AUTOCAST;
  if (!self.is_complex()) {
    if (other.isComplex()) {
      const auto c = other.toComplexDouble();
      auto eq_real = self == static_cast<float>(c.real());
      auto eq_val = static_cast<float>(c.imag()) == 0.0f ? eq_real : at::zeros_like(eq_real);
      return write_out(out, eq_val, {&self});
    }
    return write_out(out, at::eq(self, other), {&self});
  }
  auto pair = values_pair(self);
  const auto c = other.toComplexDouble();
  auto eq_val = (pair.select(-1, 0) == static_cast<float>(c.real()))
                    .logical_and(pair.select(-1, 1) == static_cast<float>(c.imag()));
  return write_out(out, eq_val, {&self});
}

constexpr int64_t kMaxIm2colIndex = int64_t{1} << 24;

bool valid_im2col_params(c10::IntArrayRef kernel_size, c10::IntArrayRef dilation, c10::IntArrayRef padding,
                         c10::IntArrayRef stride) {
  return kernel_size.size() == 2 && dilation.size() == 2 && padding.size() == 2 && stride.size() == 2 &&
         kernel_size[0] > 0 && kernel_size[1] > 0 && dilation[0] > 0 && dilation[1] > 0 && padding[0] >= 0 &&
         padding[1] >= 0 && stride[0] > 0 && stride[1] > 0;
}

std::pair<int64_t, int64_t> im2col_out_size(int64_t h, int64_t w, int64_t kh, int64_t kw, int64_t dh, int64_t dw,
                                            int64_t ph, int64_t pw, int64_t sh, int64_t sw) {
  const int64_t h_num = h + 2 * ph - dh * (kh - 1) - 1;
  const int64_t w_num = w + 2 * pw - dw * (kw - 1) - 1;
  const int64_t oh = h_num >= 0 ? (h_num / sh + 1) : 0;
  const int64_t ow = w_num >= 0 ? (w_num / sw + 1) : 0;
  return {oh, ow};
}

bool is_4d_non_overlap(int64_t h, int64_t w, int64_t kh, int64_t kw, int64_t dh, int64_t dw, int64_t ph, int64_t pw,
                       int64_t sh, int64_t sw) {
  return dh == 1 && dw == 1 && ph == 0 && pw == 0 && sh == kh && sw == kw &&
         ((w == 1 && kw == 1) || (h == 1 && kh == 1));
}

at::Tensor im2col_pu1(const at::Tensor& self, c10::IntArrayRef kernel_size, c10::IntArrayRef dilation,
                      c10::IntArrayRef padding, c10::IntArrayRef stride) {
  if (host_backed(self)) {
    return from_cpu(at::im2col(host_cpu(self), kernel_size, dilation, padding, stride), self.device());
  }
  if (!self.is_complex()) {
    return at::im2col(self, kernel_size, dilation, padding, stride);
  }
  const int64_t ndim = self.dim();
  if ((ndim != 3 && ndim != 4) || !valid_im2col_params(kernel_size, dilation, padding, stride)) {
    return at::im2col(cpu_operand(self), kernel_size, dilation, padding, stride).to(self.device());
  }
  const int64_t n = ndim == 4 ? self.size(0) : 1;
  const int64_t c = ndim == 4 ? self.size(1) : self.size(0);
  const int64_t h = ndim == 4 ? self.size(2) : self.size(1);
  const int64_t w = ndim == 4 ? self.size(3) : self.size(2);
  if (c <= 0 || h <= 0 || w <= 0) {
    return at::im2col(cpu_operand(self), kernel_size, dilation, padding, stride).to(self.device());
  }
  const int64_t kh = kernel_size[0], kw = kernel_size[1];
  const int64_t dh = dilation[0], dw = dilation[1];
  const int64_t ph = padding[0], pw = padding[1];
  const int64_t sh = stride[0], sw = stride[1];
  const auto [oh, ow] = im2col_out_size(h, w, kh, kw, dh, dw, ph, pw, sh, sw);
  if (oh <= 0 || ow <= 0) {
    return at::im2col(cpu_operand(self), kernel_size, dilation, padding, stride).to(self.device());
  }
  if (kh * kw * oh * ow > kMaxIm2colIndex) {
    cpu_fallback_calls.fetch_add(1);
    TORCH_WARN_ONCE("hpu_complex: aten::im2col on a complex HPU tensor runs on the CPU");
    return at::im2col(cpu_operand(self), kernel_size, dilation, padding, stride).to(self.device());
  }
  if (n == 0) {
    const std::vector<int64_t> empty_sizes{0, c * kh * kw, oh * ow};
    return make_complex_empty(empty_sizes, c10::contiguous_strides(empty_sizes), self.device());
  }
  HPU_COMPLEX_NO_AUTOCAST;
  auto pair = values_pair(self);
  if (ndim == 3) {
    pair = pair.unsqueeze(0);
  }
  at::Tensor res_pair;
  if (is_4d_non_overlap(h, w, kh, kw, dh, dw, ph, pw, sh, sw)) {
    if (w == 1 && kw == 1) {
      auto t = pair.reshape({n * c, h, 2});
      if (h > oh * kh) {
        t = t.narrow(1, 0, oh * kh);
      }
      res_pair = t.reshape({n * c, oh, kh, 2}).transpose(1, 2).contiguous();
    } else {
      auto t = pair.reshape({n * c, w, 2});
      if (w > ow * kw) {
        t = t.narrow(1, 0, ow * kw);
      }
      res_pair = t.reshape({n * c, ow, kw, 2}).transpose(1, 2).contiguous();
    }
  } else {
    const int64_t hp = h + 2 * ph;
    const int64_t wp = w + 2 * pw;
    auto iopts = at::TensorOptions().dtype(at::kLong).device(self.device());
    auto ky = at::arange(kh, iopts) * dh;
    auto kx = at::arange(kw, iopts) * dw;
    auto oy = at::arange(oh, iopts) * sh;
    auto ox = at::arange(ow, iopts) * sw;
    auto rows = ky.view({kh, 1, 1, 1}) + oy.view({1, 1, oh, 1});
    auto cols = kx.view({1, kw, 1, 1}) + ox.view({1, 1, 1, ow});
    auto idx = (rows * wp + cols).reshape({-1});
    auto xp = pair.reshape({n * c, h, w, 2}).permute({0, 3, 1, 2}).contiguous();
    if (ph > 0 || pw > 0) {
      xp = at::constant_pad_nd(xp, {pw, pw, ph, ph});
    }
    auto gathered = xp.reshape({n * c * 2, hp * wp}).index_select(1, idx);
    res_pair = gathered.reshape({n * c, 2, kh * kw * oh * ow}).transpose(1, 2).contiguous();
  }
  if (ndim == 3) {
    return complex_result(res_pair.reshape({c * kh * kw, oh * ow, 2}));
  }
  return complex_result(res_pair.reshape({n, c * kh * kw, oh * ow, 2}));
}

at::Tensor& im2col_out_pu1(const at::Tensor& self, c10::IntArrayRef kernel_size, c10::IntArrayRef dilation,
                           c10::IntArrayRef padding, c10::IntArrayRef stride, at::Tensor& out) {
  return write_out(out, im2col_pu1(self, kernel_size, dilation, padding, stride), {&self}, /*matrix_mm=*/true);
}

at::Tensor col2im_pu1(const at::Tensor& self, c10::IntArrayRef output_size, c10::IntArrayRef kernel_size,
                      c10::IntArrayRef dilation, c10::IntArrayRef padding, c10::IntArrayRef stride) {
  if (host_backed(self)) {
    return from_cpu(at::col2im(host_cpu(self), output_size, kernel_size, dilation, padding, stride), self.device());
  }
  if (!self.is_complex()) {
    return at::col2im(self, output_size, kernel_size, dilation, padding, stride);
  }
  const int64_t ndim = self.dim();
  if ((ndim != 2 && ndim != 3) || output_size.size() != 2 ||
      !valid_im2col_params(kernel_size, dilation, padding, stride)) {
    return at::col2im(cpu_operand(self), output_size, kernel_size, dilation, padding, stride).to(self.device());
  }
  const int64_t h = output_size[0], w = output_size[1];
  const int64_t kh = kernel_size[0], kw = kernel_size[1];
  const int64_t dh = dilation[0], dw = dilation[1];
  const int64_t ph = padding[0], pw = padding[1];
  const int64_t sh = stride[0], sw = stride[1];
  const int64_t n = ndim == 3 ? self.size(0) : 1;
  const int64_t c_kw = self.size(-2);
  const int64_t length = self.size(-1);
  if (h <= 0 || w <= 0 || c_kw <= 0 || c_kw % (kh * kw) != 0) {
    return at::col2im(cpu_operand(self), output_size, kernel_size, dilation, padding, stride).to(self.device());
  }
  const auto [oh, ow] = im2col_out_size(h, w, kh, kw, dh, dw, ph, pw, sh, sw);
  if (oh <= 0 || ow <= 0 || length != oh * ow) {
    return at::col2im(cpu_operand(self), output_size, kernel_size, dilation, padding, stride).to(self.device());
  }
  const int64_t c = c_kw / (kh * kw);
  if (kh * kw * length > kMaxIm2colIndex) {
    cpu_fallback_calls.fetch_add(1);
    TORCH_WARN_ONCE("hpu_complex: aten::col2im on a complex HPU tensor runs on the CPU");
    return at::col2im(cpu_operand(self), output_size, kernel_size, dilation, padding, stride).to(self.device());
  }
  if (n == 0) {
    const std::vector<int64_t> empty_sizes{0, c, h, w};
    return make_complex_empty(empty_sizes, c10::contiguous_strides(empty_sizes), self.device());
  }
  HPU_COMPLEX_NO_AUTOCAST;
  auto pair = values_pair(self);
  if (ndim == 2) {
    pair = pair.unsqueeze(0);
  }
  at::Tensor res_pair;
  if (is_4d_non_overlap(h, w, kh, kw, dh, dw, ph, pw, sh, sw)) {
    if (w == 1 && kw == 1) {
      auto t = pair.reshape({n * c, kh, oh, 2}).transpose(1, 2).contiguous().reshape({n * c, oh * kh, 1, 2});
      if (h > oh * kh) {
        t = at::cat({t, at::zeros({n * c, h - oh * kh, 1, 2}, t.options())}, 1);
      }
      res_pair = t;
    } else {
      auto t = pair.reshape({n * c, kw, ow, 2}).transpose(1, 2).contiguous().reshape({n * c, 1, ow * kw, 2});
      if (w > ow * kw) {
        t = at::cat({t, at::zeros({n * c, 1, w - ow * kw, 2}, t.options())}, 2);
      }
      res_pair = t;
    }
  } else {
    const int64_t hp = h + 2 * ph;
    const int64_t wp = w + 2 * pw;
    auto iopts = at::TensorOptions().dtype(at::kLong).device(self.device());
    auto ky = at::arange(kh, iopts) * dh;
    auto kx = at::arange(kw, iopts) * dw;
    auto oy = at::arange(oh, iopts) * sh;
    auto ox = at::arange(ow, iopts) * sw;
    auto rows = ky.view({kh, 1, 1, 1}) + oy.view({1, 1, oh, 1});
    auto cols = kx.view({1, kw, 1, 1}) + ox.view({1, 1, 1, ow});
    auto idx = (rows * wp + cols).reshape({-1});
    auto src = pair.reshape({n * c, kh * kw * length, 2}).transpose(1, 2).contiguous().reshape({n * c * 2, kh * kw * length});
    auto acc = at::zeros({n * c * 2, hp * wp}, src.options()).index_add(1, idx, src).reshape({n * c, 2, hp, wp});
    if (ph > 0 || pw > 0) {
      acc = acc.narrow(2, ph, h).narrow(3, pw, w);
    }
    res_pair = acc.permute({0, 2, 3, 1}).contiguous();
  }
  if (ndim == 2) {
    return complex_result(res_pair.reshape({c, h, w, 2}));
  }
  return complex_result(res_pair.reshape({n, c, h, w, 2}));
}

at::Tensor& col2im_out_pu1(const at::Tensor& self, c10::IntArrayRef output_size, c10::IntArrayRef kernel_size,
                           c10::IntArrayRef dilation, c10::IntArrayRef padding, c10::IntArrayRef stride,
                           at::Tensor& out) {
  return write_out(out, col2im_pu1(self, output_size, kernel_size, dilation, padding, stride), {&self},
                   /*matrix_mm=*/true);
}

}  // namespace

TORCH_LIBRARY(hpu_complex, m) {
  m.def("set_lazy_mode(bool enabled) -> ()");
  m.impl("set_lazy_mode", &set_lazy_mode);
  m.def("set_matmul_cpu_threshold(int macs) -> ()");
  m.impl("set_matmul_cpu_threshold", &set_matmul_cpu_threshold);
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&cpu_fallback>());
}

TORCH_LIBRARY_IMPL(_, PrivateUse2, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&cpu_fallback>());
}

TORCH_LIBRARY_IMPL(aten, AutogradHPU, m) {
  m.impl("_fft_r2c", &fft_r2c_autograd);
}

// A real HPU tensor and a complex scalar: promoted before Habana's float kernels see it (lazy mode aborts there).
constexpr const char* kComplexScalarOps[] = {
    "mul.Tensor", "mul.Scalar", "mul_.Tensor", "mul_.Scalar", "mul.out",
    "add.Tensor", "add.Scalar", "add_.Tensor", "add_.Scalar", "add.out",
    "sub.Tensor", "sub.Scalar", "sub_.Tensor", "sub_.Scalar", "sub.out",
    "rsub.Tensor", "rsub.Scalar", "mul.Scalar_out", "add.Scalar_out", "sub.Scalar_out",
    "div.Scalar_out", "div.Scalar_mode_out", "rsub.Tensor_out", "rsub.Scalar_out",
    "div.Tensor", "div.Scalar", "div_.Tensor", "div_.Scalar", "div.out",
    "div.Tensor_mode", "div.Scalar_mode", "div_.Tensor_mode", "div_.Scalar_mode",
    "div.out_mode"};

TORCH_LIBRARY_IMPL(aten, AutogradHPU, m) {
  for (const char* name : kComplexScalarOps) {
    m.impl(name, torch::CppFunction::makeFromBoxedFunction<&complex_scalar_autograd_hpu>());
  }
}

TORCH_LIBRARY_IMPL(aten, BackendSelect, m) {
  // the calls that skip AutogradHPU (torch.inference_mode, inference tensors); the hot overloads unboxed first
#define HPU_COMPLEX_BS(op, written, name) m.impl(name, TORCH_FN((BackendSelectFast<at::_ops::op, written>::call)))
  HPU_COMPLEX_BS(mul_Tensor, -1, "mul.Tensor");
  HPU_COMPLEX_BS(mul_Scalar, -1, "mul.Scalar");
  HPU_COMPLEX_BS(mul__Tensor, 0, "mul_.Tensor");
  HPU_COMPLEX_BS(mul__Scalar, 0, "mul_.Scalar");
  HPU_COMPLEX_BS(mul_out, 2, "mul.out");
  HPU_COMPLEX_BS(add_Tensor, -1, "add.Tensor");
  HPU_COMPLEX_BS(add_Scalar, -1, "add.Scalar");
  HPU_COMPLEX_BS(add__Tensor, 0, "add_.Tensor");
  HPU_COMPLEX_BS(add__Scalar, 0, "add_.Scalar");
  HPU_COMPLEX_BS(add_out, 3, "add.out");
  HPU_COMPLEX_BS(sub_Tensor, -1, "sub.Tensor");
  HPU_COMPLEX_BS(sub_Scalar, -1, "sub.Scalar");
  HPU_COMPLEX_BS(sub__Tensor, 0, "sub_.Tensor");
  HPU_COMPLEX_BS(sub__Scalar, 0, "sub_.Scalar");
  HPU_COMPLEX_BS(sub_out, 3, "sub.out");
  HPU_COMPLEX_BS(div_Tensor, -1, "div.Tensor");
  HPU_COMPLEX_BS(div_Scalar, -1, "div.Scalar");
  HPU_COMPLEX_BS(div__Tensor, 0, "div_.Tensor");
  HPU_COMPLEX_BS(div__Scalar, 0, "div_.Scalar");
  HPU_COMPLEX_BS(div_out, 2, "div.out");
  HPU_COMPLEX_BS(rsub_Tensor, -1, "rsub.Tensor");
  HPU_COMPLEX_BS(rsub_Scalar, -1, "rsub.Scalar");
  HPU_COMPLEX_BS(div_Tensor_mode, -1, "div.Tensor_mode");
  HPU_COMPLEX_BS(div_Scalar_mode, -1, "div.Scalar_mode");
  HPU_COMPLEX_BS(div__Tensor_mode, 0, "div_.Tensor_mode");
  HPU_COMPLEX_BS(div__Scalar_mode, 0, "div_.Scalar_mode");
#undef HPU_COMPLEX_BS
  // the rarer out= forms: boxed
  for (const char* name : {"mul.Scalar_out", "add.Scalar_out", "sub.Scalar_out", "div.Scalar_out",
                           "div.Scalar_mode_out", "rsub.Tensor_out", "rsub.Scalar_out", "div.out_mode"}) {
    m.impl(name, torch::CppFunction::makeFromBoxedFunction<&complex_scalar_backend_select>());
  }
  m.impl("empty.memory_format", &empty_backend_select);
  m.impl("empty_strided", &empty_strided_backend_select);
  m.impl("scalar_tensor", &scalar_tensor_backend_select);
}

TORCH_LIBRARY_IMPL(aten, HPU, m) {
  // Real HPU inputs only: these produce the tagged complex tensors.
  m.impl("_fft_r2c", &fft_r2c_hpu);
  m.impl("view_as_complex", &view_as_complex_hpu);
  m.impl("view.dtype", &view_dtype_hpu);  // real -> complex64 views get the complex key
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
  // unfold (repeat and tile call it): ATen's own view, on as_strided above.
  m.impl("unfold", TORCH_FN(at::native::unfold));
  m.impl("set_.source_Tensor", &set_tensor_pu1);
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
  m.impl("neg", &neg_pu1);
  m.impl("neg.out", &neg_out_pu1);
  m.impl("exp", &exp_pu1);
  m.impl("exp.out", &exp_out_pu1);
  m.impl("masked_fill_.Scalar", &masked_fill_pu1);
  m.impl("masked_fill_.Tensor", &masked_fill_tensor_pu1);
  m.impl("cat", &cat_pu1);
  m.impl("cat.out", &cat_out_pu1);
  m.impl("linalg_vector_norm", &vector_norm_pu1);
  m.impl("linalg_vector_norm.out", &vector_norm_out_pu1);
  m.impl("mm", &mm_pu1);
  m.impl("mm.out", &mm_out_pu1);
  m.impl("bmm", &bmm_pu1);
  m.impl("bmm.out", &bmm_out_pu1);
  m.impl("sum.dim_IntList", &sum_pu1);
  m.impl("sum.IntList_out", &sum_out_pu1);
  m.impl("_copy_from_and_resize", &copy_from_and_resize_pu1);
  m.impl("copy_", &copy_pu1);
  m.impl("fill_.Scalar", &fill_pu1);
  m.impl("zero_", &zero_pu1);
  m.impl("set_.source_Storage_storage_offset", &set_storage_pu1);
  m.impl("set_.source_Storage", &set_source_storage_pu1);
  m.impl("set_", &set_empty_pu1);
  m.impl("angle", &angle_pu1);
  m.impl("angle.out", &angle_out_pu1);
  m.impl("eq.Scalar_out", &eq_scalar_out_pu1);
  m.impl("im2col", &im2col_pu1);
  m.impl("im2col.out", &im2col_out_pu1);
  m.impl("col2im", &col2im_pu1);
  m.impl("col2im.out", &col2im_out_pu1);
}

TORCH_LIBRARY_FRAGMENT(hpu_complex, m) {  // a fragment: Team C's branch may define the namespace too
  m.def("host_deepcopy(Tensor self, Tensor? storage_of=None) -> Tensor", &host_deepcopy);
  // torch.load of a host-backed float64 tensor (hpu_complex.__init__): its CPU values back into host memory.
  m.def("host_from_cpu(Tensor cpu, Device device) -> Tensor",
        [](const at::Tensor& cpu, const at::Device& device) {
          TORCH_CHECK(cpu.is_cpu() && (cpu.scalar_type() == at::kDouble || cpu.scalar_type() == at::kComplexDouble),
                      "hpu_complex::host_from_cpu: a CPU float64 or complex128 tensor");
          return from_cpu(cpu, device);
        });
  // Whether a host-backed tensor was ever allocated: torch.compile(hpu_backend) then keeps float64 nodes eager.
  m.def("host_in_use() -> bool", []() { return host_in_use(); });
  // Whether a tensor lives in host memory (complex128, its real and dtype views, float64 results): torch.save,
  // copy.deepcopy and the torch.compile guards ask this, never the dtype.
  m.def("is_host_backed(Tensor t) -> bool", [](const at::Tensor& t) { return host_backed(t); });
  // How many calls reached the CPU fallback (warned once per op; card checks compare this count around a call).
  m.def("cpu_fallback_calls() -> int", []() { return cpu_fallback_calls.load(); });
  // Lazy mode: how many complex64 storages hold a Habana tensor (lazy_check checks that it drains).
  m.def("lazy_storages_alive() -> int", []() { return lazy_owners_alive.load(); });
  // A tensor over a Python storage slice (hpu_complex.extension): set_ then sees through slices of its storages.
  m.def("note_storage_slice(Tensor t) -> ()", &note_storage_slice);
}

// After the registrations above (same translation unit: initialized in order).
const RegisterStructured register_structured;

// CPU tests only: a fake HPU backend for stock (CPU) torch, so the extension's C++ kernels run without a Gaudi.
//
// - A no-op HPU device guard: stock torch accepts device="hpu" tensors.
// - HPU hooks: torch's accelerator API reports the fake HPU (autograd asks it for streams of HPU gradients).
// - HPU factories (empty.memory_format, empty_strided) allocate host memory that reports the HPU device (hpu:N kept).
// - HPU views and restrides are ATen's own TensorImpl kernels (as_strided, view, _reshape_alias, unfold, resize_,
//   set_), so views of a fake HPU tensor alias it as on the card.
// - Every other HPU op is a boxed fallback that runs torch's CPU kernel on CPU aliases of the same memory (one CPU
//   storage per HPU storage within a call, so torch's own overlap checks still see aliasing operands), writes in
//   place, and returns fresh results as fake HPU tensors on the first HPU operand's device.
//
// Habana's own float kernels are therefore stood in by torch's CPU kernels: the extension's dispatch, its pair
// formulas, storages, out= handling and checks run for real; Habana-specific behavior (lazy mode, its float32 GEMM,
// its layout bugs) does not, and stays with the card checks (pu1_check.py, lazy_check.py, ...).
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/EmptyTensor.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/detail/HPUHooksInterface.h>
#include <ATen/detail/MPSHooksInterface.h>
#include <ATen/native/Resize.h>
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/resize_native.h>
#include <ATen/ops/set_native.h>
#include <ATen/ops/unfold_native.h>
#include <ATen/ops/view_native.h>
#include <ATen/ops/_reshape_alias_native.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/impl/alloc_cpu.h>
#include <torch/library.h>
#include <torch/version.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <vector>

namespace {

// A no-op HPU device guard with no-op streams and events: once torch's accelerator API reports the HPU (the hooks
// below), autograd records and waits on events of HPU streams. Streams name the device they were asked for (the stock
// no-op guard says device -1, which the engine would use as an index into its per-device streams).
struct FakeHpuGuardImpl final : c10::impl::DeviceGuardImplInterface {
  static c10::Device indexed(c10::Device d) {
    return c10::Device(c10::DeviceType::HPU, d.has_index() ? d.index() : 0);
  }
  c10::DeviceType type() const override {
    return c10::DeviceType::HPU;
  }
  c10::Device exchangeDevice(c10::Device) const override {
    return c10::Device(c10::DeviceType::HPU, 0);
  }
  c10::Device getDevice() const override {
    return c10::Device(c10::DeviceType::HPU, 0);
  }
  void setDevice(c10::Device) const override {}
  void uncheckedSetDevice(c10::Device) const noexcept override {}
  c10::Stream getStream(c10::Device d) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, indexed(d));
  }
  c10::Stream getNewStream(c10::Device d, int = 0) const override {
    return c10::Stream(c10::Stream::DEFAULT, indexed(d));
  }
  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, indexed(s.device()));
  }
  c10::DeviceIndex deviceCount() const noexcept override {
    return 1;
  }
  // Host memory, one default stream: work is done when it is issued, so an event is always complete.
  void record(void**, const c10::Stream&, const c10::DeviceIndex, const c10::EventFlag) const override {}
  void block(void*, const c10::Stream&) const override {}
  bool queryEvent(void*) const override {
    return true;
  }
  void destroyEvent(void*, const c10::DeviceIndex) const noexcept override {}
  void synchronizeEvent(void*) const override {}
  bool queryStream(const c10::Stream&) const override {
    return true;
  }
  void synchronizeStream(const c10::Stream&) const override {}
  void recordDataPtrOnStream(const c10::DataPtr&, const c10::Stream&) const override {}
};

}  // namespace

C10_REGISTER_GUARD_IMPL(HPU, FakeHpuGuardImpl);

namespace {

// HPU hooks, so torch's accelerator API (at::accelerator::getAccelerator) finds the fake HPU as the habana bridge's
// hooks make it find the card. Autograd routes gradients of accelerator tensors (HPU is one) through
// at::accelerator::getCurrentStream, which throws "Cannot access accelerator device when none is available" when no
// accelerator answers. Where torch already has one (MPS on macOS, CUDA), the fake HPU stays undetected, since torch
// refuses two accelerators at once and its streams then stand in for the HPU's no-op ones.
bool other_accelerator() {
#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 8)
  return at::detail::getCUDAHooks().isBuilt() || at::detail::getMPSHooks().isBuilt();
#else
  return at::hasCUDA() || at::hasMPS();
#endif
}

struct FakeHpuHooks final : at::HPUHooksInterface {
  explicit FakeHpuHooks(at::HPUHooksArgs) {}
  void init() const override {}
  bool hasHPU() const override {
    return !other_accelerator();
  }
#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 8)
  bool isBuilt() const override {
    return !other_accelerator();
  }
  bool isAvailable() const override {
    return !other_accelerator();
  }
#endif
  c10::DeviceIndex deviceCount() const override {
    return 1;
  }
  c10::DeviceIndex getCurrentDevice() const override {
    return 0;
  }
  bool hasPrimaryContext(c10::DeviceIndex) const override {
    return true;
  }
  bool isPinnedPtr(const void*) const override {
    return false;
  }
};

}  // namespace

namespace at {
// at::detail::getHPUHooks() creates the "HPUHooks" entry (REGISTER_HPU_HOOKS would key it by the class name)
C10_REGISTER_TYPED_CLASS(HPUHooksRegistry, "HPUHooks", FakeHpuHooks);
}  // namespace at

namespace {

void fake_free(void* p) {
  c10::free_cpu(p);
}

struct FakeHpuAllocator final : c10::Allocator {
  c10::DataPtr allocate(size_t n) override {
    void* p = c10::alloc_cpu(std::max<size_t>(n, 1));
    return {p, p, &fake_free, c10::Device(c10::DeviceType::HPU, 0)};
  }
  c10::DeleterFnPtr raw_deleter() const override {
    return &fake_free;
  }
  void copy_data(void* dest, const void* src, std::size_t count) const override {
    std::memcpy(dest, src, count);
  }
};

FakeHpuAllocator& allocator() {
  static FakeHpuAllocator a;
  return a;
}

c10::Device hpu_device(std::optional<c10::Device> device) {
  if (!device || !device->has_index()) {
    return c10::Device(c10::DeviceType::HPU, 0);
  }
  return *device;
}

at::Tensor make_hpu(c10::IntArrayRef size, c10::IntArrayRef stride, caffe2::TypeMeta dtype, c10::Device device) {
  const auto nbytes = at::detail::computeStorageNbytes(size, stride, dtype.itemsize());
  auto data = allocator().allocate(nbytes);
  data.unsafe_set_device(device);
  auto storage = c10::make_intrusive<c10::StorageImpl>(c10::StorageImpl::use_byte_size_t(), nbytes, std::move(data),
                                                       &allocator(), /*resizable=*/true);
  auto t = at::detail::make_tensor<at::TensorImpl>(c10::Storage(std::move(storage)), c10::DispatchKeySet(c10::DispatchKey::HPU),
                                                   dtype);
  t.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);
  return t;
}

at::Tensor empty_hpu(c10::IntArrayRef size, std::optional<at::ScalarType> dtype, std::optional<at::Layout>,
                     std::optional<at::Device> device, std::optional<bool>,
                     std::optional<at::MemoryFormat> memory_format) {
  auto t = make_hpu(size, c10::contiguous_strides(size),
                    c10::scalarTypeToTypeMeta(dtype.value_or(at::get_default_dtype_as_scalartype())),
                    hpu_device(device));
  if (memory_format && *memory_format != at::MemoryFormat::Preserve) {
    t.unsafeGetTensorImpl()->empty_tensor_restride(*memory_format);
  }
  return t;
}

at::Tensor empty_strided_hpu(c10::IntArrayRef size, c10::IntArrayRef stride, std::optional<at::ScalarType> dtype,
                             std::optional<at::Layout>, std::optional<at::Device> device, std::optional<bool>) {
  return make_hpu(size, stride, c10::scalarTypeToTypeMeta(dtype.value_or(at::get_default_dtype_as_scalartype())),
                  hpu_device(device));
}

const at::Tensor& resize_hpu(const at::Tensor& self, c10::IntArrayRef size,
                             std::optional<at::MemoryFormat> memory_format) {
  const auto device = self.device();
  at::native::resize_(self, size, memory_format);  // the CPU kernel: grows through the storage's allocator
  if (self.has_storage() && self.storage().data_ptr().device() != device) {
    self.storage().unsafeGetStorageImpl()->mutable_data_ptr().unsafe_set_device(device);
  }
  return self;
}

at::Tensor& set_storage_hpu(at::Tensor& self, at::Storage source, int64_t offset, c10::IntArrayRef size,
                            c10::IntArrayRef stride) {
  at::native::checkSetStorage(self, std::move(source), offset, size, stride);
  at::native::setStrided(self, size, stride.empty() ? c10::IntArrayRef(c10::contiguous_strides(size)) : stride,
                         offset);
  return self;
}

at::Tensor& set_empty_hpu(at::Tensor& self) {
  auto fresh = make_hpu({0}, {1}, self.dtype(), self.device());
  return set_storage_hpu(self, fresh.storage(), 0, {0}, {});
}

at::Tensor& set_source_storage_hpu(at::Tensor& self, at::Storage source) {
  const auto n = static_cast<int64_t>(source.nbytes() / self.dtype().itemsize());
  return set_storage_hpu(self, std::move(source), 0, {n}, {});
}

// ---- the boxed fallback --------------------------------------------------------------------------

struct CpuAliases {
  std::map<c10::StorageImpl*, at::Storage> cpu_of;      // HPU storage -> its CPU storage (same memory)
  std::map<c10::StorageImpl*, at::Storage> hpu_of;      // and back
  std::optional<c10::Device> device;

  at::Tensor alias(const at::Tensor& t) {
    if (!t.defined() || t.device().type() != c10::DeviceType::HPU) {
      return t;
    }
    if (!device) {
      device = t.device();
    }
    auto* key = t.storage().unsafeGetStorageImpl();
    auto it = cpu_of.find(key);
    if (it == cpu_of.end()) {
      auto* keep = new at::Storage(t.storage());
      void* data = t.storage().mutable_data();
      at::Storage cpu(c10::Storage::use_byte_size_t(), t.storage().nbytes(),
                      c10::DataPtr(data, keep, [](void* ctx) { delete static_cast<at::Storage*>(ctx); },
                                   c10::Device(c10::DeviceType::CPU)),
                      c10::GetCPUAllocator(), /*resizable=*/true);
      it = cpu_of.emplace(key, cpu).first;
      hpu_of.emplace(cpu.unsafeGetStorageImpl(), t.storage());
    }
    auto v = at::empty({0}, at::TensorOptions().dtype(t.dtype()).device(at::kCPU));
    v.set_(it->second, t.storage_offset(), t.sizes(), t.strides());
    v._set_conj(t.is_conj());
    v._set_neg(t.is_neg());
    return v;
  }

  // A CPU result back on the fake HPU: a view of an operand's memory stays one, a fresh result is copied.
  at::Tensor back(const at::Tensor& r) {
    if (!r.defined() || !r.is_cpu()) {
      return r;
    }
    auto it = hpu_of.find(r.storage().unsafeGetStorageImpl());
    if (it != hpu_of.end()) {
      auto v = at::detail::make_tensor<at::TensorImpl>(at::Storage(it->second),
                                                       c10::DispatchKeySet(c10::DispatchKey::HPU), r.dtype());
      v.unsafeGetTensorImpl()->set_sizes_and_strides(r.sizes(), r.strides());
      v.unsafeGetTensorImpl()->set_storage_offset(r.storage_offset());
      return v;
    }
    const auto nbytes = r.storage().nbytes();
    auto out = make_hpu({static_cast<int64_t>(nbytes)}, {1}, caffe2::TypeMeta::Make<uint8_t>(),
                        device.value_or(c10::Device(c10::DeviceType::HPU, 0)));
    if (nbytes > 0) {
      std::memcpy(out.storage().mutable_data(), r.storage().data(), nbytes);
    }
    auto v = at::detail::make_tensor<at::TensorImpl>(c10::Storage(out.storage()), c10::DispatchKeySet(c10::DispatchKey::HPU),
                                                     r.dtype());
    v.unsafeGetTensorImpl()->set_sizes_and_strides(r.sizes(), r.strides());
    v.unsafeGetTensorImpl()->set_storage_offset(r.storage_offset());
    v._set_conj(r.is_conj());
    v._set_neg(r.is_neg());
    return v;
  }
};

void fake_hpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  const auto n = schema.arguments().size();
  const auto start = stack->size() - n;
  const std::vector<c10::IValue> originals(stack->begin() + start, stack->end());
  CpuAliases aliases;
  std::vector<std::pair<at::Tensor, at::Tensor>> written;  // (HPU argument, its CPU alias)
  for (size_t i = 0; i < n; ++i) {
    auto& iv = (*stack)[start + i];
    const auto* info = schema.arguments()[i].alias_info();
    const bool write = info != nullptr && info->isWrite();
    if (iv.isTensor()) {
      const auto t = iv.toTensor();
      iv = aliases.alias(t);
      if (write && t.defined() && t.device().type() == c10::DeviceType::HPU) {
        written.emplace_back(t, iv.toTensor());
      }
    } else if (iv.isTensorList()) {
      auto list = iv.toTensorList().copy();
      for (size_t j = 0; j < list.size(); ++j) {
        const at::Tensor t = list.get(j);
        list.set(j, aliases.alias(t));
        if (write && t.defined() && t.device().type() == c10::DeviceType::HPU) {
          written.emplace_back(t, list.get(j));
        }
      }
      iv = list;
    } else if (iv.isList()) {
      auto list = iv.toList().copy();
      for (size_t j = 0; j < list.size(); ++j) {
        if (list.get(j).isTensor()) {
          list.set(j, aliases.alias(list.get(j).toTensor()));
        }
      }
      iv = list;
    }
  }
  op.redispatchBoxed(c10::DispatchKeySet(c10::DispatchKey::CPU), stack);
  // A written argument the CPU kernel resized (an out=) got new CPU memory: resize the HPU tensor and copy.
  for (auto& [hpu, cpu] : written) {
    const bool moved = aliases.hpu_of.find(cpu.storage().unsafeGetStorageImpl()) == aliases.hpu_of.end() ||
                       cpu.storage().data() != hpu.storage().data();
    if (moved || !cpu.sizes().equals(hpu.sizes()) || !cpu.strides().equals(hpu.strides())) {
      if (!cpu.sizes().equals(hpu.sizes())) {
        resize_hpu(hpu, cpu.sizes(), std::nullopt);
      }
      if (moved) {
        CpuAliases fresh;
        fresh.alias(hpu).copy_(cpu);
      }
    }
  }
  const auto& returns = schema.returns();
  const auto rstart = stack->size() - returns.size();
  for (size_t i = 0; i < returns.size(); ++i) {
    auto& iv = (*stack)[rstart + i];
    const auto* info = returns[i].alias_info();
    if (info != nullptr && info->isWrite()) {
      for (size_t j = 0; j < n; ++j) {
        const auto* a = schema.arguments()[j].alias_info();
        if (a != nullptr && *a == *info) {
          iv = originals[j];
        }
      }
    } else if (iv.isTensor()) {
      iv = aliases.back(iv.toTensor());
    } else if (iv.isTensorList()) {
      std::vector<at::Tensor> moved;
      for (const at::Tensor& t : iv.toTensorList()) {
        moved.push_back(aliases.back(t));
      }
      iv = c10::List<at::Tensor>(at::ArrayRef<at::Tensor>(moved));
    }
  }
}

// copy_ (CompositeExplicitAutograd) hands a copy with an HPU side to _copy_from, which has no CPU kernel.
at::Tensor copy_from_hpu(const at::Tensor& self, const at::Tensor& dst, bool non_blocking) {
  CpuAliases aliases;
  aliases.alias(dst).copy_(aliases.alias(self), non_blocking);
  return dst;
}

at::Tensor copy_from_and_resize_hpu(const at::Tensor& self, const at::Tensor& dst) {
  if (!dst.sizes().equals(self.sizes())) {
    if (dst.device().type() == c10::DeviceType::HPU) {
      resize_hpu(dst, self.sizes(), std::nullopt);
    } else {
      dst.resize_(self.sizes());
    }
  }
  return copy_from_hpu(self, dst, false);
}

}  // namespace

TORCH_LIBRARY_IMPL(_, HPU, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&fake_hpu_fallback>());
}

TORCH_LIBRARY_IMPL(aten, HPU, m) {
  m.impl("empty.memory_format", &empty_hpu);
  m.impl("empty_strided", &empty_strided_hpu);
  m.impl("as_strided", TORCH_FN(at::native::as_strided_tensorimpl));
  m.impl("view", TORCH_FN(at::native::view));
  m.impl("_reshape_alias", TORCH_FN(at::native::_reshape_alias));
  m.impl("unfold", TORCH_FN(at::native::unfold));
  m.impl("resize_", &resize_hpu);
  m.impl("set_.source_Storage_storage_offset", &set_storage_hpu);
  m.impl("set_.source_Storage", &set_source_storage_hpu);
  m.impl("set_", &set_empty_hpu);
  m.impl("_copy_from", &copy_from_hpu);
  m.impl("_copy_from_and_resize", &copy_from_and_resize_hpu);
}

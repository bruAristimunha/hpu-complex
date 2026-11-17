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
//   HPU key (real HPU inputs): _fft_r2c, view_as_complex, complex.
//   BackendSelect (every factory passes it before a backend is chosen):
//     empty.memory_format / empty_strided with dtype complex64 on hpu; all
//     other factory calls are redispatched as the generated kernel does.
//   PrivateUse1: every op on a complex tensor. Each computes on float views
//     of the same storage, with the plain HPU key set, so Synapse never sees
//     a complex dtype. Unregistered ops fall back to the CPU (warned once).
//
// FFT: dense real DFT as matmuls (O(n^2)), bases cached per (n, norm, device).
// Norm int: 0=none, 1=by_root_n, 2=by_n (aten per-op scaling).
#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/InferSize.h>
#include <ATen/Operators.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/dispatch/DispatchKeyExtractor.h>
#include <ATen/native/CPUFallback.h>
#include <c10/core/impl/LocalDispatchKeySet.h>

#include <cmath>
#include <cstdlib>
#include <mutex>
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

DispatchKeySet hpu_float_keys() {
  static const DispatchKeySet keys = at::empty({0}, at::TensorOptions().dtype(at::kFloat).device(at::kHPU)).key_set();
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

// complex64 [...] -> float [..., 2] over the same storage, ignoring a conjugate
// bit: plain HPU keys for a tagged tensor, the stock view for a CPU one.
at::Tensor raw_pair(const at::Tensor& c) {
  TORCH_CHECK(c.scalar_type() == at::kComplexFloat, "hpu_complex: complex64 only, got ", c.scalar_type());
  if (!tagged(c)) {
    return at::view_as_real(c.is_conj() ? c.conj() : c);  // conj() of a conjugate view: the stored values
  }
  auto sizes = c.sizes().vec();
  auto strides = c.strides().vec();
  for (auto& s : strides) {
    s *= 2;
  }
  sizes.push_back(2);
  strides.push_back(1);
  return view_of(c, hpu_float_keys(), caffe2::TypeMeta::Make<float>(), sizes, strides, c.storage_offset() * 2);
}

// The pair of c's values. Conjugate views only reach copy_ (the Conjugate key
// passes copy_/clone/views through and resolves every other op's inputs first).
at::Tensor pair_view(const at::Tensor& c) {
  TORCH_CHECK(!c.is_conj(), "hpu_complex: unexpected conjugate view (resolve_conj() it first)");
  return raw_pair(c);
}

at::Tensor make_complex_empty(c10::IntArrayRef size, c10::IntArrayRef stride, at::Device device) {
  // Storage for the strided layout, in floats (two per complex element).
  int64_t extent = 1;
  for (size_t i = 0; i < size.size(); ++i) {
    if (size[i] == 0) {
      extent = 0;
      break;
    }
    extent += (size[i] - 1) * stride[i];
  }
  auto flat = at::empty({std::max<int64_t>(extent, 0) * 2}, at::TensorOptions().dtype(at::kFloat).device(device));
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

std::mutex basis_mutex;

// r2c, one GEMM: x [..., n] @ W [n, 2k] = interleaved [re0, im0, re1, ...].
at::Tensor r2c_basis(int64_t n, int64_t k, int64_t normalization, const at::Device& device) {
  static std::map<std::tuple<int64_t, int64_t, int64_t, int64_t>, at::Tensor> cache;
  std::lock_guard<std::mutex> lock(basis_mutex);
  auto key = std::make_tuple(n, k, normalization, static_cast<int64_t>(device.index()));
  auto it = cache.find(key);
  if (it != cache.end()) {
    return it->second;
  }
  auto ang = angles(n, n, k);  // [n, k]
  auto scale = norm_scale(normalization, n);
  auto w = at::stack({at::cos(ang) * scale, at::sin(ang) * -scale}, -1).reshape({n, 2 * k});
  return cache.emplace(key, w.to(at::kFloat).to(device)).first->second;
}

// c2c, one GEMM: [re, im] interleaved [..., 2n] @ W [2n, 2n] = interleaved output.
at::Tensor c2c_basis(int64_t n, bool forward, int64_t normalization, const at::Device& device) {
  static std::map<std::tuple<int64_t, bool, int64_t, int64_t>, at::Tensor> cache;
  std::lock_guard<std::mutex> lock(basis_mutex);
  auto key = std::make_tuple(n, forward, normalization, static_cast<int64_t>(device.index()));
  auto it = cache.find(key);
  if (it != cache.end()) {
    return it->second;
  }
  auto ang = angles(n, n, n);  // [j, f]
  auto scale = norm_scale(normalization, n);
  auto c = at::cos(ang) * scale;
  auto s = at::sin(ang) * (forward ? -scale : scale);  // e^{-+i}: X = sum x (c + i s)
  // out_re = re c - im s ; out_im = re s + im c, rows interleaved (re_j, im_j).
  auto from_re = at::stack({c, s}, -1);             // [j, f, 2]
  auto from_im = at::stack({-s, c}, -1);            // [j, f, 2]
  auto w = at::stack({from_re, from_im}, 1);        // [j, 2, f, 2]
  return cache.emplace(key, w.reshape({2 * n, 2 * n}).to(at::kFloat).to(device)).first->second;
}

// c2r, one GEMM from the half spectrum (Hermitian): interleaved [..., 2k] @ W [2k, n].
at::Tensor c2r_basis(int64_t n, int64_t k, int64_t normalization, const at::Device& device) {
  static std::map<std::tuple<int64_t, int64_t, int64_t, int64_t>, at::Tensor> cache;
  std::lock_guard<std::mutex> lock(basis_mutex);
  auto key = std::make_tuple(n, k, normalization, static_cast<int64_t>(device.index()));
  auto it = cache.find(key);
  if (it != cache.end()) {
    return it->second;
  }
  // x_j = sum_f w_f (re_f cos - im_f sin), w_f = 1 for DC/Nyquist, 2 for the bins counted twice.
  auto ang = angles(n, k, n);  // [f, j]
  auto weight = at::full({k, 1}, 2.0, at::TensorOptions().dtype(at::kDouble));
  weight[0] = 1.0;
  if (n % 2 == 0 && k == n / 2 + 1) {
    weight[k - 1] = 1.0;
  }
  auto scale = norm_scale(normalization, n);
  auto w = at::stack({at::cos(ang) * weight * scale, at::sin(ang) * weight * -scale}, 1);  // [f, 2, j]
  return cache.emplace(key, w.reshape({2 * k, n}).to(at::kFloat).to(device)).first->second;
}

void check_last_dim(const at::Tensor& self, c10::IntArrayRef dim) {
  TORCH_CHECK(dim.size() == 1 && (dim[0] == -1 || dim[0] == self.dim() - 1), "hpu_complex: only last-dim FFT");
}

// Autocast would run the DFT GEMMs in bfloat16: the spectra stay float32.
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

at::Tensor complex_hpu(const at::Tensor& real, const at::Tensor& imag) {
  auto shape = at::infer_size(real.sizes(), imag.sizes());
  return complex_view(at::stack({real.to(at::kFloat).expand(shape), imag.to(at::kFloat).expand(shape)}, -1));
}

// ---- views on complex tensors (PrivateUse1) -------------------------------------------

at::Tensor as_strided_pu1(const at::Tensor& self, c10::IntArrayRef size, c10::IntArrayRef stride,
                          std::optional<int64_t> storage_offset) {
  return view_of(self, self.key_set(), self.dtype(), size, stride, storage_offset.value_or(self.storage_offset()));
}

at::Tensor view_pu1(const at::Tensor& self, c10::IntArrayRef size) {
  auto inferred = at::infer_size_dv(size, self.numel());
  auto stride = at::detail::computeStride(self.sizes(), self.strides(), inferred);
  TORCH_CHECK(stride.has_value(), "view size is not compatible with input tensor's size and stride; use .reshape");
  return view_of(self, self.key_set(), self.dtype(), inferred, *stride, self.storage_offset());
}

at::Tensor reshape_alias_pu1(const at::Tensor& self, c10::IntArrayRef size, c10::IntArrayRef stride) {
  return view_of(self, self.key_set(), self.dtype(), size, stride, self.storage_offset());
}

// ---- arithmetic (PrivateUse1) ------------------------------------------------------------

// A real operand as float32 on the HPU (CPU 0-dim tensors are Python numbers).
at::Tensor hpu_float(const at::Tensor& t, const at::Device& device) {
  return t.to(device, at::kFloat);
}

// [..., 2] pair of an operand: complex -> its pair (on the HPU), real -> nullopt.
std::optional<at::Tensor> pair_of(const at::Tensor& t, const at::Device& device) {
  if (!t.is_complex()) {
    return std::nullopt;
  }
  auto p = pair_view(t);
  return p.device() == device ? p : p.to(device);
}

at::Device hpu_device_of(const at::Tensor& a, const at::Tensor& b) {
  return tagged(a) ? a.device() : b.device();
}

at::Tensor complex_result(const at::Tensor& pair) {
  return complex_view(pair.contiguous());
}

at::Tensor mul_pu1(const at::Tensor& a, const at::Tensor& b) {
  const auto device = hpu_device_of(a, b);
  auto pa = pair_of(a, device);
  auto pb = pair_of(b, device);
  if (pa && pb) {
    auto ar = pa->select(-1, 0), ai = pa->select(-1, 1), br = pb->select(-1, 0), bi = pb->select(-1, 1);
    return complex_result(at::stack({ar * br - ai * bi, ar * bi + ai * br}, -1));
  }
  // complex x real: scale both parts.
  return complex_result(pa ? *pa * hpu_float(b, device).unsqueeze(-1) : *pb * hpu_float(a, device).unsqueeze(-1));
}

at::Tensor div_pu1(const at::Tensor& a, const at::Tensor& b) {
  const auto device = hpu_device_of(a, b);
  auto pa = pair_of(a, device);
  auto pb = pair_of(b, device);
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
  TORCH_CHECK(!alpha.isComplex(), "hpu_complex: complex alpha is not supported");
  const auto device = hpu_device_of(a, b);
  const double s = alpha.toDouble();
  auto pa = pair_of(a, device);
  auto pb = pair_of(b, device);
  if (pa && pb) {
    return complex_result(s == 1.0 ? *pa + *pb : *pa + *pb * s);
  }
  // complex + real: the real part moves, the imaginary part is the complex operand's (scaled when it is b).
  at::Tensor re, im;
  if (pa) {
    re = pa->select(-1, 0) + hpu_float(b, device) * s;
    im = pa->select(-1, 1);
  } else {
    re = hpu_float(a, device) + pb->select(-1, 0) * s;
    im = pb->select(-1, 1) * s;
  }
  auto shape = at::infer_size(re.sizes(), im.sizes());
  return complex_result(at::stack({re.expand(shape), im.expand(shape)}, -1));
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
  return modulus(pair_view(self), false);
}

at::Tensor sgn_pu1(const at::Tensor& self) {
  auto pair = pair_view(self);
  // sgn(0) == 0: 0/0 is NaN there.
  return complex_result((pair / modulus(pair, true)).nan_to_num(0.0, 0.0, 0.0));
}

// A [..., 2] pair of any operand (a real one gets imaginary part 0), on the HPU.
at::Tensor as_pair(const at::Tensor& t, const at::Device& device) {
  if (auto p = pair_of(t, device)) {
    return *p;
  }
  auto re = hpu_float(t, device);
  return at::stack({re, at::zeros_like(re)}, -1);
}

// where(cond, a, b) with a complex a or b (compiled sgn backward: where(|z| == 0, 0, z / |z|)).
at::Tensor where_pu1(const at::Tensor& cond, const at::Tensor& a, const at::Tensor& b) {
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
at::Tensor& mul__pu1(at::Tensor& self, const at::Tensor& other) {
  return copy_into(self, mul_pu1(self, other));
}
at::Tensor& add__pu1(at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  return copy_into(self, add_pu1(self, other, alpha));
}

// ---- copies and fills (PrivateUse1) ------------------------------------------------------

at::Tensor& copy_pu1(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
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
  out.copy_(value.is_complex() ? pair_view(value).select(-1, 0) : value);
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
  return complex_result(pair_view(self).sum(dims, keepdim));
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

// ---- autograd of _fft_r2c on HPU: the onesided backward as one GEMM ----------------------

// The r2c basis from int32 device ops, for traced graphs: no cache or host state at trace time.
// Exact phase index: j*f < 2^31 for n up to ~65k.
at::Tensor r2c_basis_traced(int64_t n, int64_t k, int64_t normalization, const at::Device& device) {
  auto iopts = at::TensorOptions().dtype(at::kInt).device(device);
  auto idx = at::outer(at::arange(n, iopts), at::arange(k, iopts)).remainder(n);
  auto ang = idx.to(at::kFloat) * (2.0 * M_PI / static_cast<double>(n));
  auto scale = norm_scale(normalization, n);
  return at::stack({at::cos(ang) * scale, at::sin(ang) * -scale}, -1).reshape({n, 2 * k});
}

// X = x @ W with W [n, 2k] = interleaved (C, -S): the real pairs of X are linear in x, and torch's
// complex gradient is dL/dRe + i dL/dIm, so grad_x = view_as_real(grad) @ W^T. This replaces
// fft_r2c_backward's zero-padded full spectrum and its [2n, 2n] c2c.
class FftR2C : public torch::autograd::Function<FftR2C> {
 public:
  static at::Tensor forward(torch::autograd::AutogradContext* ctx, const at::Tensor& self, std::vector<int64_t> dim,
                            int64_t normalization, bool onesided) {
    ctx->saved_data["sizes"] = self.sizes().vec();
    ctx->saved_data["dim"] = dim;
    ctx->saved_data["normalization"] = normalization;
    ctx->saved_data["onesided"] = onesided;
    ctx->saved_data["dtype"] = static_cast<int64_t>(self.scalar_type());
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::_fft_r2c::call(self, dim, normalization, onesided);
  }

  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                 torch::autograd::variable_list grads) {
    const auto& grad = grads[0];
    if (!grad.defined()) {
      return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }
    HPU_COMPLEX_NO_AUTOCAST;
    auto sizes = ctx->saved_data["sizes"].toIntVector();
    auto normalization = ctx->saved_data["normalization"].toInt();
    auto dtype = static_cast<at::ScalarType>(ctx->saved_data["dtype"].toInt());
    at::Tensor grad_x;
    if (ctx->saved_data["onesided"].toBool()) {
      const int64_t n = sizes.back();
      const int64_t k = n / 2 + 1;
      // Eager gradients come from the kernels here (tagged): the cached basis. Traced ones are fake.
      auto basis = tagged(grad) ? r2c_basis(n, k, normalization, grad.device())
                                : r2c_basis_traced(n, k, normalization, grad.device());
      grad_x = at::matmul(at::view_as_real(grad).reshape({-1, 2 * k}), basis.t()).view(sizes);
    } else {  // the full spectrum: fft_r2c_backward's own formula
      auto dim = ctx->saved_data["dim"].toIntVector();
      grad_x = at::real(at::_fft_c2c(grad, dim, normalization, /*forward=*/false));
    }
    return {grad_x.to(dtype), at::Tensor(), at::Tensor(), at::Tensor()};
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
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("_fft_c2c", &fft_c2c_pu1);
  m.impl("_fft_c2r", &fft_c2r_pu1);
  m.impl("view_as_real", &view_as_real_pu1);
  m.impl("as_strided", &as_strided_pu1);
  m.impl("view", &view_pu1);
  m.impl("_reshape_alias", &reshape_alias_pu1);
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
}

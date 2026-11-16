// hpu_complex extension: real complex64 + torch.fft on Gaudi, plugin form.
//
// Registered for the fork's c10::DispatchKey::HPU (TORCH_LIBRARY_IMPL(aten,
// HPU)) -- the bridge's own key, which stock PrivateUse1 registrations
// never reach (measured: accepted, never called).
//
// A complex tensor here is a TensorImpl view (dtype complex64) over
// interleaved float HPU storage, exactly how cuComplex/MPSGraph carry
// complex. Synapse never sees the complex dtype: every kernel below
// decomposes to float ops via view_as_real first.
//
// v1 FFT is a dense DFT (matmul). Correct for all n; O(n^2). Radix-2/TPC
// is the perf phase. Norm int: 0=none, 1=by_root_n, 2=by_n (per-op scale).
#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/ops/_copy_from.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <mutex>
#include <unordered_map>

namespace {

at::Tensor make_view(
    const at::Tensor& base,
    caffe2::TypeMeta dtype,
    std::vector<int64_t> sizes,
    std::vector<int64_t> strides,
    int64_t storage_offset) {
  auto result = at::detail::make_tensor<at::TensorImpl>(
      c10::TensorImpl::VIEW, c10::Storage(base.storage()),
      base.key_set().remove(c10::DispatchKey::Conjugate).remove(
          c10::DispatchKey::Negative),
      dtype);
  auto* impl = result.unsafeGetTensorImpl();
  impl->set_sizes_and_strides(sizes, strides);
  impl->set_storage_offset(storage_offset);
  return result;
}

at::Tensor make_complex_view(const at::Tensor& pair) {
  // pair: float [..., k, 2], last stride 1
  auto sizes = pair.sizes().vec();
  auto strides = pair.strides().vec();
  sizes.pop_back();
  strides.pop_back();
  for (auto& s : strides) {
    s /= 2;
  }
  return make_view(
      pair, caffe2::TypeMeta::Make<c10::complex<float>>(), sizes, strides,
      pair.storage_offset() / 2);
}

at::Tensor make_real_view(const at::Tensor& self) {
  auto sizes = self.sizes().vec();
  auto strides = self.strides().vec();
  for (auto& s : strides) {
    s *= 2;
  }
  sizes.push_back(2);
  strides.push_back(1);
  return make_view(
      self, caffe2::TypeMeta::Make<float>(), sizes, strides,
      self.storage_offset() * 2);
}

// Plan cache (cuFFT's core trick): DFT matrices depend only on
// (n, inverse, dtype, device) -- build once, reuse every call.
struct PlanKey {
  int64_t n;
  bool inverse;
  int64_t dtype;
  int64_t device;
  bool operator==(const PlanKey& o) const {
    return n == o.n && inverse == o.inverse && dtype == o.dtype &&
        device == o.device;
  }
};
struct PlanKeyHash {
  size_t operator()(const PlanKey& k) const {
    return std::hash<int64_t>{}(k.n * 4 + (k.inverse ? 2 : 0) +
                                (k.device << 8) + k.dtype);
  }
};
struct Plan {
  at::Tensor c, s;     // DFT matrices
  at::Tensor ct, st;   // transposed (matmul-ready, no per-call transpose)
  at::Tensor cc, ci;   // [2n, n] concat for c2c: out = [re,im] @ cc / ci
};
std::unordered_map<PlanKey, Plan, PlanKeyHash> plan_cache;
std::mutex plan_mutex;

Plan dft_plan(
    int64_t n,
    bool inverse,
    const at::TensorOptions& opts) {
  PlanKey key{n, inverse, static_cast<int64_t>(opts.dtype().toScalarType()),
              static_cast<int64_t>(opts.device().index())};
  std::lock_guard<std::mutex> lock(plan_mutex);
  auto it = plan_cache.find(key);
  if (it != plan_cache.end()) {
    return it->second; // by value: later emplaces may rehash the map
  }
  if (plan_cache.size() > 32) {
    plan_cache.clear(); // ponytail: crude eviction; plans rebuild lazily
  }
  auto k = at::arange(n, opts);
  auto ang = (2.0 * M_PI / static_cast<double>(n)) *
      at::outer(k, k).to(opts.dtype());
  if (!inverse) {
    ang = -ang;
  }
  Plan plan;
  plan.c = at::cos(ang);
  plan.s = at::sin(ang);
  plan.ct = plan.c.t(); // views: Habana matmul handles transposed B
  plan.st = plan.s.t();
  plan.cc = at::cat({plan.c, -plan.s}, 0).contiguous(); // [2n, n]: xin @ cc
  plan.ci = at::cat({plan.s, plan.c}, 0).contiguous();
  return plan_cache.emplace(key, std::move(plan)).first->second;
}

// Dense DFT along last dim via one concatenated GEMM per output part
// ([re,im] @ cc, [re,im] @ ci): 2 GEMMs, zero sub/add ops.
std::pair<at::Tensor, at::Tensor> dense_fft(
    const at::Tensor& re,
    const at::Tensor& im,
    bool inverse);

std::pair<int64_t, int64_t> factorize(int64_t n) {
  const int64_t r = std::sqrt(static_cast<double>(n));
  for (int64_t d = r; d >= 8; --d) {
    if (n % d == 0) {
      return {d, n / d};
    }
  }
  return {0, 0};
}

// Four-step FFT: n = n1*n2, two small dense FFTs + twiddle multiply.
// Accumulation depth ~n1+n2 instead of n (accuracy at large n), and the
// GEMMs shrink from n x n to n1/n2 blocks. Verified vs torch.fft (1e-13).
std::pair<at::Tensor, at::Tensor> four_step_fft(
    const at::Tensor& re,
    const at::Tensor& im,
    bool inverse) {
  const int64_t n = re.size(-1);
  auto [n1, n2] = factorize(n);
  auto shape = re.sizes().vec();
  shape.pop_back();
  shape.push_back(n1);
  shape.push_back(n2);
  // Stage 1: DFT_n1 along dim -2 (via transpose to last dim).
  // .contiguous(): Habana matmul mis-handles strided batched inputs.
  auto a = dense_fft(
      re.reshape(shape).transpose(-1, -2).contiguous(),
      im.reshape(shape).transpose(-1, -2).contiguous(),
      inverse);
  auto a_re = a.first.transpose(-1, -2);
  auto a_im = a.second.transpose(-1, -2);
  // Twiddle: exp(-/+ 2i pi k1*j2 / n), cached per (n1, n2, inverse).
  struct TwKey {
    int64_t n1, n2;
    bool inverse;
    bool operator==(const TwKey& o) const {
      return n1 == o.n1 && n2 == o.n2 && inverse == o.inverse;
    }
  };
  struct TwKeyHash {
    size_t operator()(const TwKey& k) const {
      return std::hash<int64_t>{}(k.n1 * 1000003 + k.n2 * 2 + k.inverse);
    }
  };
  static std::unordered_map<TwKey, std::pair<at::Tensor, at::Tensor>, TwKeyHash>
      twiddle_cache;
  TwKey key{n1, n2, inverse};
  std::lock_guard<std::mutex> lock(plan_mutex);
  auto it = twiddle_cache.find(key);
  if (it == twiddle_cache.end()) {
    auto opts = re.options();
    auto k1 = at::arange(n1, opts).reshape({n1, 1});
    auto j2 = at::arange(n2, opts).reshape({1, n2});
    auto ang = (2.0 * M_PI / static_cast<double>(n)) * (k1 * j2);
    if (!inverse) {
      ang = -ang;
    }
    it = twiddle_cache
             .emplace(key, std::make_pair(at::cos(ang), at::sin(ang)))
             .first;
  }
  const auto& tw_c = it->second.first;
  const auto& tw_s = it->second.second;
  auto b_re = a_re * tw_c - a_im * tw_s;
  auto b_im = a_re * tw_s + a_im * tw_c;
  // Stage 2: DFT_n2 along last dim; output index k = k1 + n1*k2.
  auto out = dense_fft(b_re, b_im, inverse);
  auto out_shape = re.sizes().vec();
  // .contiguous() before reshape: Habana reshape on a transposed view
  // does not preserve logical order (measured: permuted output).
  return {
      out.first.transpose(-1, -2).contiguous().reshape(out_shape),
      out.second.transpose(-1, -2).contiguous().reshape(out_shape)};
}

bool four_step_enabled() {
  // Opt-in: four-step is ~85x more accurate at n>=4096 but launch-bound
  // slow at large batch on this stack. Dense (default) is 5e-04 @ 6000.
  static const bool enabled = std::getenv("HPU_COMPLEX_FOURSTEP") != nullptr;
  return enabled;
}

bool use_four_step(int64_t n) {
  return n >= 4096 && four_step_enabled() && factorize(n).first > 0;
}

std::pair<at::Tensor, at::Tensor> dense_fft(
    const at::Tensor& re,
    const at::Tensor& im,
    bool inverse) {
  const int64_t n = re.size(-1);
  if (use_four_step(n)) {
    auto [n1, n2] = factorize(n);
    if (n1 > 0) {
      return four_step_fft(re, im, inverse);
    }
  }
  const auto plan = dft_plan(n, inverse, re.options());
  auto xin = at::cat({re, im}, -1);
  auto out_re = at::matmul(xin, plan.cc);
  auto out_im = at::matmul(xin, plan.ci);
  if (inverse) {
    out_re = out_re / static_cast<double>(n);
    out_im = out_im / static_cast<double>(n);
  }
  return {out_re, out_im};
}

double norm_scale(int64_t normalization, int64_t n, bool inverse) {
  // aten per-op target scaling: none=1, by_root_n=1/sqrt(n), by_n=1/n.
  // dense_fft inverse already carries 1/n; compensate accordingly.
  if (normalization == 0) {
    return inverse ? static_cast<double>(n) : 1.0;
  }
  if (normalization == 1) {
    return inverse ? std::sqrt(static_cast<double>(n))
                   : 1.0 / std::sqrt(static_cast<double>(n));
  }
  return inverse ? 1.0 : 1.0 / static_cast<double>(n);
}

void check_last_dim(const at::Tensor& self, c10::IntArrayRef dim) {
  TORCH_CHECK(
      dim.size() == 1 && (dim[0] == -1 || dim[0] == self.dim() - 1),
      "hpu_complex: only last-dim FFT in v1");
}

at::Tensor fft_r2c_hpu(
    const at::Tensor& self,
    c10::IntArrayRef dim,
    int64_t normalization,
    bool onesided) {
  check_last_dim(self, dim);
  const int64_t n = self.size(-1);
  if (use_four_step(n)) {
    // Large n: four-step for accuracy (see dense_fft gate).
    auto full = four_step_fft(self, at::zeros_like(self), false);
    auto scale = norm_scale(normalization, n, false);
    if (onesided) {
      const int64_t k = n / 2 + 1;
      auto re = full.first.slice(-1, 0, k).contiguous();
      auto im = full.second.slice(-1, 0, k).contiguous();
      auto pair = scale == 1.0 ? at::stack({re, im}, -1)
                               : at::stack({re * scale, im * scale}, -1);
      return make_complex_view(pair.contiguous());
    }
    auto pair = scale == 1.0
        ? at::stack({full.first, full.second}, -1)
        : at::stack({full.first * scale, full.second * scale}, -1);
    return make_complex_view(pair.contiguous());
  }
  // Real input: the im=0 matmuls vanish, only 2 GEMMs (not 4 + a zero fill).
  const auto plan = dft_plan(n, false, self.options());
  at::Tensor re, im;
  if (onesided) {
    // Slice the plan, not the output: GEMM emits [.., k] directly.
    const int64_t k = n / 2 + 1;
    re = at::matmul(self, plan.c.slice(0, 0, k).t());
    im = at::matmul(self, plan.s.slice(0, 0, k).t());
  } else {
    re = at::matmul(self, plan.ct);
    im = at::matmul(self, plan.st);
  }
  auto scale = norm_scale(normalization, n, false);
  auto pair = scale == 1.0 ? at::stack({re, im}, -1)
                           : at::stack({re * scale, im * scale}, -1);
  return make_complex_view(pair.contiguous());
}

at::Tensor fft_c2r_hpu(
    const at::Tensor& self_in,
    c10::IntArrayRef dim,
    int64_t normalization,
    int64_t last_dim_size) {
  auto self = self_in.is_conj() ? self_in.conj_physical() : self_in;
  check_last_dim(self, dim);
  auto pair = make_real_view(self);
  auto re_half = pair.select(-1, 0);
  auto im_half = pair.select(-1, 1);
  const int64_t n = last_dim_size;
  const int64_t k = re_half.size(-1);
  // Rebuild full Hermitian spectrum, then inverse DFT.
  auto tail_re = re_half.slice(-1, 1, n - k + 1).flip({-1}).contiguous();
  auto tail_im = -im_half.slice(-1, 1, n - k + 1).flip({-1}).contiguous();
  auto re = at::cat({re_half, tail_re}, -1);
  auto im = at::cat({im_half, tail_im}, -1);
  auto out = dense_fft(re, im, true).first;
  return out * norm_scale(normalization, n, true);
}

at::Tensor fft_c2c_hpu(
    const at::Tensor& self_in,
    c10::IntArrayRef dim,
    int64_t normalization,
    bool forward) {
  auto self = self_in.is_conj() ? self_in.conj_physical() : self_in;
  check_last_dim(self, dim);
  auto pair = make_real_view(self);
  auto out = dense_fft(pair.select(-1, 0), pair.select(-1, 1), !forward);
  const int64_t n = self.size(-1);
  auto scale = norm_scale(normalization, n, !forward);
  auto opair = scale == 1.0
      ? at::stack({out.first, out.second}, -1)
      : at::stack({out.first * scale, out.second * scale}, -1);
  return make_complex_view(opair.contiguous());
}

at::Tensor complex_hpu_ext(const at::Tensor& real, const at::Tensor& imag) {
  auto pair = at::stack({real, imag}, -1);
  return make_complex_view(pair.contiguous());
}

at::Tensor abs_hpu(const at::Tensor& self) {
  if (self.is_complex()) {
    auto pair = make_real_view(self);
    auto re = pair.select(-1, 0);
    auto im = pair.select(-1, 1);
    return at::sqrt(re * re + im * im);
  }
  // ponytail: sqrt(x^2) replaces the bridge abs kernel while the ext is
  // loaded; exact except overflow at |x| > ~1e19. Bridge abs if unloaded.
  return at::sqrt(self * self);
}

at::Tensor make_complex_empty(
    c10::IntArrayRef size,
    const at::TensorOptions& options);

at::Tensor empty_memory_format_hpu(
    c10::IntArrayRef size,
    std::optional<at::ScalarType> dtype_opt,
    std::optional<at::Layout> layout_opt,
    std::optional<at::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<at::MemoryFormat> memory_format_opt) {
  auto dtype = dtype_opt.value_or(at::kFloat);
  if (dtype != at::kComplexFloat) {
    // Identical to the bridge's own empty (habana_eager/ops/empty.cpp):
    // byte-blind allocator + HPU key set, no Synapse contact.
    auto* allocator = c10::GetAllocator(c10::DeviceType::HPU);
    constexpr c10::DispatchKeySet hpu_ks(c10::DispatchKey::HPU);
    return at::detail::empty_generic(
        size, allocator, hpu_ks, dtype, memory_format_opt);
  }
  return make_complex_empty(
      size,
      at::TensorOptions().device(device_opt.value_or(at::kHPU)));
}

at::Tensor to_copy_hpu(
    const at::Tensor& self,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory,
    bool non_blocking,
    std::optional<at::MemoryFormat> memory_format) {
  auto target_dtype = dtype.value_or(self.scalar_type());
  auto target_device = device.value_or(self.device());
  if (!self.is_complex() && target_dtype != at::kComplexFloat) {
    auto out = at::empty_like(
        self,
        self.options().dtype(target_dtype).device(target_device),
        memory_format);
    out.copy_(self);
    return out;
  }
  TORCH_CHECK(
      self.is_complex() && target_dtype == at::kComplexFloat,
      "hpu_complex: only complex64 -> complex64 copies in v1");
  // Route through the float view: the bridge copies floats correctly,
  // its complex copy segfaults on our TensorImpl.
  auto src_pair = make_real_view(self);
  at::Tensor dst_pair;
  if (target_device.is_cpu()) {
    dst_pair = at::empty_like(src_pair, src_pair.options().device(target_device));
    dst_pair.copy_(src_pair);
    // Re-view the copied float storage as complex (contiguous result).
    auto result = make_complex_empty(
        self.sizes(), at::TensorOptions().device(target_device));
    make_real_view(result).copy_(dst_pair);
    return result;
  }
  dst_pair = at::empty_like(src_pair, src_pair.options().device(target_device));
  dst_pair.copy_(src_pair);
  return make_complex_view(dst_pair);
}

at::Tensor pair_of(const at::Tensor& t) {
  if (t.is_complex()) {
    return make_real_view(t);
  }
  return at::stack({t, at::zeros_like(t)}, -1);
}

void copy_pair_into(const at::Tensor& pair, at::Tensor& out_complex) {
  make_real_view(out_complex).copy_(pair);
}

// Exact float mul via GEMM: composite aliases make mul/sub/div/neg
// route into each other in this fork, so float fallbacks must only use
// matmul/bmm + Scalar ops, which never composite back.
at::Tensor float_mul(const at::Tensor& a, const at::Tensor& b) {
  // .contiguous(): broadcast operands arrive stride-0 expanded, which
  // batch_gemm rejects after reshape.
  auto af = a.contiguous().reshape({-1, 1, 1});
  auto bf = b.contiguous().reshape({-1, 1, 1});
  return at::bmm(af, bf).reshape(a.sizes());
}

void float_mul_out(at::Tensor& out, const at::Tensor& a, const at::Tensor& b) {
  // TensorIterator may hand us a permuted (non-contiguous) out: reshaping
  // it would write into a detached copy. Go through copy_ then.
  if (!out.is_contiguous() || out.scalar_type() != at::kFloat) {
    // copy_ casts (e.g. bool outs from isfinite composites); bmm cannot.
    out.copy_(float_mul(a, b));
    return;
  }
  auto out3 = out.reshape({-1, 1, 1});
  at::bmm_out(
      out3, a.contiguous().reshape({-1, 1, 1}),
      b.contiguous().reshape({-1, 1, 1}));
}

at::Tensor float_addsub(
    const at::Tensor& a,
    const at::Tensor& b,
    bool subtract) {
  auto stacked = at::stack({a, b}, -2).contiguous(); // [..., 2, n]
  // Rows cached: building ones per call costs a CPU->HPU fill (~50ms).
  static std::unordered_map<int64_t, std::pair<at::Tensor, at::Tensor>>
      row_cache;
  const int64_t dev = a.device().index();
  std::lock_guard<std::mutex> lock(plan_mutex);
  auto it = row_cache.find(dev);
  if (it == row_cache.end()) {
    auto one = at::ones({1}, a.options());
    auto plus = at::cat({one, one});   // [1, 1]
    auto minus = at::cat({one, -one}); // [1, -1]
    it = row_cache.emplace(dev, std::make_pair(plus, minus)).first;
  }
  const auto& row = subtract ? it->second.second : it->second.first;
  // bmm, not matmul: this fork's matmul mishandles 1-D x 2-D operands.
  // Flatten only batch dims (contiguous => batch-major), never across them.
  const int64_t n = a.size(-1);
  const int64_t batch = stacked.numel() / (2 * n);
  auto flat = at::bmm(
      row.reshape({1, 1, 2}).expand({batch, 1, 2}),
      stacked.reshape({batch, 2, n}));
  return flat.reshape(a.sizes());
}

void float_addsub_out(
    at::Tensor& out,
    const at::Tensor& a,
    const at::Tensor& b,
    bool subtract) {
  out.copy_(float_addsub(a, b, subtract));
}

std::pair<at::Tensor, at::Tensor> normalize_operands(
    const at::Tensor& self_in,
    const at::Tensor& other_in) {
  // This fork wraps int scalars as Long tensors in composites, and FFT
  // autograd multiplies by Bool masks: normalize non-complex to float,
  // and move CPU scalar-wrapped tensors to the HPU side explicitly.
  auto self_c = self_in.is_complex() || self_in.scalar_type() == at::kFloat
      ? self_in
      : self_in.to(at::kFloat);
  auto other_c = other_in.is_complex() ||
          other_in.scalar_type() == self_c.scalar_type()
      ? other_in
      : other_in.to(self_c.scalar_type());
  if (self_c.device() != other_c.device()) {
    if (self_c.is_cpu()) {
      self_c = self_c.to(other_c.device());
    } else {
      other_c = other_c.to(self_c.device());
    }
  }
  auto bc = at::broadcast_tensors({self_c, other_c});
  return {bc[0], bc[1]};
}

at::Tensor& mul_out_hpu(
    const at::Tensor& self_in,
    const at::Tensor& other_in,
    at::Tensor& out) {
  auto [self, other] = normalize_operands(self_in, other_in);
  if (self.is_complex() || other.is_complex()) {
    auto a = pair_of(self);
    auto b = pair_of(other);
    auto re = float_mul(a.select(-1, 0), b.select(-1, 0)) -
        float_mul(a.select(-1, 1), b.select(-1, 1));
    auto im = float_mul(a.select(-1, 0), b.select(-1, 1)) +
        float_mul(a.select(-1, 1), b.select(-1, 0));
    copy_pair_into(at::stack({re, im}, -1), out);
    return out;
  }
  float_mul_out(out, self, other);
  return out;
}

at::Tensor& div_out_hpu(
    const at::Tensor& self_in,
    const at::Tensor& other_in,
    at::Tensor& out) {
  auto [self, other] = normalize_operands(self_in, other_in);
  if (self.is_complex() || other.is_complex()) {
    auto a = pair_of(self);
    auto b = pair_of(other);
    auto denom = float_mul(b.select(-1, 0), b.select(-1, 0)) +
        float_mul(b.select(-1, 1), b.select(-1, 1));
    auto re = (float_mul(a.select(-1, 0), b.select(-1, 0)) +
               float_mul(a.select(-1, 1), b.select(-1, 1))) /
        denom;
    auto im = (float_mul(a.select(-1, 1), b.select(-1, 0)) -
               float_mul(a.select(-1, 0), b.select(-1, 1))) /
        denom;
    copy_pair_into(at::stack({re, im}, -1), out);
    return out;
  }
  float_mul_out(out, self, at::reciprocal(other));
  return out;
}

at::Tensor& add_out_hpu(
    const at::Tensor& self_in,
    const at::Tensor& other_in,
    const at::Scalar& alpha,
    at::Tensor& out) {
  auto [self, other] = normalize_operands(self_in, other_in);
  if (self.is_complex() || other.is_complex()) {
    copy_pair_into(pair_of(self) + pair_of(other) * alpha, out);
    return out;
  }
  float_addsub_out(out, self, other * alpha, false);
  return out;
}

at::Tensor& sub_out_hpu(
    const at::Tensor& self_in,
    const at::Tensor& other_in,
    const at::Scalar& alpha,
    at::Tensor& out) {
  auto [self, other] = normalize_operands(self_in, other_in);
  if (self.is_complex() || other.is_complex()) {
    copy_pair_into(pair_of(self) - pair_of(other) * alpha, out);
    return out;
  }
  float_addsub_out(out, self, other * alpha, true);
  return out;
}

at::Tensor sgn_hpu(const at::Tensor& self) {
  if (self.is_complex()) {
    auto pair = make_real_view(self);
    auto mag = at::sqrt(
        float_mul(pair.select(-1, 0), pair.select(-1, 0)) +
        float_mul(pair.select(-1, 1), pair.select(-1, 1)));
    auto out_pair = at::stack(
        {pair.select(-1, 0) / mag, pair.select(-1, 1) / mag}, -1);
    // stock sgn(0) == 0; the division above yields NaN there
    return make_complex_view(at::nan_to_num(out_pair, 0.0).contiguous());
  }
  return at::sign(self);
}

at::Tensor conj_physical_hpu(const at::Tensor& self) {
  auto pair = make_real_view(self).clone();
  pair.select(-1, 1).mul_(-1);
  return make_complex_view(pair);
}

at::Tensor make_complex_empty(
    c10::IntArrayRef size,
    const at::TensorOptions& options) {
  int64_t numel = 1;
  for (auto s : size) {
    numel *= s;
  }
  auto flat = at::empty({numel * 2}, options.dtype(at::kFloat));
  auto result = at::detail::make_tensor<at::TensorImpl>(
      c10::TensorImpl::VIEW, c10::Storage(flat.storage()),
      flat.key_set().remove(c10::DispatchKey::Conjugate).remove(
          c10::DispatchKey::Negative),
      caffe2::TypeMeta::Make<c10::complex<float>>());
  std::vector<int64_t> strides(size.size(), 1);
  for (int64_t i = static_cast<int64_t>(size.size()) - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * size[i + 1];
  }
  auto* impl = result.unsafeGetTensorImpl();
  impl->set_sizes_and_strides(size.vec(), strides);
  impl->set_storage_offset(0);
  return result;
}

at::Tensor empty_strided_hpu(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    std::optional<at::ScalarType> dtype_opt,
    std::optional<at::Layout> layout_opt,
    std::optional<at::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  auto dtype = dtype_opt.value_or(at::kFloat);
  if (dtype == at::kComplexFloat) {
    // Strides are in complex elements; empty_generic path asserts in the
    // bridge, so build the TensorImpl over float storage directly.
    auto result = make_complex_empty(
        size, at::TensorOptions().device(device_opt.value_or(at::kHPU)));
    auto* impl = result.unsafeGetTensorImpl();
    impl->set_sizes_and_strides(size.vec(), stride.vec());
    return result;
  }
  auto* allocator = c10::GetAllocator(c10::DeviceType::HPU);
  constexpr c10::DispatchKeySet hpu_ks(c10::DispatchKey::HPU);
  return at::detail::empty_strided_generic(
      size, stride, allocator, hpu_ks, dtype);
}

at::Tensor empty_like_hpu(
    const at::Tensor& self,
    std::optional<at::ScalarType> dtype_opt,
    std::optional<at::Layout> layout_opt,
    std::optional<at::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<at::MemoryFormat> memory_format_opt) {
  auto dtype = dtype_opt.value_or(self.scalar_type());
  if (dtype == at::kComplexFloat) {
    return make_complex_empty(
        self.sizes(),
        at::TensorOptions().device(device_opt.value_or(self.device())));
  }
  auto mf = memory_format_opt;
  if (mf == at::MemoryFormat::Preserve) {
    mf = self.suggest_memory_format();
  }
  return at::empty(
      self.sizes(),
      self.options().dtype(dtype).device(device_opt.value_or(self.device())),
      mf);
}

at::Tensor clone_hpu(
    const at::Tensor& self,
    std::optional<at::MemoryFormat> memory_format_opt) {
  if (self.is_complex()) {
    auto src = self.is_conj() ? self.conj_physical() : self;
    auto pair = make_real_view(src).clone();
    // ponytail: conj bit not preserved on clone; values are physically
    // conjugated, and every reader resolves the bit before use anyway.
    return make_complex_view(pair);
  }
  // Float clone via empty + bridge copy_ (clone kernel asserts on complex
  // only; this branch keeps float behavior identical without the kernel).
  auto out = at::empty_like(self);
  out.copy_(self);
  return out;
}

at::Tensor real_hpu(const at::Tensor& self) {
  auto src = self.is_conj() ? self.conj_physical() : self;
  return make_real_view(src).select(-1, 0);
}

at::Tensor imag_hpu(const at::Tensor& self) {
  auto src = self.is_conj() ? self.conj_physical() : self;
  return make_real_view(src).select(-1, 1);
}

at::Tensor new_zeros_hpu(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    std::optional<at::ScalarType> dtype_opt,
    std::optional<at::Layout> layout_opt,
    std::optional<at::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  auto dtype = dtype_opt.value_or(self.scalar_type());
  std::vector<int64_t> sizes;
  for (auto s : size) {
    sizes.push_back(s.expect_int());
  }
  if (dtype == at::kComplexFloat) {
    auto out = make_complex_empty(
        sizes,
        at::TensorOptions().device(device_opt.value_or(self.device())));
    return out.fill_(at::Scalar(c10::complex<double>(0, 0)));
  }
  return at::zeros(
      sizes,
      self.options().dtype(dtype).device(device_opt.value_or(self.device())));
}

at::Tensor& copy_hpu(
    at::Tensor& self,
    const at::Tensor& src,
    bool non_blocking) {
  if (self.is_complex() || src.is_complex()) {
    auto s = src.is_conj() ? src.conj_physical() : src;
    // Float copy_ below routes to the bridge's _copy_from primitive.
    make_real_view(self).copy_(make_real_view(s));
    return self;
  }
  at::_ops::_copy_from::call(src, self, non_blocking);
  return self;
}

at::Tensor slice_hpu(
    const at::Tensor& self,
    int64_t dim,
    std::optional<int64_t> start_opt,
    std::optional<int64_t> end_opt,
    int64_t step) {
  // True TensorImpl view (the bridge slice kernel materializes, which
  // silently detaches in-place copies in FFT backward).
  const int64_t ndim = self.dim();
  dim = dim < 0 ? dim + ndim : dim;
  const int64_t len = self.size(dim);
  int64_t start = start_opt.value_or(0);
  int64_t end = end_opt.value_or(len);
  if (start < 0) {
    start += len;
  }
  if (end < 0) {
    end += len;
  }
  start = std::clamp<int64_t>(start, 0, len);
  end = std::clamp<int64_t>(end, start, len);
  auto sizes = self.sizes().vec();
  auto strides = self.strides().vec();
  sizes[dim] = (end - start + step - 1) / step;
  strides[dim] *= step;
  auto result = at::detail::make_tensor<at::TensorImpl>(
      c10::TensorImpl::VIEW, c10::Storage(self.storage()), self.key_set(),
      self.options().dtype());
  auto* impl = result.unsafeGetTensorImpl();
  impl->set_sizes_and_strides(sizes, strides);
  impl->set_storage_offset(self.storage_offset() + start * self.stride(dim));
  // ponytail: conj bit not preserved (this fork's TensorImpl has no
  // setter); FFT backward uses slice as a copy destination, where the
  // physical values are what matter.
  return result;
}

at::Tensor& fill_scalar_hpu(at::Tensor& self, const at::Scalar& value) {
  // Via a CPU scalar tensor + bridge copy_: CPU ops are not registered
  // by this extension, and this fork wraps int scalars as Long tensors
  // in add/mul composites, which poisons naive mul_/add_ fallbacks.
  auto cpu_opts = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  // Full-shape CPU constant + same-shape copy: the bridge _copy_from
  // does not broadcast a 1-element source (garbage memory otherwise).
  if (self.is_complex()) {
    auto pair = make_real_view(self);
    auto c = value.toComplexDouble();
    auto re = pair.select(-1, 0);
    auto im = pair.select(-1, 1);
    re.copy_(at::full(re.sizes(), c.real(), cpu_opts));
    im.copy_(at::full(im.sizes(), c.imag(), cpu_opts));
    return self;
  }
  return self.copy_(at::full(self.sizes(), value, cpu_opts));
}

at::Tensor& zero_hpu(at::Tensor& self) {
  return fill_scalar_hpu(self, at::Scalar(0));
}

} // namespace

TORCH_LIBRARY_IMPL(aten, HPU, m) {
  m.impl("_fft_r2c", &fft_r2c_hpu);
  m.impl("_fft_c2r", &fft_c2r_hpu);
  m.impl("_fft_c2c", &fft_c2c_hpu);
  m.impl("view_as_complex", [](const at::Tensor& self) {
    TORCH_CHECK(self.scalar_type() == at::kFloat, "view_as_complex: float32 only");
    TORCH_CHECK(self.size(-1) == 2 && self.stride(-1) == 1,
                "view_as_complex: last dim must be 2 with stride 1");
    return make_complex_view(self);
  });
  m.impl("view_as_real", [](const at::Tensor& self) {
    TORCH_CHECK(self.is_complex(), "view_as_real: complex input only");
    return make_real_view(self);
  });
  m.impl("complex", &complex_hpu_ext);
  m.impl("abs", &abs_hpu);
  m.impl("empty.memory_format", &empty_memory_format_hpu);
  m.impl("empty_strided", &empty_strided_hpu);
  m.impl("empty_like", &empty_like_hpu);
  m.impl("clone", &clone_hpu);
  m.impl("_to_copy", &to_copy_hpu);
  m.impl("fill_.Scalar", &fill_scalar_hpu);
  m.impl("zero_", &zero_hpu);
  // HPU_COMPLEX_NO_ARITH=1: leave float mul/add/div to the bridge
  // (complex FFT still works; complex arithmetic/backward will not).
  if (std::getenv("HPU_COMPLEX_NO_ARITH") == nullptr) {
    m.impl("mul.out", &mul_out_hpu);
    m.impl("div.out", &div_out_hpu);
    m.impl("add.out", &add_out_hpu);
    m.impl("sub.out", &sub_out_hpu);
  }
  m.impl("sgn", &sgn_hpu);
  m.impl("conj_physical", &conj_physical_hpu);
  m.impl("real", &real_hpu);
  m.impl("imag", &imag_hpu);
  m.impl("new_zeros", &new_zeros_hpu);
  m.impl("copy_", &copy_hpu);
  m.impl("slice.Tensor", &slice_hpu);
  // ponytail: conj returns a physical copy, not a lazy conj-bit view;
  // mutation-aliasing semantics differ, values are correct.
  m.impl("conj", &conj_physical_hpu);
  m.impl("resolve_conj", [](const at::Tensor& self) {
    return self.is_conj() ? conj_physical_hpu(self) : self;
  });
}

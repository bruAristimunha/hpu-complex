// CPU tests only: a no-op HPU device guard, so stock (CPU) torch accepts device="hpu" tensors. With it, the
// extension's complex128 tensors, which live in host memory, run end to end without a Gaudi.
#include <c10/core/impl/DeviceGuardImplInterface.h>

C10_REGISTER_GUARD_IMPL(HPU, c10::impl::NoOpDeviceGuardImpl<c10::DeviceType::HPU>);

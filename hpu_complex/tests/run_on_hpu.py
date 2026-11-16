"""Direct runner (Gaudi image has no pytest): python3 tests/run_on_hpu.py"""
import traceback

from hpu_complex.tests import test_on_hpu as t

for name in ("test_core_fft_on_hpu", "test_stft_and_grad_on_hpu",
             "test_compile_attempt"):
    try:
        getattr(t, name)()
        print(f"{name}: DONE")
    except Exception:
        print(f"{name}: FAILED")
        traceback.print_exc()
print("HPU_TESTS_DONE")

"""Direct runner of every test in test_on_hpu.py (the Gaudi image has no pytest): python3 tests/run_on_hpu.py

Exits 1 if any test fails.
"""
import inspect
import sys
import traceback

from hpu_complex.tests import test_on_hpu as t

tests = sorted((f for name, f in inspect.getmembers(t, inspect.isfunction) if name.startswith("test_")),
               key=lambda f: f.__code__.co_firstlineno)
failed = []
for test in tests:
    try:
        test()
        print(f"{test.__name__}: DONE", flush=True)
    except Exception:  # noqa: BLE001
        failed.append(test.__name__)
        print(f"{test.__name__}: FAILED", flush=True)
        traceback.print_exc()
print(f"HPU_TESTS {len(tests) - len(failed)}/{len(tests)} passed", flush=True)
sys.exit(1 if failed else 0)

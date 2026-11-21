#!/bin/bash
S=/voyager/ceph/groups/sdp190/bpinto/spd-diag/linalg-gate/team-j/src
cd $S
for v in base new; do
  if [ $v = base ]; then R=$S/probe_j/base; export TORCH_EXTENSIONS_DIR=/tmp/ext_base; else R=$S; export TORCH_EXTENSIONS_DIR=/tmp/ext_new; fi
  for mode in 0 1; do
    for p in f64_times_1j f64_times_2j_tensor f32_times_1j f64_to_c128_then_mul save_device; do
      out=$(PYTHONPATH=$R:$PYTHONPATH PT_HPU_LAZY_MODE=$mode timeout 300 python3 tests/probe_adv_detail4_j.py $p 2>&1); rc=$?
      echo "$out" | grep -E "^(f64|f32|save_device)" | sed "s/^/$v lazy=$mode /"
      [ $rc -ne 0 ] && echo "$v lazy=$mode PROC $p exit $rc: $(echo "$out" | grep -E "Error|Segmentation|Aborted|terminate|what" | head -2 | tr '\n' ' ' | cut -c1-250)"
    done
  done
done
echo "=== lazy remaining probe cases, one process each"
CASES=$(python3 - <<'PY'
import re
names = re.findall(r"^def ([a-z0-9_]+)\(d\)", open("tests/probe_adversarial_j.py").read(), re.M)
skip = ("c64", "compile", "device_mismatch")
start = names.index("f64_times_1j")
print(" ".join(n for n in names[start:] if not any(s in n for s in skip)))
PY
)
PT_HPU_LAZY_MODE=1 bash tests/probe_adv_each_j.sh "$CASES" --rounds 2
echo "=== END"

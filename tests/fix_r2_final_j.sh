#!/bin/bash
# Fix round 2 (team J), final card pass: c128_check, save/load, then every probe case in its own process, eager and lazy.
S=/voyager/ceph/groups/sdp190/bpinto/spd-diag/linalg-gate/team-j/src
cd $S
export TORCH_EXTENSIONS_DIR=/tmp/ext_new
for mode in 0 1; do
  echo "=== c128_check lazy=$mode"
  PT_HPU_LAZY_MODE=$mode timeout 1200 python3 hpu_complex/tests/c128_check.py 2>&1 | grep -E "FAIL|ERROR|C128_CHECK|torch.save|compile"
  for p in save; do
    PT_HPU_LAZY_MODE=$mode timeout 300 python3 tests/probe_adv_detail3_j.py $p 2>&1 | grep -E "^(save|load|state_dict)" | sed "s/^/lazy=$mode /"
  done
done
CASES=$(python3 - <<'PY'
import re
print(" ".join(re.findall(r"^def ([a-z0-9_]+)\(d\)", open("tests/probe_adversarial_j.py").read(), re.M)))
PY
)
for mode in 0 1; do
  echo "=== every probe case lazy=$mode, one process each, 2 rounds"
  PT_HPU_LAZY_MODE=$mode bash tests/probe_adv_each_j.sh "$CASES" --rounds 2
done
echo "=== END"

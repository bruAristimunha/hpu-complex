#!/bin/bash
S=/voyager/ceph/groups/sdp190/bpinto/spd-diag/linalg-gate/team-j/src
cd $S
for v in base new; do
  if [ $v = base ]; then R=$S/probe_j/base; export TORCH_EXTENSIONS_DIR=/tmp/ext_base; else R=$S; export TORCH_EXTENSIONS_DIR=/tmp/ext_new; fi
  for mode in 0 1; do for p in f32 f64; do
    out=$(PYTHONPATH=$R:$PYTHONPATH PT_HPU_LAZY_MODE=$mode timeout 300 python3 tests/probe_adv_detail5_j.py $p 2>&1); rc=$?
    echo "$out" | grep -E "^f(32|64) " | sed "s/^/$v lazy=$mode /" | cut -c1-330
    [ $rc -ne 0 ] && echo "$v lazy=$mode PROC $p exit $rc: $(echo "$out" | grep -E "Error|Segmentation|Aborted|terminate|what" | head -2 | tr '\n' ' ' | cut -c1-250)"
  done; done
done
echo "=== END"

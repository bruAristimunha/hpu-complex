#!/bin/bash
# Round 2b follow-up: details, origin/main A/B of the model failures, lazy probe without complex64/compile cases.
S=/voyager/ceph/groups/sdp190/bpinto/spd-diag/linalg-gate/team-j/src
cd $S
for mode in 0 1; do
  for p in mismatch_c128 mismatch_c64 mismatch_f32 strides save; do
    out=$(PT_HPU_LAZY_MODE=$mode timeout 300 python3 tests/probe_adv_detail3_j.py $p 2>&1); rc=$?
    echo "$out" | grep -E "^(mismatch|strides|  complex64|save|load|state_dict)" | sed "s/^/lazy=$mode /"
    [ $rc -ne 0 ] && echo "lazy=$mode PROC $p exit $rc: $(echo "$out" | grep -E "Error|Segmentation|Aborted|terminate" | head -2 | tr '\n' ' ')"
  done
done
for v in base new; do
  if [ $v = base ]; then R=$S/probe_j/base; export TORCH_EXTENSIONS_DIR=/tmp/ext_base; else R=$S; export TORCH_EXTENSIONS_DIR=/tmp/ext_new; fi
  for mode in 0 1; do
    echo "=== $v green models lazy=$mode (hpu_complex $R)"
    PYTHONPATH=$R:$PYTHONPATH PT_HPU_LAZY_MODE=$mode timeout 900 python3 tests/probe_adversarial_j.py --only =none --rounds 1 --models 2>&1 | grep -E "^MODEL|^PROBE|Segmentation|Abort" | cut -c1-300
    echo "=== $v GBWAtt,TensorCSPNet lazy=$mode"
    PYTHONPATH=$R:$PYTHONPATH ONLY_MODELS=GBWAtt,TensorCSPNet PT_HPU_LAZY_MODE=$mode timeout 1200 python3 tests/linalg_steps_copy_j.py --baseline 2>&1 | grep -E "^RESULT|^PLUGINS|Segmentation|Abort" | sed -E 's/"fallback_per_step.*"status"/... "status"/' | cut -c1-400
  done
done
echo "=== lazy full probe without complex64/compile/mismatch, 2 rounds"
PT_HPU_LAZY_MODE=1 timeout 1800 python3 tests/probe_adversarial_j.py --skip c64,compile,device_mismatch,conv1d_c64,cat_c64 2>&1 | grep -E "^R[0-9]|^SUMMARY|Segmentation|Abort|what\(\)"
echo "=== END"

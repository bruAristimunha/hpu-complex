#!/bin/bash
# Round 2b (second adversarial tester): every card step of the round, one process at a time.
# bash tests/probe_adv_round2b_j.sh   (eager/lazy chosen per step; hx-j's own PT_HPU_LAZY_MODE is overridden)
cd /voyager/ceph/groups/sdp190/bpinto/spd-diag/linalg-gate/team-j/src
NEW="repeat_c128 tile_c128 device_mismatch_raises save_load_host_f64 randn_c128_on_hpu overlapping_out_raises istft_c128_roundtrip c128_to_c64_then_card_op to_hpu_noncontig_and_offset"
for mode in 0 1; do
  echo "=== compile variants lazy=$mode"
  for v in eigvals_f32 eigvals_only_f64 eigvals_f64 eigvals_f64_graph_break c128_mul_only f64_abs_only c128_input; do
    for ext in "" "--no-ext"; do
      out=$(PT_HPU_LAZY_MODE=$mode timeout 300 python3 tests/probe_adv_compile_j.py $v $ext 2>&1); rc=$?
      echo "COMPILE lazy=$mode $v ${ext:-ext} rc=$rc: $(echo "$out" | grep -E "OK|Error|Segmentation|Abort" | grep -v start | tail -1 | cut -c1-220)"
    done
  done
  echo "=== new cases lazy=$mode"
  PT_HPU_LAZY_MODE=$mode bash tests/probe_adv_each_j.sh "$NEW"
done
echo "=== full probe lazy=1 (no compile) --models"
PT_HPU_LAZY_MODE=1 timeout 1500 python3 tests/probe_adversarial_j.py --skip compile --models 2>&1 | grep -E "^R[0-9]|^MODEL|^SUMMARY|Segmentation|Abort|what\(\)"
echo "=== full probe lazy=0 (no compile) --models --hpu-ops"
PT_HPU_LAZY_MODE=0 timeout 1500 python3 tests/probe_adversarial_j.py --skip compile --models --hpu-ops 2>&1 | grep -E "^R[0-9]|^MODEL|^SUMMARY|Segmentation|Abort|what\(\)"
for mode in 0 1; do
  echo "=== spd_diag MODELS (linalg_steps, --baseline) lazy=$mode"
  PT_HPU_LAZY_MODE=$mode timeout 1500 python3 tests/linalg_steps_copy_j.py --baseline 2>&1 | grep -E "^RESULT|^PLUGINS|models|Segmentation|Abort" | cut -c1-400
done
echo "=== END"

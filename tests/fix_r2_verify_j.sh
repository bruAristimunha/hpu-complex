#!/bin/bash
# Fix round 2 (team J): re-run the round-2b tester's card probes on this branch, one process per step.
# bash tests/fix_r2_verify_j.sh [part...]   parts: compile cases green check regress (default: all)
S=/voyager/ceph/groups/sdp190/bpinto/spd-diag/linalg-gate/team-j/src
cd $S
export TORCH_EXTENSIONS_DIR=/tmp/ext_new
PARTS="${*:-compile cases green check regress}"
CASES="repeat_c128 tile_c128 unfold conv1d_c128 set_source_tensor_aliases inference_mode_tensor bool_mask_and_index grad_through_real_imag_views f32_add_inplace_c128 f32_mul_inplace_c128 f64_add_inplace_c128 mul_out_into_f32 c64_add_inplace_c128 copy_f64_from_real_then_overwrite save_load_host_f64 overlapping_out_raises device_mismatch_raises compile_eigvals_f64 compile_c128_input compile_eig_backward"
for part in $PARTS; do
  case $part in
  compile)
    for mode in 0 1; do
      echo "=== compile variants lazy=$mode"
      for v in eigvals_f32 eigvals_only_f64 eigvals_f64 eigvals_f64_graph_break c128_mul_only f64_abs_only c128_input; do
        out=$(PT_HPU_LAZY_MODE=$mode timeout 300 python3 tests/probe_adv_compile_j.py $v 2>&1); rc=$?
        echo "COMPILE lazy=$mode $v rc=$rc: $(echo "$out" | grep -E "OK|Error|Segmentation|Abort" | grep -v start | tail -1 | cut -c1-220)"
      done
    done;;
  cases)
    for mode in 0 1; do
      echo "=== tester cases lazy=$mode (2 rounds each, one process per case)"
      PT_HPU_LAZY_MODE=$mode bash tests/probe_adv_each_j.sh "$CASES" --rounds 2
    done
    for mode in 0 1; do
      for p in mismatch_c128 mismatch_c64 mismatch_f32 save; do
        out=$(PT_HPU_LAZY_MODE=$mode timeout 300 python3 tests/probe_adv_detail3_j.py $p 2>&1); rc=$?
        echo "$out" | grep -E "^(mismatch|save|load|state_dict)" | sed "s/^/lazy=$mode /"
        [ $rc -ne 0 ] && echo "lazy=$mode PROC $p exit $rc: $(echo "$out" | grep -E "Error|Segmentation|Aborted|terminate" | head -2 | tr '\n' ' ')"
      done
    done;;
  green)
    for mode in 0 1; do
      echo "=== green models lazy=$mode"
      PT_HPU_LAZY_MODE=$mode timeout 900 python3 tests/probe_adversarial_j.py --only =none --rounds 1 --models 2>&1 | grep -E "^MODEL|Segmentation|Abort" | cut -c1-300
    done;;
  check)
    for mode in 0 1; do for ops in "" "--hpu-ops"; do
      echo "=== c128_check lazy=$mode $ops"
      PT_HPU_LAZY_MODE=$mode timeout 1200 python3 hpu_complex/tests/c128_check.py $ops --time 2>&1 | grep -E "FAIL|ERROR|C128_CHECK|^TIME|compile|Segmentation|Abort"
    done; done
    echo "=== full probe lazy=0, 2 rounds"
    PT_HPU_LAZY_MODE=0 timeout 1800 python3 tests/probe_adversarial_j.py 2>&1 | grep -E "^R[0-9] (DIFF|ERR|CRASH|RACE)|^SUMMARY|Segmentation|Abort|what\(\)"
    echo "=== full probe lazy=1 without complex64/compile/mismatch, 2 rounds"
    PT_HPU_LAZY_MODE=1 timeout 1800 python3 tests/probe_adversarial_j.py --skip c64,compile,device_mismatch,conv1d_c64,cat_c64 2>&1 | grep -E "^R[0-9] (DIFF|ERR|CRASH|RACE)|^SUMMARY|Segmentation|Abort|what\(\)";;
  regress)
    PT_HPU_LAZY_MODE=0 bash probe_j/regress.sh;;
  esac
done
echo "=== END"

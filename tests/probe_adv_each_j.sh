#!/bin/bash
# Each probe case in its own process (a crash cannot hide the next): bash tests/probe_adv_each_j.sh "<cases>" [probe args]
cd /voyager/ceph/groups/sdp190/bpinto/spd-diag/linalg-gate/team-j/src
CASES="$1"; shift
for c in $CASES; do
  out=$(timeout 300 python3 tests/probe_adversarial_j.py --only "=$c" "$@" 2>&1)
  rc=$?
  echo "$out" | grep -E "^R[0-9]|^MODEL"
  if [ $rc -ne 0 ]; then echo "PROC $c exit $rc: $(echo "$out" | grep -E "what\(\)|Error|Segmentation|Aborted|terminate" | grep -v -E "^R[0-9]" | head -3 | tr '\n' ' ')"; fi
done

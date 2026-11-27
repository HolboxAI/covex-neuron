#!/usr/bin/env bash
# Refuse to launch a Trainium run unless every cheap check passes. Usage: launch_guard.sh <payload.sh> <stop_minutes>
# Prints GO and exits 0 only if: static tests pass, preflight passes, the payload hard-stops at <= stop_minutes,
# the payload pins this exact pushed HEAD (no moving branch, no placeholders), and no other Trainium instance is running (no forgotten spend).
set -uo pipefail
PAY=$1; CAP=$2; cd "$(git rev-parse --show-toplevel)"; fail=0
python3 -m pytest tests/covex -q > /tmp/guard_tests.log 2>&1 && echo "ok   tests/covex" || { echo "FAIL tests/covex (see /tmp/guard_tests.log)"; fail=1; }
benchmarks/asr/preflight.sh > /tmp/guard_preflight.log 2>&1 && echo "ok   preflight" || { echo "FAIL preflight (see /tmp/guard_preflight.log)"; fail=1; }
m=$(grep -oE 'shutdown -h \+[0-9]+' "$PAY" | head -1 | grep -oE '[0-9]+$')
[ -n "$m" ] && [ "$m" -le "$CAP" ] && echo "ok   hard stop ${m} min (cap ${CAP})" || { echo "FAIL hard stop missing or > ${CAP} min"; fail=1; }
# A rendered payload (q32/render_payload.sh) must pin exactly this pushed, clean HEAD; a template must be committed.
head=$(git rev-parse HEAD); git fetch -q origin
if grep -q '@@[A-Z_]*@@' "$PAY"; then echo "FAIL payload has unrendered placeholders (render it with exact pins)"; fail=1
elif grep -q "checkout -q $head" "$PAY" && [ -z "$(git status --porcelain)" ] && [ "$head" = "$(git rev-parse @{u} 2>/dev/null)" ]; then
  echo "ok   payload pins HEAD $head (clean, pushed)"
else echo "FAIL payload must check out HEAD $head; tree clean; HEAD pushed"; fail=1; fi
running=$(for r in us-east-1 us-east-2 us-west-2 ap-northeast-1 ap-southeast-1; do aws ec2 describe-instances --region $r \
  --filters Name=instance-type,Values='trn*' Name=instance-state-name,Values=running,pending \
  --query 'Reservations[].Instances[].InstanceId' --output text 2>/dev/null; done | tr -s '\t\n ' ' ')
[ -z "${running// /}" ] && echo "ok   no other Trainium instance running" || { echo "FAIL Trainium already running: $running"; fail=1; }
[ $fail = 0 ] && echo GO || echo "NO-GO"; exit $fail

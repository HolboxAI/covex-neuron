#!/usr/bin/env bash
# One-shot round on a trn2.48xlarge: 4 arms on chip pairs 0-3, all at once.
# 1) serve + benchmark every arm   2) full test-clean WER on S and D   3) restart with PROFILE=1 and capture stages.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ARMS=(S D DS DSM)
wait_up() { for _ in $(seq 1 360); do curl -sf "http://127.0.0.1:$1/health" > /dev/null && return 0; sleep 10; done; echo "port $1 not up" >&2; return 1; }
start_all() { for i in "${!ARMS[@]}"; do PROFILE=${1:-0} "$HERE/serve.sh" "${ARMS[$i]}" "$i" "$((8000 + i))" & done; for i in "${!ARMS[@]}"; do wait_up "$((8000 + i))" || true; done; }
stop_all() { pkill -f "vllm serve" || true; sleep 15; }

start_all 0                                              # compiles in parallel; the compile cache is reused below
for i in "${!ARMS[@]}"; do "$HERE/bench.sh" "${ARMS[$i]}" "$((8000 + i))" 200 1,4,8,16 & done; wait
for i in 0 1; do "$HERE/bench.sh" "${ARMS[$i]}-wer" "$((8000 + i))" 2620 16 & done; wait   # full test-clean WER, S and D
stop_all
start_all 1
for i in "${!ARMS[@]}"; do "$HERE/profile.sh" "${ARMS[$i]}" "$((8000 + i))" 8 || echo "profile ${ARMS[$i]} failed"; done
stop_all
echo "done: results/*.json, profiles/*/STAGES.md, logs/*"

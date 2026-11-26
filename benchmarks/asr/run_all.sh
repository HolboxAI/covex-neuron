#!/usr/bin/env bash
# One-shot round on a trn2.48xlarge: 8 arms on chip pairs 0-7 (all 16 chips), all at once.
# 1) serve + benchmark every arm   2) full test-clean WER on S and D   3) restart with PROFILE=1 and capture stages.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ARMS=(S S2 D DB DM DMB DE1 T4)
wait_up() { for _ in $(seq 1 360); do curl -sf "http://127.0.0.1:$1/health" > /dev/null && return 0; sleep 10; done; echo "port $1 not up" >&2; return 1; }
start_all() { for i in "${!ARMS[@]}"; do PROFILE=${1:-0} "$HERE/serve.sh" "${ARMS[$i]}" "$i" "$((8000 + i))" & done; for i in "${!ARMS[@]}"; do wait_up "$((8000 + i))" || true; done; }
stop_all() { pkill -f "vllm serve" || true; sleep 15; }

start_all 0                                              # compiles in parallel; the compile cache is reused below
"$HERE/bench.sh" S2-warm 8001 32 1 > /dev/null            # downloads the clips
python3 "$HERE/prompt_tokens.py" 8001 "${CLIENT_DIR:-$HOME/qwen3-asr-inference-cuda-trainium}" 200 | tee prompt_tokens.json
# 200 clips x 3 rounds = 600 requests per concurrency level (75 waves at c8); repeated 3x for the median.
for rep in 1 2 3; do
  for i in "${!ARMS[@]}"; do "$HERE/bench.sh" "${ARMS[$i]}-r$rep" "$((8000 + i))" 200 1,4,8,16 & done; wait
done
for i in "${!ARMS[@]}"; do ROUNDS=1 "$HERE/bench.sh" "${ARMS[$i]}-wer" "$((8000 + i))" 2620 16 & done; wait   # full test-clean WER, every arm
stop_all
start_all 1
for i in "${!ARMS[@]}"; do "$HERE/profile.sh" "${ARMS[$i]}" "$((8000 + i))" 8 || echo "profile ${ARMS[$i]} failed"; done
# Host-side check (is the API server the cap at c8?): 30 s py-spy sample of the S2 server while it is loaded.
if command -v py-spy > /dev/null; then
  ( "$HERE/bench.sh" S2-pyspy 8001 64 8 & ) ; sleep 20
  py-spy record -d 30 -o profiles/S2-api-server.svg -p "$(pgrep -f 'vllm serve.*--port 8001' | head -1)" || true; wait
fi
stop_all
# Neighbour check: S2 alone on the SAME 24 pinned vCPUs. If it beats S2-in-parallel by >5%, neighbours skewed
# the parallel run and the solo numbers are reported.
"$HERE/serve.sh" S2 1 8001 & wait_up 8001
for rep in 1 2 3; do "$HERE/bench.sh" "S2solo-r$rep" 8001 200 1,4,8,16; done
stop_all
# Host-size check: S2 alone, unpinned (all 192 vCPUs). If it beats pinned solo, 24 vCPUs is itself a cap.
SOLO=1 "$HERE/serve.sh" S2 1 8001 & wait_up 8001
SOLO=1 "$HERE/bench.sh" "S2free-r1" 8001 200 1,4,8,16
stop_all
python3 "$HERE/compare.py" results > RESULTS.md
stop_all
echo "done: results/*.json, profiles/*/STAGES.md, logs/*"

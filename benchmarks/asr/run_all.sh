#!/usr/bin/env bash
# One-shot round on a trn2.48xlarge: 8 arms on chip pairs 0-7 (all 16 chips), all at once.
# 1) all arms timed (3 repeats)  2) full test-clean WER, every arm  3) per-stage pass for S2 and D (stage_pass.sh)  4) solo S2 controls.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ARMS=(S2 D DB DM DL DMBL DE1 T4)   # S (Dax defaults) is already measured: 136 audio-s/s at c8
source "$HERE/../lib.sh"
CLIENT_DIR=${CLIENT_DIR:-$HOME/qwen3-asr-inference-cuda-trainium}
declare -A SPID; declare -A READY
start_one() {  # arm pair port [env...]: start, then READY only after /health and one real transcription
  local arm=$1 pair=$2 port=$3; shift 3
  env "$@" "$HERE/serve.sh" "$arm" "$pair" "$port" & SPID[$port]=$!
}
ready_one() {  # port
  local why; why=$(wait_ready "${SPID[$1]}" "$1" 3600 asr qwen3-asr-1.7b "$CLIENT_DIR/data") && READY[$1]=1
  echo "ARM port $1 $([ -n "${READY[$1]:-}" ] && echo READY || echo "NOT READY: $why")"
}
start_all() { for i in "${!ARMS[@]}"; do start_one "${ARMS[$i]}" "$i" "$((8000 + i))" PROFILE=${1:-0}; done
              for i in "${!ARMS[@]}"; do ready_one "$((8000 + i))"; done; }
stop_all() { local p; for p in "${!SPID[@]}"; do stop_arm "${SPID[$p]}" "$p"; done; SPID=(); READY=(); }   # only our arms
bench_ready() {  # bench.sh args...: run only against a READY port (arg 2); record, never hide, a skipped arm
  [ -n "${READY[$2]:-}" ] || { echo "SKIP $1: port $2 not ready" | tee -a skipped.txt; return 0; }
  "$HERE/bench.sh" "$@"; }

start_all 0                                              # compiles in parallel; the compile cache is reused below
bench_ready S2-warm 8000 32 1 > /dev/null            # downloads the clips
python3 "$HERE/prompt_tokens.py" 8000 "${CLIENT_DIR:-$HOME/qwen3-asr-inference-cuda-trainium}" 200 | tee prompt_tokens.json
# 200 clips x 3 rounds = 600 requests per concurrency level (75 waves at c8); repeated 3x for the median.
for rep in 1 2 3; do
  P=(); for i in "${!ARMS[@]}"; do bench_ready "${ARMS[$i]}-r$rep" "$((8000 + i))" 200 1,4,8,16 & P+=($!); done; wait_all "${P[@]}" || echo "BENCH FAILURES in repeat $rep (see logs)"
done
P=(); for i in "${!ARMS[@]}"; do ROUNDS=1 bench_ready "${ARMS[$i]}-wer" "$((8000 + i))" 2620 16 & P+=($!); done; wait_all "${P[@]}" || echo "BENCH FAILURES in WER pass"   # full test-clean WER, every arm
stop_all
"$HERE/stage_pass.sh"                                     # per-stage pass (S2, D at c1/c8): metrics, py-spy, device profiles
stop_all
# Neighbour check: S2 alone on the SAME 24 pinned vCPUs. If it beats S2-in-parallel by >5%, neighbours skewed
# the parallel run and the solo numbers are reported.
start_one S2 1 8001; ready_one 8001
for rep in 1 2 3; do bench_ready "S2solo-r$rep" 8001 200 1,4,8,16; done
stop_all
# Host-size check: S2 alone, unpinned (all 192 vCPUs). If it beats pinned solo, 24 vCPUs is itself a cap.
start_one S2 1 8001 SOLO=1; ready_one 8001
SOLO=1 bench_ready "S2free-r1" 8001 200 1,4,8,16
stop_all
python3 "$HERE/compare.py" results > RESULTS.md
stop_all
echo "done: results/*.json, profiles/*/STAGES.md, logs/*"

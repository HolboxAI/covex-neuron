#!/usr/bin/env bash
# Five-model round on one trn2.48xlarge: stock (T) vs COVEX (C) per model, all arms at once on disjoint chips/CPUs.
# Expects: venv active, this repo installed, models in ~/models, data in ~/data (sixdata/, asr client+clips).
# Every wait is bounded; a failed arm is recorded and skipped, the others continue.
set -uo pipefail
H=$(cd "$(dirname "$0")" && pwd); A=$H/../asr; C=$H/clients.py; D=$HOME/data/sixdata
export CLIENT_DIR=$HOME/qwen3-asr-inference-cuda-trainium
source "$H/../lib.sh"
fresh_dir results; fresh_dir profiles; mkdir -p logs   # never summarize files left by an earlier attempt
declare -A PORT=([q32-T]=8000 [q32-C]=8001 [asr-T]=8002 [asr-C]=8003 [emb-T]=8004 [emb-C]=8005 [vl-T]=8006 [vl-C]=8007 [oss-T]=8008)
declare -A CPU; i=0; for a in q32-T q32-C asr-T asr-C emb-T emb-C vl-T vl-C oss-T; do CPU[$a]="$((i*20))-$((i*20+19))"; i=$((i+1)); done
pin() { taskset -c "${CPU[$1]}" "${@:2}"; }
start() {  # arm
  case $1 in
    asr-T) SOLO=1 PROFILE=1 LOG_TAG=-T pin asr-T "$A/serve.sh" S2 2 8002 ;;
    asr-C) SOLO=1 PROFILE=1 LOG_TAG=-C pin asr-C "$A/serve.sh" DL 3 8003 ;;
    q32-*) PROFILE=1 pin "$1" "$H/serve.sh" "$1" "${PORT[$1]}" ;;
    *)     pin "$1" "$H/serve.sh" "$1" "${PORT[$1]}" ;;
  esac
}
# emb-C is KNOWN BROKEN (packed-prefill recompile); BROKEN_ARMS keeps it out until fixed. Checked by the tests.
BROKEN_ARMS=" emb-C "
declare -A SPID
# Servers never exit on their own: never `wait` on them. Every wait below names client PIDs (bare wait hung run 2).
for a in "${!PORT[@]}"; do [[ "$BROKEN_ARMS" == *" $a "* ]] && { echo "ARM $a SKIPPED (known broken)"; continue; }; start "$a" & SPID[$a]=$!; done
declare -A UP
smoke_args() {  # arm -> kind model data for the readiness request
  case $1 in emb-*) echo "emb m";; vl-*) echo "vl m $D/vl";; asr-*) echo "asr qwen3-asr-1.7b $CLIENT_DIR/data";; *) echo "gen m";; esac; }
for a in "${!PORT[@]}"; do
  [ -n "${SPID[$a]:-}" ] || continue
  # READY = /health AND one real request succeeds (<= 60 min for compile). /health alone passed for Qwen3-32B
  # stock, which then failed allocating at its first decode.
  why=$(wait_ready "${SPID[$a]}" "${PORT[$a]}" 3600 $(smoke_args $a)) && UP[$a]=1
  echo "ARM $a $([ -n "${UP[$a]:-}" ] && echo UP || echo "FAILED: $why")"
done
up() { [ -n "${UP[$1]:-}" ]; }

llm() { # arm port tokenizer conc prompts tag
  pin $1 vllm bench serve --backend vllm --base-url "http://127.0.0.1:$2" --model m --tokenizer "$3" \
    --dataset-name random --random-input-len 1024 --random-output-len 256 --ignore-eos \
    --max-concurrency "$4" --num-prompts "$5" --seed 1 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90 \
    --save-result --result-dir results --result-filename "$1-$6.json" > "logs/$1-$6.bench.log" 2>&1 || echo "BENCH $1-$6 FAILED"
}
# --- correctness: COVEX must reproduce stock's greedy output ------------------------------------------
for m in q32; do up $m-T && up $m-C || continue
  rm -f results/$m-T.gen.json results/$m-C.gen.json
  pin $m-T python3 $C gen 8000 results/$m-T.gen.json & g1=$!; pin $m-C python3 $C gen 8001 results/$m-C.gen.json & g2=$!
  if wait_all $g1 $g2 && [ -s results/$m-T.gen.json ] && [ -s results/$m-C.gen.json ]; then
    python3 $C match results/$m-T.gen.json results/$m-C.gen.json | tee results/$m.match.json
  else echo "CORRECTNESS $m FAILED (a client failed; no comparison)" | tee results/$m.match.json; fi
done

# --- stage + kernel profiles FIRST (Qwen3-32B and ASR; T and C): the step-level data must not depend on time left -------------------------------------------
for a in q32-T q32-C; do up $a || continue
  curl -sf -X POST "localhost:${PORT[$a]}/start_profile" > /dev/null && llm $a ${PORT[$a]} $HOME/models/Qwen3-32B 8 8 prof \
    && curl -sf -X POST "localhost:${PORT[$a]}/stop_profile" > /dev/null
done
for a in asr-T asr-C; do up $a || continue
  pa=$([ $a = asr-T ] && echo S2 || echo DL)
  curl -sf -X POST "localhost:${PORT[$a]}/start_profile" > /dev/null && SOLO=1 ROUNDS=1 pin $a "$A/bench.sh" $a-prof ${PORT[$a]} 32 8 \
    && curl -sf -X POST "localhost:${PORT[$a]}/stop_profile" > /dev/null
done
# --- throughput, T and C of a model at the same time ----------------------------------------------------
(
  for c in 1 8; do for a in q32-T q32-C; do up $a && llm $a ${PORT[$a]} $HOME/models/Qwen3-32B $c $((c==1?8:32)) c$c & done; wait; done
) &
CL0=$!
(
  up oss-T && for c in 1 4; do llm oss-T 8008 $HOME/models/gpt-oss-20b $c $((c==1?8:16)) c$c; done
) &
CL1=$!
(
  for t in t64 t256; do for a in emb-T emb-C; do up $a && pin $a python3 $C emb ${PORT[$a]} $D/embed/$t.jsonl 256 32 results/$a-$t.json & done; wait; done
  up emb-T && up emb-C && python3 $C match results/emb-T-t64.json results/emb-C-t64.json | tee results/emb.match.json
) &
CL2=$!
(
  for a in vl-T vl-C; do up $a && pin $a python3 $C vl ${PORT[$a]} $D/vl 40 4 results/$a.json & done; wait
  up vl-T && up vl-C && python3 $C match results/vl-T.json results/vl-C.json | tee results/vl.match.json
) &
CL3=$!
(
  for a in asr-T asr-C; do up $a && SOLO=1 ROUNDS=1 pin $a "$A/bench.sh" $a ${PORT[$a]} 100 1,8 & done; wait
) &
CL4=$!
wait_all $CL0 $CL1 $CL2 $CL3 $CL4 || echo "WORKLOAD FAILURES (see logs/*.bench.log)"

for a in "${!SPID[@]}"; do stop_arm "${SPID[$a]}" "${PORT[$a]}"; done   # only our arms
for d in profiles/*/; do neuron-explorer view -d "$d" --output-format perfetto > /dev/null 2>&1 && python3 "$A/summarize_profile.py" "$d" > "$d/STAGES.md" 2>&1; done
python3 "$H/summarize.py" results > RESULTS.md; cat RESULTS.md

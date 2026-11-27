#!/usr/bin/env bash
# Five-model round on one trn2.48xlarge: stock (T) vs COVEX (C) per model, all arms at once on disjoint chips/CPUs.
# Expects: venv active, this repo installed, models in ~/models, data in ~/data (sixdata/, asr client+clips).
# Every wait is bounded; a failed arm is recorded and skipped, the others continue; the exit status is non-zero when a
# REQUIRED arm (default: q32-T q32-C) is not ready, a required workload fails, or its comparison is incomplete.
set -uo pipefail
H=$(cd "$(dirname "$0")" && pwd); A=$H/../asr; C=$H/clients.py; D=$HOME/data/sixdata
export CLIENT_DIR=${CLIENT_DIR:-$HOME/qwen3-asr-inference-cuda-trainium}
source "$H/../lib.sh"
fresh_dir results; fresh_dir profiles; mkdir -p logs   # never summarize files left by an earlier attempt
export EVIDENCE_DIR=${EVIDENCE_DIR:-$PWD/evidence}; mkdir -p "$EVIDENCE_DIR"
ALL_ARMS="q32-T q32-C asr-T asr-C emb-T emb-C vl-T vl-C oss-T"
ARMS=${FIVE_ARMS:-$ALL_ARMS}; REQUIRED=${REQUIRED_ARMS:-"q32-T q32-C"}
declare -A PORT=([q32-T]=8000 [q32-C]=8001 [asr-T]=8002 [asr-C]=8003 [emb-T]=8004 [emb-C]=8005 [vl-T]=8006 [vl-C]=8007 [oss-T]=8008)
declare -A CPU; i=0; for a in $ALL_ARMS; do CPU[$a]=$(cpu_range $((i*20)) $((i*20+19))); i=$((i+1)); done
pin() { taskset -c "${CPU[$1]}" "${@:2}"; }
start() {  # arm
  case $1 in
    asr-T) SOLO=1 PROFILE=1 LOG_TAG=-T pin asr-T "$A/serve.sh" S2 2 8002 ;;
    asr-C) SOLO=1 PROFILE=1 LOG_TAG=-C pin asr-C "$A/serve.sh" DL 3 8003 ;;
    q32-*) PROFILE=1 pin "$1" "$H/serve.sh" "$1" "${PORT[$1]}" ;;
    *)     pin "$1" "$H/serve.sh" "$1" "${PORT[$1]}" ;;
  esac
}
status() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> results/STATUS.tsv; }   # arm, step, OK|FAILED|SKIPPED <why>
# emb-C is KNOWN BROKEN (packed-prefill recompile); BROKEN_ARMS keeps it out until fixed. Deliberate skips are recorded
# as SKIPPED, never as workload failures.
BROKEN_ARMS=" emb-C "
declare -A SPID; declare -A UP
# Servers never exit on their own: never `wait` on them. Every wait below names client PIDs via wait_all.
for a in $ARMS; do
  [[ "$BROKEN_ARMS" == *" $a "* ]] && { echo "ARM $a SKIPPED (known broken)"; status $a start "SKIPPED known broken"; continue; }
  start "$a" & SPID[$a]=$!
done
smoke_args() {  # arm -> kind model data for the readiness request
  case $1 in emb-*) echo "emb m";; vl-*) echo "vl m $D/vl";; asr-*) echo "asr qwen3-asr-1.7b $CLIENT_DIR/data";; *) echo "gen m";; esac; }
# READY = /health AND one real request succeeds (<= 60 min for compile), checked for all arms at once so the primary
# pair is not held behind a slow model. /health alone passed for Qwen3-32B stock, which then failed at first decode.
declare -A RPID
for a in "${!SPID[@]}"; do
  ( why=$(wait_ready "${SPID[$a]}" "${PORT[$a]}" 3600 $(smoke_args $a)); rc=$?; printf '%s\t%s\n' "$rc" "$why" > "logs/$a.ready" ) &
  RPID[$a]=$!
done
for a in "${!RPID[@]}"; do
  wait "${RPID[$a]}"; IFS=$'\t' read -r rc why < "logs/$a.ready"
  if [ "$rc" = 0 ]; then UP[$a]=1; echo "ARM $a UP"; status $a ready OK; else echo "ARM $a FAILED: $why"; status $a ready "FAILED $why"; fi
done
up() { [ -n "${UP[$1]:-}" ]; }

llm() { # arm port tokenizer conc prompts tag -> exit status of vllm bench serve (kept, never swallowed)
  pin $1 vllm bench serve --backend vllm --base-url "http://127.0.0.1:$2" --model m --tokenizer "$3" \
    --dataset-name random --random-input-len 1024 --random-output-len 256 --ignore-eos \
    --max-concurrency "$4" --num-prompts "$5" --seed 1 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90 \
    --save-result --result-dir results --result-filename "$1-$6.json" > "logs/$1-$6.bench.log" 2>&1
  local rc=$?; status $1 "bench-$6" "$([ $rc = 0 ] && echo OK || echo "FAILED rc=$rc")"; return $rc
}
ev_paths() {  # arm -> profile dir, server log, meta, as the arm's own launcher names them
  case $1 in   # asr arms are launched as S2 / DL with LOG_TAG -T / -C (benchmarks/asr/serve.sh)
    asr-T) echo "profiles/S2 logs/S2-T.server.log logs/S2-T.meta" ;;
    asr-C) echo "profiles/DL logs/DL-C.server.log logs/DL-C.meta" ;;
    *)     echo "profiles/$1 logs/$1.server.log logs/$1.meta" ;;
  esac; }
profile() {  # arm port cmd...: start profiling, run the load, ALWAYS stop profiling, then save the evidence at once
  local a=$1 port=$2 rc erc; shift 2
  http_post "http://127.0.0.1:$port/start_profile" 60 || { status $a profile "FAILED start_profile"; return 1; }
  "$@"; rc=$?
  http_post "http://127.0.0.1:$port/stop_profile" 300 || { status $a profile "FAILED stop_profile"; rc=1; }
  save_evidence "profile-$a" $(ev_paths $a) ${NEFF_DIR:+"$NEFF_DIR"}; erc=$?
  case $erc in 0) status $a evidence OK;; 1) status $a evidence "FAILED capture";; *) status $a evidence "FAILED upload";; esac
  status $a profile "$([ $rc = 0 ] && echo OK || echo "FAILED load rc=$rc")"; return $rc
}

# --- correctness: COVEX must reproduce stock's greedy output ------------------------------------------
for m in q32; do up $m-T && up $m-C || continue
  rm -f results/$m-T.gen.json results/$m-C.gen.json
  pin $m-T python3 $C gen 8000 results/$m-T.gen.json & g1=$!; pin $m-C python3 $C gen 8001 results/$m-C.gen.json & g2=$!
  if wait_all $g1 $g2 && [ -s results/$m-T.gen.json ] && [ -s results/$m-C.gen.json ]; then
    python3 $C match results/$m-T.gen.json results/$m-C.gen.json > results/$m.match.json; cat results/$m.match.json
  else echo '{"pass": false, "why": "a greedy client failed; no comparison"}' > results/$m.match.json; status $m correctness FAILED; fi
done

# --- stage + kernel profiles FIRST (Qwen3-32B and ASR; T and C): the step-level data must not depend on time left ------
for a in q32-T q32-C; do up $a && profile $a ${PORT[$a]} llm $a ${PORT[$a]} $HOME/models/Qwen3-32B 8 8 prof; done
for a in asr-T asr-C; do up $a && profile $a ${PORT[$a]} env SOLO=1 ROUNDS=1 taskset -c "${CPU[$a]}" "$A/bench.sh" $a-prof ${PORT[$a]} 32 8; done

# --- throughput, T and C of a model at the same time; every client PID is waited on and its status kept ------------
group() {  # name cmd...: run one workload group in the background; GPID[name]=pid
  ( "${@:2}" ) & GPID[$1]=$!; }
declare -A GPID
q32_load() { local c P a; for c in 1 8; do P=(); for a in q32-T q32-C; do up $a && { llm $a ${PORT[$a]} $HOME/models/Qwen3-32B $c $((c==1?8:32)) c$c & P+=($!); }; done
             wait_all ${P[@]+"${P[@]}"} || return 1; done; }
oss_load() { up oss-T || return 0; local c; for c in 1 4; do llm oss-T 8008 $HOME/models/gpt-oss-20b $c $((c==1?8:16)) c$c || return 1; done; }
emb_load() { local t P a; for t in t64 t256; do P=(); for a in emb-T emb-C; do up $a && { pin $a python3 $C emb ${PORT[$a]} $D/embed/$t.jsonl 256 32 results/$a-$t.json & P+=($!); }; done
             wait_all ${P[@]+"${P[@]}"} || return 1; done
             up emb-T && up emb-C && python3 $C match results/emb-T-t64.json results/emb-C-t64.json > results/emb.match.json; return 0; }
vl_load() { local P=() a; for a in vl-T vl-C; do up $a && { pin $a python3 $C vl ${PORT[$a]} $D/vl 40 4 results/$a.json & P+=($!); }; done
            wait_all ${P[@]+"${P[@]}"} || return 1
            up vl-T && up vl-C && python3 $C match results/vl-T.json results/vl-C.json > results/vl.match.json; return 0; }
asr_load() { local P=() a; for a in asr-T asr-C; do up $a && { SOLO=1 ROUNDS=1 taskset -c "${CPU[$a]}" "$A/bench.sh" $a ${PORT[$a]} 100 1,8 & P+=($!); }; done
             wait_all ${P[@]+"${P[@]}"}; }
group q32 q32_load; group oss oss_load; group emb emb_load; group vl vl_load; group asr asr_load
for g in "${!GPID[@]}"; do wait "${GPID[$g]}" && status "$g" workload OK || status "$g" workload FAILED; done

for a in "${!SPID[@]}"; do stop_arm "${SPID[$a]}" "${PORT[$a]}"; done   # only our arms
for d in profiles/*/; do [ -d "$d" ] || continue; n=$(basename "$d")
  # conversion is recorded separately: the raw traces are already saved, so a failed conversion is not data loss
  # optional after a successful raw capture: WARN, never FAILED (the run verdict ignores WARN)
  if ! command -v neuron-explorer > /dev/null; then status "$n" convert "WARN neuron-explorer not installed; raw traces in evidence/"
  elif neuron-explorer view -d "$d" --output-format perfetto > /dev/null 2>&1; then status "$n" convert OK
    python3 "$A/summarize_profile.py" "$d" > "$d/STAGES.md" 2>&1 || status "$n" stage-summary "WARN summary failed"
  else status "$n" convert "WARN conversion failed; raw traces in evidence/"; fi
done
python3 "$H/summarize.py" results --required "$REQUIRED" > RESULTS.md; srv=$?; cat RESULTS.md
fail=$srv
for a in $REQUIRED; do up $a || { echo "REQUIRED arm $a not ready"; fail=1; }; done
# every FAILED step of a required arm (ready, profile start/load/stop, evidence capture/upload, benchmarks) fails the run
while IFS=$'\t' read -r a step st; do
  [[ " $REQUIRED " == *" $a "* || " $REQUIRED " == *" $a-"* ]] || continue
  [[ $st == FAILED* ]] && { echo "REQUIRED $a $step: $st"; fail=1; }
done < results/STATUS.tsv
echo "RUN $([ $fail = 0 ] && echo OK || echo INCOMPLETE)"; exit $fail

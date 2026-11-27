#!/usr/bin/env bash
# Primary experiment: Qwen3-32B, one trn2.48xlarge, every arm on its own 8 logical cores (2 chips) and 36 host vCPUs,
# identical config (BF16, TP8, 32K max context, cap 0.5, buckets [1024,8192] x [1,2,4,8]), all arms at once.
#   U = pinned upstream vllm-neuron (user-facing baseline)   T = fork, flags off (attribution control)
#   K = in-kernel KV write only (primary intervention)       F = dynamic FA only / C = both (Q32_EXTRA_ARMS)
# Steps: ready (health + real request) -> provenance gate -> cache correctness vs U and T -> one short tagged profile per arm (evidence
# saved at once) -> Q32_WINDOWS warm, unprofiled windows at c1 and c8 (1024 in / 256 out, seed 1) -> gated summary.
# Exit 0 only when U, T and K are ready, correct and complete.
set -uo pipefail
H=$(cd "$(dirname "$0")" && pwd); F=$H/../five; C=$F/clients.py
source "$H/../lib.sh"
fresh_dir results; fresh_dir profiles; mkdir -p logs
export EVIDENCE_DIR=${EVIDENCE_DIR:-$PWD/evidence}; mkdir -p "$EVIDENCE_DIR"
M=${MODELS_DIR:-$HOME/models}; TOK=$M/Qwen3-32B
ARMS="q32-U q32-T q32-K ${Q32_EXTRA_ARMS:-}"; WINDOWS=${Q32_WINDOWS:-3}
[[ $WINDOWS =~ ^[1-9][0-9]*$ ]] || { echo "Q32_WINDOWS must be a positive integer (got '$WINDOWS')"; exit 2; }
declare -A PORT CORE CPU; i=0
for a in q32-U q32-T q32-K q32-F q32-C; do
  PORT[$a]=$((8100 + i)); CORE[$a]="$((i * 8))-$((i * 8 + 7))"; CPU[$a]=$(cpu_range $((i * 36)) $((i * 36 + 35))); i=$((i + 1))
done
pin() { taskset -c "${CPU[$1]}" "${@:2}"; }
status() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> results/STATUS.tsv; }

# Provenance: every Python process of every arm records where vllm_neuron is imported from (benchmarks/provenance);
# each arm gets its own compile cache. U runs UPSTREAM_PY (its own venv with only pinned upstream, editable from
# UPSTREAM_DIR); the fork arms run this checkout (FORK_DIR). Checked after readiness, before any measurement.
: "${UPSTREAM_PY:?set UPSTREAM_PY to the upstream venv interpreter}" "${UPSTREAM_DIR:?set UPSTREAM_DIR to the pinned upstream source}"
FORK_DIR=${FORK_DIR:-$(cd "$H/../.." && pwd)}
export COVEX_PROVENANCE_DIR=$PWD/provenance ARM_CACHE_BASE=${ARM_CACHE_BASE:-$PWD/cache} UPSTREAM_PY
fresh_dir "$COVEX_PROVENANCE_DIR"; mkdir -p "$ARM_CACHE_BASE"
declare -A SPID UP RPID
for a in $ARMS; do Q32_CORES=${CORE[$a]} PROFILE=1 MODELS_DIR=$M pin $a "$F/serve.sh" $a ${PORT[$a]} & SPID[$a]=$!; done
for a in "${!SPID[@]}"; do   # all arms checked at once; READY = /health + one real completion
  ( why=$(wait_ready "${SPID[$a]}" "${PORT[$a]}" 3600 gen m); rc=$?; printf '%s\t%s\n' "$rc" "$why" > "logs/$a.ready" ) & RPID[$a]=$!
done
for a in "${!RPID[@]}"; do
  wait "${RPID[$a]}"; IFS=$'\t' read -r rc why < "logs/$a.ready"
  if [ "$rc" = 0 ]; then UP[$a]=1; status $a ready OK; echo "ARM $a UP"; else status $a ready "FAILED $why"; echo "ARM $a FAILED: $why"; fi
done
up() { [ -n "${UP[$1]:-}" ]; }
# --- provenance gate: right tree, right flags, own cache, in >= Q32_MIN_PROCS processes (API + engine + TP workers) --
exp=(); for a in "${!UP[@]}"; do [ $a = q32-U ] && exp+=("$a=$UPSTREAM_DIR${UPSTREAM_PREFIX:+@$UPSTREAM_PREFIX}") \
  || exp+=("$a=$FORK_DIR${FORK_PREFIX:+@$FORK_PREFIX}"); done
if [ ${#exp[@]} -gt 0 ]; then
  lg=(); for a in "${!UP[@]}"; do lg+=(--log "$a=$PWD/logs/$a.server.log"); done
  python3 "$H/../provenance/verify.py" "$COVEX_PROVENANCE_DIR" "${Q32_MIN_PROCS:-10}" --tp "${Q32_TP:-8}" "${lg[@]}" "${exp[@]}" \
    > results/provenance.tsv 2> logs/provenance.err
  vrc=$?
  for a in "${!UP[@]}"; do   # exactly one verdict per launched arm; a crashed or silent verifier drops every arm
    row=$(awk -F'\t' -v a="$a" '$1==a' results/provenance.tsv)
    st=$(printf '%s' "$row" | cut -f2); why=$(printf '%s' "$row" | cut -f3-)
    if [ "$(printf '%s\n' "$row" | grep -c .)" != 1 ] || { [ "$st" != PASS ] && [ "$st" != FAIL ]; }; then
      st=FAIL; why="no single verifier verdict (verifier exit $vrc: $(tail -1 logs/provenance.err))"; fi
    [ "$st" = PASS ] && [ $vrc != 0 ] && { st=FAIL; why="verifier exit $vrc"; }
    status "$a" provenance "$st${why:+ $why}"
    [ "$st" = PASS ] || { unset "UP[$a]"; echo "ARM $a DROPPED: provenance $why"; }
  done
fi

# --- cache correctness: exact-length prompts (block/bucket/tile boundaries), c1 and padded c3, vs U and T ------------
for conc in 1 3; do P=()
  for a in $ARMS; do up $a && { pin $a python3 $C cachecheck ${PORT[$a]} results/$a.cache.c$conc.json $conc 32 & P+=($!); }; done
  wait_all ${P[@]+"${P[@]}"} || status cache "c$conc" "FAILED a client failed"
  for ref in q32-U q32-T; do for a in $ARMS; do [ $a = $ref ] && continue
    [ -s results/$ref.cache.c$conc.json ] && [ -s results/$a.cache.c$conc.json ] || continue
    python3 $C match results/$ref.cache.c$conc.json results/$a.cache.c$conc.json > results/$a.vs.$ref.cache.c$conc.match.json
  done; done
done

llm() { # arm conc prompts tag -> bench status kept
  pin $1 vllm bench serve --backend vllm --base-url "http://127.0.0.1:${PORT[$1]}" --model m --tokenizer "$TOK" \
    --dataset-name random --random-input-len 1024 --random-output-len 256 --ignore-eos \
    --max-concurrency "$2" --num-prompts "$3" --seed 1 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90 \
    --save-result --save-detailed --result-dir results --result-filename "$1-$4.json" > "logs/$1-$4.bench.log" 2>&1
  local rc=$?; status $1 "bench-$4" "$([ $rc = 0 ] && echo OK || echo "FAILED rc=$rc")"; return $rc
}
# --- one short, separately tagged profile per arm on the warmed workload; evidence saved immediately -----------------
for a in $ARMS; do up $a || continue
  llm $a 8 8 profwarm
  L0=$(wc -l < "logs/$a.server.log")
  http_post "http://127.0.0.1:${PORT[$a]}/start_profile" 60 || { status $a profile "FAILED start"; continue; }
  llm $a 8 8 prof; lrc=$?
  http_post "http://127.0.0.1:${PORT[$a]}/stop_profile" 300 || { status $a profile "FAILED stop"; lrc=1; }
  # vLLM's profiler wrapper can log a worker-side exception and still return 200: read the log written meanwhile
  perr=$(tail -n +"$((L0 + 1))" "logs/$a.server.log" | grep -iE "profil" | grep -iE "error|exception|traceback" | head -1 | cut -c1-200)
  [ -n "$perr" ] && { status $a profile "FAILED worker: $perr"; lrc=1; }
  if why=$(traces_ok "profiles/$a" "${Q32_TP:-8}"); then
    save_evidence "profile-$a" "profiles/$a" "logs/$a.server.log" "logs/$a.meta" || status $a evidence "FAILED upload/copy"
  else status $a evidence "FAILED capture: $why"; lrc=1; fi
  status $a profile "$([ $lrc = 0 ] && echo OK || echo "FAILED rc=$lrc")"
done
# compiled programs that produced those traces (all arms share one compile cache), saved once
for a in "${!UP[@]}"; do   # each arm's compiled programs, under its own cache root, must exist and be saved
  if why=$(nonempty_dir "$ARM_CACHE_BASE/$a"); then save_evidence "neff-$a" "$ARM_CACHE_BASE/$a" || status $a evidence "FAILED neff upload"
  else status $a evidence "FAILED neff: $why"; fi
done
# --- measurement: warm-up window, then WINDOWS unprofiled windows; all arms run each window at the same time --------
for c in 1 8; do n=$((c == 1 ? 8 : 32))
  for w in warm $(seq 1 "$WINDOWS"); do P=()
    for a in $ARMS; do up $a && { llm $a $c $n c$c-w$w & P+=($!); }; done
    wait_all ${P[@]+"${P[@]}"} || status bench "c$c-w$w" "FAILED a client failed"
  done
done

for a in "${!SPID[@]}"; do stop_arm "${SPID[$a]}"; done   # only our arms
python3 "$H/summarize.py" results --arms "$ARMS" --windows "$WINDOWS" --status results/STATUS.tsv > RESULTS.md; srv=$?; cat RESULTS.md
REQ_ARMS="q32-U q32-T q32-K ${Q32_EXTRA_ARMS:-}"   # a requested exploratory arm (F/C) is required to RUN
fail=$srv; for a in $REQ_ARMS; do up $a || { echo "REQUIRED arm $a not ready"; fail=1; }; done
# every FAILED step of a required arm (ready, import/provenance, profile start/load/stop, evidence, bench) fails the run
while IFS=$'\t' read -r a step st; do
  [[ " $REQ_ARMS " == *" $a "* ]] || continue
  [[ $st == FAIL* ]] && { echo "REQUIRED $a $step: $st"; fail=1; }
done < results/STATUS.tsv
if awk -F'\t' '$2=="evidence" && $3 ~ /^FAILED/{f=1} END{exit !f}' results/STATUS.tsv 2>/dev/null; then echo "profile evidence incomplete"; fail=1; fi
if awk -F'\t' '$2=="provenance" && $3 ~ /^FAIL/{f=1} END{exit !f}' results/STATUS.tsv 2>/dev/null; then echo "provenance check failed"; fail=1; fi
echo "RUN $([ $fail = 0 ] && echo OK || echo INCOMPLETE)"; exit $fail

#!/usr/bin/env bash
# Run 2 (Qwen3-32B diagnostic), one trn2.48xlarge. Every arm on its own 8 logical cores (2 chips) and 36 host vCPUs,
# identical config (BF16, TP8, 32K max context, equal 7 GiB headroom, buckets [1024,8192] x [1,2,4,8]):
#   q32-U  = pinned upstream (baseline)          q32-Ut = upstream + decode_context_length_buckets [2048, 8192]
#   q32-F  = fork, dynamic FA 8K tile (exploratory)   q32-K = fork, in-kernel KV write (correctness replay only)
# Order:
#   1. launch all arms; ready = /health + one real completion, deadline Q32_READY_SECS. No U = nothing comparable:
#      stop at once.
#   2. provenance gate.
#   3. correctness: cache check c1/c3; fixed-batch replay (U twice); needles 8,204 / 9,644 / 24,044.
#   4. K stopped (replay-only).
#   5. profile, U first; a failed U capture skips the rest. Profile failure marks stage times unknown, never fails
#      the run.
#   6. timed windows: 1024/256 at c1 and c8; 4K-6K/256 at c8.
#   7. runner/graph record.
#   8. gated summary.
# Exit 0 when every timed arm ran every step; correctness results (e.g. F differing) are findings, not run failures.
set -uo pipefail
H=$(cd "$(dirname "$0")" && pwd); F=$H/../five; C=$F/clients.py
source "$H/../lib.sh"
fresh_dir results; fresh_dir profiles; mkdir -p logs
export EVIDENCE_DIR=${EVIDENCE_DIR:-$PWD/evidence}; mkdir -p "$EVIDENCE_DIR"
M=${MODELS_DIR:-$HOME/models}; TOK=$M/Qwen3-32B
TIMED=${Q32_TIMED_ARMS:-"q32-U q32-Ut q32-F"}; REPLAY_ONLY=${Q32_REPLAY_ARMS:-"q32-K"}; ARMS="$TIMED $REPLAY_ONLY"
WINDOWS=${Q32_WINDOWS:-3}; READY_SECS=${Q32_READY_SECS:-3000}
[[ $WINDOWS =~ ^[1-9][0-9]*$ ]] || { echo "Q32_WINDOWS must be a positive integer (got '$WINDOWS')"; exit 2; }
[[ " $TIMED " == *" q32-U "* ]] || { echo "q32-U must be a timed arm (it is the reference)"; exit 2; }
declare -A PORT CORE CPU; i=0
for a in $ARMS; do   # slots in launch order: 4 arms x 36 vCPUs = 144 of 192
  PORT[$a]=$((8100 + i)); CORE[$a]="$((i * 8))-$((i * 8 + 7))"; CPU[$a]=$(cpu_range $((i * 36)) $((i * 36 + 35))); i=$((i + 1))
done
pin() { taskset -c "${CPU[$1]}" "${@:2}"; }
status() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> results/STATUS.tsv; }
url() { echo "http://127.0.0.1:${PORT[$1]}"; }

: "${UPSTREAM_PY:?set UPSTREAM_PY to the upstream venv interpreter}" "${UPSTREAM_DIR:?set UPSTREAM_DIR to the pinned upstream source}"
FORK_DIR=${FORK_DIR:-$(cd "$H/../.." && pwd)}
export COVEX_PROVENANCE_DIR=$PWD/provenance ARM_CACHE_BASE=${ARM_CACHE_BASE:-$PWD/cache} UPSTREAM_PY
fresh_dir "$COVEX_PROVENANCE_DIR"; mkdir -p "$ARM_CACHE_BASE"
declare -A SPID UP RPID
for a in $ARMS; do Q32_CORES=${CORE[$a]} PROFILE=1 MODELS_DIR=$M pin $a "$F/serve.sh" $a ${PORT[$a]} & SPID[$a]=$!; done
# One boot-relative readiness deadline for every arm (Q32_READY_DEADLINE, epoch seconds; set by the payload to boot + 20
# min). Health and the first real completion both count against it. A watchdog stops waiting at the deadline, and the
# first arm that fails ends the run at once: every arm is required, so there is nothing useful to measure without it.
DEADLINE=${Q32_READY_DEADLINE:-$(( $(date +%s) + READY_SECS ))}
for a in "${!SPID[@]}"; do
  ( left=$(( DEADLINE - $(date +%s) )); [ $left -lt 1 ] && left=1
    why=$(WAIT_READY_SMOKE_SECS=$left wait_ready "${SPID[$a]}" "${PORT[$a]}" "$left" gen m); rc=$?
    printf '%s\t%s\n' "$rc" "$why" > "logs/$a.ready" ) & RPID[$a]=$!
done
pending="${!RPID[*]}"
while [ -n "$pending" ]; do
  next=""
  for a in $pending; do
    if [ -s "logs/$a.ready" ]; then IFS=$'\t' read -r rc why < "logs/$a.ready"
      if [ "$rc" = 0 ]; then UP[$a]=1; status $a ready OK; echo "ARM $a UP"
      else status $a ready "FAILED $why"; echo "ARM $a FAILED: $why"; NOTREADY=1; fi
    else next="$next $a"; fi
  done
  pending=${next# }
  [ -n "${NOTREADY:-}" ] && break
  if [ -n "$pending" ] && [ "$(date +%s)" -ge "$DEADLINE" ]; then
    for a in $pending; do status $a ready "FAILED not ready by the boot-relative deadline"; echo "ARM $a not ready by deadline"; done
    NOTREADY=1; break; fi
  [ -n "$pending" ] && sleep 2
done
for a in "${!RPID[@]}"; do kill "${RPID[$a]}" 2>/dev/null; done
up() { [ -n "${UP[$1]:-}" ]; }
finish() {   # summary + exit status; called on every path. Nonzero unless every required step of every arm ran.
  for a in "${!SPID[@]}"; do stop_arm "${SPID[$a]}"; done
  python3 "$H/summarize.py" results --arms "$TIMED" --replay-arms "$REPLAY_ONLY" --windows "$WINDOWS" \
    --workloads "c1 c8 m5k-c8" --run2 --required "$(for a in $TIMED; do [ $a = q32-U ] || printf '%s:q32-U ' $a; done)" \
    --status results/STATUS.tsv > RESULTS.md; local srv=$?
  cat RESULTS.md
  local fail=0
  [ $srv = 0 ] || { echo "summary: required comparisons incomplete or invalid (exit $srv)"; fail=1; }
  grep -q '^## Best eligible' RESULTS.md || { echo "summary malformed"; fail=1; }
  for a in $ARMS; do awk -F'\t' -v a="$a" '$1==a && $2=="ready" && $3=="OK"{f=1} END{exit !f}' results/STATUS.tsv 2>/dev/null \
    || { echo "REQUIRED arm $a not ready"; fail=1; }; done
  # a FAILED step of any arm fails the run; profile/evidence are diagnostic (stage times unknown), not fatal
  while IFS=$'\t' read -r a step st; do
    [[ $step == profile* || $step == evidence* ]] && continue
    [[ $st == FAIL* ]] && { echo "REQUIRED $a $step: $st"; fail=1; }
  done < results/STATUS.tsv
  # every arm's replay, and every timed arm's needles and windows, must exist (a step that never ran writes no FAIL)
  for a in $ARMS; do [ -s results/$a.replay.json ] || { echo "REQUIRED $a replay: no result"; fail=1; }; done
  [ -s results/q32-U.replay2.json ] || { echo "REQUIRED q32-U repeat replay: no result"; fail=1; }
  for a in $TIMED; do [ -s results/$a.needle.json ] || { echo "REQUIRED $a needles: no result"; fail=1; }
    for tag in c1 c8 m5k-c8; do for w in $(seq 1 "$WINDOWS"); do
      [ -s results/$a-$tag-w$w.json ] || { echo "REQUIRED $a window $tag-w$w: missing"; fail=1; }; done; done; done
  echo "RUN $([ $fail = 0 ] && echo OK || echo INCOMPLETE)"; exit $fail
}
[ -n "${NOTREADY:-}" ] && { echo "an arm is not ready: stopping now"; finish; }
up q32-U || { echo "q32-U not ready: nothing to compare against; stopping now"; finish; }

# --- provenance gate ---------------------------------------------------------------------------------------------
exp=(); for a in "${!UP[@]}"; do case $a in q32-U|q32-Ut) exp+=("$a=$UPSTREAM_DIR${UPSTREAM_PREFIX:+@$UPSTREAM_PREFIX}") ;;
  *) exp+=("$a=$FORK_DIR${FORK_PREFIX:+@$FORK_PREFIX}") ;; esac; done
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
up q32-U || { echo "q32-U failed provenance; stopping now"; finish; }

# --- correctness (all arms incl. K) -------------------------------------------------------------------------------
for conc in 1 3; do P=()   # exact-length prompts at block/bucket/tile boundaries, c1 and padded c3
  for a in $ARMS; do up $a && { pin $a python3 $C cachecheck ${PORT[$a]} results/$a.cache.c$conc.json $conc 32 & P+=($!); }; done
  wait_all ${P[@]+"${P[@]}"} || status correctness "cache-c$conc" "FAILED a client failed"
  for a in $ARMS; do [ $a = q32-U ] && continue
    [ -s results/q32-U.cache.c$conc.json ] && [ -s results/$a.cache.c$conc.json ] || continue
    python3 $C match results/q32-U.cache.c$conc.json results/$a.cache.c$conc.json > results/$a.vs.q32-U.cache.c$conc.match.json
  done
done
replay() {   # a replay is valid only with all 8 prompts completed; anything else is a tool/serving failure
  pin $1 python3 "$H/replay.py" run --base-url "$(url $1)" --model m --out "results/$2" --batch 8 --tokens 32 --seed 1 \
    > "logs/$1.$2.log" 2>&1
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get("completed")==8 and len(d.get("prompts",[]))==8 else 1)' \
    "results/$2" 2>/dev/null || { status $1 replay "FAILED $(tail -c 200 "logs/$1.$2.log" | tr '\n\t' '  ')"; rm -f "results/$2"; }
}
P=(); for a in $ARMS; do up $a && { replay $a $a.replay.json & P+=($!); }; done; wait_all ${P[@]+"${P[@]}"}
replay q32-U q32-U.replay2.json   # U twice: the nondeterminism floor for attributing any diff
for a in $ARMS q32-U.rerun; do
  ref=results/q32-U.replay.json; got=results/$a.replay.json; [ $a = q32-U.rerun ] && got=results/q32-U.replay2.json
  [ $a = q32-U ] && continue; [ -s $ref ] && [ -s $got ] || continue
  python3 "$H/replay.py" compare $ref $got --out results/$a.vs.q32-U.replay.match.json > /dev/null 2>> logs/replay-compare.err
done
P=(); for a in $TIMED; do up $a && { ( pin $a python3 "$H/needles.py" --base-url "$(url $a)" --model m --tokenizer "$TOK" \
    --lengths 8204,9644,24044 --out results/$a.needle.json > logs/$a.needle.log 2>&1; rc=$?
    # A wrong answer is a finding (pass=false, a real answer in "got"). A tool/serving error (no JSON, wrong lengths, a
    # row without an answer) is a failed step.
    v=$(python3 - "results/$a.needle.json" <<'PY' 2>/dev/null
import json, sys
d = json.load(open(sys.argv[1])); rows = d.get("results") or []
ok = sorted(r.get("len") for r in rows) == [8204, 9644, 24044] and all(isinstance(r.get("got"), str) for r in rows)
print("ANSWERED" if ok else "TOOL " + str(d.get("reason") or [r.get("reason") for r in rows if not isinstance(r.get("got"), str)])[:150])
json.dump({"pass": bool(d.get("pass")) and ok, "n": len(rows), "lengths": [r.get("len") for r in rows]},
          open(sys.argv[1].replace(".needle.json", ".vs.q32-U.needle.match.json"), "w"))
PY
)
    [ "$v" = ANSWERED ] || status $a needles "FAILED rc=$rc ${v:-no result}" ) & P+=($!); }; done
wait_all ${P[@]+"${P[@]}"}
for a in $REPLAY_ONLY; do [ -n "${SPID[$a]:-}" ] && stop_arm "${SPID[$a]}"; unset "UP[$a]"; done   # replay-only arms are done

llm() { # arm conc prompts tag [input-len range-ratio]
  pin $1 vllm bench serve --backend vllm --base-url "$(url $1)" --model m --tokenizer "$TOK" \
    --dataset-name random --random-input-len "${5:-1024}" --random-range-ratio "${6:-0}" --random-output-len 256 --ignore-eos \
    --max-concurrency "$2" --num-prompts "$3" --seed 1 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90 \
    --save-result --save-detailed --result-dir results --result-filename "$1-$4.json" > "logs/$1-$4.bench.log" 2>&1
  local rc=$?; status $1 "bench-$4" "$([ $rc = 0 ] && echo OK || echo "FAILED rc=$rc")"; return $rc
}
# --- profile: U first; diagnostic only ----------------------------------------------------------------------------
PROF_OK=1
for a in $TIMED; do up $a || continue
  [ $PROF_OK = 1 ] || { status $a profile "SKIPPED (U capture failed: stage times unknown)"; continue; }
  llm $a 8 8 profwarm
  L0=$(wc -l < "logs/$a.server.log"); lrc=0
  if http_post "$(url $a)/start_profile" 60; then
    llm $a 8 8 prof || lrc=1
    http_post "$(url $a)/stop_profile" 300 || lrc=1   # always stopped, even when the load client failed
  else lrc=1; fi
  perr=$(tail -n +"$((L0 + 1))" "logs/$a.server.log" | grep -iE "profil" | grep -iE "error|exception|traceback" | head -1 | cut -c1-200)
  [ -n "$perr" ] && lrc=1
  if [ $lrc = 0 ] && why=$(traces_ok "profiles/$a" "${Q32_TP:-8}") && nwhy=$(nonempty_dir "$ARM_CACHE_BASE/$a"); then
    save_evidence "profile-$a" "profiles/$a" "logs/$a.server.log" "logs/$a.meta" || status $a evidence "FAILED upload/copy"
    save_evidence "neff-$a" "$ARM_CACHE_BASE/$a" || status $a evidence "FAILED neff upload"
    status $a profile OK
  else
    status $a profile "FAILED (stage times unknown): ${perr:-${why:-${nwhy:-start/load/stop failed}}}"
    [ $a = q32-U ] && PROF_OK=0
  fi
  [ -n "${SPID[$a]:-}" ] && ! kill -0 "${SPID[$a]}" 2>/dev/null && { unset "UP[$a]"; status $a ready "FAILED died during profile"; }
done
# --- timed windows: all timed arms run each window at the same time ------------------------------------------------
# vLLM 0.29 applies a single --random-range-ratio to input AND output lengths (datasets/utils.py _resolve_range_ratios);
# the JSON form varies only the input, so every request still produces exactly 256 tokens (checked by bench_ok).
for spec in "c1 1 8 1024 0" "c8 8 32 1024 0" 'm5k-c8 8 16 5120 {"input":0.2,"output":0}'; do read -r tag c n il rr <<< "$spec"
  for w in warm $(seq 1 "$WINDOWS"); do P=()
    if [ -n "${Q32_DEADLINE:-}" ] && [ "$(date +%s)" -ge "$Q32_DEADLINE" ]; then   # leave time to summarize + upload
      status bench "$tag-w$w" "FAILED skipped: run deadline reached"; continue; fi
    for a in $TIMED; do up $a && { llm $a $c $n $tag-w$w $il $rr & P+=($!); }; done
    wait_all ${P[@]+"${P[@]}"} || status bench "$tag-w$w" "FAILED a client failed"
  done
done
# --- runner / graph record from the server logs ---------------------------------------------------------------------
for a in $ARMS; do L=logs/$a.server.log; [ -s $L ] || continue
  r=$(grep -q "Using the standalone V2 Neuron model runner" $L && echo MRv2 || { grep -q "Neuron Model Runner V2 does not yet support" $L && echo MRv1 || echo unknown; })
  b=$(grep -m1 -oE "Using decode_context_length_buckets from config: .*" $L | cut -d: -f2- | tr -d ' ' ); b=${b:-none}
  k=$(grep -m1 -oE "KV cache budget: .*" $L | cut -c1-160); k=${k:-unknown}
  # the selected decode graph per workload is not logged at INFO by e16d424 / the fork: recorded as unknown, never guessed
  printf '%s\t%s\t%s\t%s\t%s\n' "$a" "$r" "$b" "c1/c8/m5k-c8 decode graph: unknown (not logged)" "$k" >> results/runner.tsv
done
finish

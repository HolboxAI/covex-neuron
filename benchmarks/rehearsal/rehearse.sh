#!/usr/bin/env bash
# x86 CPU rehearsal, no Trainium. Two tracks run at the same time:
#  A. CPU mode (VLLM_NEURON_CPU_MODE=1): every arm with its exact flags and config, on small checkpoints of the same
#     architecture, plus real requests through the real clients. Catches engine/runtime errors (e.g. the emb-C recompile,
#     ASR multi-prefill rejection). NKI kernels run as their PyTorch reference here, so kernel numerics are not tested.
#  B. CPU compile (VLLM_NEURON_CPU_COMPILE=1, target trn2): real checkpoints, exact flags and config, real neuronx-cc.
#     Catches trace/compile/HBM errors (e.g. q32-C NCC_EOOM002, dynamo trace errors). No execution.
# Writes REHEARSAL.tsv (arm, track, PASS/FAIL, reason) and REHEARSAL.md. Every wait is bounded.
set -uo pipefail
H=$(cd "$(dirname "$0")/.." && pwd); F=$H/five; A=$H/asr; C=$F/clients.py; D=$HOME/data/sixdata
export CLIENT_DIR=$HOME/qwen3-asr-inference-cuda-trainium
# fresh results every attempt: a stale probe output must never reach a verdict
[ -d results ] && [ -n "$(ls -A results 2>/dev/null)" ] && mv results "results.prev.$(date +%s)"
mkdir -p results logs; : > REHEARSAL.tsv
# A recompile word, not a substring: "Precompile: OFF" (an info line in every successful stock log) must not match.
RECOMPILE_RE='fail_on_recompile|(^|[^[:alnum:]_])recompil'

port() { case $1 in q32-T) echo 8000;; q32-C) echo 8001;; q32-U) echo 8008;; q32-K) echo 8009;; q32-F) echo 8010;; q32-Ut) echo 8011;; asr-T) echo 8002;; asr-C) echo 8003;;
                    emb-T) echo 8004;; emb-C) echo 8005;; vl-T) echo 8006;; vl-C) echo 8007;; esac; }
log_of() { case $1 in asr-T) echo "logs/S2-$2-T.server.log";; asr-C) echo "logs/DL-$2-C.server.log";; *) echo "logs/$1-$2.server.log";; esac; }
NCPU=$(nproc)
cpuset() {  # first last -> that range, or all CPUs when the host is smaller (taskset rejects absent CPUs)
  [ "$2" -lt "$NCPU" ] && echo "$1-$2" || echo "0-$((NCPU - 1))"; }
serve() {  # track arm port_offset modeldir cpus
  local tr=$1 a=$2 p=$(( $(port $2) + $3 )) md=$4 cpus=$5
  case $a in
    asr-T) SOLO=1 LOG_TAG=-$tr-T MODEL_DIR=$md/Qwen3-ASR-1.7B taskset -c "$cpus" "$A/serve.sh" S2 2 $p ;;
    asr-C) SOLO=1 LOG_TAG=-$tr-C MODEL_DIR=$md/Qwen3-ASR-1.7B taskset -c "$cpus" "$A/serve.sh" DL 3 $p ;;
    *)     MODELS_DIR=$md LOG_TAG=-$tr taskset -c "$cpus" "$F/serve.sh" $a $p ;;
  esac
}
first_err() { grep -m1 -E "NCC_E[A-Z]+[0-9]+|Error:|Exception:|Traceback|FAILED|Killed|out of memory" "$1" 2>/dev/null \
              | sed -E 's/^\([^)]*\) //' | cut -c1-220 | tr '\t' ' '; }
record() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" | tee -a REHEARSAL.tsv; }
declare -A SPID
wait_up() {  # arm port seconds -> 0 up, 1 died, 2 timeout. Liveness = the serve PID (serve.sh execs vllm, same PID);
  # never pgrep for "--port": that string only appears after exec, so an early check calls a starting server dead.
  for _ in $(seq 1 $(( $3 / 10 ))); do
    curl -sf "localhost:$2/health" > /dev/null && return 0
    kill -0 "${SPID[$1]}" 2> /dev/null || return 1
    sleep 10
  done; return 2
}
need() {  # wait <= 25 min for the real checkpoint download marker
  local m; case $1 in q32-*) m=Qwen3-32B;; vl-*) m=Qwen3-VL-32B-Instruct;; emb-*) m=Qwen3-Embedding-8B;; asr-*) m=Qwen3-ASR-1.7B;; esac
  for _ in $(seq 1 150); do [ -f "$HOME/models/$m/.done" ] && return 0; sleep 10; done; return 1; }
killtree() { local c; for c in $(pgrep -P "$1" 2>/dev/null); do killtree "$c" "$2"; done; kill "-$2" "$1" 2>/dev/null; true; }
stop() {  # port arm: kill the arm's process tree by PID (no command-line patterns; they can match other processes)
  [ -n "${2:-}" ] && [ -n "${SPID[$2]:-}" ] || return 0
  killtree "${SPID[$2]}" TERM; local i; for i in 1 2 3 4 5 6; do kill -0 "${SPID[$2]}" 2>/dev/null || break; sleep 1; done
  killtree "${SPID[$2]}" KILL; true; }

in_pairs() { [[ " ${PAIRS[*]} " == *" $1 "* ]]; }          # exact arm name: q32-U must not match q32-Ut
is_one_of() { local x=$1; shift; [[ " $* " == *" $x "* ]]; }
LONG_ARMS=${REHEARSAL_LONG_ARMS:-"q32-U q32-Ut q32-F"}

# ---------------- Track A: CPU mode, small same-architecture checkpoints ----------------
cpu_probe() {  # arm port
  case $1 in
    # same token-id API as the Trainium cache check (return_token_ids), sized for CPU: PROBE_N prompts x PROBE_TOKENS ids, c3
    q32-*) rm -f results/$1.cpu.gen.json; timeout 900 python3 $C cachecheck $2 results/$1.cpu.gen.json 3 "${PROBE_TOKENS:-8}" ;;
    emb-*) timeout 900 python3 $C emb $2 $D/embed/t256.jsonl 32 32 results/$1.cpu.json ;;   # 32 concurrent -> packed prefills on C
    vl-*)  timeout 900 python3 $C vl $2 $D/vl 4 2 results/$1.cpu.json ;;
    asr-*) ROUNDS=1 SOLO=1 timeout 900 "$A/bench.sh" $1-cpu $2 4 1,4 ;;
  esac
}
track_a() {
  # CPU decode runs ~1 tok/s: probes check that real requests complete, not speed (4 prompts x 8 tokens, 4 images, 4 clips).
  export VLLM_NEURON_CPU_MODE=1 REHEARSAL_EXTRA="--gpu-memory-utilization 0.2" PROBE_N=4 PROBE_TOKENS=8
  # per-process provenance (benchmarks/provenance): proves U runs pinned upstream and T/K the fork, before any Trainium
  export COVEX_PROVENANCE_DIR=$PWD/provenance-cpu ARM_CACHE_BASE=$HOME/arm_cache_cpu; rm -rf "$COVEX_PROVENANCE_DIR"
  # REHEARSAL_PAIRS_A / _B (";"-separated) narrow the arms, e.g. for the CI harness test.
  # No vl-* in CPU mode: vllm-neuron's torch fallback for block-KV attention rejects GQA with 2 KV heads/rank
  # (Qwen3-VL at TP4; rehearsal 1). VL is covered by the compile track; its runtime only on Trainium.
  IFS=";" read -r -a PAIRS <<< "${REHEARSAL_PAIRS_A:-q32-T q32-K;q32-U q32-Ut;q32-F}"
  for pair in "${PAIRS[@]}"; do
    i=0; for a in $pair; do serve cpu $a 100 $HOME/models-small "$(cpuset $((i*8)) $((i*8+7)))" & SPID[$a]=$!; i=$((i+1)); done
    for a in $pair; do p=$(( $(port $a) + 100 )); L=$(log_of $a cpu)
      wait_up $a $p 1200; rc=$?
      if [ $rc != 0 ]; then record $a cpu FAIL "$([ $rc = 2 ] && echo 'not up in 20 min' || { [ -s "$L" ] && echo 'server died' || echo 'did not start (empty log)'; }): $(first_err $L)"; stop $p $a; continue; fi
      if cpu_probe $a $p > logs/$a.cpu.probe.log 2>&1 && curl -sf "localhost:$p/health" > /dev/null; then record $a cpu PASS "served real requests"
      else record $a cpu FAIL "requests: $(first_err $L)$(tail -1 logs/$a.cpu.probe.log | cut -c1-120)"; fi
      if is_one_of $a $LONG_ARMS; then   # prompts over max_num_batched_tokens (8192): segmented prefill, unproven in run 1
        if lr=$(timeout 2400 python3 "$H/rehearsal/long_prompt.py" $p "${LONG_LENGTHS:-9644,24044}" 8 2> logs/$a.cpu.long.log) \
           && curl -sf "localhost:$p/health" > /dev/null; then record $a long-prompt PASS "$lr"
        else record $a long-prompt FAIL "$(tail -1 logs/$a.cpu.long.log | cut -c1-160) $(first_err $L)"; fi
      fi
      stop $p $a
    done
  done
  # provenance verdict for the q32 arms that ran (U against UPSTREAM_DIR, the rest against this checkout)
  exp=(); for a in q32-U q32-Ut q32-T q32-K q32-C q32-F; do in_pairs $a || continue
    case $a in q32-U|q32-Ut) exp+=("$a=${UPSTREAM_DIR:-/nonexistent}");; *) exp+=("$a=${FORK_DIR:-$(cd "$H/.." && pwd)}");; esac; done
  if [ ${#exp[@]} -gt 0 ]; then
    python3 "$H/provenance/verify.py" "$COVEX_PROVENANCE_DIR" "${REHEARSAL_MIN_PROCS:-9}" "${exp[@]}" > results/provenance-cpu.tsv
    while IFS=$'\t' read -r a st why; do record "$a" provenance "$st" "$why"; done < results/provenance-cpu.tsv
  fi
  unset COVEX_PROVENANCE_DIR
  # greedy agreement with the stock fork arm, from THIS attempt's probes: exact count, non-empty, identical
  for a in q32-K q32-U q32-Ut q32-C q32-F; do
    in_pairs $a || continue
    if [ -s results/q32-T.cpu.gen.json ] && [ -s results/$a.cpu.gen.json ]; then
      m=$(python3 $C match results/q32-T.cpu.gen.json results/$a.cpu.gen.json 2>&1)
      v=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); n=int(sys.argv[2]); print("PASS" if d.get("pass") and d.get("n")==n else "FAIL")' "$m" "${PROBE_N:-4}" 2>/dev/null || echo FAIL)
      record $a cpu-match "$v" "vs q32-T: $m (expect n=${PROBE_N:-4})"
    else record $a cpu-match FAIL "missing probe output (q32-T or $a)"; fi
  done
}

# ---------------- Track B: CPU compile for trn2, real checkpoints ----------------
track_b() {
  unset VLLM_NEURON_CPU_MODE REHEARSAL_EXTRA
  # per-arm compile caches (serve.sh ARM_CACHE_BASE): U and T compile identical graphs and must not share NEFFs
  export VLLM_NEURON_CPU_COMPILE=1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2 NEURON_LOGICAL_NC_CONFIG=2 \
         ARM_CACHE_BASE=$HOME/neff_cache NEURON_LIBTORCH_PARALLEL_COMPILE_WORKERS=8
  mkdir -p $HOME/neff_cache
  # asr-T/asr-C/emb-T already compiled and served on Trainium (2026-11-27); spend the time on the arms that never did.
  IFS=";" read -r -a PAIRS <<< "${REHEARSAL_PAIRS_B:-q32-K q32-T;q32-U q32-Ut;q32-F}"
  for pair in "${PAIRS[@]}"; do
    for a in $pair; do need $a || { record $a compile FAIL "checkpoint not downloaded"; continue 2; }; done
    i=0; for a in $pair; do
      : > "$(log_of "$a" compile)"  # a launcher that dies before exec must leave no evidence from an earlier attempt
      serve compile $a 200 $HOME/models "$(cpuset $((16+i*24)) $((16+i*24+23)))" & SPID[$a]=$!; i=$((i+1))
    done
    for a in $pair; do p=$(( $(port $a) + 200 )); L=$(log_of $a compile)
      exit_rc=0; wait_up $a $p 1800; rc=$?
      if [ "$rc" = 1 ]; then
        wait "${SPID[$a]}"; exit_rc=$?
        unset 'SPID[$a]'  # the child has been reaped
      fi
      bad=$(grep -m1 -oE "NCC_E[A-Z]+[0-9]+[^.]*" "$L" | cut -c1-200)
      # PASS needs positive evidence of compilation in the log, never just the absence of errors.
      ngraphs=$(grep -cE "Extracting graphs for" "$L" 2>/dev/null); ngraphs=${ngraphs:-0}
      if [ ! -s "$L" ] || [ "$ngraphs" = 0 ]; then record $a compile FAIL "no compile output ($([ -s "$L" ] && echo 'log has no graph extraction' || echo 'empty log')): $(first_err $L)"
      elif [ "$rc" = 1 ] && [ "$exit_rc" != 0 ]; then record $a compile FAIL "launcher exited $exit_rc: ${bad:-$(first_err "$L")}"
      elif [ "$rc" = 2 ]; then record $a compile FAIL "not done in 30 min: ${bad:-$(first_err "$L")}"
      elif ! grep -qF "Model warmup completed:" "$L"; then record $a compile FAIL "no post-compilation completion marker: ${bad:-$(first_err "$L")}"
      elif grep -qiE "$RECOMPILE_RE" "$L"; then record $a compile FAIL "recompile: $(grep -m1 -iE "$RECOMPILE_RE" "$L" | cut -c1-200)"
      elif [ $rc = 0 ] && [ -z "$bad" ]; then record $a compile PASS "compiled, server up"
      elif [ $rc = 1 ] && [ -z "$bad" ] && ! grep -qE "Traceback|Error:" "$L"; then record $a compile PASS "compiled, exited clean"
      else record $a compile FAIL "${bad:-$(first_err "$L")}"; fi
      stop $p $a
    done
    # KV aliasing on every lane x rank (alias_check.py). q32-T is the stock reference (upstream code path, flags off):
    # its structure is checked; every other q32 arm (U, K, F, C) must match its mutated-input list lane by lane.
    cfgj=$HOME/models/Qwen3-32B/config.json; LT=$(log_of q32-T compile)
    if [[ " $pair " == *" q32-T "* ]]; then
      res=$(python3 "$H/rehearsal/alias_check.py" "$LT" "$cfgj" 6 8 2>&1)
      # the reference row reports structure only; it is not promoted to PASS on identity grounds
      case "${res%%	*}" in FAIL) st=FAIL;; UNVERIFIED) st=PASS; res="x	reference structure OK (counts, all lanes x ranks consistent): ${res#*	}";; *) st=FAIL; res="x	alias_check error: $res";; esac
      record q32-T compile-alias $st "${res#*	}"
      record q32-T kv-identity UNVERIFIED "compile log has parameter ordinals only; named K/V bindings not available"
    fi
    for a in $pair; do [[ $a == q32-* && $a != q32-T ]] || continue
      if [ $a = q32-Ut ]; then   # extra (batch x context-bucket) decode graphs: lanes differ from stock's 6, so no
        # lane-matched parity; require every lane present on all 8 ranks with full KV aliasing (count + SPMD checks)
        # Expected targets from the pinned compile plan, not from what was observed: 2 prefill buckets + 4 decode batch
        # sizes x 3 contexts (2,048, 8,192 and the 32,768 fallback) = 14 lanes, each on all 8 ranks.
        La=$(log_of $a compile); nl=$(grep -oE "\[trace lane=[0-9]+ " "$La" 2>/dev/null | sort -u | wc -l | tr -d ' ')
        want=${Q32_UT_LANES:-14}
        res=$(python3 "$H/rehearsal/alias_check.py" "$La" "$cfgj" "$want" 8 2>&1)
        case "${res%%	*}" in UNVERIFIED) [ "${nl:-0}" = "$want" ] && st=PASS || st=FAIL; res="x	$nl lanes (exactly $want expected), all ranks, full aliasing; no lane-matched stock reference: ${res#*	}";;
          FAIL) st=FAIL;; *) st=FAIL; res="x	alias_check error: $res";; esac
        record $a compile-alias $st "${res#*	}"; record $a kv-identity UNVERIFIED "count + SPMD only (extra decode-bucket graphs)"; continue
      fi
      if [ -s "$LT" ]; then res=$(python3 "$H/rehearsal/alias_check.py" "$(log_of $a compile)" "$cfgj" 6 8 --ref "$LT" 2>&1)
      else res=$(python3 "$H/rehearsal/alias_check.py" "$(log_of $a compile)" "$cfgj" 6 8 2>&1); fi
      case "${res%%	*}" in PASS|FAIL|UNVERIFIED) st=${res%%	*};; *) st=FAIL; res="x	alias_check error: $res";; esac
      record $a compile-alias $st "${res#*	}"
      record $a kv-identity UNVERIFIED "structural parity only; named K/V bindings not in the compile log"
    done
  done
}

track_a > logs/track_a.log 2>&1 & TA=$!
track_b > logs/track_b.log 2>&1 & TB=$!
wait $TA $TB
{ echo "# CPU rehearsal $(date -u +%F' '%H:%MZ)"; echo; echo "covex-neuron $(git -C "$H/.." rev-parse --short HEAD)"; echo
  echo "| Arm | Track | Result | Detail |"; echo "|---|---|---|---|"
  sed -E 's/\|/\//g; s/\t/ | /g; s/^/| /; s/$/ |/' REHEARSAL.tsv
  echo; echo "Track cpu: CPU mode, small same-architecture checkpoints, real requests (kernels run as PyTorch reference)."
  echo "Track compile: CPU compile for trn2, real checkpoints, real neuronx-cc, no execution."
} > REHEARSAL.md
# Overall: every REQUIRED row must exist and PASS. Known-broken arms keep their detailed rows but are not required.
REQUIRED=${REHEARSAL_REQUIRED:-"q32-T:cpu q32-K:cpu q32-U:cpu q32-T:provenance q32-K:provenance q32-U:provenance q32-K:cpu-match q32-U:cpu-match q32-T:compile q32-K:compile q32-U:compile q32-T:compile-alias q32-K:compile-alias q32-U:compile-alias"}
# a scheduled arm is a required arm: F (dynamic FA, exploratory) must run, pass provenance and compile; only its token
# agreement stays informational (dynamic FA may legitimately change greedy tokens)
if [[ " ${REHEARSAL_PAIRS_A:-q32-F} ${REHEARSAL_PAIRS_B:-q32-F} " == *q32-F* ]] && [ -z "${REHEARSAL_REQUIRED_SET:-}" ]; then
  REQUIRED="$REQUIRED q32-F:cpu q32-F:provenance q32-F:compile q32-F:compile-alias"; fi
# run 2: tuned upstream (decode_context_length_buckets) is a scheduled arm, and long prompts (> 8192 batched tokens)
# must be served by every arm that gets them on Trainium. Arm lists are ";"-separated pairs; compare exact names.
PA=" $(echo "${REHEARSAL_PAIRS_A:-q32-T q32-K;q32-U q32-Ut;q32-F}" | tr ';' ' ') "
PB=" $(echo "${REHEARSAL_PAIRS_B:-q32-K q32-T;q32-U q32-Ut;q32-F}" | tr ';' ' ') "
if [ -z "${REHEARSAL_REQUIRED_SET:-}" ]; then
  [[ "$PA" == *" q32-Ut "* ]] && REQUIRED="$REQUIRED q32-Ut:cpu q32-Ut:provenance q32-Ut:cpu-match"
  [[ "$PB" == *" q32-Ut "* ]] && REQUIRED="$REQUIRED q32-Ut:compile q32-Ut:compile-alias"
  for a in $LONG_ARMS; do [[ "$PA" == *" $a "* ]] && REQUIRED="$REQUIRED $a:long-prompt"; done
fi
overall=0
for r in $REQUIRED; do
  st=$(awk -F'\t' -v a="${r%%:*}" -v t="${r#*:}" '$1==a && $2==t {print $3}' REHEARSAL.tsv | tail -1)
  [ "$st" = PASS ] || { echo "REQUIRED $r: ${st:-MISSING (incomplete)}" | tee -a REHEARSAL.md; overall=1; }
done
echo "OVERALL $([ $overall = 0 ] && echo PASS || echo FAIL)" | tee -a REHEARSAL.md
cat REHEARSAL.md
exit $overall

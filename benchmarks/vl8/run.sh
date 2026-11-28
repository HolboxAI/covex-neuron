#!/usr/bin/env bash
# Qwen3-VL-8B pass 1 (Astra B's VL lane): vanilla stock vs tuned stock vs COVEX, one chip (TP4) per arm, same images,
# questions and c4 load. COVEX = buckets selected from real text+image token counts; tuned stock = the same buckets
# on upstream (attribution). Records completed answers, output tok/s, exact prompt tokens (padding) and the server's
# own per-request prefill / decode time (vLLM /metrics deltas per window), plus the serial image-cost probe.
set -uo pipefail
H=$(cd "$(dirname "$0")" && pwd); F=$H/../five; C=$F/clients.py; D=${VL_DATA:-$HOME/data/sixdata/vl}
source "$H/../lib.sh"
fresh_dir results; mkdir -p logs metrics
ARMS=${VL8_ARMS:-"vl8-U vl8-Ub vl8-C"}; WINDOWS=${VL8_WINDOWS:-3}; N=${VL8_N:-40}; READY=${VL8_READY_SECS:-2400}
: "${UPSTREAM_PY:?}" "${UPSTREAM_DIR:?}"; FORK_DIR=${FORK_DIR:-$(cd "$H/../.." && pwd)}
export COVEX_PROVENANCE_DIR=$PWD/provenance ARM_CACHE_BASE=${ARM_CACHE_BASE:-$PWD/cache} UPSTREAM_PY
fresh_dir "$COVEX_PROVENANCE_DIR"; mkdir -p "$ARM_CACHE_BASE"
declare -A PORT CORE CPU SPID UP RPID; i=0
for a in $ARMS; do PORT[$a]=$((8400 + i)); CORE[$a]="$((i * 4))-$((i * 4 + 3))"; CPU[$a]=$(cpu_range $((i * 24)) $((i * 24 + 23))); i=$((i + 1)); done
pin() { taskset -c "${CPU[$1]}" "${@:2}"; }
status() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> results/STATUS.tsv; }
for a in $ARMS; do VL8_CORES=${CORE[$a]} MODELS_DIR=${MODELS_DIR:-$HOME/models} pin $a "$F/serve.sh" $a ${PORT[$a]} & SPID[$a]=$!; done
for a in "${!SPID[@]}"; do
  ( why=$(wait_ready "${SPID[$a]}" "${PORT[$a]}" "$READY" vl m "$D"); printf '%s\t%s\n' "$?" "$why" > "logs/$a.ready" ) & RPID[$a]=$!
done
for a in "${!RPID[@]}"; do wait "${RPID[$a]}"; IFS=$'\t' read -r rc why < "logs/$a.ready"
  [ "$rc" = 0 ] && { UP[$a]=1; status $a ready OK; } || status $a ready "FAILED $why"; done
up() { [ -n "${UP[$1]:-}" ]; }
exp=(); lg=(); for a in "${!UP[@]}"; do case $a in vl8-U|vl8-Ub) exp+=("$a=$UPSTREAM_DIR${UPSTREAM_PREFIX:+@$UPSTREAM_PREFIX}") ;;
  *) exp+=("$a=$FORK_DIR${FORK_PREFIX:+@$FORK_PREFIX}") ;; esac; lg+=(--log "$a=$PWD/logs/$a.server.log"); done
if [ ${#exp[@]} -gt 0 ]; then
  python3 "$H/../provenance/verify.py" "$COVEX_PROVENANCE_DIR" "${VL8_MIN_PROCS:-6}" --tp 4 "${lg[@]}" "${exp[@]}" > results/provenance.tsv 2> logs/provenance.err; vrc=$?
  for a in "${!UP[@]}"; do row=$(awk -F'\t' -v a="$a" '$1==a' results/provenance.tsv); st=$(printf '%s' "$row" | cut -f2)
    { [ "$(printf '%s\n' "$row" | grep -c .)" = 1 ] && [ "$st" = PASS ] && [ $vrc = 0 ]; } || st=FAIL
    status "$a" provenance "$st $(printf '%s' "$row" | cut -f3-)"; [ "$st" = PASS ] || unset "UP[$a]"; done
fi
scrape() { curl -sf --max-time 10 "http://127.0.0.1:${PORT[$1]}/metrics" > "$2" || : > "$2"; }
P=(); for a in $ARMS; do up $a && { pin $a python3 $C vlstage ${PORT[$a]} "$D" 10 results/$a-stage.json > logs/$a.stage.log 2>&1 & P+=($!); }; done
wait_all ${P[@]+"${P[@]}"} || status stage probe "FAILED a probe failed"
for w in warm $(seq 1 "$WINDOWS"); do P=()
  for a in $ARMS; do up $a || continue
    ( scrape $a metrics/$a-w$w.before.prom
      pin $a python3 $C vl ${PORT[$a]} "$D" "$N" 4 results/$a-c4-w$w.json > logs/$a-w$w.log 2>&1; rc=$?
      scrape $a metrics/$a-w$w.after.prom; exit $rc ) & P+=($!); done
  wait_all ${P[@]+"${P[@]}"} || status bench "c4-w$w" "FAILED a client failed"
done
for a in "${!SPID[@]}"; do stop_arm "${SPID[$a]}"; done
python3 "$H/summarize.py" results metrics --arms "$ARMS" --windows "$WINDOWS" --n "$N" --status results/STATUS.tsv > RESULTS.md; srv=$?
cat RESULTS.md; fail=$srv
for a in $ARMS; do up $a || { echo "REQUIRED arm $a not ready/provenance"; fail=1; }; done
echo "RUN $([ $fail = 0 ] && echo OK || echo INCOMPLETE)"; exit $fail

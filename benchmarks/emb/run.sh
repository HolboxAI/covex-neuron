#!/usr/bin/env bash
# Embedding packing sweep: Qwen3-Embedding-8B, TP8 (2 chips) per arm, identical config, all arms at once.
#   U = pinned upstream (1 prefill/step)  T = fork, 1/step  P4 / P8 / P16 = fork packing N prefills per step
# Per input length (t64, t256, t1024): one warm pass, then EMB_WINDOWS windows of 256 requests at c32.
# Correctness: every returned vector vs U (and vs T), cosine >= 0.999 on ALL vectors (min over the set).
# Exit 0 only when U, T and every requested P arm are ready, provenance-clean, complete and correct.
set -uo pipefail
H=$(cd "$(dirname "$0")" && pwd); F=$H/../five; C=$F/clients.py; D=${EMB_DATA:-$HOME/data/sixdata/embed}
source "$H/../lib.sh"
fresh_dir results; mkdir -p logs
ARMS=${EMB_ARMS:-"emb-U emb-T emb-P4 emb-P8 emb-P16"}; WINDOWS=${EMB_WINDOWS:-3}; LENS=${EMB_LENS:-"t64 t256 t1024"}
[[ $WINDOWS =~ ^[1-9][0-9]*$ ]] || { echo "EMB_WINDOWS must be a positive integer"; exit 2; }
: "${UPSTREAM_PY:?}" "${UPSTREAM_DIR:?}"
FORK_DIR=${FORK_DIR:-$(cd "$H/../.." && pwd)}
export COVEX_PROVENANCE_DIR=$PWD/provenance ARM_CACHE_BASE=${ARM_CACHE_BASE:-$PWD/cache} UPSTREAM_PY
fresh_dir "$COVEX_PROVENANCE_DIR"; mkdir -p "$ARM_CACHE_BASE"
declare -A PORT CORE CPU SPID UP RPID; i=0
for a in $ARMS; do PORT[$a]=$((8200 + i)); CORE[$a]="$((i * 8))-$((i * 8 + 7))"; CPU[$a]=$(cpu_range $((i * 36)) $((i * 36 + 35))); i=$((i + 1)); done
pin() { taskset -c "${CPU[$1]}" "${@:2}"; }
status() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> results/STATUS.tsv; }

for a in $ARMS; do EMB_CORES=${CORE[$a]} MODELS_DIR=${MODELS_DIR:-$HOME/models} pin $a "$F/serve.sh" $a ${PORT[$a]} & SPID[$a]=$!; done
for a in "${!SPID[@]}"; do
  ( why=$(wait_ready "${SPID[$a]}" "${PORT[$a]}" 3600 emb m); rc=$?; printf '%s\t%s\n' "$rc" "$why" > "logs/$a.ready" ) & RPID[$a]=$!
done
for a in "${!RPID[@]}"; do
  wait "${RPID[$a]}"; IFS=$'\t' read -r rc why < "logs/$a.ready"
  [ "$rc" = 0 ] && { UP[$a]=1; status $a ready OK; } || status $a ready "FAILED $why"
done
up() { [ -n "${UP[$1]:-}" ]; }
# provenance gate (same verifier as q32): one verdict per launched arm, crash -> every arm dropped
exp=(); lg=(); for a in "${!UP[@]}"; do [ $a = emb-U ] && exp+=("$a=$UPSTREAM_DIR") || exp+=("$a=$FORK_DIR"); lg+=(--log "$a=$PWD/logs/$a.server.log"); done
if [ ${#exp[@]} -gt 0 ]; then
  python3 "$H/../provenance/verify.py" "$COVEX_PROVENANCE_DIR" "${EMB_MIN_PROCS:-10}" --tp "${EMB_TP:-8}" "${lg[@]}" "${exp[@]}" \
    > results/provenance.tsv 2> logs/provenance.err; vrc=$?
  for a in "${!UP[@]}"; do
    row=$(awk -F'\t' -v a="$a" '$1==a' results/provenance.tsv); st=$(printf '%s' "$row" | cut -f2); why=$(printf '%s' "$row" | cut -f3-)
    { [ "$(printf '%s\n' "$row" | grep -c .)" = 1 ] && [ "$st" = PASS ] && [ $vrc = 0 ]; } || { st=FAIL; why=${why:-"verifier exit $vrc"}; }
    status "$a" provenance "$st${why:+ $why}"; [ "$st" = PASS ] || unset "UP[$a]"
  done
fi

for t in $LENS; do
  for w in warm $(seq 1 "$WINDOWS"); do P=()
    for a in $ARMS; do up $a && { EMB_ALL_VECTORS=$([ $w = 1 ] && echo 1 || echo 0) pin $a python3 $C emb ${PORT[$a]} $D/$t.jsonl 256 32 results/$a-$t-w$w.json & P+=($!); }; done
    wait_all ${P[@]+"${P[@]}"} || status bench "$t-w$w" "FAILED a client failed"
  done
  for ref in emb-U emb-T; do for a in $ARMS; do [ $a = $ref ] && continue
    [ -s results/$ref-$t-w1.json ] && [ -s results/$a-$t-w1.json ] || continue
    EMB_COS_MIN=0.999 python3 $C match results/$ref-$t-w1.json results/$a-$t-w1.json > results/$a.vs.$ref.$t.match.json
  done; done
done

for a in "${!SPID[@]}"; do stop_arm "${SPID[$a]}"; done
python3 "$H/summarize.py" results --arms "$ARMS" --lens "$LENS" --windows "$WINDOWS" --status results/STATUS.tsv > RESULTS.md; srv=$?
cat RESULTS.md; fail=$srv
for a in $ARMS; do up $a || { echo "REQUIRED arm $a not ready"; fail=1; }; done
echo "RUN $([ $fail = 0 ] && echo OK || echo INCOMPLETE)"; exit $fail

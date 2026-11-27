#!/usr/bin/env bash
# Run Dax's benchmark client (pinned commit) against one running arm.
# Usage: bench.sh <ARM> <PORT> [NUM_SAMPLES=200] [CONCURRENCY=1,4,8,16]
set -euo pipefail
ARM=$1; PORT=$2; N=${3:-200}; CONC=${4:-1,4,8,16}; ROUNDS=${ROUNDS:-3}
# The client runs on the same pinned CPU set as its arm (PAIR = PORT - 8000).
PIN=(taskset -c "$(( (PORT - 8000) * 24 ))-$(( (PORT - 8000) * 24 + 23 ))"); [ "${SOLO:-0}" = 1 ] && PIN=()
CLIENT_DIR=${CLIENT_DIR:-$HOME/qwen3-asr-inference-cuda-trainium}
[ -d "$CLIENT_DIR" ] || git clone -q https://github.com/HolboxAI/qwen3-asr-inference-cuda-trainium "$CLIENT_DIR"
[ -d "$CLIENT_DIR/.git" ] && git -C "$CLIENT_DIR" checkout -q 4ce1ddb   # same client and dataset slice as the CUDA (1xL4, 8xL4) baselines
mkdir -p results metrics
curl -sf "http://127.0.0.1:$PORT/health" > /dev/null || { echo "arm $ARM (port $PORT) is not up" >&2; exit 3; }   # never report success for a skipped arm
run_client() {  # out concurrency phases warmup rounds samples
  "${PIN[@]}" python3 "$CLIENT_DIR/benchmark/benchmark.py" --base-url "http://127.0.0.1:$PORT" --model qwen3-asr-1.7b \
    --data-dir "$CLIENT_DIR/data" --num-samples "$6" --concurrency "$2" --phases "$3" --warmup "$4" --rounds "$5" \
    --label "trn2-$ARM" --output "$1"
}
# 1) Warm every phase once BEFORE any counter snapshot: first requests pay one-time loading (51 s on one arm,
#    2026-11-27), which otherwise lands in the server histograms and fakes a stage regression.
run_client "results/$ARM.prewarm.json" 1 transcribe,chat_stream 3 1 4
# 2) One snapshot pair per (concurrency, phase), so server-side stage times are per level, warmup-free.
#    The client's own warmup is 0 here because the server is already warm; its stats exclude warmup either way.
parts=()
for c in ${CONC//,/ }; do for ph in transcribe chat_stream; do
  # Prewarm THIS (phase, concurrency) first: batch-size-specific first-use work (decode bucket, sampler shapes)
  # must not land inside the measured snapshot. 2 waves at this concurrency, discarded.
  run_client "results/$ARM.c$c.$ph.prewarm.json" "$c" "$ph" 0 1 "$(( c * 2 > 4 ? c * 2 : 4 ))"
  curl -sf "http://127.0.0.1:$PORT/metrics" > "metrics/$ARM.c$c.$ph.before.prom" || true
  run_client "results/$ARM.c$c.$ph.json" "$c" "$ph" 0 "$ROUNDS" "$N"
  curl -sf "http://127.0.0.1:$PORT/metrics" > "metrics/$ARM.c$c.$ph.after.prom" || true
  parts+=("results/$ARM.c$c.$ph.json")
done; done
# Merge into the single per-arm JSON that compare.py reads (same schema as one multi-level client call).
python3 - "results/$ARM.json" "${parts[@]}" <<'PY'
import json, sys
out, parts = sys.argv[1], sys.argv[2:]
base = json.load(open(parts[0])); base["phases"] = {}
for f in parts:
    for ph, levels in json.load(open(f))["phases"].items():
        base["phases"].setdefault(ph, []).extend(levels)
json.dump(base, open(out, "w"))
PY

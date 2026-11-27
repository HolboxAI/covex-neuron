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
  ${PIN[@]+"${PIN[@]}"} python3 "$CLIENT_DIR/benchmark/benchmark.py" --base-url "http://127.0.0.1:$PORT" --model qwen3-asr-1.7b \
    --data-dir "$CLIENT_DIR/data" --num-samples "$6" --concurrency "$2" --phases "$3" --warmup "$4" --rounds "$5" \
    --label "trn2-$ARM" --output "$1"
}
scrape() {  # file: /metrics with a deadline; an empty file marks a failed scrape (stage attribution then unavailable)
  curl -sf --max-time 10 "http://127.0.0.1:$PORT/metrics" > "$1" || : > "$1"
}
e2e_count() { awk '/^vllm:e2e_request_latency_seconds_count/ {s += $NF} END {print s + 0}' "$1" 2>/dev/null || echo 0; }
scrape_after() {  # before after expected: metrics publication can lag the last response; poll <= 10 s for the count
  local i; for i in 1 2 3 4 5 6 7 8 9 10; do
    scrape "$2"; [ -s "$1" ] && [ -s "$2" ] || return 0
    awk -v b="$(e2e_count "$1")" -v a="$(e2e_count "$2")" -v n="$3" 'BEGIN {exit !(a - b >= n)}' && return 0
    sleep 1; done
}
# 1) Warm every phase once BEFORE any counter snapshot: first requests pay one-time loading (51 s on one arm,
#    2026-11-27), which otherwise lands in the server histograms and fakes a stage regression.
run_client "results/$ARM.prewarm.json" 1 transcribe,chat_stream 3 1 4
# 2) One snapshot pair per (concurrency, phase): prewarm THIS level, scrape, measure with client warmup 0, scrape.
parts=()
for c in ${CONC//,/ }; do for ph in transcribe chat_stream; do
  run_client "results/$ARM.c$c.$ph.prewarm.json" "$c" "$ph" 0 1 "$(( c * 2 > 4 ? c * 2 : 4 ))"
  scrape "metrics/$ARM.c$c.$ph.before.prom"
  run_client "results/$ARM.c$c.$ph.json" "$c" "$ph" 0 "$ROUNDS" "$N"
  scrape_after "metrics/$ARM.c$c.$ph.before.prom" "metrics/$ARM.c$c.$ph.after.prom" "$(( N * ROUNDS ))"
  parts+=("results/$ARM.c$c.$ph.json")
done; done
# 3) Merge into the single per-arm JSON compare.py reads (phases, raw_results, concurrency_levels), then validate every
#    measured level: the real client exits 0 even when every request failed.
python3 - "results/$ARM.json" "$(( N * ROUNDS ))" "$CONC" "${parts[@]}" <<'PY'
import json, sys
out, expected, concs, parts = sys.argv[1], int(sys.argv[2]), [int(c) for c in sys.argv[3].split(",")], sys.argv[4:]
base = json.load(open(parts[0])); base["phases"] = {}; base["raw_results"] = []; levels = set(); bad = []
for f in parts:
    d = json.load(open(f))
    for ph, lv in d.get("phases", {}).items():
        base["phases"].setdefault(ph, []).extend(lv)
        for r in lv:
            levels.add(r.get("concurrency"))
            if r.get("n_total") != expected or r.get("n_success") != expected or r.get("n_error"):
                bad.append(f"{ph} c{r.get('concurrency')}: success {r.get('n_success')}/{r.get('n_total')} "
                           f"(expected {expected}), errors {r.get('n_error')}")
    raw = d.get("raw_results", [])
    base["raw_results"].extend(raw if isinstance(raw, list) else [raw])
base["concurrency_levels"] = sorted(x for x in levels if x is not None)
# exactly one measured result for every expected (phase, concurrency); an empty or partial merge is invalid
for ph in ("transcribe", "chat_stream"):
    for c in concs:
        k = sum(1 for r in base["phases"].get(ph, []) if r.get("concurrency") == c)
        if k != 1:
            bad.append(f"{ph} c{c}: {k} results (expected exactly 1)")
base["validation"] = {"expected_per_level": expected, "problems": bad}
json.dump(base, open(out, "w"))
for b in bad:
    print("INVALID", b, file=sys.stderr)
sys.exit(4 if bad else 0)
PY

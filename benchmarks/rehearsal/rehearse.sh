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
mkdir -p results logs; : > REHEARSAL.tsv
port() { case $1 in q32-T) echo 8000;; q32-C) echo 8001;; asr-T) echo 8002;; asr-C) echo 8003;;
                    emb-T) echo 8004;; emb-C) echo 8005;; vl-T) echo 8006;; vl-C) echo 8007;; esac; }
log_of() { case $1 in asr-T) echo "logs/S2-$2-T.server.log";; asr-C) echo "logs/DL-$2-C.server.log";; *) echo "logs/$1-$2.server.log";; esac; }
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
wait_up() {  # port seconds -> 0 up, 1 died, 2 timeout
  for _ in $(seq 1 $(( $2 / 10 ))); do
    curl -sf "localhost:$1/health" > /dev/null && return 0
    pgrep -f -- "--port $1( |$)" > /dev/null || return 1
    sleep 10
  done; return 2
}
need() {  # wait <= 25 min for the real checkpoint download marker
  local m; case $1 in q32-*) m=Qwen3-32B;; vl-*) m=Qwen3-VL-32B-Instruct;; emb-*) m=Qwen3-Embedding-8B;; asr-*) m=Qwen3-ASR-1.7B;; esac
  for _ in $(seq 1 150); do [ -f "$HOME/models/$m/.done" ] && return 0; sleep 10; done; return 1; }
stop() { pkill -f -- "--port $1( |$)"; sleep 5; pkill -9 -f -- "--port $1( |$)"; true; }

# ---------------- Track A: CPU mode, small same-architecture checkpoints ----------------
cpu_probe() {  # arm port
  case $1 in
    q32-*) timeout 900 python3 $C gen $2 results/$1.cpu.gen.json ;;
    emb-*) timeout 900 python3 $C emb $2 $D/embed/t256.jsonl 64 32 results/$1.cpu.json ;;   # 32 concurrent -> packed prefills on C
    vl-*)  timeout 900 python3 $C vl $2 $D/vl 4 2 results/$1.cpu.json ;;
    asr-*) ROUNDS=1 SOLO=1 timeout 900 "$A/bench.sh" $1-cpu $2 8 1,4 ;;
  esac
}
track_a() {
  export VLLM_NEURON_CPU_MODE=1 REHEARSAL_EXTRA="--gpu-memory-utilization 0.2"
  for pair in "q32-T q32-C" "emb-T emb-C" "vl-T vl-C" "asr-T asr-C"; do
    i=0; for a in $pair; do serve cpu $a 100 $HOME/models-small "$((i*8))-$((i*8+7))" & i=$((i+1)); done
    for a in $pair; do p=$(( $(port $a) + 100 )); L=$(log_of $a cpu)
      wait_up $p 1200; rc=$?
      if [ $rc != 0 ]; then record $a cpu FAIL "$([ $rc = 2 ] && echo 'not up in 20 min' || echo 'server died'): $(first_err $L)"; stop $p; continue; fi
      if cpu_probe $a $p > logs/$a.cpu.probe.log 2>&1 && curl -sf "localhost:$p/health" > /dev/null; then record $a cpu PASS "served real requests"
      else record $a cpu FAIL "requests: $(first_err $L)$(tail -1 logs/$a.cpu.probe.log | cut -c1-120)"; fi
      stop $p
    done
    case "$pair" in q32*) [ -s results/q32-T.cpu.gen.json ] && [ -s results/q32-C.cpu.gen.json ] && python3 $C match results/q32-T.cpu.gen.json results/q32-C.cpu.gen.json > results/q32.cpu.match.json 2>&1 ;; esac
  done
}

# ---------------- Track B: CPU compile for trn2, real checkpoints ----------------
track_b() {
  unset VLLM_NEURON_CPU_MODE REHEARSAL_EXTRA
  export VLLM_NEURON_CPU_COMPILE=1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2 NEURON_LOGICAL_NC_CONFIG=2 \
         NEURON_LIBTORCH_REMOTE_CACHE=$HOME/neff_cache NEURON_LIBTORCH_PARALLEL_COMPILE_WORKERS=8
  mkdir -p $HOME/neff_cache
  # asr-T/asr-C/emb-T already compiled and served on Trainium (2026-11-27); spend the time on the arms that never did.
  for pair in "q32-C q32-T" "vl-C vl-T" "emb-C"; do
    for a in $pair; do need $a || { record $a compile FAIL "checkpoint not downloaded"; continue 2; }; done
    i=0; for a in $pair; do serve compile $a 200 $HOME/models "$((16+i*24))-$((16+i*24+23))" & i=$((i+1)); done
    for a in $pair; do p=$(( $(port $a) + 200 )); L=$(log_of $a compile)
      wait_up $p 1800; rc=$?
      bad=$(grep -m1 -oE "NCC_E[A-Z]+[0-9]+[^.]*" "$L" | cut -c1-200)
      if [ $rc = 0 ] && [ -z "$bad" ]; then record $a compile PASS "compiled, server up"
      elif [ $rc = 1 ] && [ -z "$bad" ] && ! grep -qE "Traceback|Error:" "$L"; then record $a compile PASS "compiled, exited clean"
      else record $a compile FAIL "${bad:-$( [ $rc = 2 ] && echo 'not done in 30 min: ' )$(first_err $L)}"; fi
      # decode graphs must keep the KV caches as in-place (mutated) inputs; [1] alone = the q32-C aliasing bug
      if [[ $a == q32-* ]]; then n=$(grep -E "Mutated inputs: \[1\], " "$L" | wc -l); m=$(grep -cE "Mutated inputs: \[1, " "$L")
        record $a compile-alias "$([ "$n" = 0 ] && [ "$m" -gt 0 ] && echo PASS || echo FAIL)" "lanes with KV-cache aliasing=$m, without=$n"; fi
      stop $p
    done
  done
}

track_a > logs/track_a.log 2>&1 &
track_b > logs/track_b.log 2>&1 &
wait
{ echo "# CPU rehearsal $(date -u +%F' '%H:%MZ)"; echo; echo "covex-neuron $(git -C "$H/.." rev-parse --short HEAD)"; echo
  echo "| Arm | Track | Result | Detail |"; echo "|---|---|---|---|"
  sed -E 's/\|/\//g; s/\t/ | /g; s/^/| /; s/$/ |/' REHEARSAL.tsv
  echo; echo "Track cpu: CPU mode, small same-architecture checkpoints, real requests (kernels run as PyTorch reference)."
  echo "Track compile: CPU compile for trn2, real checkpoints, real neuronx-cc, no execution."
  [ -f results/q32.cpu.match.json ] && { echo; echo "q32 T vs C greedy match (CPU mode): $(cat results/q32.cpu.match.json)"; }
} > REHEARSAL.md
cat REHEARSAL.md

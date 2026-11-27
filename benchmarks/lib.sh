# Shared process helpers for the benchmark runners (five/run.sh, asr/run_all.sh). Source it; needs bash >= 4.
# Rules learned on Trainium (2026-11-27):
#  - liveness is the PID we started, never `pgrep --port` (misses a server until serve.sh execs vllm);
#  - /health is not readiness: an arm is READY only after a real inference request succeeds;
#  - never a bare `wait` (servers never exit) and never `wait a b` (returns only b's status): use wait_all;
#  - stop only our own arms, by PID tree and port, never `pkill -f "vllm serve"`.
LIB_CLIENTS=${LIB_CLIENTS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/five/clients.py}

killtree() {  # pid signal: signal the pid and all its descendants, children first
  local c; for c in $(pgrep -P "$1" 2>/dev/null); do killtree "$c" "$2"; done
  kill "-$2" "$1" 2>/dev/null; true
}

wait_health() {  # pid port seconds -> 0 healthy, 1 process died, 2 timeout
  local i
  for ((i = 0; i < $3; i += 5)); do
    curl -sf "http://127.0.0.1:$2/health" > /dev/null && return 0
    kill -0 "$1" 2> /dev/null || return 1
    sleep 5
  done
  return 2
}

wait_ready() {  # pid port seconds kind [model] [data] -> 0 ready; prints the reason on failure
  local pid=$1 port=$2 secs=$3 kind=$4 model=${5:-m} data=${6:-} rc
  wait_health "$pid" "$port" "$secs"; rc=$?
  case $rc in 1) echo "process died before /health"; return 1;; 2) echo "no /health within ${secs}s"; return 2;; esac
  # A server can pass /health and still fail its first real request (Qwen3-32B: allocation failure at first decode).
  local out; out=$(timeout 900 python3 "$LIB_CLIENTS" smoke "$kind" "$port" "$model" "$data" 2>&1) \
    || { echo "smoke request failed: $(echo "$out" | tail -1 | cut -c1-200)"; return 3; }
  kill -0 "$pid" 2> /dev/null || { echo "process died after smoke request"; return 1; }
  return 0
}

stop_arm() {  # pid port: stop exactly this arm
  [ -n "${1:-}" ] && killtree "$1" TERM
  pkill -f -- "--port $2( |$)" 2>/dev/null; sleep 3
  [ -n "${1:-}" ] && killtree "$1" KILL
  pkill -9 -f -- "--port $2( |$)" 2>/dev/null; true
}

wait_all() {  # pid... -> 0 only if every pid exited 0 (bash `wait a b` reports only b)
  local rc=0 p; for p in "$@"; do wait "$p" || rc=1; done; return $rc
}

fresh_dir() {  # dir: move any previous contents aside so stale files never reach a summary
  [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ] && mv "$1" "$1.prev.$(date +%s)"
  mkdir -p "$1"
}

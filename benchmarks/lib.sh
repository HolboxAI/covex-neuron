# Shared process helpers for the benchmark runners (five/run.sh, asr/run_all.sh). Source it; needs bash >= 4.
# Rules learned on Trainium (2026-11-27):
#  - liveness is the PID we started, never `pgrep --port` (misses a server until serve.sh execs vllm);
#  - /health is not readiness: an arm is READY only after a real inference request succeeds;
#  - never a bare `wait` (servers never exit) and never `wait a b` (returns only b's status): use wait_all;
#  - stop only our own arms, by PID tree, never by command-line pattern (`pkill -f`).
LIB_CLIENTS=${LIB_CLIENTS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/five/clients.py}

killtree() {  # pid signal: signal the pid and all its descendants, children first
  local c; for c in $(pgrep -P "$1" 2>/dev/null); do killtree "$c" "$2"; done
  kill "-$2" "$1" 2>/dev/null; true
}

wait_health() {  # pid port seconds -> 0 healthy, 1 process died, 2 timeout
  local i
  for ((i = 0; i < $3; i += 5)); do
    curl -sf --max-time 5 "http://127.0.0.1:$2/health" > /dev/null && return 0
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

stop_arm() {  # pid [port]: stop exactly this arm. PID tree only: a `pkill -f "--port N"` pattern also matches any
  # shell whose command line contains that text (it killed the test harness itself), so it is not used.
  [ -n "${1:-}" ] || return 0
  killtree "$1" TERM
  local i; for i in 1 2 3 4 5 6; do kill -0 "$1" 2>/dev/null || break; sleep 1; done
  killtree "$1" KILL; wait "$1" 2>/dev/null; true
}

wait_all() {  # pid... -> 0 only if every pid exited 0 (bash `wait a b` reports only b)
  local rc=0 p; for p in "$@"; do wait "$p" || rc=1; done; return $rc
}

fresh_dir() {  # dir: move any previous contents aside so stale files never reach a summary
  [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ] && mv "$1" "$1.prev.$(date +%s)"
  mkdir -p "$1"
}

cpu_range() {  # first last -> that range, or all CPUs when the host is smaller (taskset rejects absent CPUs)
  local n; n=$(nproc); [ "$2" -lt "$n" ] && echo "$1-$2" || echo "0-$((n - 1))"
}

http_post() {  # url [max_seconds]: POST with a deadline; never hangs a runner
  curl -sf --max-time "${2:-60}" -X POST "$1" > /dev/null
}

save_evidence() {  # tag path...: copy paths into $EVIDENCE_DIR/<tag> and upload at once via $EVIDENCE_UPLOAD <dir> <tag>.
  # Called right after each profile capture, so a spot interruption later cannot take the device evidence with it.
  local tag=$1; shift; local dst=${EVIDENCE_DIR:-evidence}/$tag rc=0
  mkdir -p "$dst"
  local p miss=""; for p in "$@"; do [ -e "$p" ] && cp -a "$p" "$dst/" || { miss="$miss $p"; rc=1; }; done
  [ -n "$miss" ] && echo "evidence $tag: capture missing:$miss" >&2
  if [ -n "${EVIDENCE_UPLOAD:-}" ]; then $EVIDENCE_UPLOAD "$dst" "$tag" || { echo "evidence $tag: upload failed" >&2; rc=2; }; fi
  # rc 1 = capture incomplete (a path the server should have written is missing); rc 2 = upload failed
  printf '%s\t%s\n' "$tag" "$(case $rc in 0) echo OK;; 1) echo "FAILED rc=1 capture missing:$miss";; *) echo "FAILED rc=2 upload";; esac)" \
    >> "${EVIDENCE_DIR:-evidence}/EVIDENCE.tsv"
  return $rc
}

traces_ok() {  # dir min_files: at least min non-empty files under dir (one device trace per TP rank, at minimum)
  local n; n=$(find "$1" -type f -size +0 2>/dev/null | wc -l | tr -d ' '); [ "$n" -ge "$2" ] || { echo "$n non-empty trace files < $2"; return 1; }
}
nonempty_dir() {  # dir: at least one non-empty file (an empty NEFF/cache root is not evidence)
  [ -n "$(find "$1" -type f -size +0 2>/dev/null | head -1)" ] || { echo "no non-empty files in $1"; return 1; }
}

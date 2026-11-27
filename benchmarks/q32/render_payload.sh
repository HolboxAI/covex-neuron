#!/usr/bin/env bash
# Render payload.sh with exact pins. Usage: render_payload.sh <covex_sha> <upstream_sha> <model_rev> <out> [extra_arms] [windows]
# Refuses short or moving refs and any unreplaced @@NAME@@. Never launches anything.
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd); TEMPLATE=${TEMPLATE:-$H/payload.sh}   # TEMPLATE=... renders the rehearsal payload
covex=$1 up=$2 rev=$3 out=$4 extra=${5:-} windows=${6:-3} ready=${READY_SECS:-3000} readymin=${READY_MIN:-20}
for v in "$covex" "$up" "$rev"; do [[ $v =~ ^[0-9a-f]{40}$ ]] || { echo "pin '$v' is not a full 40-hex commit/revision" >&2; exit 2; }; done
[[ $windows =~ ^[1-9][0-9]*$ ]] || { echo "windows must be a positive integer" >&2; exit 2; }
[[ $ready =~ ^[1-9][0-9]*$ ]] || { echo "READY_SECS must be a positive integer" >&2; exit 2; }
[[ $readymin =~ ^[1-9][0-9]*$ ]] && [ "$readymin" -le 60 ] || { echo "READY_MIN must be 1-60 (minutes from boot)" >&2; exit 2; }
for a in $extra; do [[ $a == q32-F || $a == q32-C ]] || { echo "unknown extra arm $a" >&2; exit 2; }; done
sed -e "s/@@COVEX_SHA@@/$covex/g" -e "s/@@UPSTREAM_SHA@@/$up/g" -e "s/@@MODEL_REV@@/$rev/g" \
    -e "s/@@EXTRA_ARMS@@/$extra/g" -e "s/@@WINDOWS@@/$windows/g" -e "s/@@READY_SECS@@/$ready/g" -e "s/@@READY_MIN@@/$readymin/g" "$TEMPLATE" > "$out.tmp"
sha=$(shasum -a 256 "$out.tmp" 2>/dev/null || sha256sum "$out.tmp"); sha=${sha%% *}
sed -e "s/@@PAYLOAD_SHA@@/$sha/g" "$out.tmp" > "$out"; rm -f "$out.tmp"
if grep -q '@@[A-Z_]*@@' "$out"; then echo "unresolved placeholder in $out" >&2; rm -f "$out"; exit 3; fi
bash -n "$out"; echo "rendered $out (payload sha256 before self-reference: $sha)"

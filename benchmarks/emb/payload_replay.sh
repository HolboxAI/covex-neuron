#!/bin/bash
# Embedding pair replay (Qwen3-Embedding-8B, offline engine, pack-8, one arm): which change moves B (trn2.48xlarge Spot, Neuron DLAMI SDK 2.33.1). Rendered by render_payload.sh:
# every double-at placeholder is replaced by an exact pin; the renderer refuses output with any left.
shutdown -h +30   # hard stop: 30 min x $8.60/h Spot = $4.30
BOOT=$(date +%s)
B=covex-neuron-runs-992382417943; R=us-east-2
# IMDSv2 (token) with a UUID fallback: the prefix must be unique even when metadata is unavailable
TOK=$(curl -s --max-time 5 -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 600")
md() { curl -sf --max-time 5 -H "X-aws-ec2-metadata-token: $TOK" "http://169.254.169.254/latest/meta-data/$1"; }
IID=$(md instance-id) && [[ $IID == i-* ]] || IID="noimds-$(cat /proc/sys/kernel/random/uuid)"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$IID"; P="emb-replay/$RUN_ID"     # distinct prefix per attempt: nothing is overwritten
echo "$(md instance-type || echo unknown) $(md ami-id || echo unknown)" > /home/ubuntu/instance.txt 2>/dev/null || true
exec > /var/log/covex-run.log 2>&1
mkdir -p /home/ubuntu/run && chown ubuntu /home/ubuntu/run && echo "$P" > /home/ubuntu/run/PREFIX && echo "$BOOT" > /home/ubuntu/run/BOOT
# periodic sync of results/logs; device evidence is uploaded by the runner itself right after each capture
( while true; do aws s3 sync /home/ubuntu/run s3://$B/$P/ --region $R --only-show-errors --exclude "evidence/*" --exclude "profiles/*";
  aws s3 cp /var/log/covex-run.log s3://$B/$P/run.log --region $R --only-show-errors; sleep 60; done ) &
sudo -iu ubuntu bash <<'EOS'
set -x
fail() { echo "SETUP FAILED $1" | tee ~/run/STATUS; aws s3 cp ~/run/STATUS "s3://$B/$P/STATUS" --region "$R" --only-show-errors; exit 1; }
B=covex-neuron-runs-992382417943; R=us-east-2; P=$(cat ~/run/PREFIX)   # defined here: sudo -i may strip passed vars
V=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0
# Setup shares the boot-relative readiness deadline: a stalled clone, pip or model download stops the attempt
# (status uploaded, then shutdown) instead of running past it before the runner's watchdog starts.
READY_AT=$(( $(cat ~/run/BOOT) + @@READY_MIN@@ * 60 ))
T() { local l=$(( READY_AT + 900 - $(date +%s) )); [ $l -gt 0 ] || { echo "setup past the readiness deadline"; return 124; }; timeout "$l" "$@"; }
cd ~ && T git clone -q https://github.com/HolboxAI/covex-neuron && git -C covex-neuron checkout -q @@COVEX_SHA@@ || fail fork
T git clone -q https://github.com/vllm-project/vllm-neuron upstream && git -C upstream checkout -q @@UPSTREAM_SHA@@ || fail upstream
# Arm U gets its own environment: a copy of the pristine DLAMI venv (same Neuron stack, taken BEFORE the fork is
# installed) with only pinned upstream installed, run by absolute interpreter. The fork goes into the original venv.
T cp -a "$V" ~/venv-upstream && T ~/venv-upstream/bin/python -m pip install -q --no-deps -e ~/upstream || fail "upstream venv"
source "$V/bin/activate"
T pip install -q --no-deps -e covex-neuron && T pip install -q httpx huggingface_hub || fail pip
~/venv-upstream/bin/python -m pip freeze > ~/run/pip_freeze_upstream.txt; pip freeze > ~/run/pip_freeze.txt
T hf download Qwen/Qwen3-Embedding-8B --revision @@MODEL_REV@@ --local-dir ~/models/Qwen3-Embedding-8B > /dev/null && ls ~/models/Qwen3-Embedding-8B/config.json || fail model
mkdir -p ~/data && T aws s3 cp s3://$B/inputs/sixdata.tgz - --region $R | tar xz -C ~/data && ls ~/data/sixdata/embed/t64.jsonl || fail data
# Run manifest: this is one offline fork diagnostic, not a stock-versus-fork throughput or correctness run.
python3 - > ~/run/manifest.json <<PY
import hashlib, json, subprocess as sp
sh = lambda c: sp.run(c, shell=True, capture_output=True, text=True).stdout.strip()
fhash = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()     # hash the saved file bytes
freeze = open("/home/ubuntu/run/pip_freeze.txt").read()
it = (open("/home/ubuntu/instance.txt").read().split() + ["unknown", "unknown"])[:2]
print(json.dumps({"run_prefix": "$P", "instance_type": it[0], "ami": it[1], "region": "$R",
  "covex_neuron": sh("git -C ~/covex-neuron rev-parse HEAD"), "upstream_vllm_neuron": sh("git -C ~/upstream rev-parse HEAD"),
  "upstream_python": sh("readlink -f ~/venv-upstream/bin/python"), "fork_python": sh("readlink -f $(which python3)"),
  "upstream_pip_freeze_sha256": fhash("/home/ubuntu/run/pip_freeze_upstream.txt"),
  "model": "Qwen/Qwen3-Embedding-8B", "model_revision": "@@MODEL_REV@@",
  "neuron_packages": [l for l in freeze.splitlines() if any(k in l.lower() for k in ("neuron", "nki", "libneuronxla", "vllm", "torch"))],
  "pip_freeze_sha256": fhash("/home/ubuntu/run/pip_freeze.txt"),
  "arms": ["offline-emb-P8"], "diagnostic_only": True, "repeat": 3,
  "cases": ["B", "A+B", "A'+B", "B+A", "B+A'", "X+B"], "chips_per_arm": 2,
  "ready_deadline_min_from_boot": int("@@READY_MIN@@"),
  "correctness": "diagnostic of packed-step peer dependence; this run alone cannot establish stock vector parity or claim a speedup",
  "payload_template_render_sha256": "@@PAYLOAD_SHA@@"}, indent=1))   # digest of the rendered script BEFORE this field is filled
PY
aws s3 cp ~/run/manifest.json s3://$B/$P/manifest.json --region $R --only-show-errors
cat > ~/upload_evidence.sh <<UP
#!/bin/bash
aws s3 sync "\$1" "s3://$B/$P/evidence/\$2/" --region $R --only-show-errors
UP
chmod +x ~/upload_evidence.sh
cd ~/run && mkdir -p results logs && export NEURON_VISIBLE_DEVICES=0-7 COVEX_PACK_TRACE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  VLLM_NEURON_RESERVED_HEADROOM_GIB=4 && T python3 ~/covex-neuron/benchmarks/emb/replay_pairs.py --model ~/models/Qwen3-Embedding-8B \
  --dataset ~/data/sixdata/embed/t64.jsonl --tokenizer ~/models/Qwen3-Embedding-8B --packing 8 --repeat 3 \
  --out results/pairs.json --log logs/replay.log >> logs/replay.log 2>&1
echo "RUN_EXIT=$?" | tee ~/run/STATUS
aws s3 sync ~/run s3://$B/$P/ --region $R --only-show-errors --exclude "evidence/*"
echo "ALLDONE $(cat ~/run/STATUS 2>/dev/null)"
EOS
sleep 60; sudo shutdown -h now

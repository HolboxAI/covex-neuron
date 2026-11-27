#!/bin/bash
# Run 2 (Qwen3-32B diagnostic) payload (trn2.48xlarge Spot, Neuron DLAMI SDK 2.33.1). Rendered by render_payload.sh:
# every double-at placeholder is replaced by an exact pin; the renderer refuses output with any left.
shutdown -h +76   # hard stop: 76 min x $8.60/h Spot = $10.9, inside the $11 cap
BOOT=$(date +%s)
B=covex-neuron-runs-992382417943; R=us-east-2
# IMDSv2 (token) with a UUID fallback: the prefix must be unique even when metadata is unavailable
TOK=$(curl -s --max-time 5 -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 600")
md() { curl -sf --max-time 5 -H "X-aws-ec2-metadata-token: $TOK" "http://169.254.169.254/latest/meta-data/$1"; }
IID=$(md instance-id) && [[ $IID == i-* ]] || IID="noimds-$(cat /proc/sys/kernel/random/uuid)"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$IID"; P="q32/$RUN_ID"     # distinct prefix per attempt: nothing is overwritten
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
cd ~ && git clone -q https://github.com/HolboxAI/covex-neuron && git -C covex-neuron checkout -q @@COVEX_SHA@@ || fail fork
git clone -q https://github.com/vllm-project/vllm-neuron upstream && git -C upstream checkout -q @@UPSTREAM_SHA@@ || fail upstream
# Arm U gets its own environment: a copy of the pristine DLAMI venv (same Neuron stack, taken BEFORE the fork is
# installed) with only pinned upstream installed, run by absolute interpreter. The fork goes into the original venv.
cp -a "$V" ~/venv-upstream && ~/venv-upstream/bin/python -m pip install -q --no-deps -e ~/upstream || fail "upstream venv"
source "$V/bin/activate"
pip install -q --no-deps -e covex-neuron && pip install -q httpx huggingface_hub || fail pip
~/venv-upstream/bin/python -m pip freeze > ~/run/pip_freeze_upstream.txt; pip freeze > ~/run/pip_freeze.txt
hf download Qwen/Qwen3-32B --revision @@MODEL_REV@@ --local-dir ~/models/Qwen3-32B > /dev/null && ls ~/models/Qwen3-32B/config.json || fail model
# run manifest: everything needed to reproduce or audit this attempt
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
  "model": "Qwen/Qwen3-32B", "model_revision": "@@MODEL_REV@@",
  "neuron_packages": [l for l in freeze.splitlines() if any(k in l.lower() for k in ("neuron", "nki", "libneuronxla", "vllm", "torch"))],
  "pip_freeze_sha256": fhash("/home/ubuntu/run/pip_freeze.txt"),
  "timed_arms": ["q32-U", "q32-Ut", "q32-F"], "replay_only_arms": ["q32-K"], "windows": int("@@WINDOWS@@"),
  "headroom_gib": 7, "decode_context_length_buckets_Ut": [2048, 8192],
  "workloads": {"c1": {"in": 1024, "out": 256, "n": 8}, "c8": {"in": 1024, "out": 256, "n": 32},
                "m5k-c8": {"in": "5120 +-20%", "out": 256, "n": 16}}, "seed": 1, "dataset": "random",
  "correctness": {"cache": [1, 3], "replay": {"batch": 8, "tokens": 32, "u_twice": True}, "needles": [8204, 9644, 24044]},
  "limits": {"1024/256": "p90 TTFT <= 2 s, p90 TPOT <= 50 ms", "m5k-c8": "<= 1.1x U p90 TTFT and TPOT"},
  "payload_template_render_sha256": "@@PAYLOAD_SHA@@"}, indent=1))   # digest of the rendered script BEFORE this field is filled
PY
aws s3 cp ~/run/manifest.json s3://$B/$P/manifest.json --region $R --only-show-errors
cat > ~/upload_evidence.sh <<UP
#!/bin/bash
aws s3 sync "\$1" "s3://$B/$P/evidence/\$2/" --region $R --only-show-errors
UP
chmod +x ~/upload_evidence.sh
cd ~/run && UPSTREAM_DIR=~/upstream UPSTREAM_PY=~/venv-upstream/bin/python ARM_CACHE_BASE=/var/tmp/covex-arm-cache \
  EVIDENCE_UPLOAD=~/upload_evidence.sh \
  Q32_WINDOWS=@@WINDOWS@@ Q32_READY_SECS=@@READY_SECS@@ Q32_DEADLINE=$(( $(cat ~/run/BOOT) + 68 * 60 )) \
  bash ~/covex-neuron/benchmarks/q32/run2.sh
echo "RUN_EXIT=$?" | tee ~/run/STATUS
aws s3 sync ~/run s3://$B/$P/ --region $R --only-show-errors --exclude "evidence/*"
echo "ALLDONE $(cat ~/run/STATUS 2>/dev/null)"
EOS
sleep 60; sudo shutdown -h now

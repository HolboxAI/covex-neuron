#!/bin/bash
# K9 engine check payload: x86 Neuron DLAMI (m7i.16xlarge), CPU mode, small real Qwen3 (0.6B, untied like the rehearsal).
# Predicted-output decoding (k9-P) must equal plain greedy (k9-T) exactly. Hard stop 45 min; terminates on shutdown.
shutdown -h +45
B=covex-neuron-runs-992382417943; R=us-east-2
P="k9-cpu/$(date -u +%Y%m%dT%H%M%SZ)-$(cat /proc/sys/kernel/random/uuid | cut -c1-8)"
exec > /var/log/covex-run.log 2>&1
mkdir -p /home/ubuntu/run && chown ubuntu /home/ubuntu/run && echo "$P" > /home/ubuntu/run/PREFIX
( while true; do aws s3 sync /home/ubuntu/run s3://$B/$P/ --region $R --only-show-errors; aws s3 cp /var/log/covex-run.log s3://$B/$P/run.log --region $R --only-show-errors; sleep 60; done ) &
sudo -iu ubuntu bash <<'EOS'
set -x
B=covex-neuron-runs-992382417943; R=us-east-2; P=$(cat ~/run/PREFIX)
fail() { echo "SETUP FAILED $1" | tee ~/run/STATUS; aws s3 cp ~/run/STATUS "s3://$B/$P/STATUS" --region "$R" --only-show-errors; exit 1; }
V=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0
source "$V/bin/activate"
cd ~ && git clone -q https://github.com/HolboxAI/covex-neuron && git -C covex-neuron checkout -q @@COVEX_SHA@@ || fail fork
pip install -q --no-deps -e covex-neuron && pip install -q httpx huggingface_hub || fail pip
pip freeze > ~/run/pip_freeze.txt
mkdir -p ~/models-small ~/hf-small
hf download Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca --local-dir ~/hf-small/Qwen3-0.6B > /dev/null || fail "small model"
python3 ~/covex-neuron/benchmarks/rehearsal/untie.py ~/hf-small/Qwen3-0.6B ~/models-small/Qwen3-32B || fail untie
printf '{"covex": "%s", "model": "Qwen/Qwen3-0.6B@c1899de2 (untied)", "payload_template_render_sha256": "@@PAYLOAD_SHA@@", "upstream_unused": "@@UPSTREAM_SHA@@ @@MODEL_REV@@ @@WINDOWS@@ @@READY_MIN@@"}\n' \
  "$(git -C ~/covex-neuron rev-parse HEAD)" > ~/run/manifest.json
cd ~/run && K9_READY_SECS=1800 bash ~/covex-neuron/benchmarks/k9/cpu_check.sh; echo "RUN_EXIT=$?" | tee ~/run/STATUS
aws s3 sync ~/run s3://$B/$P/ --region $R --only-show-errors || true
echo "ALLDONE $(cat ~/run/STATUS 2>/dev/null)"
EOS
sleep 60; sudo shutdown -h now

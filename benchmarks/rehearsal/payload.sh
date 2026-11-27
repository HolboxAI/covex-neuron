#!/bin/bash
# x86 CPU rehearsal payload (m7i.16xlarge, Neuron DLAMI). Hard stop 65 min; the instance terminates on shutdown.
shutdown -h +65
B=covex-neuron-runs-992382417943; R=us-east-2
P="rehearsal/$(date -u +%Y%m%dT%H%M%SZ)-$(cat /proc/sys/kernel/random/uuid | cut -c1-8)"   # unique per attempt
exec > /var/log/covex-run.log 2>&1
mkdir -p /home/ubuntu/run && chown ubuntu /home/ubuntu/run && echo "$P" > /home/ubuntu/run/PREFIX
( while true; do aws s3 sync /home/ubuntu/run s3://$B/$P/ --region $R --only-show-errors --exclude "neff/*"; aws s3 cp /var/log/covex-run.log s3://$B/$P/run.log --region $R --only-show-errors; free -g | sed -n 2p >> /home/ubuntu/run/mem.log; sleep 60; done ) &
sudo -iu ubuntu bash <<'EOS'
set -x
B=covex-neuron-runs-992382417943; R=us-east-2; P=$(cat ~/run/PREFIX)   # the inner shell does not see the outer P
fail() { echo "SETUP FAILED $1" | tee ~/run/STATUS; aws s3 cp ~/run/STATUS "s3://$B/$P/STATUS" --region "$R" --only-show-errors; exit 1; }
V=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0
source "$V/bin/activate"
pip freeze > ~/run/pip_freeze.txt
cd ~ && git clone -q https://github.com/HolboxAI/covex-neuron && git -C covex-neuron checkout -q @@COVEX_SHA@@ || fail "fork"
git clone -q https://github.com/vllm-project/vllm-neuron ~/upstream && git -C ~/upstream checkout -q @@UPSTREAM_SHA@@ || fail "upstream"
# arm U: copy of the pristine venv (before the fork install below) with only pinned upstream installed
cp -a "$V" ~/venv-upstream && ~/venv-upstream/bin/python -m pip install -q --no-deps -e ~/upstream || fail "upstream venv"
cd covex-neuron && git rev-parse HEAD > ~/run/covex_commit.txt && git -C ~/upstream rev-parse HEAD > ~/run/upstream_commit.txt
pip install -q --no-deps -e . && pip install -q httpx soundfile jiwer librosa huggingface_hub || fail pip
mkdir -p ~/data ~/models ~/models-small ~/hf-small ~/qwen3-asr-inference-cuda-trainium
aws s3 cp s3://$B/inputs/sixdata.tgz - --region $R | tar xz -C ~/data
aws s3 cp s3://$B/inputs/asr_client.tgz - --region $R | tar xz -C ~/qwen3-asr-inference-cuda-trainium
aws s3 cp s3://$B/inputs/asr_data.tgz - --region $R | tar xz -C ~/qwen3-asr-inference-cuda-trainium
ls ~/data/sixdata/vl/manifest.json ~/qwen3-asr-inference-cuda-trainium/data/manifest.json || fail data
# Qwen3 only (the spot run's model). Small same-architecture checkpoint for CPU mode, pinned and untied to the 32B
# code path; the real Qwen3-32B at the SAME revision as the spot run for CPU compile. Revisions go into the manifest.
hf download Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca --local-dir ~/hf-small/Qwen3-0.6B > /dev/null || fail "small model"
python3 ~/covex-neuron/benchmarks/rehearsal/untie.py ~/hf-small/Qwen3-0.6B ~/models-small/Qwen3-32B || fail untie
ls ~/models-small/Qwen3-32B/config.json || fail "small model config"
( hf download Qwen/Qwen3-32B --revision @@MODEL_REV@@ --local-dir ~/models/Qwen3-32B > /dev/null && ls ~/models/Qwen3-32B/config.json \
  && touch ~/models/Qwen3-32B/.done ) &
printf '{"qwen3_0_6b_revision": "c1899de289a04d12100db370d81485cdf75e47ca", "qwen3_32b_revision": "@@MODEL_REV@@", "covex": "%s", "upstream": "%s"}\n' \
  "$(git -C ~/covex-neuron rev-parse HEAD)" "$(git -C ~/upstream rev-parse HEAD)" > ~/run/manifest.json
cd ~/run && UPSTREAM_DIR=~/upstream UPSTREAM_PY=~/venv-upstream/bin/python bash ~/covex-neuron/benchmarks/rehearsal/rehearse.sh; echo "REHEARSAL_EXIT=$?" | tee ~/run/STATUS   # synced with the diagnostics
mkdir -p ~/run/neff && cp -r ~/neff_cache ~/run/neff/ 2>/dev/null; aws s3 sync ~/neff_cache s3://$B/compile-cache/rehearsal/ --region $R --only-show-errors || true
aws s3 sync ~/run s3://$B/$P/ --region $R --only-show-errors --exclude "neff/*" || true
echo "ALLDONE $(cat ~/run/STATUS 2>/dev/null)"
EOS
sleep 90; sudo shutdown -h now
